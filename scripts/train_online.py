#!/usr/bin/env python3
"""Online Eagle3 training orchestrator.

Coordinates datagen, sync, and streaming training processes.

Single-node mode:
    python scripts/train_online.py --mode single-node \
        --datagen-gpus 0,1 --train-gpus 2,3,4,5,6,7 \
        --target-model-path /data/models/Qwen3-32B \
        --train-data-path /data/novita/sharegpt.jsonl \
        --output-dir /data/output/qwen3_online

Multi-node mode:
    python scripts/train_online.py --mode multi-node \
        --datagen-nodes host-10-83-115-21 \
        --train-node host-10-83-115-18 \
        --target-model-path /data/models/Qwen3-32B \
        --train-data-path /data/novita/sharegpt.jsonl \
        --output-dir /data/output/qwen3_online
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [orchestrator] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


class ProcessGroup:
    """Manages a group of subprocesses with graceful shutdown."""

    def __init__(self):
        self.processes: list[tuple[str, subprocess.Popen]] = []

    def spawn(self, name: str, cmd: list[str], env: dict | None = None) -> subprocess.Popen:
        merged_env = {**os.environ, **(env or {})}
        log.info(f"Starting [{name}]: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, env=merged_env)
        self.processes.append((name, proc))
        return proc

    def spawn_ssh(self, name: str, host: str, remote_cmd: str) -> subprocess.Popen:
        cmd = ["ssh", "-o", "StrictHostKeyChecking=no", host, remote_cmd]
        return self.spawn(name, cmd)

    def wait_any(self, timeout: float = 5.0) -> tuple[str, int] | None:
        """Check if any process has exited. Non-blocking."""
        for name, proc in self.processes:
            ret = proc.poll()
            if ret is not None:
                return name, ret
        return None

    def shutdown(self):
        """Send SIGTERM to all running processes, then SIGKILL after 10s."""
        for name, proc in self.processes:
            if proc.poll() is None:
                log.info(f"Stopping [{name}] (pid={proc.pid})")
                proc.terminate()
        deadline = time.time() + 10
        for name, proc in self.processes:
            remaining = max(0, deadline - time.time())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                log.warning(f"Force-killing [{name}] (pid={proc.pid})")
                proc.kill()


def run_single_node(args):
    """Single-node: datagen on some GPUs, training on others."""
    pg = ProcessGroup()
    gen_dir = os.path.join(args.output_dir, "gen")
    manifest_path = os.path.join(gen_dir, "manifest.json")
    os.makedirs(gen_dir, exist_ok=True)

    # Start datagen
    datagen_cmd = [
        sys.executable, "scripts/data_generation_offline.py",
        "--target-model-path", args.target_model_path,
        "--train-data-path", args.train_data_path,
        "--output-dir", gen_dir,
        "--manifest-path", manifest_path,
        "--tensor-parallel-size", str(args.datagen_tp),
        "--seq-length", str(args.seq_length),
        "--batch-size", str(args.datagen_batch_size),
    ]
    if args.max_samples:
        datagen_cmd += ["--max-samples", str(args.max_samples)]

    datagen_env = {"CUDA_VISIBLE_DEVICES": args.datagen_gpus}
    pg.spawn("datagen", datagen_cmd, env=datagen_env)

    # Wait for min_samples before starting training
    log.info(f"Waiting for {args.min_samples} samples before starting training...")
    while True:
        from speculators.train import manifest as manifest_mod
        m = manifest_mod.read(manifest_path)
        if len(m["files"]) >= args.min_samples:
            break
        time.sleep(args.poll_interval)

    # Start streaming training
    train_gpus = args.train_gpus
    num_train_gpus = len(train_gpus.split(","))
    train_cmd = [
        "torchrun",
        f"--nproc_per_node={num_train_gpus}",
        "scripts/train_streaming.py",
        "--verifier-name-or-path", args.target_model_path,
        "--manifest-path", manifest_path,
        "--data-path", gen_dir,
        "--save-path", os.path.join(args.output_dir, "checkpoints"),
        "--total-seq-len", str(args.seq_length),
        "--final-epochs", str(args.final_epochs),
        "--min-samples", str(args.min_samples),
        "--lr", str(args.lr),
        "--num-workers", str(args.num_workers),
        "--prefetch-factor", str(args.prefetch_factor),
    ]
    train_env = {"CUDA_VISIBLE_DEVICES": train_gpus}
    pg.spawn("training", train_cmd, env=train_env)

    # Monitor processes
    def handle_signal(sig, frame):
        log.info("Received signal, shutting down...")
        pg.shutdown()
        sys.exit(1)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    while True:
        result = pg.wait_any()
        if result:
            name, retcode = result
            if name == "training":
                log.info(f"Training finished (exit code {retcode})")
                pg.shutdown()
                return retcode
            elif name == "datagen":
                if retcode != 0:
                    log.error(f"Datagen failed (exit code {retcode})")
                    pg.shutdown()
                    return retcode
                log.info("Datagen completed, training continues...")
        time.sleep(5)


def run_multi_node(args):
    """Multi-node: SSH-based datagen + sync + training."""
    pg = ProcessGroup()
    gen_dir = os.path.join(args.output_dir, "gen")
    manifest_path = os.path.join(gen_dir, "manifest.json")

    datagen_nodes = [n.strip() for n in args.datagen_nodes.split(",")]
    num_shards = len(datagen_nodes)

    # Start datagen on each remote node
    for i, node in enumerate(datagen_nodes):
        remote_cmd = (
            f"cd {args.remote_workdir} && "
            f"python scripts/data_generation_offline.py "
            f"--target-model-path {args.target_model_path} "
            f"--train-data-path {args.train_data_path} "
            f"--output-dir {args.remote_gen_dir} "
            f"--manifest-path {args.remote_gen_dir}/manifest_shard{i}.json "
            f"--tensor-parallel-size {args.datagen_tp} "
            f"--seq-length {args.seq_length} "
            f"--batch-size {args.datagen_batch_size} "
            f"--shard-id {i} --num-shards {num_shards}"
        )
        if args.max_samples:
            remote_cmd += f" --max-samples {args.max_samples}"
        pg.spawn_ssh(f"datagen-{node}", node, remote_cmd)

    # Start sync agent
    sync_cmd = [
        "bash", "scripts/sync_datagen.sh",
        "--datagen-nodes", args.datagen_nodes,
        "--remote-dir", args.remote_gen_dir,
        "--local-dir", gen_dir,
        "--manifest-path", manifest_path,
        "--poll-interval", str(args.sync_interval),
    ]
    pg.spawn("sync", sync_cmd)

    # Wait for min_samples
    log.info(f"Waiting for {args.min_samples} samples on training node...")
    while True:
        from speculators.train import manifest as manifest_mod
        m = manifest_mod.read(manifest_path)
        if len(m["files"]) >= args.min_samples:
            break
        time.sleep(args.poll_interval)

    # Start streaming training (locally or on train_node)
    train_cmd_str = (
        f"torchrun --nproc_per_node={args.train_gpus_count} "
        f"scripts/train_streaming.py "
        f"--verifier-name-or-path {args.target_model_path} "
        f"--manifest-path {manifest_path} "
        f"--data-path {gen_dir} "
        f"--save-path {os.path.join(args.output_dir, 'checkpoints')} "
        f"--total-seq-len {args.seq_length} "
        f"--final-epochs {args.final_epochs} "
        f"--min-samples {args.min_samples} "
        f"--lr {args.lr} "
        f"--num-workers {args.num_workers} "
        f"--prefetch-factor {args.prefetch_factor}"
    )

    if args.train_node:
        pg.spawn_ssh("training", args.train_node, f"cd {args.remote_workdir} && {train_cmd_str}")
    else:
        pg.spawn("training", train_cmd_str.split())

    # Monitor
    def handle_signal(sig, frame):
        log.info("Received signal, shutting down...")
        pg.shutdown()
        sys.exit(1)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    while True:
        result = pg.wait_any()
        if result:
            name, retcode = result
            if name == "training":
                log.info(f"Training finished (exit code {retcode})")
                pg.shutdown()
                return retcode
            elif retcode != 0 and "datagen" in name:
                log.error(f"{name} failed (exit code {retcode})")
                pg.shutdown()
                return retcode
            else:
                log.info(f"{name} exited (code {retcode})")
        time.sleep(5)


def parse_args():
    parser = argparse.ArgumentParser(description="Online Eagle3 training orchestrator")
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["single-node", "multi-node"],
    )

    # Model / data
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--train-data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seq-length", type=int, default=8192)

    # Single-node GPU splits
    parser.add_argument("--datagen-gpus", type=str, default="0,1")
    parser.add_argument("--train-gpus", type=str, default="2,3,4,5,6,7")
    parser.add_argument("--datagen-tp", type=int, default=2)

    # Multi-node
    parser.add_argument("--datagen-nodes", type=str, default="")
    parser.add_argument("--train-node", type=str, default="")
    parser.add_argument("--train-gpus-count", type=int, default=8)
    parser.add_argument("--remote-workdir", type=str, default="/root/develop/speculators")
    parser.add_argument("--remote-gen-dir", type=str, default="/data/output/gen")
    parser.add_argument("--sync-interval", type=int, default=30)

    # Training
    parser.add_argument("--min-samples", type=int, default=500)
    parser.add_argument("--final-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--datagen-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--poll-interval", type=float, default=10.0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "single-node":
        sys.exit(run_single_node(args) or 0)
    else:
        sys.exit(run_multi_node(args) or 0)
