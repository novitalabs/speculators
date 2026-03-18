"""Streaming Eagle3 training script.

Reads file lists from a manifest.json (written by datagen / sync agent),
and dynamically grows the training set each epoch as new data arrives.
Validation set is fixed at first initialization.

Usage:
    torchrun --nnodes=1 --nproc_per_node=8 scripts/train_streaming.py \
        --verifier-name-or-path /data/models/Qwen3-32B \
        --manifest-path /data/output/gen/manifest.json \
        --data-path /data/output/gen \
        --save-path /data/output/checkpoints \
        --final-epochs 3
"""

import argparse
import logging
import os
from pathlib import Path
import random
import time
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import LlamaConfig, PretrainedConfig
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from speculators.model import SpeculatorModel
from speculators.train import manifest as manifest_mod
from speculators.train.data import (
    Eagle3SampleFileDataset,
    create_collate_fn,
    standardize_data_v1,
)
from speculators.train.distributed_batch_sampler import (
    MultipackDistributedBatchSamplerV2,
)
from speculators.train.logger import setup_metric_logger, setup_root_logger
from speculators.train.noise_transforms import AddUniformNoise
from speculators.train.trainer import Trainer, TrainerConfig
from speculators.train.utils import maybe_destroy_distributed, maybe_setup_distributed

root_logger = logging.getLogger("speculators")

DRAFT_ARCH_CONFIGS: dict[str, type] = {
    "llama": LlamaConfig,
    "qwen3": Qwen3Config,
}

MIN_FILES_FOR_VAL = 10  # Need at least this many files to split train/val


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_dataloader(
    file_list: list[str],
    total_seq_len: int,
    world_size: int,
    local_rank: int,
    add_noise: bool = True,
    noise_std: float = 0.05,
    num_workers: int = 12,
    prefetch_factor: int = 4,
) -> DataLoader:
    noise_transform = (
        AddUniformNoise(
            std=noise_std, tensors=("hidden_states", "verifier_last_hidden_states")
        )
        if add_noise
        else None
    )
    dataset = Eagle3SampleFileDataset(
        file_list=file_list,
        max_len=total_seq_len,
        transform=noise_transform,
        standardize_fn=standardize_data_v1,
    )
    batch_sampler = MultipackDistributedBatchSamplerV2(
        batch_max_length=total_seq_len,
        lengths=dataset.approx_lengths,
        num_replicas=world_size,
        rank=local_rank,
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        collate_fn=create_collate_fn(total_seq_len),
        persistent_workers=True,
    )


def create_transformer_layer_config(
    verifier_name_or_path: str, num_layers: int, draft_arch: str = "llama"
) -> PretrainedConfig:
    if draft_arch not in DRAFT_ARCH_CONFIGS:
        raise ValueError(
            f"Unknown draft architecture: {draft_arch}. "
            f"Available: {list(DRAFT_ARCH_CONFIGS.keys())}"
        )
    if draft_arch != "llama":
        warnings.warn(
            f"Draft architecture '{draft_arch}' is not yet supported in vLLM. "
            "The trained model may not be usable for inference in vLLM.",
            stacklevel=2,
        )
    config_class = DRAFT_ARCH_CONFIGS[draft_arch]
    verifier_config = AutoConfig.from_pretrained(
        verifier_name_or_path, trust_remote_code=True
    )
    if hasattr(verifier_config, "text_config"):
        verifier_config = verifier_config.text_config
    transformer_layer_config = config_class(
        vocab_size=verifier_config.vocab_size,
        hidden_size=verifier_config.hidden_size,
        intermediate_size=verifier_config.intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=verifier_config.num_attention_heads,
        num_key_value_heads=verifier_config.num_key_value_heads,
        hidden_act=verifier_config.hidden_act,
        max_position_embeddings=verifier_config.max_position_embeddings,
        initializer_range=verifier_config.initializer_range,
        rms_norm_eps=verifier_config.rms_norm_eps,
        head_dim=getattr(verifier_config, "head_dim", None),
    )
    transformer_layer_config._attn_implementation = "simple_flex_attention"  # noqa: SLF001
    return transformer_layer_config


def resolve_file_paths(data_path: str, manifest_files: list[dict]) -> list[str]:
    """Resolve manifest file entries to absolute paths, filtering to only existing files."""
    paths = []
    for f in manifest_files:
        p = os.path.join(data_path, f["path"])
        if os.path.exists(p):
            paths.append(p)
    return paths


def wait_for_min_files(
    manifest_path: str,
    data_path: str,
    min_files: int,
    poll_interval: float = 10.0,
) -> dict:
    """Block until at least min_files .pt files actually exist on disk."""
    while True:
        m = manifest_mod.read(manifest_path)
        existing = resolve_file_paths(data_path, m["files"])
        if len(existing) >= min_files:
            return m
        root_logger.info(
            f"Waiting for data: {len(existing)}/{min_files} files on disk "
            f"({len(m['files'])} in manifest), polling every {poll_interval}s..."
        )
        time.sleep(poll_interval)


def main(args: argparse.Namespace):
    set_seed(args.seed, args.deterministic_cuda)
    setup_root_logger()
    setup_metric_logger(
        loggers=args.logger, run_name=args.run_name, output_dir=args.log_dir
    )

    local_rank, world_size, rank, is_distributed = maybe_setup_distributed()
    device = torch.device(local_rank)

    # Vocab mapping
    if args.d2t_path or args.t2d_path:
        if not (args.d2t_path and args.t2d_path):
            raise ValueError("Both t2d and d2t must be provided together.")
        d2t = torch.from_numpy(np.load(args.d2t_path)).to(device)
        t2d = torch.from_numpy(np.load(args.t2d_path)).to(device)
        draft_vocab_size = d2t.shape[0]
    else:
        d2t = None
        t2d = None
        verifier_config = AutoConfig.from_pretrained(
            args.verifier_name_or_path, trust_remote_code=True
        )
        if hasattr(verifier_config, "text_config"):
            verifier_config = verifier_config.text_config
        draft_vocab_size = verifier_config.vocab_size

    transformer_layer_config = create_transformer_layer_config(
        args.verifier_name_or_path, args.num_layers, draft_arch=args.draft_arch
    )

    if SpeculatorModel.registry_auto_discovery:
        SpeculatorModel.auto_populate_registry()
    if args.speculator_type not in SpeculatorModel.registry:
        raise ValueError(
            f"Unknown speculator type: {args.speculator_type}. "
            f"Available: {list(SpeculatorModel.registry.keys())}"
        )

    model_class = SpeculatorModel.registry[args.speculator_type]
    draft_model = model_class.from_training_args(
        verifier_config=transformer_layer_config,
        t2d=t2d,
        d2t=d2t,
        draft_vocab_size=draft_vocab_size,
        **vars(args),
    )

    train_call_kwargs, val_call_kwargs = model_class.get_trainer_kwargs(**vars(args))

    # ---- Streaming loop ----
    root_logger.info(f"Streaming mode: reading manifest from {args.manifest_path}")

    # Wait for enough data to start
    manifest = wait_for_min_files(
        args.manifest_path, args.data_path, args.min_samples,
        poll_interval=args.poll_interval,
    )
    all_files = resolve_file_paths(args.data_path, manifest["files"])

    # Fix val set on first read
    random.seed(args.seed)
    shuffled = list(all_files)
    random.shuffle(shuffled)
    num_val = int(len(shuffled) * args.val_ratio)
    num_val = min(num_val, args.max_val_files)
    num_train = len(shuffled) - num_val
    val_files = shuffled[num_train:]
    val_file_set = set(val_files)
    train_files = [f for f in all_files if f not in val_file_set]

    root_logger.info(
        f"Initial split: {len(train_files)} train, {len(val_files)} val files"
    )

    # Build val loader (fixed for all epochs)
    val_loader = setup_dataloader(
        val_files,
        args.total_seq_len,
        world_size,
        local_rank,
        add_noise=False,
        noise_std=args.noise_std,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    ) if val_files else None

    # Build first train loader
    train_loader = setup_dataloader(
        train_files,
        args.total_seq_len,
        world_size,
        local_rank,
        add_noise=True,
        noise_std=args.noise_std,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )

    # Create trainer (handles model setup, optimizer, etc.)
    trainer_config = TrainerConfig(
        num_epochs=999,  # streaming: we control the loop
        save_path=args.save_path,
        lr=args.lr,
        resume_from_checkpoint=not args.no_resume_from_checkpoint,
        is_distributed=is_distributed,
        local_rank=local_rank,
        train_call_kwargs=train_call_kwargs,
        val_call_kwargs=val_call_kwargs,
        scheduler_type="none",  # streaming: use constant lr
        val_every_steps=args.val_every_steps,
    )
    trainer = Trainer(draft_model, trainer_config, train_loader, val_loader)

    epoch = trainer.current_epoch
    final_countdown = args.final_epochs
    datagen_complete = manifest["status"] == "complete"

    epoch_lock_path = os.path.join(args.data_path, ".epoch_in_progress")

    # Track unique files ever seen for global epoch computation
    files_ever_seen: set[str] = set()
    global_epoch_count = 0

    while True:
        root_logger.info(
            f"Epoch {epoch}: {len(train_files)} train files, "
            f"datagen_complete={datagen_complete}"
        )

        # Signal to buffer_cleanup that an epoch is in progress
        Path(epoch_lock_path).write_text(str(epoch))

        trainer.train_epoch(epoch)

        torch.cuda.empty_cache()
        root_logger.info(f"Epoch {epoch}: training complete, starting validation...")

        if val_loader is not None:
            trainer.val_epoch(epoch)

        root_logger.info(f"Epoch {epoch}: validation complete, saving checkpoint...")

        trainer.save_checkpoint(epoch)

        root_logger.info(f"Epoch {epoch}: checkpoint saved.")

        # Update manifest train_count for trained files
        trained_basenames = {os.path.basename(f) for f in train_files}
        files_ever_seen.update(trained_basenames)
        manifest_mod.increment_train_count(args.manifest_path, trained_basenames)

        # Read back manifest to get total_remote_files
        manifest = manifest_mod.read(args.manifest_path)
        total_remote = manifest.get("total_remote_files", 0)
        global_epoch_count = (
            len(files_ever_seen) // total_remote if total_remote > 0 else 0
        )

        root_logger.info(
            f"Global progress: {len(files_ever_seen)}/{total_remote} unique files, "
            f"global_epoch={global_epoch_count}"
        )

        # Write progress to manifest
        extra = {k: v for k, v in manifest.items()
                 if k not in ("status", "files", "updated_at")}
        extra.update(
            total_remote_files=total_remote,
            files_ever_seen_count=len(files_ever_seen),
            global_epoch=global_epoch_count,
        )
        manifest_mod.write(
            args.manifest_path, manifest["files"], manifest["status"], **extra
        )

        # Release epoch lock — cleanup can now safely delete trained files
        Path(epoch_lock_path).unlink(missing_ok=True)
        root_logger.info("Epoch lock released, waiting for cleanup + sync cycle...")

        epoch += 1

        # Check termination
        if datagen_complete and total_remote > 0 and args.target_global_epochs > 0:
            if global_epoch_count >= args.target_global_epochs:
                root_logger.info(
                    f"Reached {args.target_global_epochs} global epochs. Done!"
                )
                break
        elif datagen_complete:
            # Fallback to final_epochs countdown
            final_countdown -= 1
            root_logger.info(
                f"Datagen complete. Final epochs remaining: {final_countdown}"
            )
            if final_countdown <= 0:
                break

        # Wait for cleanup to finish and sync to replenish files
        # This ensures enough train files exist before rebuilding the DataLoader
        while True:
            time.sleep(args.poll_interval)
            manifest = manifest_mod.read(args.manifest_path)
            all_files = resolve_file_paths(args.data_path, manifest["files"])
            train_files = [f for f in all_files if f not in val_file_set]
            if len(train_files) >= args.min_samples:
                break
            root_logger.info(
                f"Waiting for files: {len(train_files)}/{args.min_samples} "
                f"train files on disk, polling every {args.poll_interval}s..."
            )

        if manifest["status"] == "complete":
            datagen_complete = True

        # Rebuild train DataLoader with new files
        # Shutdown old workers first
        del trainer.train_loader
        train_loader = setup_dataloader(
            train_files,
            args.total_seq_len,
            world_size,
            local_rank,
            add_noise=True,
            noise_std=args.noise_std,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
        )
        trainer.train_loader = train_loader

    root_logger.info("Streaming training complete!")
    maybe_destroy_distributed()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Streaming Eagle3 training (reads manifest for dynamic data)"
    )

    # Core args (same as train.py)
    parser.add_argument("--verifier-name-or-path", type=str, required=True)
    parser.add_argument("--speculator-type", type=str, default="eagle3")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, default="./checkpoints")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no-resume-from-checkpoint", action="store_true")
    parser.add_argument("--logger", type=str, default="")
    parser.add_argument("--total-seq-len", type=int, default=8192)
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument(
        "--draft-arch", type=str, default="llama",
        choices=list(DRAFT_ARCH_CONFIGS.keys()),
    )
    parser.add_argument("--d2t-path", type=str, default=None)
    parser.add_argument("--t2d-path", type=str, default=None)
    parser.add_argument("--ttt-steps", type=int, default=3)
    parser.add_argument("--ttt-step-loss-decay", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic-cuda", action="store_true", default=False)
    parser.add_argument("--use-off-policy-tokens", action="store_true", default=False)
    parser.add_argument(
        "--norm-before-residual", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--embed-requires-grad", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--noise-std", type=float, default=0.05)

    # Streaming-specific args
    parser.add_argument(
        "--manifest-path", type=str, required=True,
        help="Path to manifest.json written by datagen/sync",
    )
    parser.add_argument(
        "--min-samples", type=int, default=100,
        help="Minimum number of samples before starting training",
    )
    parser.add_argument(
        "--final-epochs", type=int, default=3,
        help="Number of epochs to run after datagen is complete",
    )
    parser.add_argument(
        "--poll-interval", type=float, default=10.0,
        help="Seconds between manifest polls when waiting for data",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1,
        help="Fraction of initial files to use for validation (default: 0.1)",
    )
    parser.add_argument(
        "--max-val-files", type=int, default=200,
        help="Maximum number of validation files (default: 200)",
    )
    parser.add_argument(
        "--val-every-steps", type=int, default=0,
        help="Run validation every N training steps (0 = only at epoch end)",
    )
    parser.add_argument(
        "--target-global-epochs", type=int, default=10,
        help="Terminate when this many global epochs are reached (0 = use --final-epochs fallback)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)


# RUN WITH:
# torchrun --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_streaming.py \
#     --verifier-name-or-path <model> --manifest-path <path> --data-path <dir>
