#!/usr/bin/env python3
"""
Offline EAGLE Training Data Generation Pipeline

This script generates training data for EAGLE models by:
1. Automatically preprocessing data if needed (or loading from cache)
2. Using vLLM to extract hidden states from target model
3. Saving each data point as a separate .pt file

Preprocessing is cached automatically by HuggingFace datasets.
Token frequencies are saved in the current directory by default.

Usage:
    python data_generation_offline.py \
        --target-model-path meta-llama/Llama-3.1-8B-Instruct \
        --train-data-path sharegpt \
        --output-dir ./training_data \
        --hf-cache-dir /path/to/cache \
        --max-samples 5000
"""

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from datasets import config as datasets_config
from tqdm import tqdm  # type: ignore[import-untyped]

# Set vLLM to use 'spawn' instead of 'fork'
# to prevent "Cannot re-initialize CUDA in forked subprocess" errors
from vllm import envs

envs.VLLM_WORKER_MULTIPROC_METHOD = "spawn"

from speculators.data_generation.config_generator import (  # noqa: E402
    DataGenerationConfig,
)
from speculators.train import manifest as manifest_mod  # noqa: E402
from speculators.data_generation.logging_utils import PipelineLogger  # noqa: E402
from speculators.data_generation.preprocessing import (  # noqa: E402
    load_and_preprocess_dataset,
)
from speculators.data_generation.vllm_hidden_states_generator import (  # noqa: E402
    VllmHiddenStatesGenerator,
)

# Constants
MAX_IO_WORKERS = 4  # Number of parallel file save operations

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = PipelineLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate EAGLE training data offline")

    # Model arguments
    parser.add_argument(
        "--target-model-path",
        type=str,
        required=True,
        help="HuggingFace model ID or local path for target model",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=torch.accelerator.device_count(),
        help="Tensor parallel size for target model (default: 1)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="Target GPU memory utilization (default: 0.8)",
    )

    # Data arguments
    parser.add_argument(
        "--train-data-path",
        type=str,
        required=True,
        help="Path to training data (same as used in preprocessing)",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length for preprocessing and model (default: 2048)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: None, process all)",
    )
    parser.add_argument(
        "--token-freq-path",
        type=str,
        default="./token_freq.pt",
        help="Path to save token frequency distribution (default: ./token_freq.pt)",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default=None,
        help=(
            "Directory for HuggingFace datasets cache. "
            "If not specified, uses HF_DATASETS_CACHE env var or default location. "
            "(default: None)"
        ),
    )
    parser.add_argument(
        "--assistant-pattern",
        type=str,
        default=None,
        help=(
            "Custom regex pattern for matching assistant responses. "
            "If not provided, auto-detected from chat template."
        ),
    )
    parser.add_argument(
        "--turn-dropout",
        action="store_true",
        help=(
            "Enable turn dropout: randomly keeps first N consecutive turns "
            "per conversation for data augmentation."
        ),
    )

    # Output arguments
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save .pt files"
    )

    # Hidden states generation arguments
    parser.add_argument(
        "--layer-ids",
        type=int,
        nargs="+",
        default=None,
        help=(
            "List of layer IDs from which to capture hidden states "
            "(default: auto-select)"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for hidden states generation (default: 8)",
    )

    # Processing arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (must match preprocessing seed, default: 0)",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Starting index for output files (default: 0)",
    )
    parser.add_argument(
        "--num-preprocessing-workers",
        type=int,
        default=8,
        help="Number of CPU processes for dataset preprocessing (default: 8)",
    )

    # Buffer control
    parser.add_argument(
        "--max-output-size-gb",
        type=float,
        default=1024,
        help="Max output directory size in GB. Pauses generation when exceeded (default: 1024)",
    )

    # Online/streaming training support
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Path to manifest.json for online training coordination (default: None)",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="Shard ID for multi-node datagen (default: 0)",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of datagen shards (default: 1)",
    )

    # Continuous datagen mode
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run datagen in continuous loop mode. Re-shuffles and regenerates "
             "dataset each round, using difficulty weights from training feedback.",
    )
    parser.add_argument(
        "--difficulty-scores-path",
        type=str,
        default=None,
        help="Path to difficulty_scores.json (written by training). "
             "Used in --continuous mode for weighted resampling.",
    )
    return parser.parse_args()


def find_last_checkpoint(output_dir: str) -> int:
    """Find the last successfully saved file index by scanning existing files."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return 0

    max_index = -1
    for file_path in output_path.iterdir():
        if file_path.name.startswith("data_") and file_path.name.endswith(".pt"):
            index_str = file_path.stem[5:]  # Remove "data_" prefix
            try:
                index = int(index_str)
                max_index = max(max_index, index)
            except ValueError:
                continue

    return max_index + 1


def get_dir_size_gb(path: str) -> float:
    """Get total size of files in a directory in GB (non-recursive, fast)."""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                total += entry.stat().st_size
    except OSError:
        pass
    return total / (1024**3)


def _cleanup_oldest_files(output_dir: str, target_files: int):
    """Delete oldest data_*.pt files, keeping only the newest target_files."""
    import re

    entries = []
    try:
        for entry in os.scandir(output_dir):
            m = re.match(r"data_(\d+)\.pt$", entry.name)
            if m and entry.is_file(follow_symlinks=False):
                entries.append((int(m.group(1)), entry.path))
    except OSError:
        return 0

    if len(entries) <= target_files:
        return 0

    entries.sort()
    to_delete = entries[: len(entries) - target_files]
    deleted = 0
    for _, path in to_delete:
        try:
            os.remove(path)
            deleted += 1
        except OSError:
            pass
    return deleted


def _start_output_cleanup_thread(
    output_dir: str, max_gb: float, interval: int = 300
):
    """Start a daemon thread that evicts oldest files when output dir exceeds max_gb.

    The training node's buffer cleanup only cleans its own local copy — it does NOT
    delete files on the datagen node. Without datagen-side cleanup, the output
    directory grows unbounded, hits --max-output-size-gb, and generation stalls.

    Uses size-based eviction: when directory reaches 85% of budget, delete oldest
    files until directory is at 70% of budget. This avoids relying on estimated
    file sizes which vary widely across experiments.
    """
    import re
    import threading

    high_watermark = max_gb * 0.85  # start evicting
    low_watermark = max_gb * 0.70   # stop evicting

    def _evict_to_budget(target_gb: float) -> int:
        """Delete oldest data_*.pt files until directory is under target_gb."""
        entries = []
        try:
            for entry in os.scandir(output_dir):
                m = re.match(r"data_(\d+)\.pt$", entry.name)
                if m and entry.is_file(follow_symlinks=False):
                    entries.append((int(m.group(1)), entry.path, entry.stat().st_size))
        except OSError:
            return 0

        entries.sort()  # oldest first
        current_gb = sum(s for _, _, s in entries) / (1024**3)
        if current_gb <= target_gb:
            return 0

        deleted = 0
        for _, path, size in entries:
            if current_gb <= target_gb:
                break
            try:
                os.remove(path)
                current_gb -= size / (1024**3)
                deleted += 1
            except OSError:
                pass
        return deleted

    def _loop():
        while True:
            time.sleep(interval)
            size_gb = get_dir_size_gb(output_dir)
            if size_gb >= high_watermark:
                deleted = _evict_to_budget(low_watermark)
                if deleted > 0:
                    new_size = get_dir_size_gb(output_dir)
                    log.info(
                        f"[cleanup] Evicted {deleted} old files "
                        f"({size_gb:.0f}GB -> {new_size:.0f}GB, "
                        f"budget={max_gb:.0f}GB)"
                    )

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    log.info(
        f"[cleanup] Background cleanup started "
        f"(high={high_watermark:.0f}GB, low={low_watermark:.0f}GB, "
        f"interval={interval}s)"
    )
    return t


def wait_for_output_budget(output_dir: str, max_gb: float, poll_interval: int = 60):
    """Block until output directory is under the size limit."""
    while True:
        size_gb = get_dir_size_gb(output_dir)
        if size_gb < max_gb:
            return
        log.warning(
            f"Output dir {size_gb:.1f}GB >= {max_gb:.0f}GB limit, "
            f"waiting {poll_interval}s for space to free up..."
        )
        time.sleep(poll_interval)


def load_difficulty_weights(
    scores_path: str | None, num_samples: int
) -> list[float] | None:
    """Load per-sample difficulty weights from training feedback.

    Returns None for uniform sampling (round 1 or no scores file).
    Higher loss → higher weight → more likely to be regenerated.
    """
    if scores_path is None or not os.path.exists(scores_path):
        return None

    try:
        with open(scores_path) as f:
            scores = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    if not scores:
        return None

    # Map file-level scores to sample indices
    weights = []
    values = list(scores.values())
    min_loss = min(values)
    max_loss = max(values)
    loss_range = max_loss - min_loss if max_loss > min_loss else 1.0

    for i in range(num_samples):
        key = f"data_{i}.pt"
        if key in scores:
            normalized = (scores[key] - min_loss) / loss_range
            # Min 50% weight for easy samples, max 150% for hard
            weights.append(0.5 + normalized)
        else:
            weights.append(1.0)  # unseen → default

    log.info(
        f"[difficulty] Loaded {len(scores)} scores, "
        f"loss range [{min_loss:.3f}, {max_loss:.3f}]"
    )
    return weights


def weighted_sample_order(
    num_samples: int, weights: list[float] | None, seed: int
) -> list[int]:
    """Return sample indices shuffled by weighted probability.

    If weights is None, returns a simple shuffle (uniform).
    Uses Gumbel-max trick for weighted shuffle without replacement.
    """
    import numpy as _np

    rng = _np.random.default_rng(seed)
    indices = _np.arange(num_samples)

    if weights is None:
        rng.shuffle(indices)
        return indices.tolist()

    # Gumbel-max trick: add Gumbel noise scaled by log(weight), sort descending
    w = _np.array(weights, dtype=_np.float64)
    w = _np.maximum(w, 1e-6)
    keys = _np.log(w) + rng.gumbel(size=num_samples)
    order = _np.argsort(-keys)
    return order.tolist()


def save_sample_to_disk(data_dict, output_path):
    """Save a single sample to disk for async execution."""
    torch.save(data_dict, output_path)
    return output_path


def save_config(args, generator, num_samples, output_dir):
    """Save metadata config file for reproducibility."""
    log.subsection("Saving configuration metadata")

    cache_dir = (
        args.hf_cache_dir if args.hf_cache_dir else datasets_config.HF_DATASETS_CACHE
    )

    config = DataGenerationConfig.from_generator(
        generator=generator,
        train_data_path=args.train_data_path,
        seq_length=args.seq_length,
        cache_dir=str(cache_dir),
        num_samples=num_samples,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    config_path = Path(output_dir) / "data_config.json"
    config_path.write_text(json.dumps(config.to_dict(), indent=2))
    log.info(f"Saved config v{config.version} to {config_path}")


def generate_and_save_hidden_states(
    args, dataset, *, file_idx_start: int | None = None, generator=None
):
    """Generate hidden states and save each sample as a .pt file.

    Args:
        file_idx_start: Override starting file index (for continuous mode).
            If None, auto-detect from existing files.
        generator: Reuse existing VllmHiddenStatesGenerator (for continuous mode).

    Returns:
        (num_saved, final_file_idx, generator) tuple.
    """
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if file_idx_start is not None:
        start_file_idx = file_idx_start
    else:
        start_file_idx = find_last_checkpoint(args.output_dir)

    # Load existing sample lengths to preserve them on resume
    sample_lengths_output_path = Path(args.output_dir) / "sample_lengths.json"
    if start_file_idx > 0 and sample_lengths_output_path.exists():
        with open(sample_lengths_output_path) as f:
            sample_lengths = json.load(f)
        log.subsection(
            f"Resuming: {start_file_idx} files already exist, "
            f"loaded {len(sample_lengths)} existing sample lengths"
        )
    else:
        sample_lengths = {}
        if start_file_idx > 0:
            log.subsection(f"Resuming: {start_file_idx} files already exist")

    num_samples = len(dataset)
    start_sample_idx = start_file_idx - args.start_idx

    if file_idx_start is None and start_sample_idx >= num_samples:
        log.info("All samples already processed!")
        return 0, start_file_idx, generator

    # In continuous mode, always process from sample 0 (dataset is re-ordered)
    if file_idx_start is not None:
        start_sample_idx = 0

    if generator is None:
        log.subsection("Initializing vLLM hidden states generator")
        generator = VllmHiddenStatesGenerator(
            model_path=args.target_model_path,
            layer_ids=args.layer_ids,
            max_model_len=args.seq_length,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
        )

    log.info(f"Processing {num_samples - start_sample_idx}/{num_samples} samples")
    file_idx = start_file_idx

    num_batches = (
        num_samples - start_sample_idx + args.batch_size - 1
    ) // args.batch_size

    # Start background cleanup thread to evict old files before disk fills up
    if args.max_output_size_gb > 0:
        _start_output_cleanup_thread(args.output_dir, args.max_output_size_gb)

    # Use ThreadPoolExecutor for async file I/O
    max_io_workers = MAX_IO_WORKERS

    pbar = tqdm(
        range(start_sample_idx, num_samples, args.batch_size),
        desc="Generating hidden states",
        total=num_batches,
    )

    with ThreadPoolExecutor(max_workers=max_io_workers) as thread_executor:
        futures = []

        for i in pbar:
            if args.max_output_size_gb > 0:
                wait_for_output_budget(args.output_dir, args.max_output_size_gb)

            batch_end = min(i + args.batch_size, num_samples)
            batch = dataset[i:batch_end]
            batch_input_ids = batch["input_ids"]
            batch_loss_mask = batch["loss_mask"]

            results = generator.generate(batch_input_ids)

            # Submit save operations to thread pool (async I/O)
            for j, result in enumerate(results):
                # Truncate loss_mask to match input_ids length (generator may truncate)
                input_len = len(result["input_ids"])
                sample_lengths[str(file_idx)] = input_len
                loss_mask = batch_loss_mask[j][:input_len]

                result_cleaned = {
                    "input_ids": result["input_ids"],
                    "hidden_states": [h.contiguous() for h in result["hidden_states"]],
                    "loss_mask": loss_mask,
                }
                output_path = Path(args.output_dir) / f"data_{file_idx}.pt"
                future = thread_executor.submit(
                    save_sample_to_disk, result_cleaned, output_path
                )
                futures.append(future)
                file_idx += 1

            # Update manifest after each batch
            if args.manifest_path:
                manifest_files = [
                    {"idx": int(k), "path": f"data_{k}.pt", "length": v, "train_count": 0}
                    for k, v in sample_lengths.items()
                ]
                manifest_mod.write(args.manifest_path, manifest_files, "generating")

        log.info("Waiting for remaining file saves to complete...")
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Saving files"
        ):
            future.result()

    samples_saved = file_idx - start_file_idx

    with open(sample_lengths_output_path, "w") as f:
        json.dump(sample_lengths, f, indent=2)

    log.info(f"Saved {samples_saved} new data points to {args.output_dir}")

    save_config(args, generator, num_samples, args.output_dir)

    return samples_saved, file_idx, generator


def main():
    args = parse_args()

    log.section("EAGLE Offline Data Generation")
    log.config(
        {
            "Target Model": args.target_model_path,
            "Dataset": args.train_data_path,
            "Output Dir": args.output_dir,
            "Tensor Parallel": args.tensor_parallel_size,
            "Batch Size": args.batch_size,
        }
    )

    dataset, _ = load_and_preprocess_dataset(
        target_model_path=args.target_model_path,
        train_data_path=args.train_data_path,
        seq_length=args.seq_length,
        build_dataset_num_proc=args.num_preprocessing_workers,
        seed=args.seed,
        max_samples=args.max_samples,
        token_freq_path=args.token_freq_path,
        cache_dir=args.hf_cache_dir,
        assistant_pattern=args.assistant_pattern,
        turn_dropout=args.turn_dropout,
    )
    # Apply sharding if requested
    if args.num_shards > 1:
        total = len(dataset)
        shard_size = total // args.num_shards
        start = args.shard_id * shard_size
        end = start + shard_size if args.shard_id < args.num_shards - 1 else total
        dataset = dataset.select(range(start, end))
        log.info(
            f"Shard {args.shard_id}/{args.num_shards}: "
            f"samples {start}-{end} ({len(dataset)} total)"
        )

    if args.continuous:
        # Continuous mode: loop over dataset with difficulty-weighted resampling
        datagen_round = 0
        file_idx = find_last_checkpoint(args.output_dir)
        generator = None

        while True:
            datagen_round += 1
            log.section(f"Continuous datagen: Round {datagen_round}")

            weights = load_difficulty_weights(
                args.difficulty_scores_path, len(dataset)
            )
            if weights is not None:
                log.info(f"Using difficulty-weighted sampling (round {datagen_round})")
            else:
                log.info(f"Using uniform sampling (round {datagen_round})")

            sample_order = weighted_sample_order(
                len(dataset), weights, seed=args.seed + datagen_round
            )
            reordered = dataset.select(sample_order)

            num_saved, file_idx, generator = generate_and_save_hidden_states(
                args, reordered, file_idx_start=file_idx, generator=generator
            )
            log.info(
                f"Round {datagen_round} complete: {num_saved} files saved, "
                f"next file_idx={file_idx}"
            )
    else:
        # Single-pass mode (original behavior)
        num_saved, _, _ = generate_and_save_hidden_states(args, dataset)

        if args.manifest_path:
            manifest_mod.mark_complete(args.manifest_path)

        log.section("Data generation complete!")
        log.info(f"Saved {num_saved} files to {args.output_dir}")


if __name__ == "__main__":
    main()
