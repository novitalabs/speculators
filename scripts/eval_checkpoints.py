#!/usr/bin/env python3
"""Evaluate multiple checkpoints on validation set and report per-layer metrics.

Usage:
    torchrun --standalone --nproc_per_node=8 scripts/eval_checkpoints.py \
        --verifier-name-or-path /data/models/MiniMax-M2.5 \
        --manifest-path /data/output/minimax_m2.5_eagle3_novita_full_v2/gen/manifest.json \
        --data-path /data/output/minimax_m2.5_eagle3_novita_full_v2/gen \
        --checkpoint-dir /data/output/minimax_m2.5_eagle3_novita_full_v2/checkpoints \
        --checkpoints 19 20 21 22 \
        --total-seq-len 8192 \
        --max-val-files 200
"""

import argparse
import json
import os
import random
import sys

import torch
import torch.distributed as dist

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speculators.model import SpeculatorModel
from speculators.train.utils import maybe_setup_distributed, maybe_destroy_distributed
from speculators.train.logger import setup_root_logger
from speculators.train.utils import apply_fully_sharded
from speculators.train.checkpointer import DistributedCheckpointer, load_safetensors_state_dict, convert_float_dtype
from speculators.train import manifest as manifest_mod

from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

from scripts.train_streaming import (
    create_transformer_layer_config,
    resolve_file_paths,
    setup_dataloader,
)

import logging
root_logger = logging.getLogger("speculators")


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_checkpoint_into_model(model, checkpoint_dir: str, epoch: int):
    """Load a specific checkpoint into an FSDP model."""
    model_path = os.path.join(checkpoint_dir, str(epoch), "model.safetensors")
    full_state_dict = load_safetensors_state_dict(model_path, "cpu")
    full_state_dict = convert_float_dtype(full_state_dict, model.dtype)
    set_model_state_dict(
        model,
        full_state_dict,
        options=StateDictOptions(
            full_state_dict=True, broadcast_from_rank0=True, strict=False
        ),
    )
    dist.barrier()


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on validation set")
    parser.add_argument("--verifier-name-or-path", type=str, required=True)
    parser.add_argument("--manifest-path", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--checkpoints", type=int, nargs="+", required=True)
    parser.add_argument("--total-seq-len", type=int, default=8192)
    parser.add_argument("--max-val-files", type=int, default=200)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speculator-type", type=str, default="eagle3")
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--draft-arch", type=str, default="llama")
    parser.add_argument("--ttt-steps", type=int, default=3)
    parser.add_argument("--ttt-step-loss-decay", type=float, default=1.0)
    parser.add_argument("--use-off-policy-tokens", action="store_true", default=False)
    parser.add_argument("--norm-before-residual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--embed-requires-grad", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--noise-std", type=float, default=0.05)
    # Architecture overrides (for draft model different from verifier)
    parser.add_argument(
        "--override-num-attention-heads", type=int, default=None,
        help="Override num_attention_heads in draft transformer layer config",
    )
    parser.add_argument(
        "--override-intermediate-size", type=int, default=None,
        help="Override intermediate_size in draft transformer layer config",
    )
    parser.add_argument(
        "--override-rope-theta", type=float, default=None,
        help="Override rope_theta in draft transformer layer config",
    )
    parser.add_argument(
        "--draft-vocab-size", type=int, default=None,
        help="Override draft vocab size (default: verifier vocab size)",
    )
    parser.add_argument("--d2t-path", type=str, default=None,
        help="Path to draft-to-target vocab mapping (npy)")
    parser.add_argument("--t2d-path", type=str, default=None,
        help="Path to target-to-draft vocab mapping (npy)")
    args = parser.parse_args()

    set_seed(args.seed)
    setup_root_logger()

    local_rank, world_size, rank, is_distributed = maybe_setup_distributed()

    # Model setup
    from transformers import AutoConfig
    verifier_config = AutoConfig.from_pretrained(
        args.verifier_name_or_path, trust_remote_code=True
    )
    if hasattr(verifier_config, "text_config"):
        verifier_config = verifier_config.text_config
    draft_vocab_size = args.draft_vocab_size or verifier_config.vocab_size

    # Load vocab mappings if provided
    t2d = None
    d2t = None
    if args.t2d_path:
        import numpy as np
        t2d = torch.from_numpy(np.load(args.t2d_path))
        if rank == 0:
            root_logger.info(f"Loaded t2d mapping: {t2d.shape}")
    if args.d2t_path:
        import numpy as np
        d2t = torch.from_numpy(np.load(args.d2t_path))
        if rank == 0:
            root_logger.info(f"Loaded d2t mapping: {d2t.shape}")

    transformer_layer_config = create_transformer_layer_config(
        args.verifier_name_or_path, args.num_layers, draft_arch=args.draft_arch,
        override_num_attention_heads=args.override_num_attention_heads,
        override_intermediate_size=args.override_intermediate_size,
        override_rope_theta=args.override_rope_theta,
    )

    if SpeculatorModel.registry_auto_discovery:
        SpeculatorModel.auto_populate_registry()

    model_class = SpeculatorModel.registry[args.speculator_type]
    _, val_call_kwargs = model_class.get_trainer_kwargs(**vars(args))

    # Build val set (same logic as train_streaming.py — deterministic split)
    manifest = manifest_mod.read(args.manifest_path)
    all_files = resolve_file_paths(args.data_path, manifest["files"])

    random.seed(args.seed)
    shuffled = list(all_files)
    random.shuffle(shuffled)
    num_val = int(len(shuffled) * args.val_ratio)
    num_val = min(num_val, args.max_val_files)
    num_train = len(shuffled) - num_val
    val_files = shuffled[num_train:]

    # Filter out files that don't exist on disk (partial sync)
    val_files = [f for f in val_files if os.path.exists(f)]

    if rank == 0:
        root_logger.info(f"Validation set: {len(val_files)} files")

    val_loader = setup_dataloader(
        val_files,
        args.total_seq_len,
        world_size,
        local_rank,
        add_noise=False,
        noise_std=args.noise_std,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )

    results = {}

    for ckpt_epoch in sorted(args.checkpoints):
        ckpt_path = os.path.join(args.checkpoint_dir, str(ckpt_epoch))
        if not os.path.exists(ckpt_path):
            if rank == 0:
                root_logger.warning(f"Checkpoint {ckpt_epoch} not found, skipping")
            continue

        if rank == 0:
            root_logger.info(f"Evaluating checkpoint {ckpt_epoch}")

        # Create fresh model and apply FSDP
        model_args = vars(args).copy()
        model_args["draft_vocab_size"] = draft_vocab_size
        model = model_class.from_training_args(
            verifier_config=transformer_layer_config,
            t2d=t2d,
            d2t=d2t,
            **model_args,
        )
        apply_fully_sharded(model)

        # Load specific checkpoint
        load_checkpoint_into_model(model, args.checkpoint_dir, ckpt_epoch)

        # Run validation
        model.eval()
        val_metrics: dict[str, float] = {}
        num_batches = len(val_loader)

        with torch.no_grad():
            for batch in val_loader:
                gpu_batch = {
                    k: v.to(local_rank, non_blocking=True)
                    if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                _draft_tokens, _loss, metrics = model(
                    **gpu_batch, **val_call_kwargs
                )
                if is_distributed:
                    for v in metrics.values():
                        dist.reduce(v, dst=0, op=dist.ReduceOp.AVG)
                for k, v in metrics.items():
                    val_metrics[k] = val_metrics.get(k, 0.0) + v.item()

        val_metrics = {k: v / num_batches for k, v in val_metrics.items()}
        results[ckpt_epoch] = val_metrics

        if rank == 0:
            root_logger.info(f"Checkpoint {ckpt_epoch} results:")
            for k, v in sorted(val_metrics.items()):
                root_logger.info(f"  {k}: {v:.4f}")

        # Cleanup model to free GPU memory
        del model
        torch.cuda.empty_cache()

    # Print summary table
    if rank == 0 and results:
        print("\n" + "=" * 80)
        print("VALIDATION RESULTS SUMMARY")
        print("=" * 80)

        all_keys = sorted(next(iter(results.values())).keys())

        header = f"{'Checkpoint':>12}"
        for k in all_keys:
            header += f"  {k:>14}"
        print(header)
        print("-" * len(header))

        for ckpt_epoch in sorted(results.keys()):
            row = f"{ckpt_epoch:>12}"
            for k in all_keys:
                row += f"  {results[ckpt_epoch][k]:>14.4f}"
            print(row)

        loss_key = "loss"
        if loss_key in all_keys:
            best_epoch = min(results.keys(), key=lambda e: results[e][loss_key])
            print(f"\nBest checkpoint by total loss: {best_epoch} (loss={results[best_epoch][loss_key]:.4f})")

        output_path = os.path.join(args.checkpoint_dir, "eval_results.json")
        with open(output_path, "w") as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2)
        print(f"\nResults saved to {output_path}")

    maybe_destroy_distributed()


if __name__ == "__main__":
    main()
