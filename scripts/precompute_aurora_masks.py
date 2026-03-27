"""Precompute static Aurora accept/reject masks from a reference draft model.

Runs a trained draft model checkpoint on all .pt data files and saves per-step
accepted masks (argmax match between draft logits and verifier targets).

The masks are saved in post-shift coordinates (length seq_len-1), matching the
coordinate space of loss_mask after shift_batch.

Usage:
    python scripts/precompute_aurora_masks.py \
        --data-dir /data/output/gen \
        --mask-output-dir /data/output/masks \
        --checkpoint-path /data/output/checkpoints/ckpt_67 \
        --verifier-name-or-path /data/models/MiniMax-M2.5 \
        --d2t-path /data/output/vocab_mapping/d2t.npy \
        --t2d-path /data/output/vocab_mapping/t2d.npy \
        --override-num-attention-heads 24 \
        --override-intermediate-size 8192 \
        --override-rope-theta 5000000
"""

import argparse
import glob
import logging
import os
import sys

import numpy as np
import torch
from transformers.models.auto.configuration_auto import AutoConfig

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Disable torch.compile for precompute — we need dynamic shapes for variable-length inputs
torch._dynamo.config.disable = True

from speculators.model import SpeculatorModel
from speculators.train.data import shift_batch, slice_and_pad_to_length, standardize_data_v1

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def create_transformer_layer_config(args):
    """Create transformer layer config (same logic as train.py)."""
    from transformers import LlamaConfig
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    DRAFT_ARCH_CONFIGS = {"llama": LlamaConfig, "qwen3": Qwen3Config}
    config_class = DRAFT_ARCH_CONFIGS[args.draft_arch]

    verifier_config = AutoConfig.from_pretrained(
        args.verifier_name_or_path, trust_remote_code=True
    )
    if hasattr(verifier_config, "text_config"):
        verifier_config = verifier_config.text_config

    num_attention_heads = args.override_num_attention_heads or verifier_config.num_attention_heads
    intermediate_size = args.override_intermediate_size or verifier_config.intermediate_size
    rope_theta = args.override_rope_theta or getattr(verifier_config, "rope_theta", 10000.0)

    transformer_layer_config = config_class(
        vocab_size=verifier_config.vocab_size,
        hidden_size=verifier_config.hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=verifier_config.num_key_value_heads,
        hidden_act=verifier_config.hidden_act,
        max_position_embeddings=verifier_config.max_position_embeddings,
        initializer_range=verifier_config.initializer_range,
        rms_norm_eps=verifier_config.rms_norm_eps,
        head_dim=getattr(verifier_config, "head_dim", None),
        rope_theta=rope_theta,
    )
    transformer_layer_config._attn_implementation = "simple_flex_attention"  # noqa: SLF001
    return transformer_layer_config


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocab mappings
    if args.d2t_path and args.t2d_path:
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

    # Build model
    transformer_layer_config = create_transformer_layer_config(args)

    if SpeculatorModel.registry_auto_discovery:
        SpeculatorModel.auto_populate_registry()

    model_class = SpeculatorModel.registry["eagle3"]
    model = model_class.from_training_args(
        verifier_config=transformer_layer_config,
        t2d=t2d,
        d2t=d2t,
        draft_vocab_size=draft_vocab_size,
        verifier_name_or_path=args.verifier_name_or_path,
        ttt_steps=args.ttt_steps,
        norm_before_residual=True,
        embed_requires_grad=False,
    )

    # Load checkpoint (safetensors format)
    logger.info(f"Loading checkpoint from {args.checkpoint_path}")
    from safetensors import safe_open
    ckpt_file = os.path.join(args.checkpoint_path, "model.safetensors")
    ckpt = {}
    with safe_open(ckpt_file, framework="pt", device="cpu") as f:
        for key in f.keys():  # noqa: SIM118
            ckpt[key] = f.get_tensor(key)
    model.load_state_dict(ckpt, strict=False)
    model = model.to(device).eval()
    logger.info("Model loaded and set to eval mode (torch.compile disabled)")

    # Find all data files
    data_files = sorted(glob.glob(os.path.join(args.data_dir, "data_*.pt")))
    logger.info(f"Found {len(data_files)} data files")

    os.makedirs(args.mask_output_dir, exist_ok=True)

    ttt_steps = args.ttt_steps

    for file_idx_i, data_path in enumerate(data_files):
        # Extract index from filename: data_42.pt -> 42
        basename = os.path.basename(data_path)
        file_idx = basename.replace("data_", "").replace(".pt", "")

        mask_output_path = os.path.join(args.mask_output_dir, f"mask_{file_idx}.pt")
        if os.path.exists(mask_output_path) and not args.overwrite:
            if file_idx_i % 500 == 0:
                logger.info(f"[{file_idx_i}/{len(data_files)}] Skipping {basename} (exists)")
            continue

        # Load and preprocess data
        data = torch.load(data_path, mmap=True, weights_only=True, map_location="cpu")
        data = standardize_data_v1(data)

        # Replace NaN values
        for key in ("hidden_states", "verifier_last_hidden_states"):
            if key in data and data[key].isnan().any():
                data[key] = torch.nan_to_num(data[key], nan=0.0)

        # Add lengths and position_ids (same as Eagle3SampleFileDataset.__getitem__)
        pre_shift_seq_len = data["input_ids"].shape[0]
        data["lengths"] = torch.tensor([pre_shift_seq_len], dtype=torch.long)
        data["position_ids"] = torch.arange(pre_shift_seq_len, dtype=torch.long)

        shifted = shift_batch(data)
        seq_len = shifted["input_ids"].shape[0]  # post-shift length (S-1)

        # Pad to max_len and add batch dim (same as collate_fn)
        max_len = args.total_seq_len
        batch = {}
        for k, v in shifted.items():
            if k == "lengths":
                continue
            batch[k] = slice_and_pad_to_length(v, max_len).unsqueeze(0).to(device)
        batch["lengths"] = shifted["lengths"].to(device)

        with torch.no_grad():
            # Call model forward — returns draft_tokens per step + loss + metrics
            draft_tokens, loss, metrics = model(
                hidden_states=batch["hidden_states"].float(),
                input_ids=batch["input_ids"],
                lengths=batch["lengths"],
                loss_mask=batch["loss_mask"],
                position_ids=batch["position_ids"],
                verifier_last_hidden_states=batch["verifier_last_hidden_states"].float(),
                ttt_steps=ttt_steps,
            )

            # Compute targets for mask comparison
            targets = model.verifier_lm_head(
                model.verifier_norm(batch["verifier_last_hidden_states"].float())
            )
            target_tokens = targets.argmax(dim=-1)  # [1, seq_len]

            # Compute per-step masks using draft_tokens and aligned targets
            mask_dict = {}
            for s in range(ttt_steps):
                # align_for_step logic: draft[:-s] vs target[s:]
                if s > 0:
                    d_tokens = draft_tokens[s][:, :-s]  # [1, seq_len-s]
                    t_tokens = target_tokens[:, s:]  # [1, seq_len-s]
                else:
                    d_tokens = draft_tokens[s]  # [1, seq_len]
                    t_tokens = target_tokens  # [1, seq_len]

                accepted = (d_tokens == t_tokens)[0]  # [padded_aligned_len]

                # Take only the real positions (seq_len), discard padding
                # For step s, aligned_len = seq_len - s
                real_len = seq_len - s
                padded = torch.zeros(seq_len, dtype=torch.bool, device=device)
                padded[:real_len] = accepted[:real_len].to(torch.bool)
                mask_dict[f"accepted_mask_{s}"] = padded.cpu()

        torch.save(mask_dict, mask_output_path)

        if file_idx_i % 100 == 0:
            # Log acceptance ratios for sanity check
            ratios = [mask_dict[f"accepted_mask_{s}"].float().mean().item() for s in range(ttt_steps)]
            logger.info(
                f"[{file_idx_i}/{len(data_files)}] {basename} -> mask_{file_idx}.pt  "
                f"accept_ratios={[f'{r:.3f}' for r in ratios]}"
            )

    logger.info(f"Done! Masks saved to {args.mask_output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute static Aurora accept/reject masks from a reference model"
    )
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing data_*.pt files")
    parser.add_argument("--mask-output-dir", type=str, required=True,
                        help="Directory to save mask_*.pt files")
    parser.add_argument("--checkpoint-path", type=str, required=True,
                        help="Path to checkpoint directory (containing model.pt)")
    parser.add_argument("--verifier-name-or-path", type=str, required=True)
    parser.add_argument("--d2t-path", type=str, default=None)
    parser.add_argument("--t2d-path", type=str, default=None)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--ttt-steps", type=int, default=3)
    parser.add_argument("--total-seq-len", type=int, default=8192,
                        help="Pad inputs to this length (must match training seq len)")
    parser.add_argument("--draft-arch", type=str, default="llama")
    parser.add_argument("--override-num-attention-heads", type=int, default=None)
    parser.add_argument("--override-intermediate-size", type=int, default=None)
    parser.add_argument("--override-rope-theta", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Overwrite existing mask files")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
