"""Convert speculators Eagle3DraftModel to vLLM Eagle3LlamaForCausalLM format.

Usage:
    python convert_speculators_to_vllm_eagle3.py \
        --input /path/to/speculators/checkpoint \
        --output /path/to/vllm/eagle3/model
"""

import argparse
import json
import shutil
from pathlib import Path

import safetensors.torch as st
import torch


def convert(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load speculators config
    with open(input_dir / "config.json") as f:
        spec_config = json.load(f)

    transformer_config = spec_config["transformer_layer_config"]
    draft_vocab_size = spec_config.get("draft_vocab_size", transformer_config["vocab_size"])

    # Build vLLM Eagle3 config (LlamaConfig-based)
    vllm_config = {
        "architectures": ["LlamaForCausalLMEagle3"],
        "model_type": "llama",
        "hidden_size": transformer_config["hidden_size"],
        "intermediate_size": transformer_config["intermediate_size"],
        "num_attention_heads": transformer_config["num_attention_heads"],
        "num_key_value_heads": transformer_config["num_key_value_heads"],
        "num_hidden_layers": transformer_config["num_hidden_layers"],
        "head_dim": transformer_config.get("head_dim", transformer_config["hidden_size"] // transformer_config["num_attention_heads"]),
        "hidden_act": transformer_config.get("hidden_act", "silu"),
        "rms_norm_eps": transformer_config.get("rms_norm_eps", 1e-6),
        "vocab_size": transformer_config["vocab_size"],
        "draft_vocab_size": draft_vocab_size,
        "max_position_embeddings": transformer_config.get("max_position_embeddings", 196608),
        "rope_theta": transformer_config.get("rope_theta", 10000.0),
        "rope_scaling": transformer_config.get("rope_scaling", None),
        "attention_bias": transformer_config.get("attention_bias", False),
        "attention_dropout": transformer_config.get("attention_dropout", 0.0),
        "mlp_bias": transformer_config.get("mlp_bias", False),
        "tie_word_embeddings": False,
        "use_cache": True,
        "dtype": "bfloat16",
        "norm_before_residual": spec_config.get("norm_before_residual", True),
        "bos_token_id": 1,
        "eos_token_id": 2,
        "transformers_version": "4.57.6",
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(vllm_config, f, indent=2)
    print(f"[CONVERT] Config saved to {output_dir / 'config.json'}")

    # Load and convert weights
    weights = st.load_file(str(input_dir / "model.safetensors"))

    new_weights = {}
    for name, tensor in weights.items():
        new_name = name
        # Rename layers.0.* -> midlayer.*
        # (vLLM's load_weights reverses this: midlayer.* -> layers.0.*)
        if name.startswith("layers.0."):
            new_name = name.replace("layers.0.", "midlayer.", 1)

        # Convert to bfloat16 if needed
        if tensor.dtype == torch.float32:
            tensor = tensor.to(torch.bfloat16)

        new_weights[new_name] = tensor
        if new_name != name:
            print(f"  {name} -> {new_name}  {tensor.shape}")

    # Add draft-to-target vocab mapping (d2t) if not present
    # When draft_vocab_size == target_vocab_size, use identity mapping
    # When draft_vocab_size < target_vocab_size, a proper mapping is needed
    if "d2t" not in new_weights:
        target_vocab_size = transformer_config["vocab_size"]
        if draft_vocab_size == target_vocab_size:
            # Identity mapping: draft token i -> target token i
            new_weights["draft_id_to_target_id"] = torch.arange(
                draft_vocab_size, dtype=torch.long
            )
            print(f"[CONVERT] Added identity d2t mapping ({draft_vocab_size} tokens)")
        else:
            print(f"[CONVERT] WARNING: draft_vocab_size ({draft_vocab_size}) != "
                  f"target_vocab_size ({target_vocab_size}), d2t mapping needed but not found")

    # Print weight summary
    print(f"\n[CONVERT] Weight summary:")
    for k, v in sorted(new_weights.items()):
        print(f"  {k}: {v.shape} {v.dtype}")

    st.save_file(new_weights, str(output_dir / "model.safetensors"))
    print(f"\n[CONVERT] Weights saved to {output_dir / 'model.safetensors'}")
    print(f"[CONVERT] Done! Model ready at {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to speculators Eagle3DraftModel checkpoint")
    parser.add_argument("--output", type=Path, required=True,
                        help="Path to output vLLM Eagle3 model directory")
    args = parser.parse_args()

    convert(args.input, args.output)
