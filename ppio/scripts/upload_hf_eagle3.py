import json
import os
import sys
import torch
from safetensors.torch import load_file, save_file
from huggingface_hub import HfApi, create_repo

HF_TOKEN = os.environ["HF_TOKEN"]
api = HfApi(token=HF_TOKEN)

def convert_and_upload(ckpt_path, repo_id, exp_name, readme_text):
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Uploading {exp_name} -> {repo_id}")
    print(sep)

    work_dir = f"/tmp/hf_upload_{exp_name}"
    os.makedirs(work_dir, exist_ok=True)

    # 1. Convert weights: layers.0.* -> midlayer.*
    print("[1/4] Converting weights...")
    tensors = load_file(os.path.join(ckpt_path, "model.safetensors"))
    new_tensors = {}
    for k, v in tensors.items():
        new_key = k.replace("layers.0.", "midlayer.")
        new_tensors[new_key] = v
    save_file(new_tensors, os.path.join(work_dir, "model.safetensors"))
    print(f"  Converted {len(tensors)} tensors")

    # 2. Create Aurora-format config.json
    print("[2/4] Creating config.json...")
    config = {
        "architectures": ["LlamaForCausalLMEagle3"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "draft_vocab_size": 32000,
        "dtype": "bfloat16",
        "eos_token_id": 2,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 3072,
        "initializer_range": 0.02,
        "intermediate_size": 8192,
        "max_position_embeddings": 196608,
        "mlp_bias": False,
        "model_type": "llama",
        "num_attention_heads": 24,
        "num_hidden_layers": 1,
        "num_key_value_heads": 8,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-06,
        "rope_scaling": None,
        "rope_theta": 5000000,
        "tie_word_embeddings": False,
        "transformers_version": "4.57.6",
        "use_cache": True,
        "vocab_size": 200064
    }
    with open(os.path.join(work_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # 3. Create README.md
    print("[3/4] Creating README.md...")
    with open(os.path.join(work_dir, "README.md"), "w") as f:
        f.write(readme_text)

    # 4. Upload to HF
    print("[4/4] Uploading to HuggingFace...")
    create_repo(repo_id, token=HF_TOKEN, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=work_dir,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"  Done! https://huggingface.co/{repo_id}")


EXP14_README = """---
language:
- en
- zh
license: apache-2.0
tags:
- speculative-decoding
- eagle3
- minimax
base_model: MiniMax/MiniMax-M2.5
pipeline_tag: text-generation
---

# Eagle3-Spec-Minimax-M2.5-Exp14

EAGLE3 draft model for speculative decoding with MiniMax-M2.5, trained with the [speculators](https://github.com/gpu-mode/speculators) framework on Novita API conversation data (52K samples from novita20260309).

## Architecture

Same architecture as [Aurora-Spec-Minimax-M2.1](https://huggingface.co/togethercomputer/Aurora-Spec-Minimax-M2.1): single-layer Transformer decoder with 24 attention heads, 8192 intermediate size, 32K draft vocab.

## Performance

### Novita Chat Prompts (10 prompts x 512 tokens, TP=4)

| Model | Tokens/s | Speedup | Acc@0 | Acceptance Length |
|-------|----------|---------|-------|-------------------|
| Baseline (no spec) | 70.1 | 1.00x | - | - |
| Aurora-Spec-M2.1 | 75.0 | 1.07x | 46.2% | 1.757 |
| **This model (ckpt54)** | **79.9** | **1.14x** | **52.7%** | **1.945** |

### ZClawBench Agent Prompts (116 prompts x 512 tokens, TP=4)

| Model | Tokens/s | Speedup | Acc@0 | Acceptance Length |
|-------|----------|---------|-------|-------------------|
| Baseline (no spec) | 215.3 | 1.00x | - | - |
| Aurora-Spec-M2.1 | 198.7 | 0.92x | 29.8% | 1.465 |
| **This model (ckpt54)** | **219.2** | **1.02x** | **34.1%** | **1.476** |

## Usage with vLLM

```python
from vllm import LLM

llm = LLM(
    model="MiniMax/MiniMax-M2.5",
    speculative_config={
        "model": "novita/Eagle3-Spec-Minimax-M2.5-Exp14",
        "num_speculative_tokens": 3,
        "method": "eagle3",
    },
    tensor_parallel_size=4,
    trust_remote_code=True,
)
```

## Training Details

- **Framework**: [speculators](https://github.com/gpu-mode/speculators) (Eagle3 streaming training)
- **Data**: 52K Novita API conversations (novita20260309)
- **Hardware**: 8x NVIDIA H200 (FSDP)
- **Epochs**: 54 (best checkpoint by throughput from 69 evaluated)
- **Learning Rate**: 3e-5
"""

EXP15_README = """---
language:
- en
- zh
license: apache-2.0
tags:
- speculative-decoding
- eagle3
- minimax
base_model: MiniMax/MiniMax-M2.5
pipeline_tag: text-generation
---

# Eagle3-Spec-Minimax-M2.5-Exp15

EAGLE3 draft model for speculative decoding with MiniMax-M2.5, trained with the [speculators](https://github.com/gpu-mode/speculators) framework on a larger Novita API conversation dataset (114K samples from novita20260320).

## Architecture

Same architecture as [Aurora-Spec-Minimax-M2.1](https://huggingface.co/togethercomputer/Aurora-Spec-Minimax-M2.1): single-layer Transformer decoder with 24 attention heads, 8192 intermediate size, 32K draft vocab.

## Performance

### Novita Chat Prompts (10 prompts x 512 tokens, TP=4)

| Model | Tokens/s | Speedup | Acc@0 | Acceptance Length |
|-------|----------|---------|-------|-------------------|
| Baseline (no spec) | 531.9 | 1.00x | - | - |
| Aurora-Spec-M2.1 | 503.2 | 0.95x | 49.4% | 1.765 |
| **This model (ckpt67)** | **538.5** | **1.01x** | **63.2%** | **2.135** |

### ZClawBench Agent Prompts (116 prompts x 512 tokens, TP=4)

| Model | Tokens/s | Speedup | Acc@0 | Acceptance Length |
|-------|----------|---------|-------|-------------------|
| Baseline (no spec) | 215.3 | 1.00x | - | - |
| Aurora-Spec-M2.1 | 198.7 | 0.92x | 29.8% | 1.465 |
| **This model (ckpt67)** | **225.8** | **1.05x** | **39.6%** | **1.589** |

## Usage with vLLM

```python
from vllm import LLM

llm = LLM(
    model="MiniMax/MiniMax-M2.5",
    speculative_config={
        "model": "novita/Eagle3-Spec-Minimax-M2.5-Exp15",
        "num_speculative_tokens": 3,
        "method": "eagle3",
    },
    tensor_parallel_size=4,
    trust_remote_code=True,
)
```

## Training Details

- **Framework**: [speculators](https://github.com/gpu-mode/speculators) (Eagle3 streaming training)
- **Data**: 114K Novita API conversations (novita20260320, with turn dropout augmentation -> 221K samples)
- **Hardware**: 8x NVIDIA H200 (FSDP)
- **Epochs**: 67 (best checkpoint)
- **Learning Rate**: 3e-5
"""

convert_and_upload(
    "/data/output/minimax_m2.5_eagle3_aurora_arch/checkpoints/54",
    "novita/Eagle3-Spec-Minimax-M2.5-Exp14",
    "exp14_ckpt54",
    EXP14_README,
)

convert_and_upload(
    "/data/output/minimax_m2.5_eagle3_novita0320/checkpoints/67",
    "novita/Eagle3-Spec-Minimax-M2.5-Exp15",
    "exp15_ckpt67",
    EXP15_README,
)
