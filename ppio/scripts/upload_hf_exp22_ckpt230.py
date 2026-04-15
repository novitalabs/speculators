"""Upload Aurora-Spec-Minimax-M2.5-Exp22 (ckpt230) to novita/Eagle3-Spec-Minimax-M2.5-Exp22 (private)."""

import os
from huggingface_hub import HfApi, create_repo

HF_TOKEN = os.environ["HF_TOKEN"]
REPO_ID   = "novita/Eagle3-Spec-Minimax-M2.5-Exp22"
MODEL_DIR = "/data/models/Aurora-Spec-Minimax-M2.5-Exp22"

README = """---
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

# Eagle3-Spec-Minimax-M2.5-Exp22

EAGLE3 draft model for speculative decoding with [MiniMax-M2.5](https://huggingface.co/MiniMax/MiniMax-M2.5),
trained with the [speculators](https://github.com/gpu-mode/speculators) framework on a combined
English (novita_merged, 724K) + Chinese (nemotron-v2-chinese, 195K) conversation dataset (~919K total).

## Architecture

Same architecture as [Aurora-Spec-Minimax-M2.1](https://huggingface.co/togethercomputer/Aurora-Spec-Minimax-M2.1):
single-layer Transformer decoder with 24 attention heads, 8192 intermediate size, 32K draft vocab, RoPE theta 5M.

## Performance (ckpt230, TP=4, 512 tokens, seed=42)

### novita_merged_eval (100 prompts)

| Model | Tokens/s | Speedup | Acc@0 | Acceptance Length |
|-------|----------|---------|-------|-------------------|
| Baseline (no spec) | 2704.4 | 1.00x | — | — |
| **This model (ckpt230)** | **4506.1** | **1.67x** | **67.4%** | **2.320** |

### ZClawBench (116 prompts, Chinese agent tasks)

| Model | Tokens/s | Speedup | Acc@0 | Acceptance Length |
|-------|----------|---------|-------|-------------------|
| Baseline (no spec) | 4237.9 | 1.00x | — | — |
| **This model (ckpt230)** | **4458.6** | **1.05x** | **52.1%** | **1.868** |

## Usage with vLLM

```python
from vllm import LLM

llm = LLM(
    model="MiniMax/MiniMax-M2.5",
    speculative_config={
        "model": "novita/Eagle3-Spec-Minimax-M2.5-Exp22",
        "num_speculative_tokens": 3,
        "method": "eagle3",
    },
    tensor_parallel_size=4,
    trust_remote_code=True,
)
```

## Training Details

- **Framework**: [speculators](https://github.com/gpu-mode/speculators) (Eagle3 streaming training)
- **Data**: novita_merged (724K) + nemotron-v2-chinese (195K) = ~919K conversations
- **Hardware**: 8x NVIDIA H200 (FSDP)
- **Checkpoint**: 230 (val_acc@0=0.710)
- **Learning Rate**: 3e-5
"""

api = HfApi(token=HF_TOKEN)

print(f"[1/2] Creating repo {REPO_ID} (private)...")
create_repo(REPO_ID, token=HF_TOKEN, exist_ok=True, repo_type="model", private=True)

# Write README alongside model files
readme_path = f"{MODEL_DIR}/README.md"
with open(readme_path, "w") as f:
    f.write(README)

print(f"[2/2] Uploading {MODEL_DIR} -> {REPO_ID}...")
api.upload_folder(
    folder_path=MODEL_DIR,
    repo_id=REPO_ID,
    repo_type="model",
)
print(f"\nDone! https://huggingface.co/{REPO_ID}")
