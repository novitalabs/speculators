# Eagle3 Speculative Decoding Experiments

## Overview

Training Eagle3 speculative decoding draft models for Qwen3-32B using different data sources,
targeting vLLM inference acceleration.

**Pipeline**: data_generation_offline.py → build_vocab_mapping.py → train.py (FSDP)

---

## Experiments

| # | Experiment | Model | Data | Best Top-1 Acc | Status |
|---|-----------|-------|------|---------------|--------|
| [1](experiments/exp01-qwen3-32b-sharegpt-baseline.md) | Qwen3-32B + ShareGPT/UltraChat (Baseline) | Qwen3-32B | ShareGPT+UC 10K | 52% | COMPLETED |
| [2](experiments/exp02-qwen3-32b-novita-v1.md) | Qwen3-32B + Novita v1 (local logs) | Qwen3-32B | Novita 812 | 20.2% | COMPLETED |
| [3](experiments/exp03-qwen3-32b-novita-v2.md) | Qwen3-32B + Novita v2 (HuggingFace) | Qwen3-32B | Novita 5K | - | IN PROGRESS |
| [4](experiments/exp04-minimax-sharegpt-ultrachat.md) | MiniMax-M2.5 + ShareGPT/UltraChat | MiniMax-M2.5 | ShareGPT+UC 10K | 56.0% | COMPLETED |
| [5](experiments/exp05-minimax-streaming-quicktest.md) | MiniMax-M2.5 Streaming (Quicktest) | MiniMax-M2.5 | ShareGPT 100 | ~2% | COMPLETED |
| [6](experiments/exp06-minimax-streaming-5k.md) | MiniMax-M2.5 Streaming (5K) | MiniMax-M2.5 | ShareGPT 5K | 52.9% | COMPLETED |
| [7](experiments/exp07-minimax-novita-812.md) | MiniMax-M2.5 Streaming (Novita 812) | MiniMax-M2.5 | Novita 812 | 81.5%* | COMPLETED |
| [8](experiments/exp08-minimax-streaming-50k.md) | MiniMax-M2.5 Streaming (50K ShareGPT) | MiniMax-M2.5 | ShareGPT 50K | 74.9% | COMPLETED |
| [9](experiments/exp09-minimax-eval-v1.md) | Inference Eval v1 (vLLM Spec Decode) | MiniMax-M2.5 | - | Aurora 1.17x | COMPLETED |
| [10](experiments/exp10-minimax-novita-5k.md) | MiniMax-M2.5 Training (Novita 5K) | MiniMax-M2.5 | Novita 5K | 72.5% | COMPLETED |
| [11](experiments/exp11-minimax-eval-v2.md) | Inference Eval v2 (Novita2 vs Aurora) | MiniMax-M2.5 | - | Aurora 1.01x | COMPLETED |
| [12](experiments/exp12-minimax-novita-full-38k.md) | Full Novita 38K (Online Streaming) | MiniMax-M2.5 | Novita 38K | - | STOPPED |
| [13](experiments/exp13-minimax-novita-full-56k.md) | Full Novita ~56K (No Turn Filter) | MiniMax-M2.5 | Novita ~56K | - | PENDING |

*Domain-specific accuracy (coding agent conversations), not directly comparable.

---

## Reference

- [Infrastructure Notes](experiments/infrastructure-notes.md) — CUDA, NCCL, rsync, vLLM, and other operational notes
