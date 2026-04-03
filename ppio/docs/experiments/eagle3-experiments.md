# Eagle3 Speculative Decoding Experiments

## Overview

Training Eagle3 speculative decoding draft models for Qwen3-32B and MiniMax-M2.5 using different data sources,
targeting vLLM inference acceleration.

**Pipeline**: data_generation_offline.py → build_vocab_mapping.py → train.py (FSDP)

---

## Experiments

| # | Experiment | Model | Data | Best Top-1 Acc | Status |
|---|-----------|-------|------|---------------|--------|
| [1](exp01-qwen3-32b-sharegpt-baseline.md) | Qwen3-32B + ShareGPT/UltraChat (Baseline) | Qwen3-32B | ShareGPT+UC 10K | 52% | COMPLETED |
| [2](exp02-qwen3-32b-novita-v1.md) | Qwen3-32B + Novita v1 (local logs) | Qwen3-32B | Novita 812 | 20.2% | COMPLETED |
| [3](exp03-qwen3-32b-novita-v2.md) | Qwen3-32B + Novita v2 (HuggingFace) | Qwen3-32B | Novita 5K | - | IN PROGRESS |
| [4](exp04-minimax-sharegpt-ultrachat.md) | MiniMax-M2.5 + ShareGPT/UltraChat | MiniMax-M2.5 | ShareGPT+UC 10K | 56.0% | COMPLETED |
| [5](exp05-minimax-streaming-quicktest.md) | MiniMax-M2.5 Streaming (Quicktest) | MiniMax-M2.5 | ShareGPT 100 | ~2% | COMPLETED |
| [6](exp06-minimax-streaming-5k.md) | MiniMax-M2.5 Streaming (5K) | MiniMax-M2.5 | ShareGPT 5K | 52.9% | COMPLETED |
| [7](exp07-minimax-novita-812.md) | MiniMax-M2.5 Streaming (Novita 812) | MiniMax-M2.5 | Novita 812 | 81.5%* | COMPLETED |
| [8](exp08-minimax-streaming-50k.md) | MiniMax-M2.5 Streaming (50K ShareGPT) | MiniMax-M2.5 | ShareGPT 50K | 74.9% | COMPLETED |
| [9](exp09-minimax-eval-v1.md) | Inference Eval v1 (vLLM Spec Decode) | MiniMax-M2.5 | - | Aurora 1.17x | COMPLETED |
| [10](exp10-minimax-novita-5k.md) | MiniMax-M2.5 Training (Novita 5K) | MiniMax-M2.5 | Novita 5K | 72.5% | COMPLETED |
| [11](exp11-minimax-eval-v2.md) | Inference Eval v2 (Novita2 vs Aurora) | MiniMax-M2.5 | - | Aurora 1.01x | COMPLETED |
| [12](exp12-minimax-novita-full-38k.md) | Full Novita 38K (Online Streaming) | MiniMax-M2.5 | Novita 38K | - | STOPPED |
| [13](exp13-minimax-novita-full-56k.md) | Full Novita ~56K (No Turn Filter) | MiniMax-M2.5 | Novita ~52K | 55.3% | COMPLETED |
| [14](exp14-minimax-aurora-arch.md) | Aurora-like Architecture (24 heads) | MiniMax-M2.5 | Novita ~52K | 52.7% | COMPLETED |
| [15](exp15-minimax-novita0320-aurora-arch.md) | Aurora-Arch + novita20260320 (114K) | MiniMax-M2.5 | Novita 114K | 63.2% | COMPLETED |
| [16](exp16-aurora-loss.md) | Aurora Accept/Discard Loss | MiniMax-M2.5 | Novita 6K | 31.7% | COMPLETED |
| [17](exp17-minimax-novita0327-aurora-arch.md) | Aurora-Arch + novita20260327 | MiniMax-M2.5 | Novita 0327 | 43.1% | STOPPED |
| [18](exp18-minimax-merged-diversity.md) | Aurora-Arch + Merged Novita (session-dedup) | MiniMax-M2.5 | Merged 3x Novita | 60.4% | STOPPED |
| [19](exp19-continuous-datagen.md) | Continuous Datagen + Difficulty Resampling | MiniMax-M2.5 | Merged 3x Novita | - | IN PROGRESS |

*Domain-specific accuracy (coding agent conversations), not directly comparable.

---

## Reference

- [Infrastructure Notes](infrastructure-notes.md) — CUDA, NCCL, rsync, vLLM, and other operational notes
