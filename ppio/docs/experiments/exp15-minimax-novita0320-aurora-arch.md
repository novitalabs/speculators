# Experiment 15: MiniMax-M2.5 Eagle3 Aurora-Arch with novita20260320 (Larger Dataset)

Scales up Exp14's Aurora-architecture Eagle3 draft model with a new, larger dataset `weilan55/novita20260320`. Same architecture as Exp14, only the dataset changes.

- **Nodes**: .28 (datagen, 8x H200), .10 (training + eval, 8x H200)
- **Image**: speculators:v0.17.0
- **Dataset**: `weilan55/novita20260320` (downloaded from HF, preprocessed with `--min-turns 2` → 114K conversations, 221K samples with turn dropout)
- **Architecture**: Aurora-like (same as Exp14):

  | Parameter | Value |
  |-----------|-------|
  | num_attention_heads | 24 |
  | intermediate_size | 8192 |
  | draft_vocab_size | 32000 |
  | rope_theta | 5000000 |
  | hidden_size | 3072 |
  | num_kv_heads | 8 |
  | head_dim | 128 |

- **Config**:
  - Datagen: TP=4, batch_size=4, seq_length=8192 on .28
  - Training: 8 GPU FSDP, lr=3e-5, streaming with buffer cleanup on .10
  - Vocab mapping: `build_vocab_mapping.py` with `token_freq.pt` (fallback from Exp14, same verifier)
  - Data synced from .28 to .10 via rsync
- **Output**: `/data/output/minimax_m2.5_eagle3_novita0320/`
- **Key Files**:
  - `k8s/run_minimax_m2.5_novita0320_datagen.sh` — datagen script (downloads HF dataset + generates)
  - `k8s/k8s-minimax-m2.5-novita0320-datagen.yaml` — datagen pod (.28)
  - `k8s/run_minimax_m2.5_novita0320_train.sh` — training script
  - `k8s/k8s-minimax-m2.5-novita0320-train.yaml` — training pod (.10)
  - `k8s/run_minimax_m2.5_eval_exp15.sh` — eval script
  - `k8s/k8s-minimax-m2.5-eval-exp15.yaml` — eval pod (.10)
- **Changes from Exp 14**:
  - New dataset: `weilan55/novita20260320` (114K conversations vs Exp14's 52K from novita20260309)
  - Separate datagen on .28 (Exp14 reused Exp13's datagen data)
  - Training on .10 (Exp14 used .18)
  - Vocab mapping uses Exp14's token_freq.pt (same verifier, similar distribution)
- **Status**: COMPLETE — training epoch 67, eval done
- **Result**: **ckpt67 beats Aurora-Spec** — 63.2% Acc@0 vs 49.4%, 538.5 tok/s (1.01x baseline)

## Datagen

- **Dataset**: `weilan55/novita20260320` → 6.7GB tar.gz → 51GB JSON → 114,013 conversations (--min-turns 2)
- **Samples**: 221,596 (with turn dropout)
- **Output**: 118,917 .pt files on .28 (hit disk full, but sufficient)
- **Duration**: ~8.5 hours (2026-03-21 03:00–12:02 UTC)

## Training Metrics

Training ran on .10 with streaming data sync from .28. Multiple restarts due to disk full (3x) and corrupt file (rsync race condition).

| Epoch | train/loss | train/cond_acc_2 | Notes |
|-------|-----------|-----------------|-------|
| 0 | 9.952 | 0.398 | Initial |
| 5 | 1.861 | 0.327 | |
| 17 | 0.990 | 0.853 | |
| 22 | 0.584 | 0.895 | Peak acc in early training |
| 25 | 0.392 | — | Disk full crash #1 |
| 30 | — | 0.867 | Recovered |
| 37 | 0.214 | 0.664 | Lowest train loss |
| 44 | 0.309 | 0.910 | Peak accuracy |
| 47 | — | — | Disk full crash #2+3, buffer reduced 1024→500→300GB |
| 55 | — | 0.887 | |
| 67 | — | — | Final checkpoint saved |

## Eval Results (Inference)

Evaluated on .10 with TP=4, gpu_memory_utilization=0.95, vLLM v1, 10 Novita prompts × 512 tokens.

| Model | Tokens/s | Speedup | Acc Len | Acc@0 | Acc@1 | Acc@2 |
|-------|----------|---------|---------|-------|-------|-------|
| baseline (no spec) | 531.9 | 1.00x | — | — | — | — |
| Aurora-Spec-M2.1 | 503.2 | 0.95x | 1.765 | 49.4% | 18.8% | 8.3% |
| **exp15_ckpt66** | 441.3 | 0.83x | 1.776 | 45.1% | 22.6% | 9.9% |
| **exp15_ckpt67** | **538.5** | **1.01x** | **2.135** | **63.2%** | **34.8%** | **15.5%** |

**Key findings**:
- **ckpt67 is the best checkpoint** — significantly better than ckpt66 (likely val loss improvement between epochs)
- **ckpt67 beats Aurora-Spec on all metrics**: 63.2% vs 49.4% Acc@0, 2.135 vs 1.765 acceptance length
- **ckpt67 achieves slight speedup over baseline** (1.01x) with torch.compile working — Aurora-Spec was actually slower than baseline (0.95x)
- ckpt66 underperforms, suggesting checkpoint selection matters — need proper val-set eval to find the true best checkpoint
- The larger novita20260320 dataset (114K vs 52K) with Aurora architecture produces a competitive speculative decoding model

## Progress

- **2026-03-21 03:00**: Datagen pod deployed on .28. Downloaded dataset (6.7GB), extracted, preprocessed 114K conversations.
- **2026-03-21 03:24**: Datagen started generating .pt files (~1.3 it/s).
- **2026-03-21 03:46**: Training pod deployed on .10 after 1.7K files generated. Multiple issues: HF cache race condition (MiniMaxM2Config), manifest desync, token_freq.pt wait.
- **2026-03-21 04:00**: Training started producing loss. torch.compile took ~30 min on first step.
- **2026-03-21 07:41**: First disk full crash (epoch 25). Gen buffer grew to 994GB. Freed space, reduced buffer 1024→500GB.
- **2026-03-21 10:55**: Second disk full crash (epoch 30). Buffer cleanup not enforcing limit. Reduced to 300GB.
- **2026-03-21 11:00**: Recurring manifest desync issue — rsync too slow with 100K+ source files, manifest not updating. Set up automatic manifest fix cron.
- **2026-03-21 12:02**: Datagen completed (118,917 .pt files, hit disk full on .28).
- **2026-03-21 12:55**: Training crashed with corrupt .pt file (rsync race condition). Multiple restart attempts.
- **2026-03-21 13:20**: Training stable again, reached epoch 67. Final crash on corrupt file after epoch 68 attempt.
- **2026-03-23 03:50**: Eval deployed on .10. Initial failures (stale GPU processes, TP=8 sharding error). Fixed with TP=4, gpu_memory_utilization=0.95.
- **2026-03-23 04:14**: Eval complete. ckpt67 beats Aurora-Spec and baseline.

## Issue Log

### Issue 1: Disk Full Crashes (3x)
**Root cause**: Buffer cleanup (`buffer_cleanup.py`) not enforcing size limit reliably. Gen directory grew to 994GB despite 500GB/300GB limits.
**Impact**: Training crashed 3 times at epochs 25, 30, and 47. Corrupted checkpoints on crash.
**Mitigation**: Reduced buffer limit progressively (1024→500→300GB), manually deleted old gen files and checkpoints after each crash.

### Issue 2: Manifest Desync
**Root cause**: `sync_datagen.sh` only updates manifest after rsync completes a full cycle. With 100K+ files on source, first rsync never completes quickly, so new files on disk aren't reflected in manifest.
**Impact**: Training stalled between epochs ("Waiting for files: N/5000") despite having enough files on disk.
**Mitigation**: Set up automatic manifest rebuild cron (every 3 min) to detect stalls and fix.

### Issue 3: Corrupt .pt Files (rsync race condition)
**Root cause**: rsync writes files in-place. Training DataLoader reads a file while rsync is still writing it → `PytorchStreamReader failed reading zip archive`.
**Impact**: Training crashes with `FileNotFoundError` or `RuntimeError` mid-epoch.
**Mitigation**: Restart training (resumes from last checkpoint). Files are eventually fully synced.

### Issue 4: GPU Memory (TP=4 vs TP=8)
**Root cause**: MiniMax MoE expert FFN dimensions (192) not divisible by TP=8 quantization block (128). TP=4 works but requires 0.95 gpu_memory_utilization on H200.
**Impact**: Eval failed with TP=8 (sharding error) and TP=4 at 0.85 utilization (OOM).
**Fix**: Use TP=4 with gpu_memory_utilization=0.95.

## ZClawBench Eval Results (2026-03-24, .18, TP=4, 116 agent prompts × 512 tokens)

Evaluated on [ZClawBench](https://huggingface.co/datasets/zai-org/ZClawBench) — 116 real agent task prompts covering code, office tasks, data analysis, automation, security. Tests speculative decoding on out-of-distribution agent scenarios.

| Model | Tokens/s | Speedup | Acc Len | Acc@0 | Acc@1 | Acc@2 |
|-------|----------|---------|---------|-------|-------|-------|
| baseline (no spec) | 215.3 | 1.00x | — | — | — | — |
| Aurora-Spec-M2.1 | 198.7 | 0.92x | 1.465 | 29.8% | 11.8% | 5.0% |
| Exp14 ckpt54 | 219.2 | 1.02x | 1.476 | 34.1% | 10.1% | 3.4% |
| **Exp15 ckpt67** | **225.8** | **1.05x** | **1.589** | **39.6%** | **14.2%** | **5.1%** |

**Key findings (ZClawBench)**:
- **Exp15 ckpt67 is the best model on agent tasks** — 1.05x speedup, 39.6% Acc@0
- **Aurora-Spec slows down inference** on agent prompts (0.92x) — Acc@0 only 29.8%
- **Exp15 beats Aurora by +9.8pp Acc@0** and turns a 0.92x slowdown into a 1.05x speedup
- Agent scenario Acc@0 much lower than chat across all models (39.6% vs 63.2% for Exp15) due to structured output (tool calls, code)
- **Larger training dataset pays off** — Exp15 (114K novita0320) > Exp14 (52K novita0309) on agent tasks too

### ZClawBench Eval on M2.1 (2026-03-25, .18, TP=4, 116 agent prompts × 512 tokens)

Cross-model test: draft models trained for M2.5, evaluated on M2.1 as base model.

| Model | Tokens/s | Speedup | Acc Len | Acc@0 | Acc@1 | Acc@2 |
|-------|----------|---------|---------|-------|-------|-------|
| baseline (M2.1) | 237.1 | 1.00x | — | — | — | — |
| Aurora-Spec (M2.1 native) | 209.0 | 0.88x | 1.377 | 23.3% | 9.9% | 4.6% |
| Exp14 ckpt54 (M2.5 trained) | 225.4 | 0.95x | 1.432 | 30.9% | 9.3% | 3.0% |
| **Exp15 ckpt67** (M2.5 trained) | **231.4** | **0.98x** | **1.513** | **35.4%** | **11.8%** | **4.1%** |

**Key findings (M2.1)**:
- Exp15 ckpt67 nearly matches baseline on M2.1 (0.98x), best among all draft models
- Aurora-Spec, even on its native M2.1, slows inference on agent prompts (0.88x)
- Exp15 transfers well across models: 35.4% Acc@0 on M2.1 vs 39.6% on M2.5

### Cross-benchmark comparison (all on ZClawBench agent prompts)

| Draft Model | M2.5 Speedup | M2.1 Speedup | M2.5 Acc@0 | M2.1 Acc@0 |
|-------------|-------------|-------------|-----------|-----------|
| Aurora-Spec-M2.1 (1.7GB) | 0.92x | 0.88x | 29.8% | 23.3% |
| Aurora-Spec-M2.5 (5.1GB) | 0.23x* | — | 32.9% | — |
| Exp14 ckpt54 (1.7GB) | 1.02x | 0.95x | 34.1% | 30.9% |
| **Exp15 ckpt67 (1.7GB)** | **1.05x** | **0.98x** | **39.6%** | **35.4%** |

*Aurora-Spec-M2.5 throughput (224.8 tok/s) measured in a separate session with different baseline (960.7 tok/s vs 215.3). The 3x larger model size creates excessive spec decode overhead despite slightly better Acc@0 than M2.1 version.

## Next Steps

1. Run `eval_checkpoints.py` on more checkpoints (e.g., 56-67) with fixed val set to find the true best checkpoint
2. Test with higher `num_speculative_tokens` (e.g., 5) to see if acceptance length scales
3. Consider production deployment — Exp15 ckpt67 provides consistent speedup across both chat and agent workloads
