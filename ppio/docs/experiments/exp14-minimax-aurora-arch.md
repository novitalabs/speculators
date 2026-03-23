# Experiment 14: MiniMax-M2.5 Eagle3 with Aurora-like Architecture

Retrain Eagle3 draft model with Aurora-Spec-M2.1-identical architecture to resolve torch.compile incompatibility from Exp13.

Exp13 checkpoint 60 achieved higher acceptance rate than Aurora (55.3% vs 48.7% Acc@0), but throughput was 2.5x worse (11.4 vs 28.7 tok/s) because the 48-head architecture causes torch.compile cache conflicts, forcing `enforce_eager`. Aurora uses 24 heads which is torch.compile compatible.

- **Nodes**: .18 (training, 8x H200), .17 (datagen source via rsync)
- **Image**: speculators:v0.17.0
- **Data**: Reuses Exp13 datagen data (52K files on .17/.18)
- **Architecture Changes** (match Aurora-Spec-M2.1):

  | Parameter | Exp13 | Exp14 (Aurora-like) |
  |-----------|-------|---------------------|
  | num_attention_heads | 48 | **24** |
  | intermediate_size | 1536 | **8192** |
  | draft_vocab_size | 200064 | **32000** |
  | rope_theta | 10000 (bug) | **5000000** |
  | hidden_size | 3072 | 3072 (same) |
  | num_kv_heads | 8 | 8 (same) |
  | head_dim | 128 | 128 (same) |

- **Config**:
  - Training: 8 GPU FSDP, lr=3e-5, streaming with buffer cleanup
  - Vocab mapping: `build_vocab_mapping.py` with `token_freq.pt` → d2t [32000], t2d [200064]
  - Architecture overrides: `--override-num-attention-heads 24 --override-intermediate-size 8192 --override-rope-theta 5000000`
- **Output**: `/data/output/minimax_m2.5_eagle3_aurora_arch/`
- **Key Files**:
  - `k8s/run_minimax_m2.5_aurora_arch_train.sh` — training script
  - `k8s/k8s-minimax-m2.5-aurora-arch-train.yaml` — training pod (.18)
  - `scripts/train_streaming.py` — added `--override-*` args + rope_theta bug fix
  - `scripts/train.py` — same changes for consistency
- **Changes from Exp 13**:
  - Draft architecture: 48 heads → 24 heads (torch.compile compatible)
  - Draft intermediate: 1536 → 8192 (match Aurora)
  - Draft vocab: 200064 → 32000 (with d2t/t2d mapping from token frequency)
  - rope_theta: 10000 (LlamaConfig default, bug) → 5000000 (verifier's actual value)
  - New output path: `minimax_m2.5_eagle3_aurora_arch`
- **Status**: EVAL COMPLETE — training continuing on .18 (epoch 150+), best checkpoints identified
- **Outcome**: Exp14 ckpt54 beats Aurora-Spec on all metrics: 1.14x vs 1.07x speedup, 52.7% vs 46.2% Acc@0, torch.compile fully compatible
- **Retained Checkpoints**: 49, 54, 59, 62, 65, 68 (others deleted to free disk)

## Progress

- **2026-03-19**: Deployed training pod on .18. First 2 hours blocked waiting for data — rsync was doing a full 52K-file initial sync from .17, and `sync_datagen.sh` only generates manifest after rsync completes. Manually created manifest from 18K local files to unblock training. Fixed `sync_datagen.sh` to bootstrap manifest before first rsync (see Issue 2).
- **2026-03-19**: torch.compile compilation took ~30 minutes on first step (flex_attention backward pass codegen for new 24-head architecture, no cache). Training started producing loss at ~11:44 UTC.
- **2026-03-19**: Training progressing well. Loss dropped from 11.5 (step 1) to ~0.3 by epoch 25. Val metrics at epoch 24: val/loss=3.157, val/full_acc_0=0.806 (already exceeds Exp13 best of 0.782).
- **2026-03-20 07:28**: OOM crash (SIGKILL) at epoch 35. All 8 ranks killed simultaneously by OOM killer. Likely caused by larger model size (intermediate_size 8192 vs 1536) increasing memory footprint during checkpoint save + dataloader workers. Checkpoints 0-34 saved successfully.
- **2026-03-20 08:14**: Restarted pod, resumed from checkpoint 34 into epoch 35. torch.compile recompilation took ~30 min again (no persistent cache across pod restarts).
- **2026-03-20-21**: Training stable through epoch 67. No further OOM incidents.

### Val Metrics (streaming val, unstable set — use `eval_checkpoints.py` for final numbers)

| Epoch | val/loss | val/loss_0 | val/loss_1 | val/loss_2 | val/full_acc_0 | val/full_acc_1 | val/full_acc_2 |
|-------|----------|-----------|-----------|-----------|---------------|---------------|---------------|
| 24 | 3.157 | 0.596 | 1.086 | 1.475 | 0.806 | 0.653 | 0.543 |
| 27 | 3.105 | — | — | — | 0.811 | — | — |
| 41 | 3.141 | — | — | — | — | — | — |
| 54 | **3.077** | 0.548 | 1.062 | 1.467 | 0.793 | 0.641 | 0.540 |
| 60 | 3.103 | 0.545 | 1.069 | 1.489 | 0.798 | 0.644 | 0.540 |
| 66 | 3.198 | — | — | — | 0.791 | — | — |

**Observations**:
- Val loss plateaued around **3.1** from epoch ~40 onward, with best at epoch 54 (3.077)
- val/full_acc_0 ≈ 0.79-0.81 throughout, consistently above Exp13 best (0.782)
- Note: val loss ~3.1 here vs Exp13's 0.729 — **not directly comparable** because Exp14 uses 32K draft vocab (cross-entropy over 32K classes) vs Exp13's 200K vocab. The loss scales are fundamentally different.
- Slight overfitting trend: val loss rising from 3.077 (epoch 54) to 3.198 (epoch 66), while train loss continues dropping
- As with Exp13, streaming val set is unstable due to buffer cleanup/sync. Need fixed-set eval via `eval_checkpoints.py` for reliable comparison.

### Inference Eval Results (2026-03-21, .17, TP=4, 10 Novita prompts × 512 tokens)

| Model | Tokens/s | Speedup | Acc Len | Acc@0 | Acc@1 | Acc@2 |
|-------|----------|---------|---------|-------|-------|-------|
| baseline (no spec) | 70.1 | 1.00x | — | — | — | — |
| Aurora-Spec-M2.1 | 75.0 | 1.07x | 1.757 | 46.2% | 20.2% | 9.3% |
| **Exp14 ckpt54** | **79.9** | **1.14x** | **1.945** | **52.7%** | **26.6%** | **15.3%** |
| Exp14 ckpt60 | 76.2 | 1.09x | 1.860 | 48.2% | 25.2% | 12.6% |
| Exp14 ckpt66 | 53.1 | 0.76x | 1.940 | 51.7% | 27.1% | 15.2% |

**Key findings**:
- **Exp14 ckpt54 beats Aurora-Spec on all metrics**: +6.5% Acc@0 (52.7% vs 46.2%), +10.7% acceptance length (1.945 vs 1.757), +6.6% throughput (79.9 vs 75.0 tok/s), 1.14x vs 1.07x speedup
- **torch.compile fully compatible**: Eagle3 head compilation took only 6.2s (same as Aurora), confirming the architecture change works
- **ckpt60 regresses slightly** vs ckpt54 — consistent with val loss overfitting trend
- **ckpt66 throughput anomaly**: Acc@0 is good (51.7%) but throughput dropped to 53.1 tok/s (0.76x) — possible CUDAGraph issue or transient system load, needs investigation
- Overall: Aurora architecture + domain training = best of both worlds (Aurora's torch.compile compatibility + Exp13's higher acceptance rate)

### Offline Val Eval — All Checkpoints (2026-03-21, .17, 8x H200 FSDP, 20 val files)

Evaluated all 69 checkpoints (0-68) using `eval_checkpoints.py` with fixed val set (seed=42, 200-file split filtered to 20 available files on .17).

| Ckpt | Loss | Loss@0 | Loss@1 | Loss@2 | Acc@0 | Acc@1 | Acc@2 |
|------|------|--------|--------|--------|-------|-------|-------|
| 0 | 6.493 | 1.527 | 2.212 | 2.754 | 66.2% | 40.9% | 26.8% |
| 5 | 3.701 | 0.733 | 1.273 | 1.696 | 77.9% | 57.9% | 44.8% |
| 10 | 3.216 | 0.595 | 1.101 | 1.521 | 80.7% | 62.3% | 50.6% |
| 17 | 2.975 | 0.525 | 1.010 | 1.441 | 81.7% | 64.6% | 52.5% |
| 24 | 2.741 | 0.483 | 0.939 | 1.319 | 82.2% | 65.8% | 55.0% |
| 33 | 2.695 | 0.453 | 0.907 | 1.335 | 84.3% | 68.2% | 57.5% |
| 43 | 2.695 | 0.443 | 0.909 | 1.343 | 84.0% | 67.8% | 57.1% |
| **49** | **2.690** | **0.428** | 0.914 | 1.348 | **85.2%** | **68.9%** | **58.1%** |
| **54** | 2.736 | 0.435 | 0.924 | 1.378 | 84.0% | 67.4% | 56.7% |
| **59** | **2.657** | **0.421** | **0.893** | **1.343** | 84.3% | 69.1% | 58.3% |
| **62** | 2.672 | 0.426 | 0.891 | 1.355 | 84.6% | 69.3% | 59.1% |
| **65** | **2.653** | 0.427 | **0.890** | **1.336** | 83.6% | 68.2% | 57.3% |
| **68** | 2.719 | 0.434 | 0.912 | 1.373 | 84.1% | 69.1% | 58.5% |

**Key findings from offline eval**:
- **Best by total loss**: ckpt 65 (2.653)
- **Best by Acc@0**: ckpt 49 (85.2%)
- Loss curve shows convergence around epoch 30-35, with slow improvement through epoch 65
- Slight overfitting after ckpt 65 (loss rising to 2.72 at ckpt 68)
- All checkpoints in 49-68 range are very close in quality (Acc@0 83.6-85.2%, loss 2.65-2.74)

### Checkpoint Cleanup (2026-03-21)

Deleted all checkpoints except **49, 54, 59, 62, 65, 68** on both .17 and .18.
- .18: 171GB → 16GB (saved 155GB)
- .17: 111GB → 9.6GB + removed 1.3TB gen data
- Rationale: retained checkpoints cover the best-performing range with good spacing; ckpt 54 is best for inference throughput, ckpt 49/65 are best for offline val metrics

### Next Steps

1. Run extended inference eval on ckpt 49 and 65 (the offline eval winners) to see if they beat ckpt 54's throughput
2. Consider stopping training — loss plateau reached, further epochs show diminishing returns
3. Package best checkpoint for production deployment

## Issue Log

### Issue 1: No Manifest on First Start — Training Blocked 2 Hours

**When**: Initial deployment, 2026-03-19 09:24 UTC
**Symptom**: Training stuck in `Waiting for data: 0/5000 files on disk (0 in manifest)` for 2+ hours despite 18K .pt files already synced to local disk.
**Root cause**: `sync_datagen.sh` only calls `update_manifest()` after each rsync loop completes. First full rsync of 52K files from .17 took hours, so no manifest existed during that time. Training requires a manifest to discover files.
**Fix**:
1. Manually created manifest from existing local files to unblock training immediately.
2. Added `update_manifest` call before the sync loop in `sync_datagen.sh` to bootstrap manifest from any pre-existing local files on startup. This ensures training can start immediately with whatever data is already on disk, while rsync brings in more files in the background.

### Issue 2: `build_vocab_mapping.py` CLI Arg Mismatch

**When**: First deployment attempt, 2026-03-19
**Symptom**: `build_vocab_mapping.py: error: the following arguments are required: --output-path`
**Root cause**: Training script used `--token-freq` and `--output-dir`, but the actual CLI args are `--token-freq-path` and `--output-path`.
**Fix**: Corrected arg names in `run_minimax_m2.5_aurora_arch_train.sh`.

### Issue 3: OOM Crash at Epoch 35

**When**: 2026-03-20 07:28 UTC
**Symptom**: All 8 ranks killed by SIGKILL (OOM killer) simultaneously during epoch 35.
**Root cause**: Aurora architecture uses intermediate_size=8192 (5.3x larger than Exp13's 1536), significantly increasing model parameter count and memory footprint. Memory peak likely occurred during checkpoint save (model state dict materialization) combined with dataloader worker memory usage.
**Fix**: Restarted pod — training resumed from checkpoint 34. No recurrence through epoch 67 (may have been a transient memory spike). If recurs, consider reducing `--num-workers` from 4 to 2 to lower baseline memory usage.
