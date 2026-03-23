# Experiment 13: MiniMax-M2.5 + Full Novita ~56K (No Turn Filter, Online Streaming)

Full-scale online training with ALL Novita API production logs (~56K conversations, no turn filter).
Exp 12 used `--min-turns 4` which reduced dataset from ~56K to 38K. This experiment uses `--min-turns 2` (default) to keep all conversations with at least 1 user + 1 assistant message.

- **Nodes**: .17 (datagen, 8x H200) + .18 (training, 8x H200)
- **Image**: speculators:v0.17.0
- **Data**: weilan55/novita20260309 full dataset (no turn filter)
  - Raw: 799,622 API log records → 52,313 conversations (--min-turns 2)
  - Preprocessed: `preprocess_novita_logs.py --max-samples 0 --min-turns 2`
  - JSONL: `conversations_full_no_filter.jsonl`
- **Config**:
  - Datagen: TP=4, batch_size=4, seq_len=8192, gpu_memory_utilization=0.85
  - Training: 8 GPU FSDP, lr=3e-5, final_epochs=10, max_val_files=200
  - Ring buffer: 1TB cap on .18, buffer_cleanup.py with min_train_count=2
  - NCCL_TIMEOUT=3600
- **Pipeline**: Online streaming — datagen on .17 → rsync → train_streaming.py on .18
- **Output**: `/data/output/minimax_m2.5_eagle3_novita_full_v2/`
- **Key Files**:
  - `k8s/run_minimax_m2.5_novita_full_datagen_v2.sh` — datagen script
  - `k8s/run_minimax_m2.5_novita_full_train_v2.sh` — training + rsync + cleanup
  - `k8s/k8s-minimax-m2.5-novita-full-datagen-v2.yaml` — datagen pod (.17)
  - `k8s/k8s-minimax-m2.5-novita-full-train-v2.yaml` — training pod (.18)
- **Changes from Exp 12**:
  - `--min-turns 4` → `--min-turns 2` (no turn filter, ~56K vs 38K conversations)
  - Datagen node: .17, Training node: .18 (originally planned .21/.22 but those have production dynamo pods)
  - New output path: `minimax_m2.5_eagle3_novita_full_v2`
- **Status**: COMPLETE — best model at checkpoint 60, val loss 0.729
- **Final Model**: Checkpoint 60 at `/data/output/minimax_m2.5_eagle3_novita_full_v2/checkpoints/60/`
- **Final Metrics** (checkpoint 60, fixed val set):

  | Metric | Layer 0 | Layer 1 | Layer 2 | Total |
  |--------|---------|---------|---------|-------|
  | loss | 0.112 | 0.247 | 0.369 | **0.729** |
  | cond_acc | 0.782 | 0.742 | 0.744 | — |
  | full_acc | 0.782 | 0.692 | 0.618 | — |

- **Progress**:
  - Datagen (attempt 1): .17 — generated 52,313 files, but disk cleaned up, only 5K remained. .17 now occupied by PROD.
  - Datagen (attempt 2): .21 — new datagen started, completed ~13K batches
  - Training: Epoch 0-9 on .18, migrated to .23 after .18 GPU taken by mofeite
  - Training: Epoch 10-12 on .23, with two NCCL incidents (Issues 6, 8) and one insufficient-files crash (Issue 7)
  - Checkpoints 0-12 saved. Checkpoint 13 lost due to validation hang.
  - V4 buffer system verified working: eviction ledger, train_count tracking, cleanup coordination
  - Note: Used `hostNetwork: true` on training pod for apt-get proxy + hostname resolution fix
  - Note: .23 may have NCCL hardware instability — consider migrating to another node if issues persist
  - **2026-03-18**: Resumed on .17 + .18. Synced checkpoints 10-12 from .23→.18, synced 26K gen files from .23→.17 and .23→.18. Cleaned stale manifest on .17 (had `status:complete` with 52313 entries but only 5K .pt files, causing datagen to skip). Datagen restarted fresh on .17, training resumed from checkpoint 12 into epoch 13 on .18 with 18864 local files.
  - **2026-03-19**: Training ran to epoch 135. Observed apparent overfitting in training-time val loss (rose from 1.79 at epoch 20 to 2.99 at epoch 134), but this was caused by **unstable val set** — buffer cleanup/sync changed the files available each epoch, so val set composition shifted over time. Re-evaluated checkpoints 19-135 with a **fixed val set** (100 files, deterministic split using `scripts/eval_checkpoints.py`). True val loss curve: 1.007 (ckpt 19) → 0.729 (ckpt 60, best) → 0.823 (ckpt 135). Overfitting starts around epoch 60-80, primarily in layers 1 and 2.
- **Inference Eval** (2026-03-19, novita20260312_eval held-out data, 10 prompts × 512 tokens, vLLM on .18):

  | Model | Tokens/s | Acc Length | Acc@0 | Acc@1 | Acc@2 | Notes |
  |-------|----------|-----------|-------|-------|-------|-------|
  | Baseline (no spec) | 103.8 | — | — | — | — | torch.compile |
  | **Exp13 ckpt60** | 11.4 | **1.887** | **55.3%** | **23.1%** | **10.3%** | enforce_eager |
  | Aurora-Spec-M2.1 | 28.7 | 1.749 | 48.7% | 18.8% | 7.5% | torch.compile |

  **Acceptance rate**: Ckpt60 beats Aurora by +6.6% Acc@0 (55.3% vs 48.7%) and +7.9% acceptance length (1.887 vs 1.749), showing domain-trained models significantly outperform general-purpose Aurora on Novita traffic.

  **Throughput issue (TODO)**: Both speculative models are slower than baseline — this is wrong and needs investigation:
  - Ckpt60 (11.4 tok/s): Uses `enforce_eager` due to 48-head architecture causing torch.compile cache conflict with target model. The 48 heads match MiniMax-M2.5's `num_attention_heads=48`, but Aurora uses 24 heads which avoids the conflict. **Fix**: Retrain with `num_attention_heads=24` to enable torch.compile, or investigate the torch.compile cache conflict root cause.
  - Aurora (28.7 tok/s): Despite torch.compile, still slower than baseline (103.8). Likely due to speculative decoding overhead on MoE models — the draft+verify cycle adds latency that isn't offset by the ~49% acceptance rate. May need higher acceptance rates (>60%) or more speculative tokens to achieve speedup on MoE.
  - Both results may also be affected by first-run compilation overhead amortized over only 10 short prompts. Need to test with more prompts and exclude warmup.

- **Lessons learned**:
  - Online streaming training with buffer eviction causes val set instability — val metrics logged during training are unreliable. Always re-evaluate with a fixed val set.
  - Created `scripts/eval_checkpoints.py` for offline checkpoint evaluation with FSDP support.
  - Domain-specific training (52K Novita conversations) yields significantly higher acceptance rates than general-purpose Aurora-Spec, but architectural choices (num_attention_heads) impact torch.compile compatibility and throughput.

## Issue Log

### Issue 1: Epoch 1 Buffer Cleanup Race — All Files Deleted (V3)

**When**: First training attempt on .18
**Symptom**: Training crashed at start of Epoch 1 with cascading `FileNotFoundError`
**Root cause**: `buffer_cleanup.py` runs every 60s in background. After Epoch 0, `increment_train_count`
bumped all 11045 files to train_count=2. Cleanup deleted ALL of them in a single pass (no safety limits).
**Fix**:
1. `train_streaming.py`: writes `.epoch_in_progress` lock file during training
2. `buffer_cleanup.py`: skips cleanup when lock exists + `--max-delete-per-cycle 5000` + `--min-retain-count 1000`
3. `data.py`: fallback retries increased from 1 to 5

### Issue 2: V4 Manifest Overwrite — train_count Always 0

**When**: After V4 deployment, epochs 6-9 on .18
**Symptom**: Manifest showed `train_count=0` for all files despite 9 completed epochs. V4 extra fields (`total_remote_files`, `global_epoch`, `files_ever_seen_count`) missing.
**Root cause**: `sync_datagen.sh` rsync command included `--include='manifest.json'`, which copied the REMOTE manifest (from datagen node, where all train_count=0) to local dir every 30s, overwriting training's `increment_train_count` writes.
**Fix**: Removed `--include='manifest.json'` from rsync in `sync_datagen.sh`. The local `update_manifest()` function handles manifest generation from local file scan.

### Issue 3: sync_datagen.sh Crash — `du` + `pipefail`

**When**: After fixing Issue 2, restarting sync process
**Symptom**: Sync process silently crashed immediately after restart
**Root cause**: `du -sk` on the gen directory encountered a temporary rsync file (`.data_17653.pt.HXYi6A`) that disappeared during scan. With `set -euo pipefail`, the non-zero exit code from `du` propagated through the awk pipeline and killed the script.
**Fix**: Wrapped du in subshell with `|| true`: `current_size_kb=$( (du -sk "$LOCAL_DIR" 2>/dev/null || true) | awk '{print $1}')`

### Issue 4: Epoch 10 Crash — Cleanup Evicted Files During Training

**When**: Epoch 9→10 transition on .18
**Symptom**: `FileNotFoundError: data_1932.pt` during epoch 10 training
**Root cause**: Cleanup checked epoch lock during the brief window when it was released between epochs (line 349), then spent ~45s evicting 5000 files while epoch 10 was already running. The epoch lock was re-set at the top of the loop (line 303), but cleanup didn't re-check it during its eviction loop.
**Fix** (3 changes):
1. `buffer_cleanup.py`: Re-check epoch lock inside eviction loop — stops early if new epoch starts
2. `train_streaming.py`: Added 90s sleep after releasing epoch lock to let cleanup finish before rebuilding DataLoader
3. `data.py`: Increased max_retries from 5 to 50, with random index fallback after first 5 sequential retries

### Issue 5: .18 GPU Taken — Node Migration

**When**: After epoch 10 crash, attempting to restart
**Symptom**: `UnexpectedAdmissionError: Requested: 8, Available: 0` — all 8 GPUs occupied by `mofeite-vllm-kimi-25-h200`
**Fix**: Migrated training to .23 — synced model, code, checkpoints, manifest from .18

### Issue 6: NCCL Timeout on .23 — Epoch 10

**When**: First training attempt on .23, epoch 10 (resumed from checkpoint 9)
**Symptom**: Training hung for 3600s during forward pass, NCCL watchdog triggered: `SeqNum=3511`, all GPUs at 100% but no progress
**Root cause**: Transient GPU communication failure on .23 (possibly hardware-related)
**Fix**: Restarted pod — training resumed from checkpoint 9

### Issue 7: Epoch 12→13 Crash — Insufficient Files After Cleanup

**When**: Transition from epoch 12 to epoch 13 on .23
**Symptom**: `IndexError` in `MultipackDistributedBatchSamplerV2` — only 404 train files remained after cleanup eviction
**Root cause**: After epoch lock release, cleanup evicted files aggressively, but the fixed 90s sleep wasn't enough for sync's `update_manifest()` to run and replenish file list. DataLoader was rebuilt with only 404 files (below `min_samples=5000`).
**Fix**: Replaced fixed 90s sleep with a `wait_for_min_files` polling loop in `train_streaming.py` that waits until `min_samples` train files exist on disk before rebuilding the DataLoader.

### Issue 8: Validation Hang After Epoch 13

**When**: After epoch 13 training completed on .23
**Symptom**: Validation stuck for 30+ minutes — GPU at 100% utilization but no log output. Likely NCCL deadlock during `dist.reduce` in validation.
**Root cause**: Second NCCL instability incident on .23 (similar to Issue 6), this time during validation's distributed reduction
**Fix**: Restarted pod. Checkpoint 13 was not saved (hang occurred before save). Resumed from checkpoint 12.
