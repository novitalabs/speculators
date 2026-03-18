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
- **Status**: RUNNING on .23 (resumed from checkpoint 9)
- **Progress**:
  - Datagen (attempt 1): .17 — generated 52,313 files, but disk cleaned up, only 5K remained. .17 now occupied by PROD.
  - Datagen (attempt 2): .21 — new datagen started, completed ~13K batches
  - Training: Epoch 0-9 on .18, migrated to .23 after .18 GPU taken by mofeite
  - V4 buffer system verified working: eviction ledger, train_count tracking, cleanup coordination
  - Note: Used `hostNetwork: true` on training pod for apt-get proxy + hostname resolution fix

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
