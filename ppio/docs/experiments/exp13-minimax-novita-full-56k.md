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
- **Status**: RESTARTING (Epoch 1 crash, fix applied)
- **Progress**:
  - Datagen: COMPLETE on .17 (52,313 files generated)
  - Training (attempt 1): Crashed at Epoch 1 — buffer_cleanup deleted all 11045 training files mid-epoch
  - Training (attempt 2): Preparing restart with coordination fix
  - Note: Used `hostNetwork: true` on training pod for apt-get proxy + hostname resolution fix

## Crash Report: Epoch 1 Buffer Cleanup Race Condition

**Symptom**: Training crashed at start of Epoch 1 with cascading `FileNotFoundError`

**Root cause**: `buffer_cleanup.py` runs every 60s in background. After Epoch 0, `increment_train_count`
bumped all 11045 files to train_count=2. Before the DataLoader could load any file in Epoch 1,
cleanup deleted ALL of them in a single pass. The single-retry fallback in `data.py` also failed
because every alternative file was also deleted.

**Fix applied** (branch `minimax`):
1. `train_streaming.py`: writes `.epoch_in_progress` lock file during training, removed after `increment_train_count`
2. `buffer_cleanup.py`: skips cleanup when lock exists + `--max-delete-per-cycle 5000` + `--min-retain-count 1000`
3. `data.py`: fallback retries increased from 1 to 5
4. `k8s/run_minimax_m2.5_novita_full_train_v2.sh`: re-enabled cleanup with safety flags

**Restart plan**:
- Stopped full rsync (was syncing all 52K files = 10TB, unnecessary)
- Cleaned .18 disk, syncing only latest 5000 files (~940GB) via `rsync --files-from`
- After sync: generate fresh manifest on .18, restart training pod with fixed code
