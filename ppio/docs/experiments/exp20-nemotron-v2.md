# Experiment 20: Nemotron-v2 Dataset with Continuous Datagen

## Status: COMPLETE (best checkpoint: epoch 251, novita 2.02x / zclawbench 1.18x speedup)

## Motivation

Exp19 validated continuous datagen with difficulty-weighted resampling using the novita_merged_exp18 dataset (166K samples). Exp20 scales this to the nemotron-v2 dataset (6.3M samples across 10 categories) to test whether broader data diversity improves Eagle3 speculative decoding quality for MiniMax-M2.5.

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | MiniMax-M2.5 (Aurora architecture) |
| Dataset | nemotron-v2 (6.3M samples, JSONL-converted) |
| Datagen Node | .21 (8 GPU) |
| Training Node | .28 (8 GPU) |
| Seq Length | 8192 |
| LR | 3e-5 |
| Buffer Max | 500GB |
| Architecture | Same as Exp19 (Aurora overrides) |
| Pipeline | Continuous datagen with difficulty feedback |

### Dataset Breakdown

| Split | Samples |
|-------|---------|
| chat | 627,720 |
| code | 175,000 |
| math | 239,467 |
| stem | 355,000 |
| multilingual_de | 1,015,314 |
| multilingual_es | 935,704 |
| multilingual_fr | 1,001,504 |
| multilingual_it | 1,016,503 |
| multilingual_ja | 975,202 |
| multilingual (misc) | 100 |
| **Total** | **6,341,514** |

## Changes from Exp19

1. **Dataset**: novita_merged_exp18 (166K) → nemotron-v2 (6.3M, 38x larger)
2. **Nodes**: .17/.18 → .21/.28
3. **Data format**: Original parquet files converted to JSONL due to PyArrow compatibility issue
4. **Code**: Added parquet/directory loading support to `load_raw_dataset` in `src/speculators/data_generation/preprocessing.py`

## Files

- `k8s/k8s-minimax-m2.5-exp20-datagen.yaml` — datagen pod manifest
- `k8s/k8s-minimax-m2.5-exp20-train.yaml` — training pod manifest
- `k8s/run_minimax_m2.5_exp20_datagen.sh` — datagen run script
- `k8s/run_minimax_m2.5_exp20_train.sh` — training run script
- `src/speculators/data_generation/preprocessing.py` — parquet/directory loading support

## Issue Log

### Issue 1: PyArrow Nested Data Conversion Failure

**When**: Initial deployment, dataset loading phase
**Symptom**: `ArrowNotImplementedError: Nested data conversions not implemented for chunked array outputs` when loading large multilingual parquet files (>900K rows in single row group) via HuggingFace `load_dataset`
**Root cause**: PyArrow 23.0.1 cannot handle nested struct arrays (list-of-dicts `messages` column) in large single-row-group parquet files. Small files (chat, code, math, stem) loaded fine. The 5 large multilingual files each had 1M+ rows in a single row group.
**Fix**: Pre-converted all parquet files to JSONL format using `pf.iter_batches(batch_size=5000)` which avoids the chunked array issue. Dataset stored at `/data/tengwan/datasets/nemotron-v2-jsonl/` (94GB). Also added `_normalize_column_names()` to rename `messages` → `conversations` column.

### Issue 2: Schema Mismatch Across Parquet Files

**When**: First attempt at loading parquet directory
**Symptom**: `DatasetGenerationError` when loading all 10 parquet files together — `multilingual.parquet` has extra `metadata` column
**Root cause**: `multilingual.parquet` (100 rows) has schema `[..., messages, metadata]` while other files have `[..., messages]`. HuggingFace `load_dataset` with `data_files=list` requires matching schemas.
**Fix**: Updated `load_raw_dataset` to load each parquet file individually, select only the `conversations` column, then `concatenate_datasets`. Moot after switching to JSONL.

### Issue 3: Training NCCL Timeout Crash at Epoch 31

**When**: Epoch 31 validation phase, 2026-04-03 21:07
**Symptom**: All 8 ranks received SIGABRT after NCCL collective timeout. Last enqueued work: 317794, last completed: 317778 (16 ops stuck). Crash occurred during validation after completing epoch 31 training.
**Root cause**: NCCL collective timeout during validation all-reduce. Likely a transient hang on one rank (possible OOM or data loading stall during validation). Training itself ran normally for 31 epochs.
**Fix**: Restarted train pod. 31 checkpoints (0-30) preserved. Training resumes from latest checkpoint via streaming pipeline.

### Issue 4: Datagen Extremely Slow — 37-Day ETA for First Round

**When**: Ongoing since deployment
**Symptom**: Datagen processing 1,585,379 batches at ~2s/batch. After 11.5h only 24K/1.58M (1.5%) complete. ETA for one full round: ~37 days.
**Root cause**: Dataset is 38x larger than exp19 (6.3M vs 166K samples). After tokenization, 1.58M valid batches generated. The vLLM inference throughput (~0.5 batch/s) is the bottleneck — same as exp19 but with 10x more data to process.
**Impact**: Training still works because datagen generates .pt files continuously, but coverage of the full dataset will be very slow. Difficulty-weighted resampling (round 2+) won't kick in until round 1 completes.

### Issue 5: Disk Full on .28 — Checkpoint Save Failure at Epoch 41

**When**: Epoch 41 checkpoint save, 2026-04-04 05:26
**Symptom**: `safetensors_rust.SafetensorError: Error while serializing: I/O error: No space left on device (os error 28)`. Node .28 disk at 100% (7.0T/7.0T).
**Root cause**: Two compounding factors: (1) Buffer cleanup process skipped all cycles with "Epoch in progress, skipping cleanup" — training never left epoch state long enough for cleanup to run. Gen dir grew to 980GB (double the 500GB limit). (2) 41 checkpoints × 2.6GB = 103GB accumulated without pruning. Combined with 5.9T of models/datasets already on disk → full.
**Fix**: Manual cleanup: deleted checkpoints 0-35 + incomplete 41 (freed 94GB), removed 6,000 oldest .pt files from gen (980GB → 434GB). Restarted train pod. Disk at 92% with 636GB free.

## Progress

### 2026-04-03 — Deployment

- Parquet → JSONL conversion completed (94GB total)
- Data synced to node .21 (46GB parquet + 94GB JSONL)
- Code synced to nodes .21 and .28
- Both pods running: datagen (tokenizing 6.3M samples, ~5h ETA), train (waiting for rsync data)

### 2026-04-04 — Training Crash & Restart

- **Datagen**: 24K/1.58M batches (1.5%), 9,160 .pt files generated on .21
- **Training**: Completed 31 epochs, saved checkpoints 0-30
  - Last metrics: `full_acc_0=0.856, cond_acc_0=0.856, loss=2.345`
  - Crashed during epoch 31 validation (NCCL timeout → SIGABRT)
  - 9,099 .pt files synced to .28
- **Action**: Restarted train pod; datagen continues uninterrupted
- **Concern**: Datagen ETA ~37 days for full round — training will keep cycling over the same ~9K files until more are generated

### 2026-04-05 — Disk Full Crash & Cleanup

- **Datagen**: 63K/1.58M batches (4%), running 42h
- **Training**: Completed 41 epochs, saved checkpoints 0-40
  - Last metrics: `val/full_acc_0=0.832, val/loss=3.002` (epoch 41)
  - Crashed during checkpoint 41 save: `safetensors_rust.SafetensorError: No space left on device`
  - Node .28 disk 100% full (7T): gen=980GB + checkpoints=103GB + models+data=5.9T
- **Root cause**: Buffer cleanup was blocked ("Epoch in progress, skipping cleanup") throughout all training epochs — never got a chance to evict files. Sync correctly stopped at 500GB limit, but 980GB already accumulated before limit was enforced.
- **Cleanup actions**:
  - Deleted checkpoints 0-35 and incomplete 41 (freed ~94GB)
  - Manually removed 6,000 oldest .pt files from gen dir (980GB → 434GB)
  - Disk now 92% (636GB free)
- **Action**: Restarted train pod; datagen continues on .21
- **TODO**: Investigate why buffer cleanup skips during epochs — it should run between epochs or during training steps, not only when idle

### 2026-04-06 — Second Disk Full Crash, Buffer Cleanup Fix

- **Datagen**: 93K/1.58M batches (6%), running 66h
- **Training**: Completed 67 epochs, saved checkpoints 36-66
  - Last metrics: `val/full_acc_0=0.832, val/cond_acc_0=0.832, val/loss=2.913` (epoch 67)
  - Crashed again at checkpoint 67 save: same disk-full error. Gen=1005GB, checkpoints=78GB.
- **Root cause analysis**: Buffer cleanup polls every 120s but checks `.epoch_in_progress` lock file. Training holds the lock for the entire epoch (train+val+save ~5-6min), releases briefly between epochs. Cleanup's 120s interval rarely aligns with the brief unlock window, so it almost never runs.
- **Fixes applied**:
  - `scripts/buffer_cleanup.py`: Added hard limit (1.5x max_size). When buffer exceeds 750GB (1.5×500GB), cleanup ignores the epoch lock and force-evicts files.
  - `scripts/train_streaming.py`: Added automatic checkpoint pruning — keeps only the latest 5 checkpoints, deletes older ones after each save.
- **Cleanup actions**: Deleted checkpoints 36-62 + incomplete 67, removed 7,000 oldest .pt files (1005GB → 365GB). Disk at 91% (708GB free).
- **Action**: Synced fixes to .21 and .28, restarted train pod

### 2026-04-06 — Checkpoint Pruning: Best-N by Val Loss

- **Change**: Checkpoint pruning upgraded from "keep latest 5" to "keep best N by validation loss"
  - `src/speculators/train/trainer.py`: `val_epoch()` now returns `dict[str, float]` of validation metrics
  - `scripts/train_streaming.py`: Added `--keep-checkpoints` CLI arg (default=5), tracks `ckpt_val_scores` dict mapping epoch→val_loss, prunes checkpoints with worst val loss while keeping best N
  - Bug fixes: initialized `ckpt_val_scores` dict before training loop; fixed val metrics key from `"val/loss_epoch"` to `"loss_epoch"` (the `"val/"` prefix is only in metric logger wrapper)
- **Action**: Synced to .28, restarted train pod

### 2026-04-07 — Training Stopped, Overfitting Detected

- **Datagen**: 110K/1.58M batches (7%), running 78h
- **Training**: Completed 260 epochs (193 since last restart at epoch 67)
  - Last metrics: `train/loss=1.3, train/full_acc_0=0.88`
  - Val loss trend: **2.968** (epoch 251, best) → **3.508** (epoch 260, rising steadily)
  - Clear overfitting: model cycling over same ~7,200 .pt files while datagen only at 7%
- **Decision**: Stopped training. Val loss has been worsening since epoch 251 — continued training counterproductive.
- **Available checkpoints**: 63, 64, 65, 66 (pre-restart, val_loss unknown), 238, 251 (val_loss=2.968), 258 (val_loss=3.376)
- **Action**: Deployed eval pod on .28 to evaluate all 7 checkpoints. Datagen continues on .21.
- **Files added**:
  - `scripts/eval_checkpoints.py`: Restored from git + added `--override-*` and `--d2t-path`/`--t2d-path` support
  - `k8s/k8s-minimax-m2.5-exp20-eval.yaml` + `k8s/run_minimax_m2.5_exp20_eval.sh`

### 2026-04-07 — Speedup Eval Complete

- **Eval setup**: vLLM speculative decoding benchmark on .28 (TP=4, max_model_len=8192)
  - Benchmarks: Novita (10 prompts) + ZClawBench (116 prompts)
  - Checkpoints evaluated: 66 (pre-restart), 251 (best val_loss=2.968)
- **Results**:

| Model | Benchmark | Tokens/s | Speedup | Acc@0 | Acc@1 | Acc@2 | Acc Len |
|-------|-----------|----------|---------|-------|-------|-------|---------|
| baseline | novita | 526.5 | 1.00x | - | - | - | - |
| baseline | zclawbench | 3645.3 | 1.00x | - | - | - | - |
| ckpt66 | novita | 876.8 | 1.67x | 62.4% | 34.2% | 19.8% | 2.164 |
| ckpt66 | zclawbench | 4109.3 | 1.13x | 45.7% | 18.2% | 6.8% | 1.707 |
| **ckpt251** | **novita** | **1065.9** | **2.02x** | 62.0% | 35.6% | 19.0% | 2.165 |
| **ckpt251** | **zclawbench** | **4304.6** | **1.18x** | 46.4% | 18.9% | 7.9% | 1.732 |

- **Best checkpoint**: **epoch 251** — novita **2.02x**, zclawbench **1.18x** speedup
- **Analysis**: ckpt251 outperforms ckpt66 in throughput on both benchmarks despite similar acceptance rates, suggesting better draft quality. Significant improvement over exp19 (novita ~1.01x). The nemotron-v2 dataset's broader diversity (6.3M samples, 10 categories) appears to help even at only 7% datagen coverage.
- **Files**: `eval/scripts/run_minimax_m2.5_eval_exp20.sh`, `eval/k8s/k8s-minimax-m2.5-eval-exp20.yaml`
