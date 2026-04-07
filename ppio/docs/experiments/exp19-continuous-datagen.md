# Experiment 19: Continuous Datagen with Difficulty-Weighted Resampling

## Status: COMPLETE (80 epochs trained, epoch 79 eval done)

## Motivation

Exp18 validated that session-level dedup is critical (60.4% Acc@0 vs Exp17's 43.1%), but exposed a fundamental pipeline flaw:

1. **Datagen runs one pass and exits** — generates 165K samples, then stops
2. **Buffer cleanup continuously evicts** — with 500GB buffer, old files are deleted before training uses them enough
3. **90% data loss** — only 17K of 165K samples survived on .18; training overfit on a shrinking pool
4. **No feedback loop** — all samples treated equally; easy/hard samples get the same generation frequency

The root cause is that the pipeline was designed for a "generate once, train once" workflow, but effective Eagle3 training needs **continuous data refresh with intelligent resampling**.

## Design

### Core Principle

```
Datagen (continuous)  →  Buffer (prioritized)  →  Training (with loss feedback)
     ↑                                                    │
     └──────── difficulty scores (per-sample loss) ────────┘
```

### Component 1: Continuous Datagen Loop

**File: `scripts/data_generation_offline.py`**

Current behavior: iterate over dataset once → exit.

New behavior: iterate over dataset in rounds, never exit.

```python
# Current (single-pass):
for i in pbar:
    generate_and_save(dataset[i])
# exits here

# New (continuous loop):
round_num = 0
while True:
    round_num += 1
    weights = load_difficulty_weights()  # from training feedback
    sample_order = weighted_shuffle(dataset, weights, seed=round_num)

    for i in sample_order:
        wait_for_output_budget(...)  # existing budget control
        generate_and_save(dataset[i])

    log.info(f"Round {round_num} complete, starting next round...")
```

**Key changes:**
- Add `--continuous` flag (default False for backward compat)
- After each round, reload difficulty weights from a shared file
- Round 1: uniform weights (ensure full coverage)
- Round 2+: weight proportional to difficulty score
- File naming: `data_{global_idx}.pt` with monotonically increasing index across rounds
- Manifest status stays "generating" indefinitely (never "complete")

### Component 2: Per-Sample Difficulty Tracking

**Files: `src/speculators/train/trainer.py`, `src/speculators/models/eagle3/core.py`**

Training already computes per-token KL divergence loss. Currently aggregated to a scalar. Need to:

1. **Return per-sample loss from model forward**

```python
# In core.py loss_function():
# Currently returns: batch_loss.mean()  (scalar)
# Change to return: batch_loss  (shape [B])
# Caller (trainer.py) handles averaging for backward, but also records per-sample values
```

2. **Track sample boundaries through packed batches**

The collate function packs multiple samples into one sequence. Need to track which tokens belong to which source file.

```python
# In data.py collate_fn:
# Currently returns: {"input_ids": [1, total_seq_len], ...}
# Add: {"sample_boundaries": [(start, end, file_idx), ...]}
```

3. **Write per-file loss to a shared difficulty file**

```python
# In trainer.py, after each step:
# Aggregate per-sample loss back to source files
# Periodically write to: <data_dir>/difficulty_scores.json
# Format: {"data_0.pt": 2.34, "data_1.pt": 0.87, ...}
```

**Update frequency**: Every N steps (e.g., 100) to avoid I/O overhead. Use exponential moving average to smooth across epochs.

### Component 3: Difficulty-Weighted Resampling in Datagen

**File: `scripts/data_generation_offline.py`**

```python
def load_difficulty_weights(data_dir, dataset_size, default_weight=1.0):
    """Load per-conversation difficulty weights from training feedback."""
    scores_path = os.path.join(data_dir, "difficulty_scores.json")
    if not os.path.exists(scores_path):
        return [default_weight] * dataset_size  # Round 1: uniform

    scores = json.load(open(scores_path))
    # Map .pt file scores back to conversation indices
    # Higher loss → higher weight (resample more often)
    weights = []
    for i in range(dataset_size):
        key = f"data_{i}.pt"
        if key in scores:
            weights.append(scores[key])
        else:
            weights.append(default_weight)  # unseen → default priority
    return weights

def weighted_shuffle(indices, weights, seed):
    """Shuffle indices with probability proportional to weights."""
    rng = random.Random(seed)
    # Weighted sampling without replacement
    return rng.choices(indices, weights=weights, k=len(indices))
```

**Difficulty-to-weight mapping:**
- Normalize loss values to [0, 1] range
- `weight = 0.5 + 0.5 * normalized_loss` — ensures easy samples still get some coverage (min 50% weight) but hard samples are 2x more likely
- Round 1 always uniform (no difficulty data yet)

### Component 4: Prioritized Buffer Eviction

**File: `scripts/buffer_cleanup.py`**

Current: evicts highest `train_count` files first.

New: evict **low-difficulty + high-train-count** files first (easy, well-trained samples are least valuable).

```python
# Current eviction sort:
all_files.sort(key=lambda f: (-f.get("train_count", 0), f["idx"]))

# New eviction sort:
def eviction_priority(f):
    tc = f.get("train_count", 0)
    loss = f.get("avg_loss", float("inf"))  # unknown loss → keep
    # Low loss + high train_count → evict first (high priority number)
    # High loss + low train_count → keep (low priority number)
    return (-loss, tc)  # sort: lowest loss first, then highest tc

all_files.sort(key=eviction_priority)
```

**Manifest extension**: Add `avg_loss` field to each file entry, updated by training after each epoch.

### Component 5: seq_length Increase to 32K

**Exp18 analysis**: 9.4% of conversations (15.6K) exceed 8K tokens, truncated to opening portion only.

Changes:
- `--seq-length 32768` in datagen script
- `--batch-size 1` or `--batch-size 2` (less samples per vLLM call to fit in GPU memory)
- `--total-seq-len 32768` in training script
- May need `--gpu-memory-utilization 0.90` for larger KV cache

**Risk**: Training with 32K seq_len needs more GPU memory for attention. FSDP with 8x H200 (144GB each) should handle it, but need to test. Fallback: keep 8192 for datagen, only increase for training via padding.

## Implementation Plan

### Phase 1: Continuous Datagen (core loop)

Modify `scripts/data_generation_offline.py`:
- Add `--continuous` flag
- Wrap generation loop in outer `while True`
- After each round: log stats, re-shuffle dataset
- File index continues monotonically across rounds
- Keep manifest status as "generating"

**Estimated changes**: ~50 lines in `data_generation_offline.py`

### Phase 2: Per-Sample Loss Tracking

Modify training pipeline:
- `eagle3/core.py`: return per-sample loss vector alongside scalar
- `train/data.py`: pass sample file indices through collate
- `train/trainer.py`: aggregate per-sample loss to file level, write `difficulty_scores.json`

**Estimated changes**: ~30 lines in `core.py`, ~20 lines in `data.py`, ~50 lines in `trainer.py`

### Phase 3: Difficulty-Weighted Resampling

Modify datagen:
- Load `difficulty_scores.json` at start of each round
- Weighted shuffle/sampling of conversation indices
- Higher loss → higher sampling probability

**Estimated changes**: ~40 lines in `data_generation_offline.py`

### Phase 4: Prioritized Eviction

Modify `buffer_cleanup.py`:
- Read `avg_loss` from manifest
- Sort by (low loss, high train_count) for eviction priority
- Extend manifest write in training to include `avg_loss`

**Estimated changes**: ~20 lines in `buffer_cleanup.py`, ~10 lines in `train_streaming.py`

### Phase 5: seq_length 32K (optional, can be separate experiment)

Modify K8s configs:
- `--seq-length 32768 --batch-size 1` in datagen
- `--total-seq-len 32768` in training
- Test GPU memory on H200 before committing

## Data Flow (Complete)

```
┌─────────────────────────────────────────────────────────────┐
│                    Datagen Node (.17)                        │
│                                                             │
│  Round 1: uniform weights → generate all 165K samples       │
│  Round 2: load difficulty_scores.json → weighted resample   │
│  Round 3: updated scores → focus on hard samples            │
│  ...never exits...                                          │
│                                                             │
│  Cleanup thread: evict oldest when >85% budget              │
│  Output: data_{idx}.pt files (monotonic idx across rounds)  │
└──────────────────────┬──────────────────────────────────────┘
                       │ rsync
┌──────────────────────▼──────────────────────────────────────┐
│                    Training Node (.18)                       │
│                                                             │
│  Streaming training (8 GPU FSDP):                           │
│  - Each epoch: train on buffer files                        │
│  - Record per-sample loss → difficulty_scores.json          │
│  - increment_train_count + update avg_loss in manifest      │
│                                                             │
│  Buffer cleanup:                                            │
│  - Evict low-loss + high-train-count files first            │
│  - Respect epoch lock                                       │
│                                                             │
│  Rsync difficulty_scores.json → datagen node                │
│  (datagen reads it at start of each round)                  │
└─────────────────────────────────────────────────────────────┘
```

## Key Metrics to Track

| Metric | Where | Purpose |
|--------|-------|---------|
| Round number | datagen log | Track how many full passes over dataset |
| Per-file avg_loss | manifest + difficulty_scores.json | Difficulty signal for resampling |
| Files ever seen | manifest (existing) | Global coverage |
| Eviction loss distribution | cleanup log | Verify easy samples evicted first |
| Val acc@0 over time | training log | Convergence quality |
| Inference Acc@0 | eval (periodic) | True performance metric |

## Expected Outcome

- **No data loss**: Datagen continuously regenerates, evicted samples come back in next round
- **Curriculum effect**: Hard samples trained more, easy samples less → better generalization
- **Higher Acc@0**: Training on full diversity of data instead of shrinking pool
- **Longer sequences**: 32K captures full coding agent conversations (currently 9.4% truncated)

## Dependencies

- Exp18's merged dataset (`/data/datasets/novita_merged_exp18/conversations.jsonl`) — reuse as-is
- Same Aurora architecture (24 heads, 8192 intermediate)
- Same nodes (.17 datagen + .18 training)

## Risks

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Difficulty scores stale across rounds | Suboptimal resampling | EMA smoothing + periodic refresh |
| Per-sample loss tracking overhead | Training slowdown | Update every 100 steps, not every step |
| 32K seq_length OOM | Cannot train | Fallback to 8192; test memory usage first |
| Difficulty feedback loop diverges | Model ignores easy samples entirely | Min weight floor (50%) ensures coverage |
| File index overflow across rounds | Disk naming collision | Use `round_{R}_data_{idx}.pt` naming |

## Eval Results

### 2026-04-03 — ckpt5 (Epoch 6) Simple Eval

Novita (10 prompts) + ZClawBench (116 short prompts), `max_model_len=8192`:

| Benchmark | Baseline tok/s | Spec tok/s | Speedup | Acc@0 | Acc@1 | Acc@2 | AccLen |
|-----------|---------------|------------|---------|-------|-------|-------|--------|
| Novita | 561.4 | 682.5 | **1.22x** | 49.3% | 23.3% | 10.3% | 1.829 |
| ZClawBench | 3613.9 | 3703.3 | **1.02x** | 33.8% | 10.8% | 3.8% | 1.485 |

Training metrics at time of eval: `full_acc_0 ≈ 55-60%`, `loss ≈ 3.2-6.3`, epoch 6.

### 2026-04-03 — ckpt5 (Epoch 6) Full ZClawBench Eval

650 multi-turn agent trajectories from [zai-org/ZClawBench](https://huggingface.co/datasets/zai-org/ZClawBench), `max_model_len=32768`, bucketed by input length:

| Bucket | #Prompts | Avg Input Tokens | Baseline tok/s | Spec tok/s | Speedup | Acc@0 | AccLen |
|--------|----------|-----------------|---------------|------------|---------|-------|--------|
| short (0-2K) | 210 | 847 | 5267.6 | 3183.3 | **0.60x** | 32.3% | 1.455 |
| med (2K-8K) | 320 | 4601 | 2312.9 | 2187.2 | **0.95x** | 30.8% | 1.440 |
| long (8K-32K) | 119 | 12177 | 903.0 | 915.8 | **1.01x** | 31.7% | 1.429 |

**Observations:**
- Acc@0 ~31-32% across all buckets — far below the ~55%+ threshold needed for speedup
- Short prompts **slow down 40%** due to spec decode overhead exceeding acceptance benefit at high batch concurrency
- Medium prompts nearly break even (0.95x), still net negative
- Long prompts barely break even (1.01x) — decode-bound regime where spec decode has more room but Acc@0 too low
- Novita eval shows higher Acc@0 (49.3%) because Novita data is closer to training distribution (novita_merged_exp18)
- ZClawBench is harder: diverse agent tasks (6 categories), long tool-use trajectories, different from training data

**Conclusion:** ckpt5 is too early (epoch 6). Draft model accuracy insufficient for speedup. Continue training; re-eval at higher epoch when `full_acc_0` stabilizes above 60%.

**Eval infrastructure:**
- Created `speculators-eval` submodule with organized eval scripts, K8s configs, tools, and data
- Added `eval/scripts/run_minimax_m2.5_eval_exp19.sh` (simple eval)
- Added `eval/scripts/run_minimax_m2.5_eval_exp19_zclawbench_full.sh` (full 650-prompt bucketed eval)
- Downloaded full ZClawBench dataset (696 trajectories → 650 valid) to `/data/datasets/zclawbench/zclawbench_full_696.json`

### 2026-04-04 — Training Progress (Epoch 28) + NCCL Crash

**Training metrics at epoch 28 (last step before crash):**
- `full_acc_0 = 78.0%` (up from ~55-60% at ckpt5/epoch 6)
- `loss = 1.187` (down from ~3.2-6.3 at ckpt5)
- Checkpoints saved: epoch 24, 25, 26, 27 (all overwritten into `checkpoints/5/`)

### Issue: NCCL Timeout Crash at Epoch 28 Validation

**When**: 2026-04-03 21:49 UTC, immediately after epoch 28 training completed, during validation/checkpoint save phase
**Symptom**: All 8 ranks hit NCCL `_ALLGATHER_BASE` timeout (30 min / 1800s). Root cause rank 5 received SIGABRT first, then ranks 0-4,6,7 followed. Pod status: `Error`.
**Root cause**: NCCL collective operation hung during FSDP all-gather in validation phase. Likely transient GPU communication issue or memory pressure during checkpoint save. SeqNum=276118 on most ranks.
**Fix**: Delete failed pod and recreate — training script auto-resumes from last saved checkpoint (epoch 27). Datagen pod on .17 still running (39.5%, 16374/41443 samples).
**Data preserved**: Last checkpoint is epoch 27 in `checkpoints/5/`. Epoch 28 training completed but checkpoint was NOT saved.

### 2026-04-06 — Training Progress (Epoch 73) + 2nd NCCL Crash

**Training metrics at epoch 73 (last steps before crash):**
- `full_acc_0 ≈ 45-78%` (high batch-level variance), `loss ≈ 0.2-0.35`
- Trained from epoch 28 to 73 after first restart (45 epochs in ~33 hours)
- Checkpoints saved: epoch 43-72 (all overwritten into `checkpoints/5/`)
- Loss dropped significantly: 1.2 → 0.3, but full_acc_0 plateau around 70-80% with high variance

### Issue 2: NCCL Timeout Crash at Epoch 73 Validation

**When**: 2026-04-05 13:08 UTC, after epoch 73 training completed, during validation/checkpoint save
**Symptom**: Same pattern as Issue 1 — NCCL `_ALLGATHER_BASE` timeout across all 8 ranks. Root cause rank 2 SIGABRT first.
**Root cause**: Recurring NCCL timeout during FSDP all-gather in validation. This is the 2nd occurrence — likely a systematic issue with .18 node GPU interconnect or memory pressure during checkpoint save.
**Fix**: Delete and recreate pod — resumes from epoch 72 checkpoint.
**Data preserved**: Last checkpoint is epoch 72 in `checkpoints/5/`.

### 2026-04-07 — Training Progress (Epoch 80) + 3rd NCCL Crash

**Training metrics at epoch 80 (last steps before crash):**
- `full_acc_0 ≈ 43-86%` (still high variance), `loss ≈ 0.2-0.6`
- Trained from epoch 73 to 80 after 2nd restart (7 epochs in ~5 hours)
- Checkpoints saved: epoch 73-79 (overwritten into `checkpoints/5/`)
- **No significant improvement since epoch ~50** — model appears converged

### Issue 3: NCCL Timeout Crash at Epoch 80 Validation

**When**: 2026-04-06 11:31 UTC, after epoch 80 training completed, during validation/checkpoint save
**Symptom**: Same NCCL timeout pattern. Root cause rank 0.
**Root cause**: 3rd occurrence of identical failure. This is a **systematic issue** with .18 node — NCCL hangs during FSDP all-gather in validation/checkpoint save phase every ~20-45 epochs.
**Fix**: Delete and recreate pod — resumes from epoch 79 checkpoint.
**Data preserved**: Last checkpoint is epoch 79 in `checkpoints/5/`.

**Note on convergence**: Training metrics have plateaued since ~epoch 50. full_acc_0 oscillates 43-86% (batch-level), loss ~0.2-0.5. Should run eval on current checkpoint (epoch 79) to assess actual inference speedup before continuing.

### 2026-04-07 — Epoch 79 Final Eval

Stopped training after 80 epochs (3 NCCL crashes, model converged). Evaluated epoch 79 checkpoint.

#### Simple Eval (Novita 10 prompts + ZClawBench 116 prompts, `max_model_len=8192`)

| Benchmark | Baseline tok/s | Spec tok/s | Speedup | Acc@0 | Acc@1 | Acc@2 | AccLen |
|-----------|---------------|------------|---------|-------|-------|-------|--------|
| Novita | 521.6 | 798.9 | **1.53x** | 65.8% | 35.2% | 18.5% | 2.196 |
| ZClawBench | 3612.3 | 4015.0 | **1.11x** | 43.1% | 16.2% | 6.4% | 1.657 |

#### Full ZClawBench Eval (650 prompts, `max_model_len=32768`, bucketed by input length)

| Bucket | #Prompts | Avg Input | Baseline tok/s | Spec tok/s | Speedup | Acc@0 | AccLen |
|--------|----------|-----------|---------------|------------|---------|-------|--------|
| short (0-2K) | 210 | 847 | 5296.7 | 3616.8 | **0.68x** | 43.8% | 1.649 |
| med (2K-8K) | 320 | 4601 | 2328.9 | 2415.0 | **1.04x** | 47.4% | 1.712 |
| long (8K-32K) | 119 | 12177 | 907.3 | 985.0 | **1.09x** | 47.6% | 1.702 |

#### Comparison: ckpt5 (Epoch 6) vs ckpt79 (Epoch 79)

| Metric | ckpt5 (Epoch 6) | ckpt79 (Epoch 79) | Delta |
|--------|----------------|-------------------|-------|
| Novita Acc@0 | 49.3% | 65.8% | +16.5pp |
| Novita Speedup | 1.22x | 1.53x | +0.31x |
| ZClawBench Acc@0 | 33.8% | 43.1% | +9.3pp |
| ZClawBench Speedup | 1.02x | 1.11x | +0.09x |
| Full ZClaw short | 0.60x | 0.68x | +0.08x |
| Full ZClaw med | 0.95x | 1.04x | +0.09x |
| Full ZClaw long | 1.01x | 1.09x | +0.08x |

#### Analysis

- **Novita (in-distribution)**: Strong improvement — 1.53x speedup with 65.8% Acc@0. Demonstrates the model learns well on training-distribution data.
- **ZClawBench (out-of-distribution)**: Moderate improvement — Acc@0 rose from 33.8% to 43.1%, but still insufficient for meaningful speedup in high-concurrency short-prompt scenarios (0.68x slowdown).
- **Length dependence**: Short prompts slow down (spec decode overhead > acceptance benefit at high batch throughput). Med/long prompts just barely break even (1.04-1.09x).
- **Bottleneck**: OOD generalization. Training data (Novita conversations) differs significantly from eval data (ZClawBench agent tasks with tool_use/tool_result/thinking blocks). Acc@0 needs >55% across all buckets for consistent speedup.
- **Training saturation**: 80 epochs with no improvement since ~epoch 50. More training on the same data will not help. Next steps should focus on expanding training data diversity (e.g., agent task trajectories, tool-use conversations) or architecture changes.
