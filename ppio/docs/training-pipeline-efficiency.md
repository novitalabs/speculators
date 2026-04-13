# Training Pipeline Efficiency Analysis

> Based on Exp20-3 (Chinese+Novita, 361K dataset, .17/.18 nodes, 2026-04-10)

## 1. Per-Epoch Time Breakdown

Measured from training log timestamps across recent epochs (256-263).

| Phase | Duration | Share | Notes |
|-------|----------|-------|-------|
| **Training steps** | ~290s | 83.2% | ~400 steps × 0.73s/step, 8-GPU FSDP |
| **Validation** | ~5s | 1.4% | 9-10 val steps @ 4 it/s |
| **Checkpoint save** | ~3s | 0.9% | safetensors write to hostPath |
| **DataLoader prefetch** | ~30s | 8.6% | rsync pull + shuffle + build next epoch |
| **Lock handshake overhead** | ~2s | 0.6% | epoch lock release + signal to datagen |
| **Total** | **~330s (5:30)** | 100% | — |

> Checkpoint save and DataLoader prefetch run **in parallel**: lock is released immediately after
> checkpoint write, so datagen updates difficulty scores while training prefetches the next epoch.

### Training Step Speed

- ~400 steps/epoch at 0.73s/step
- Sequence length: 8192 tokens
- 8 GPUs FSDP with Aurora architecture (24 heads, 8192 intermediate, 32K vocab)
- Epoch size varies ~326-433 steps depending on buffer contents (difficulty-weighted sampling)

---

## 2. Two-Pod Coordination Protocol

```
Training (.18)                       Datagen (.17)
─────────────────────                ──────────────────────────
train steps (~290s)                  generating hidden states
  (DataLoader from local buffer)       writing .pt files to /gen/
validation (~5s)
checkpoint save (~3s)
release epoch lock ────────────────→ receive lock signal
begin DataLoader prefetch            update difficulty scores
  rsync pull .pt from .17 (~15s)     re-weight sampling distribution
  shuffle + build DataLoader (~15s)  prepare next batch
"lock released after DataLoader"
train next epoch...
```

### Lock Release Patterns

Two variants observed in logs (`train_streaming.py:438` vs `:483`):

- `"Epoch lock released, waiting for..."` → briefly polled before data was ready
- `"Epoch lock released after DataLoader"` → data was already available immediately

Both complete in ~30-32s total. In the current run (epoch 200+), the pipeline is **never data-starved** — all epochs resolve "after DataLoader" within 30s.

---

## 3. Datagen Throughput

| Metric | Value |
|--------|-------|
| Dataset size | 361,395 conversations (Chinese 195K + Novita 166K) |
| Round 1 samples | 90,349 (each conversation → multiple turn-level samples) |
| Round 1 completion | ~46h total (~2.5s/sample avg) |
| Speed range | 2.3–3.3s/sample (varies with prompt length) |
| Buffer size | ~12K files (~500GB cap) |
| Round 2+ | Difficulty-weighted resampling, same ~46h/round |

**Coverage**: Datagen processes ~25% of conversations per round in terms of file count;
full semantic coverage (all conversations sampled at least once) completes in ~1 round.

---

## 4. Data Pipeline Steady State

After initial buffer fill (~5000 files, ~4h from cold start):

- Training consumes ~12 files/epoch from the buffer
- Datagen produces ~1 file every 2.5s = ~1440 files/hour
- Buffer maintains 10-13K files — well above training demand
- **No data starvation observed** in epochs 69–263+

The buffer provides ~8+ hours of training headroom even if datagen stops.

---

## 5. Efficiency Bottlenecks

### 5.1 NCCL Timeout Crash (.18 Node)

| Attribute | Detail |
|-----------|--------|
| Frequency | Every 40–70 epochs (~3.5–6h) |
| Trigger | FSDP `_ALLGATHER_BASE` during validation/checkpoint phase |
| Timeout | 1800s (30min watchdog) |
| Recovery | Manual pod delete + recreate; auto-resumes from last checkpoint |
| Time lost | ~30min per crash (watchdog timeout) + ~1 epoch |
| Total waste | ~7% of training time at current frequency |
| Root cause | Suspected .18 node GPU NVLink/PCIe interconnect degradation |

Crashes observed: epoch 29 (04-08 11:51), epoch 69 (04-09), and recurring.
Workaround: none — `NCCL_TIMEOUT` cannot be reduced below the validation allgather window.

### 5.2 DataLoader Prefetch Overhead

30s per epoch (8.6%) spent on:
1. rsync pull of new `.pt` files from .17 over 10GbE (~15s)
2. Shuffle + difficulty-weighted DataLoader construction (~15s)

Potential improvement: pre-fetch next epoch asynchronously during training steps
(currently prefetch only starts after checkpoint is saved).

### 5.3 Validation Cost (Minor)

5s per epoch (1.4%) — negligible. Only becomes a problem when NCCL hangs during allgather.

---

## 6. Overall Efficiency Summary

| Category | Efficiency |
|----------|------------|
| GPU active training time | ~83% |
| Pipeline idle (DataLoader prefetch) | ~9% |
| NCCL crash overhead (amortized) | ~7% |
| Validation + checkpoint | ~2% |
| **Effective GPU utilization** | **~83%** |

The pipeline is well-optimized for steady-state operation. The dominant inefficiency is the
recurring .18 node NCCL crash. Migrating training to a stable node would recover ~7% throughput
and eliminate the manual intervention requirement.

---

## 7. Comparison: Exp20-2 vs Exp20-3

Both experiments ran on .17/.18 with the same pipeline.

| Metric | Exp20-2 (125K dataset) | Exp20-3 (361K dataset) |
|--------|------------------------|------------------------|
| Epoch duration | ~5:30 | ~5:30 |
| Steps/epoch | ~336 | ~400 |
| Convergence epoch | ~30 | ~200+ (ongoing) |
| NCCL crashes | 3 | 2 (so far) |
| Datagen round time | ~14h | ~46h |
| Data starvation | None after epoch 5 | None after epoch 5 |

Larger dataset → more steps/epoch, slower convergence (more unique samples per round),
but better generalization (val_loss 3.361 vs Exp20-2's floor of 3.643).
