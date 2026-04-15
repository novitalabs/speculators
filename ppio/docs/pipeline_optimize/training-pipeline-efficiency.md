# Training Pipeline Efficiency Analysis

> Originally based on Exp20-3 (2026-04-10). Updated with Exp22 findings (2026-04-14).

## 1. Per-Epoch Time Breakdown

Measured from training log timestamps across recent epochs (Exp20-3: 256–263, Exp22: 220–231).

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

Both complete in ~30-32s total. In steady state (epoch 200+), the pipeline is **never data-starved** by the epoch-level DataLoader — all epochs resolve "after DataLoader" within 30s.

---

## 3. Datagen Throughput

### Exp20-3 (361K dataset, TP=4)

| Metric | Value |
|--------|-------|
| Dataset size | 361,395 conversations |
| Round 1 samples | 90,349 |
| Round 1 completion | ~46h (~2.5s/sample avg) |
| Speed range | 2.3–3.3s/sample |
| Buffer size | ~12K files (~500GB cap) |

### Exp22 (919K dataset, TP=4) — **Bottlenecked**

| Metric | Value |
|--------|-------|
| Dataset size | 919,820 conversations (724K novita + 195K Chinese) |
| Round 1 batches | 229,955 (BS=4) |
| Throughput | ~2,500 files/hour (TP=4, GPU 0-3 only) |
| Round 1 ETA | ~92h (~4 days) at current speed |
| After 24h | 60K / 229K = **26% complete** |

---

## 4. Data Pipeline Steady State

### Exp20-3: Healthy Balance

After initial buffer fill (~5000 files, ~4h from cold start):

- Training consumes ~12 files/epoch from the buffer
- Datagen produces ~1440 files/hour (TP=4)
- Buffer maintains 10–13K files — well above training demand
- **No data starvation observed** in epochs 69–263+

### Exp22: Datagen Bottleneck (Train >> Datagen)

**Problem**: Training consumes samples ~3× faster than datagen produces them.

| Metric | Value |
|--------|-------|
| files_ever_seen (training) | 185,903 unique files |
| Datagen files generated | ~60,000 (round 1 only) |
| Ratio | Training consumed **3× more** than generated |
| Buffer files (current) | 8,867 |
| Buffer mean train_count | 13.0 (in-buffer) |
| Evicted files | 206,170 |
| Evicted mean train_count | **26.7** (target: 10) |
| Evicted p50 train_count | 22 |
| Evicted max train_count | 108 |

`TARGET_TRAIN_COUNT=10` is the re-sync exclusion threshold, **not a hard training cap**.
Because datagen is slow, samples stay in the buffer long enough to accumulate 20–30+ training passes before eviction.

**Root cause**: 919K dataset is 2.5× larger than Exp20-3 (361K), but TP is the same (4 GPUs).
Datagen cannot keep pace with training from the start.

---

## 5. Efficiency Bottlenecks

### 5.1 NCCL Timeout Crash (.18 Node)

| Attribute | Detail |
|-----------|--------|
| Frequency | Every 40–70 epochs (~3.5–6h) in Exp20-3; once so far at epoch 33 in Exp22 |
| Trigger | FSDP `_ALLGATHER_BASE` during validation/checkpoint phase |
| Timeout | 1800s (30min watchdog) |
| Recovery | Manual pod delete + recreate; auto-resumes from last checkpoint |
| Time lost | ~30min per crash (watchdog timeout) + ~1 epoch |
| Total waste | ~7% of training time at Exp20-3 frequency |
| Root cause | Suspected .18 node GPU NVLink/PCIe interconnect degradation |

### 5.2 DataLoader Prefetch Overhead

30s per epoch (8.6%) — not a bottleneck in current experiments.

### 5.3 Datagen Under-Provisioning (Exp22 New Finding)

When the training dataset is large (>500K conversations) and datagen runs TP=4, training exhausts
fresh data rapidly and falls back to reusing already-trained samples. This leads to:

- Excessive sample reuse (mean 26.7× vs target 10×)
- Potential overfitting on the fraction of data seen early
- Difficulty-weighted resampling only activates after round 1 completes — never triggered in Exp22 so far

---

## 6. Optimization Plan

### 6.1 Datagen TP=8 (Highest Priority, Zero-Cost)

The datagen pod on .17 requests **8 GPUs** but `DATAGEN_TP` defaults to 4.
Switch to TP=8 to use all GPUs:

```yaml
# k8s-minimax-m2.5-exp22-datagen.yaml
env:
  - name: DATAGEN_TP
    value: "8"
```

Expected effect: **2× throughput** (~5,000 files/hour), round 1 completion in ~46h instead of ~92h.

### 6.2 Multi-Node Datagen

`sync_datagen.sh` supports `--datagen-nodes` with multiple IPs:

```bash
--datagen-nodes "10.83.115.17 10.83.115.XX"
```

Run a second datagen pod on an idle node (e.g., .21/.22) pointing to the same `exp22_merged` dataset.
Combined with 6.1, can achieve **4× baseline throughput**.

### 6.3 Reduce seq_len (Trade-off)

Reducing `SEQ_LENGTH` from 8192 → 4096 would ~2× datagen speed, but loses long-context training samples.
Not recommended for ZClawBench workloads (long system prompts). Use only if nodes are unavailable.

### Priority

| Option | Throughput gain | Action required | Recommended |
|--------|----------------|-----------------|-------------|
| TP=8 on .17 | ×2 | Restart datagen pod with new env | **Yes, immediately** |
| Second datagen node | ×2 additional | Find idle node + new pod yaml | Yes, if node available |
| seq_len 4096 | ×2 | Code change + restart | No (ZClaw regression risk) |

---

## 7. Overall Efficiency Summary

| Category | Exp20-3 | Exp22 |
|----------|---------|-------|
| GPU active training time | ~83% | ~83% |
| Pipeline idle (DataLoader) | ~9% | ~9% |
| NCCL crash overhead (amortized) | ~7% | ~1% (1 crash so far) |
| Validation + checkpoint | ~2% | ~2% |
| **Datagen bottleneck** | None | **Severe** (sample reuse 26.7×) |
| **Effective GPU utilization** | **~83%** | **~83% (but training stale data)** |

---

## 8. Experiment Comparison

| Metric | Exp20-2 (125K) | Exp20-3 (361K) | Exp22 (919K) |
|--------|----------------|----------------|--------------|
| Epoch duration | ~5:30 | ~5:30 | ~5:30 |
| Steps/epoch | ~336 | ~400 | ~400 |
| Datagen TP | 4 | 4 | 4 |
| Datagen round time | ~14h | ~46h | ~92h (est.) |
| Data starvation | None | None | **Severe** (3× over-training) |
| NCCL crashes | 3 | 2 | 1 (epoch 33) |
| Best val_acc@0 | 0.66 | — | **0.714** (epoch 120) |
| Best novita speedup | — | — | 1.67× (ckpt230) |
| Best ZClaw speedup | 1.34× | — | 1.12× (ckpt224) |

**Key lesson**: Dataset size must be matched with datagen capacity.
For datasets >500K conversations, TP=4 on a single node is insufficient — use TP=8 or multi-node datagen.
