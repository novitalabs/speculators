# Experiment 21: Expanded Novita Data (6-Source Merged)

## Status: STOPPED (deployed 2026-04-03, stopped 2026-04-07, best ckpt epoch 62)

## Motivation

Exp19 uses `novita_merged_exp18/conversations.jsonl` (165K conversations from 3 sources: 0309, 0320, 0327). Exp21 scales up the training data to 724K conversations from 6 sources, with proper deduplication and train/eval split.

Key improvements over exp19's dataset:
1. **4.4x more data** — 724K train conversations (vs 165K in exp18/19)
2. **6 novita log sources** — 0309, 0312_eval, 0320, 0325, 0327, 0401 (vs 3 sources)
3. **Proper task-level dedup** — fingerprint by first user message, max 5 per task
4. **Clean train/eval split** — 5,920 eval conversations with zero task-level leakage
5. **Stratified eval sampling** — maintains system prompt distribution across splits

## Data Pipeline

```
novita20260309  (52K convs)  ──┐
novita20260312_eval (1.4K)  ──┤
novita20260320  (222K convs) ──┤
novita20260325  (1.97M convs) ─┼─→ merge_novita_datasets.py ──→ conversations.jsonl (747K)
novita20260327  (825K convs) ──┤       (task-level dedup,         ├─→ train.jsonl (724K)
novita20260401  (613K convs) ──┘        max 5 per task)           └─→ eval.jsonl  (5.9K)
```

Total raw: 3.68M conversations → 747K after dedup → 724K train + 5.9K eval

## Design

Same continuous datagen + difficulty-weighted resampling pipeline as exp19. Only differences:

| Parameter | Exp19 | Exp21 |
|-----------|-------|-------|
| Training data | novita_merged_exp18 (165K) | novita_merged/train.jsonl (724K) |
| Eval data | none | novita_merged/eval.jsonl (5.9K) |
| Datagen node | .17 | **.14** |
| Train node | .18 | **.23** |
| Output path | exp19 | exp21 |

Architecture, LR, seq_length, buffer size, difficulty feedback — all identical to exp19.

## Infrastructure

- **Datagen**: node .14 (10.83.115.14), 8x GPU, TP=4
- **Training**: node .23 (10.83.115.23), 8x GPU
- **Image**: speculators:v0.17.0

## Deployment

```bash
# Deploy datagen first, then training
kubectl apply -f k8s/k8s-minimax-m2.5-exp21-datagen.yaml
# Wait for datagen to start generating, then:
kubectl apply -f k8s/k8s-minimax-m2.5-exp21-train.yaml
```

## Files

- `k8s/k8s-minimax-m2.5-exp21-datagen.yaml`
- `k8s/k8s-minimax-m2.5-exp21-train.yaml`
- `k8s/run_minimax_m2.5_exp21_datagen.sh`
- `k8s/run_minimax_m2.5_exp21_train.sh`
- Data: `/data/datasets/novita_merged/train.jsonl`
- Eval: `/data/datasets/novita_merged/eval.jsonl`
- Output: `/data/output/minimax_m2.5_eagle3_exp21/`

## Results

### Training Timeline

| Date | Hours | Epoch | Event |
|------|-------|-------|-------|
| 2026-04-03 11:14 | 0 | 0 | Deployed datagen (.14) + train (.23) |
| 2026-04-04 03:37 | 16 | 27 | val_acc@0=0.632, val_loss=4.126 |
| 2026-04-05 ~06:00 | 42 | 62 | **Best val_acc@0=0.668**, val_loss=4.282 |
| 2026-04-05 ~12:00 | 48 | ~70 | val_acc@0 stabilizes ~0.65-0.66 |
| 2026-04-06 05:45 | 66 | 105 | val_acc@0=0.655, val_loss=3.976 |
| 2026-04-06 20:52 | 81 | 138 | Last checkpoint before crash |
| 2026-04-06 21:49 | 82 | 139 | **NCCL AllGather timeout** — train killed |
| 2026-04-07 02:18 | 87 | 139 | Restarted from ckpt 138, then stopped |

### Validation Metrics (key epochs)

| Epoch | val_acc@0 | val_loss | Notes |
|-------|-----------|----------|-------|
| 23 | 0.626 | 4.192 | First val after data accumulated |
| 24 | 0.637 | 3.855 | Early peak |
| 27 | 0.632 | 4.126 | |
| **62** | **0.668** | **4.282** | **Best val_acc@0** |
| 63 | 0.664 | 4.392 | |
| 103 | 0.662 | 3.678 | Best val_loss |
| 105 | 0.655 | 3.976 | |
| 138 | 0.651 | — | Last before crash |

### Datagen Progress

- Round 1: 84% complete (151K/181K samples) at crash time
- Difficulty-weighted resampling not yet activated (requires round 2)
- ~20K data files generated total

### Crash Details

- **Time**: 2026-04-06 21:49 UTC (epoch 139, mid-training)
- **Cause**: NCCL AllGather timeout (30 min watchdog) on rank 5
- **Not OOM** — likely transient GPU communication failure
- Recovered from ckpt 138 after restart

### Analysis

- **Best checkpoint: epoch 62** (val_acc@0=0.668)
- val_acc@0 peaked early (epoch 62) then slowly declined to ~0.65, suggesting:
  - Possible overfitting on the continuous data stream
  - Distribution shift as datagen progresses through dataset
  - Difficulty-weighted resampling never activated (still in round 1)
- Compared to exp19 (val_acc@0 ~0.55-0.60 at ckpt5), exp21 consistently better → larger dataset helps
- val_loss improved over time (4.2 → 3.7) even as acc@0 plateaued, indicating better calibration on later tokens
### Eval Results (ckpt 62, node .23)

| Config | Benchmark | #Prompts | Tokens/s | Speedup | Acc@0 | Acc@1 | Acc@2 | Acc Len |
|--------|-----------|----------|----------|---------|-------|-------|-------|---------|
| Baseline (no spec) | Novita | 50 | 426.0 | 1.00x | — | — | — | — |
| **exp21 ckpt62** | **Novita** | **50** | **536.2** | **1.26x** | **0.589** | **0.345** | **0.182** | **2.12** |
| Baseline (no spec) | ZClawBench | 116 | 2,759.0 | 1.00x | — | — | — | — |
| **exp21 ckpt62** | **ZClawBench** | **116** | **1,133.5** | **0.41x** | **0.427** | **0.169** | **0.071** | **1.67** |

**Observations:**
- **Novita**: 1.26x speedup with Acc@0=0.589, acceptance length=2.12. Positive but modest gain
- **ZClawBench**: 0.41x — spec decode is **slower** than baseline. Baseline batched throughput (2759 tok/s) is very high; draft overhead exceeds savings at high batch sizes. Acc@0=0.427 also indicates poor draft prediction on multi-turn agent trajectories
- Compared to exp19 (ckpt5): Novita Acc@0 comparable (~0.59 vs ~0.55-0.60), 4.4x more data did **not** significantly improve acceptance rate
- The benefit of spec decode is batch-size dependent — online serving (batch=1) would show larger speedups than this batched eval
