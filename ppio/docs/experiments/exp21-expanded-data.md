# Experiment 21: Expanded Novita Data (6-Source Merged)

## Status: DEPLOYING

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

_Pending deployment_
