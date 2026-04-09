# Experiment 20-3: Chinese + Novita Dataset with Continuous Datagen

## Status: RUNNING (redeployed on .17/.18, 2026-04-08)

## Motivation

Exp20 achieved 2.02x novita speedup with nemotron-v2 (6.3M samples) but suffered overfitting at 7% datagen coverage — dataset too large for timely round completion. Exp20-3 uses a focused ~361K sample dataset combining nemotron-v2 Chinese subset with exp19's novita data, enabling full-coverage datagen rounds in ~2 days instead of ~37.

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | MiniMax-M2.5 (Aurora architecture) |
| Dataset | nemotron-v2-chinese (195K) + novita_merged_exp18 (166K) = 361K, shuffled |
| Datagen Node | .17 (8 GPU) |
| Training Node | .18 (8 GPU) |
| Seq Length | 8192 |
| LR | 3e-5 |
| Buffer Max | 500GB |
| Architecture | Aurora overrides (heads=24, intermediate=8192, rope_theta=5M) |
| Pipeline | Continuous datagen with difficulty feedback |

### Dataset Breakdown

| Source | Samples |
|--------|---------|
| nemotron-v2-chinese | 195,624 |
| novita_merged_exp18 | 165,771 |
| **Total (shuffled)** | **361,395** |

## Changes from Exp20

1. **Dataset**: nemotron-v2 full (6.3M) → chinese subset (195K) + novita (166K) = 361K
2. **Nodes**: .21/.28 → .22/.26
3. **Expected datagen speed**: ~2 days per round (vs ~37 days in exp20)
4. **Checkpoint pruning**: Best-N by validation loss (--keep-checkpoints 5)

## Files

- `k8s/k8s-minimax-m2.5-exp20-3-datagen.yaml` — datagen pod manifest
- `k8s/k8s-minimax-m2.5-exp20-3-train.yaml` — training pod manifest
- `k8s/run_minimax_m2.5_exp20_3_datagen.sh` — datagen run script
- `k8s/run_minimax_m2.5_exp20_3_train.sh` — training run script

## Issue Log

(none yet)

## Progress

### 2026-04-07 — Setup

- Merged nemotron-v2-chinese (195K) + novita (166K) → `/data/tengwan/datasets/exp20_3_merged/conversations.jsonl` (361K, shuffled, 5.6GB)
- Initial deployment on .22/.26 (stopped)

### 2026-04-08 — Redeployed on .17/.18

- Redeployed datagen on .17, training on .18 (nodes freed after Exp20-2 training stopped)
- Dataset already present on .17, model on both nodes
- Both pods running: datagen (.17) + train (.18)

### Issue: NCCL Timeout Crash (2026-04-08 11:51)

- **Epoch**: 29 (validation/checkpoint save phase)
- **Error**: `_ALLGATHER_BASE` timeout after 1800s (30min), ranks 1/4/5
- **Last saved checkpoint**: epoch 28
- **Root cause**: Same systematic .18 node GPU interconnect instability (identical to Exp19 crashes)
- **Fix**: Deleted failed pod, redeployed — training auto-resumes from checkpoint 28
