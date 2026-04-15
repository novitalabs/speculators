# Experiment 23: Dual TP=4 Datagen

## Status: RUNNING (datagen starting, training pending)

## Motivation

Exp22 (919K combined dataset) demonstrated a severe datagen bottleneck: single TP=4 on .17 produced ~2,500 files/hour while training consumed data ~3x faster, leading to mean sample reuse of 26.7x (target 10x). This over-training likely contributed to ZClaw underperformance (1.12x vs target 1.20x).

**Hypothesis**: Running two parallel TP=4 datagen processes on .17 (using all 8 GPUs) doubles throughput to ~5,000 files/hour, significantly reducing sample over-training and improving model quality, especially on ZClawBench.

## Dataset

Same as exp22: `/data/datasets/exp22_merged/` (~919K conversations)

| Source | Conversations |
|--------|---------------|
| novita_merged/train.jsonl | 724K |
| nemotron-v2-chinese/conversations.jsonl | 195K |
| **Total** | **~919K** |

## Design

### Dual Datagen Architecture

One pod on .17 (8 GPUs) runs two parallel `data_generation_offline.py` processes:

| Component | GPUs | Output Dir | Index Range |
|-----------|------|------------|-------------|
| Process A | 0-3 (`CUDA_VISIBLE_DEVICES=0,1,2,3`) | `gen/` | 0 – 4,999,999 |
| Process B | 4-7 (`CUDA_VISIBLE_DEVICES=4,5,6,7`) | `gen_b/` | 5,000,000+ (after merge) |
| Merger | — | — | Moves `gen_b/*.pt` → `gen/` with +5M offset |

**Key design decisions:**
- Separate output dirs avoid `find_last_checkpoint()` race conditions (no file locking in datagen)
- Merger adds fixed 5M index offset to gen_b files before moving to gen/ (collision after ~21 rounds = thousands of epochs, safe)
- `difficulty_scores.json` symlinked from gen_b/ → gen/ so both processes benefit from training feedback
- Training side syncs from a single `gen/` dir (no changes to `sync_datagen.sh`)

### Training

Same as exp22 — 8-GPU FSDP on .18, Aurora architecture.

| Parameter | Value |
|-----------|-------|
| Architecture | Aurora (24 heads, 8192 intermediate, 32K draft vocab, RoPE theta 5M) |
| LR | 3e-5 |
| Seq length | 8192 |
| Buffer max | 500GB |
| Target train count | 10 |
| Val every steps | 500 |

## Infrastructure

- **Datagen**: node .17 (10.83.115.17), 8x GPU (dual TP=4)
- **Training**: node .18 (10.83.115.18), 8x GPU
- **Image**: speculators:v0.17.0

## Files

- `k8s/k8s-minimax-m2.5-exp23-datagen.yaml`
- `k8s/k8s-minimax-m2.5-exp23-train.yaml`
- `k8s/run_minimax_m2.5_exp23_datagen.sh`
- `k8s/run_minimax_m2.5_exp23_train.sh`
- Output: `/data/output/minimax_m2.5_eagle3_exp23/`

## Expected Outcome

| Metric | Exp22 | Expected Exp23 |
|--------|-------|----------------|
| Datagen throughput | ~2,500 files/hr | ~5,000 files/hr |
| Mean sample train_count | 26.7x | ≤15x |
| novita_merged_eval speedup | 1.67x (ckpt230) | ≥1.50x |
| ZClaw speedup | 1.12x (ckpt224) | ≥1.20x |
| ZClaw Acc@0 | 53.1% (ckpt224) | ≥50% |

## Results

### 2026-04-15 — Deployed

Exp22 stopped (epoch 384, NCCL crash). Exp23 datagen pod deployed with dual TP=4 processes on .17. Both processes tokenizing dataset (919K samples). Training pod pending — will deploy after datagen generates sufficient data (~5000 files).
