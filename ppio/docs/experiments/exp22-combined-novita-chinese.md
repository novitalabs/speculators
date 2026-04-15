# Experiment 22: Combined Novita + Chinese Data

## Status: STOPPED (epoch 384, replaced by exp23 with dual datagen)

## Motivation

Exp21 achieves 1.57x on Novita (English) using 724K novita conversations.
Exp20-2 achieves 1.34x on ZClawBench (Chinese) using 195K Chinese nemotron conversations.
Both excel in their respective domains but at the cost of the other.

**Hypothesis**: Training on the union of both datasets (919K conversations) produces a draft model that handles both English (Novita) and Chinese (ZClawBench) trajectories well, without sacrificing either.

## Dataset

| Source | Path | Conversations |
|--------|------|---------------|
| Exp21 novita | `/data/datasets/novita_merged/train.jsonl` | 724K |
| Exp20-2 Chinese | `/data/tengwan/datasets/nemotron-v2-chinese/conversations.jsonl` | 195K |
| **Total** | `/data/datasets/exp22_merged/` | **~919K** |

The combined directory is created at runtime via symlinks in the datagen script:
- `exp22_merged/novita_merged_train.jsonl` → `novita_merged/train.jsonl`
- `exp22_merged/nemotron_chinese.jsonl` → `nemotron-v2-chinese/conversations.jsonl`

## Design

Same continuous datagen + difficulty-weighted resampling pipeline as exp19/21.

| Parameter | Exp21 | Exp20-2 | **Exp22** |
|-----------|-------|---------|-----------|
| Training data | novita_merged (724K) | nemotron-v2 Chinese (195K) | **both (919K)** |
| Datagen node | .14 | .17 | **.17** |
| Train node | .23 | .18 | **.18** |
| Output path | exp21 | exp20_2 | **exp22** |

Architecture, LR, seq_length, buffer size — all identical to exp21.

## Infrastructure

- **Datagen**: node .17 (10.83.115.17), 8x GPU, TP=4
- **Training**: node .18 (10.83.115.18), 8x GPU
- **Image**: speculators:v0.17.0

## Deployment

```bash
kubectl apply -f k8s/k8s-minimax-m2.5-exp22-datagen.yaml
# Wait for datagen to accumulate data, then:
kubectl apply -f k8s/k8s-minimax-m2.5-exp22-train.yaml
```

## Files

- `k8s/k8s-minimax-m2.5-exp22-datagen.yaml`
- `k8s/k8s-minimax-m2.5-exp22-train.yaml`
- `k8s/run_minimax_m2.5_exp22_datagen.sh`
- `k8s/run_minimax_m2.5_exp22_train.sh`
- Data: `/data/datasets/exp22_merged/` (symlinks to both source datasets)
- Output: `/data/output/minimax_m2.5_eagle3_exp22/`

## Expected Outcome

| Benchmark | Exp21 | Exp20-2 | Expected Exp22 | Exp22 best |
|-----------|-------|---------|----------------|------------|
| novita0312_eval speedup | — | — | ≥1.40x | **1.35x** ✗ (ckpt230) |
| novita_merged_eval speedup | 1.57x | not eval'd | ≥1.40x | **1.67x** ✓ (ckpt230) |
| ZClaw speedup | 0.41x (batched) | 1.34x | ≥1.20x | **1.12x** ✗ (ckpt224) |
| ZClaw Acc@0 | 42.7% | 48.6% | ≥45% | **53.1%** ✓ (ckpt224) |

*(ckpt381 eval: novita_merged 4649 tok/s abs; ZClaw 1.07x / 51.7%)*

## Results

### 2026-04-13 — Deployed

Datagen (node .17) and training (node .18) pods deployed. Combined dataset (~919K conversations): novita_merged/train.jsonl (724K) + nemotron-v2-chinese/conversations.jsonl (195K). Symlink directory created at `/data/datasets/exp22_merged/` by datagen script on first run.

### 2026-04-13 — Training progress (epoch 33 crash)

Strong early convergence — val_acc@0 reached 0.669 at epoch 30, outpacing exp21 (which hit 0.632 at epoch 27).

| Epoch | val_acc@0 | val_loss |
|-------|-----------|----------|
| 0 | 0.402 | 9.413 |
| 10 | 0.613 | 5.117 |
| 17 | 0.641 | 4.613 |
| 30 | **0.669** | 4.018 |
| 32 | 0.660 | 3.907 |

### Issue 1: NCCL AllGather Timeout (epoch 33)

**When**: 2026-04-13 11:48 UTC, epoch 33, mid-training  
**Symptom**: SIGABRT on rank 5 first, then all ranks. `WorkNCCL(OpType=REDUCE) ran for 1800077ms before timing out` (30-min watchdog).  
**Root cause**: Transient GPU communication failure on node .18 — same pattern as exp21 crash at epoch 139.  
**Fix**: Restarted train pod; auto-resumes from epoch 32 checkpoint.

### 2026-04-14 — Training progress (epoch 383, stable post-peak)

Resumed from ckpt32. Strong continued improvement post-crash. **Peak val_acc@0 = 0.714 at epoch 120** (no checkpoint saved). Model past its peak since ~epoch 150; val_acc@0 stabilized ~0.708–0.710, val_loss oscillating 3.6–4.0. Datagen at ~26% of round 1 (difficulty-weighted resampling not yet active).

Key epochs (post-crash):

| Epoch | val_acc@0 | val_loss | Notes |
|-------|-----------|----------|-------|
| 39 | 0.699 | 3.693 | |
| 58 | 0.703 | 3.631 | |
| 90 | 0.699 | 3.643 | |
| **120** | **0.714** | 4.012 | **best val_acc@0 (no ckpt saved)** |
| 128 | 0.711 | 3.561 | |
| **148** | 0.709 | **3.561** | **best val_loss (no ckpt saved)** |
| 165 | 0.703 | 3.779 | |
| 188 | 0.689 | 4.688 | |
| 221 | 0.710 | 3.836 | |
| **224** | **0.710** | **3.617** | **lowest val_loss of saved ckpts** |
| 225 | 0.710 | 3.648 | |
| 228 | 0.710 | 3.768 | |
| 229 | — | 4.063 | |
| **230** | **0.710** | 3.994 | |
| 344 | — | — | ckpt saved |
| 365 | — | — | ckpt saved |
| **381** | — | — | **ckpt saved + evaled** |
| 382 | — | — | ckpt saved |
| 383 | — | — | ckpt saved (latest before crash) |
| 384 | 0.708 | 3.907 | NCCL crash mid-epoch |

Saved checkpoints on .18: 17, 30, 32, 221, 225, 228, 230, 344, 365, 381, 382, 383  
Transferred to .17 for eval: 224, 230, 381

**Comparison vs baselines (val_acc@0):**

| Exp | Best val_acc@0 | Epoch |
|-----|---------------|-------|
| Exp19 | ~0.60 | ckpt5 |
| Exp21 | 0.668 | 62 |
| **Exp22** | **0.714** | **120** |

Exp22 peak is **+4.6pp above exp21** with combined Novita+Chinese data.

### 2026-04-14 — Eval started (ckpt 165 + 188, node .17 GPU 4-7)

Running Novita (100 prompts, seed=42) + ZClawBench (116 prompts), TP=4, max_model_len=8192.
Output: `/data/output/minimax_m2.5_eval_exp22/`

**Note**: ckpt165 and ckpt188 were superseded by continued training. Eval ran on ckpt224 (val_acc@0=0.710, past-peak but best-transferred checkpoint).

### 2026-04-14 — Eval complete: ckpt224 results

Ckpt224 evaluated on Novita (100 prompts) + ZClawBench (116 prompts), TP=4, max_model_len=8192. Run inside datagen pod on .17 GPU 4-7 (`CUDA_VISIBLE_DEVICES=4,5,6,7`).

Novita eval set: `/data/datasets/novita_merged/eval.jsonl` (labeled `novita_merged_eval`)

| Name | Benchmark | Tok/s | Speedup | Acc@0 | Acc@1 | Acc@2 | AccLen |
|------|-----------|-------|---------|-------|-------|-------|--------|
| baseline | novita_merged_eval | 2704.4 | 1.00x | — | — | — | — |
| baseline | zclawbench | 4237.9 | 1.00x | — | — | — | — |
| exp22_ckpt224 | novita_merged_eval | 4119.1 | **1.52x** | 0.683 | 0.427 | 0.244 | 2.354 |
| exp22_ckpt224 | zclawbench | 4745.9 | **1.12x** | 0.531 | 0.247 | 0.117 | 1.896 |

**Comparison vs expected and baselines:**

| Benchmark | Exp21 | Exp20-2 | Expected Exp22 | Exp22 ckpt224 | Exp22 ckpt230 |
|-----------|-------|---------|----------------|---------------|---------------|
| novita_merged_eval speedup | 1.57x | — | ≥1.40x | 1.52x ✓ | **1.67x** ✓ |
| ZClaw speedup | 0.41x | 1.34x | ≥1.20x | **1.12x** (below target) | 1.05x |
| ZClaw Acc@0 | 42.7% | 48.6% | ≥45% | **53.1%** ✓ | 52.1% ✓ |

**Analysis**: Combined training confirmed to improve both languages vs exp21 (0.41x → 1.12x on ZClaw). Novita 1.52x is slightly below exp21's 1.57x — expected since ckpt224 is past peak (best val_acc@0=0.714 at epoch 120, no checkpoint saved). ZClaw 1.12x is below the ≥1.20x target and below exp20-2's 1.34x, likely because: (a) past-peak checkpoint, (b) Chinese data is only 21% of the combined dataset (195K/919K).

### 2026-04-14 — Eval ckpt230 (novita_merged_eval + zclawbench)

Novita eval set: `/data/datasets/novita_merged/eval.jsonl` (labeled `novita_merged_eval`)

| Name | Benchmark | Tok/s | Speedup | Acc@0 | Acc@1 | Acc@2 | AccLen |
|------|-----------|-------|---------|-------|-------|-------|--------|
| baseline | novita_merged_eval | 2704.4 | 1.00x | — | — | — | — |
| baseline | zclawbench | 4237.9 | 1.00x | — | — | — | — |
| exp22_ckpt230 | novita_merged_eval | 4506.1 | **1.67x** | 0.674 | 0.410 | 0.236 | 2.320 |
| exp22_ckpt230 | zclawbench | 4446.6 | **1.05x** | 0.521 | 0.236 | 0.111 | 1.868 |

**ckpt224 vs ckpt230 对比：**

| Checkpoint | val_loss | novita_merged_eval | zclawbench |
|-----------|----------|-----------------|------------|
| ckpt224 | 3.617 | 1.52x | 1.12x |
| ckpt230 | 3.994 | **1.67x** | 1.05x |

ckpt230 novita_merged_eval 反而更高（1.67x > 1.52x），ZClaw 略降（1.05x < 1.12x）。val_loss 与 speedup 并非单调相关，可能存在评估噪声或 acceptance 率与任务分布的交互。**注意：以上 novita 结果使用的是 `novita_merged/eval.jsonl`，非正式 novita0312_eval。**

### 2026-04-14 — Eval ckpt230 (novita0312_eval 正式，/data/datasets/novita20260312_eval/conversations.jsonl)

| Name | Benchmark | Tok/s | Speedup | Acc@0 | Acc@1 | Acc@2 | AccLen |
|------|-----------|-------|---------|-------|-------|-------|--------|
| baseline | novita0312_eval | 3807.0 | 1.00x | — | — | — | — |
| baseline | zclawbench | 4235.6 | 1.00x | — | — | — | — |
| exp22_ckpt230 | novita0312_eval | 5153.9 | **1.35x** | 0.699 | 0.417 | 0.263 | 2.378 |
| exp22_ckpt230 | zclawbench | 4458.6 | **1.05x** | 0.521 | 0.236 | 0.111 | 1.868 |

novita0312_eval 1.35x，低于 novita_merged_eval 上的 1.67x，也低于目标 ≥1.40x。novita20260312 是线上真实流量样本（ZClaw 风格的 agent 任务），与 novita_merged 训练分布有差异，导致 acceptance 率偏低。

### 2026-04-15 — Eval ckpt381 (novita_merged_eval + zclawbench)

Ckpt381 evaluated on .17 GPU 4-7. **Note**: novita baseline tok/s varied vs prior runs (3557 vs 2704 tok/s) due to GPU load variation from datagen. ZClaw baseline stable (4240 tok/s). Absolute novita throughput (4649) is similar to ckpt230 (4506).

| Name | Benchmark | Tok/s | Speedup | Acc@0 | Acc@1 | Acc@2 | AccLen |
|------|-----------|-------|---------|-------|-------|-------|--------|
| baseline | novita_merged_eval | 3557.4 | 1.00x | — | — | — | — |
| baseline | zclawbench | 4239.8 | 1.00x | — | — | — | — |
| exp22_ckpt381 | novita_merged_eval | 4648.7 | **1.31x** | 0.692 | 0.436 | 0.250 | 2.378 |
| exp22_ckpt381 | zclawbench | 4519.9 | **1.07x** | 0.517 | 0.235 | 0.109 | 1.861 |

**ckpt230 vs ckpt381 对比（absolute throughput）：**

| Checkpoint | novita abs tok/s | zclaw speedup | ZClaw Acc@0 |
|-----------|-----------------|---------------|-------------|
| ckpt230 | 4506 | 1.05x | 52.1% |
| ckpt381 | 4649 | **1.07x** | 51.7% |

ckpt381 ZClaw 略有改善（1.07x vs 1.05x）；novita 绝对吞吐也略高（4649 vs 4506）。训练仍在收益，但提升已趋于平稳。

### Issue 2: NCCL AllGather Timeout (epoch 384)

**When**: 2026-04-14 21:00 UTC, epoch 384, mid-training (during or after validation)  
**Symptom**: SIGABRT on rank 5 first, then all ranks. Same 30-min watchdog pattern.  
**Root cause**: Transient GPU communication failure on node .18 — recurring pattern (also epoch 33, exp21 epoch 139).  
**Fix**: Deleted + recreated train pod; will auto-resume from ckpt383.

### 2026-04-14 — Datagen 瓶颈分析（样本复用过多）

**现象：** 训练样本平均被训练次数远超预期：

| 统计口径 | files | mean train_count | p50 | max |
|--------|-------|-----------------|-----|-----|
| 当前缓冲区内（.18） | 8,867 | 13.0 | 15 | 19 |
| 已驱逐（eviction_ledger） | 206,170 | **26.7** | 22 | 108 |

设计目标 `TARGET_TRAIN_COUNT=10`（驱逐后不再 re-sync 的阈值），但实际均值 26.7x，说明样本在被驱逐前已被训练多次，而驱逐后仍留在 buffer 中继续被训练。

**根本原因：datagen 速度严重滞后于训练速度。**

- Datagen 进度：60,459 / 229,955（**26%**），~2,500 files/hour（TP=4，GPU 0-3）
- 训练进度：epoch ~232，`files_ever_seen` = 185,903，已消费 datagen 生成量的 ~3倍
- 新鲜数据供给不足 → 旧样本反复被复用 → 过拟合风险

**优化方案（优先级排序）：**

1. **TP=8（最高优先级，零成本）**：datagen pod 已申请 8 GPU（当前仅用 4 个，TP=4）。将 `DATAGEN_TP=8` 重启 datagen，预计吞吐量翻倍（~5,000 files/hour）。修改 k8s yaml 加入 `DATAGEN_TP: "8"` env var 或重启时传参。

2. **增加第二个 datagen 节点**：`sync_datagen.sh` 支持 `--datagen-nodes` 传多个节点。在另一台空闲节点（如 .21/.22）起第二个 datagen pod，`DATAGEN_NODE` 改为两个 IP，吞吐再翻倍。

3. **降低 seq_len**：当前 `SEQ_LENGTH=8192`，改为 4096 可使 datagen 吞吐翻倍，但会丢失长上下文训练样本，对 ZClaw（含长 system prompt）可能有负面影响，谨慎。
