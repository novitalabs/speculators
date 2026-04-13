# Experiment 20-3: Chinese + Novita Dataset with Continuous Datagen

## Status: COMPLETE (ckpt799 final eval, 2026-04-13)

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

### Issue 1: NCCL Timeout Crash (2026-04-08 11:51)

**When**: Epoch 29, validation/checkpoint save phase  
**Symptom**: `_ALLGATHER_BASE` timeout after 1800s (30min), ranks 1/4/5 SIGABRT  
**Root cause**: Same systematic .18 node GPU interconnect instability (identical to Exp19 crashes)  
**Fix**: Deleted failed pod, redeployed — training auto-resumes from checkpoint 28

### Issue 2: NCCL Timeout Crash (2026-04-09)

**When**: Epoch 69, validation/checkpoint save phase  
**Symptom**: `_ALLGATHER_BASE` timeout 1800s, rank 7 SIGABRT  
**Root cause**: Same .18 node GPU interconnect instability  
**Fix**: Deleted failed pod, redeployed — resumes from checkpoint 68

## Progress

### 2026-04-07 — Setup

- Merged nemotron-v2-chinese (195K) + novita (166K) → `/data/tengwan/datasets/exp20_3_merged/conversations.jsonl` (361K, shuffled, 5.6GB)
- Initial deployment on .22/.26 (stopped)

### 2026-04-08 — Redeployed on .17/.18

- Redeployed datagen on .17, training on .18 (nodes freed after Exp20-2 training stopped)
- Dataset already present on .17, model on both nodes
- Both pods running: datagen (.17) + train (.18)

### 2026-04-08 — NCCL crash, restarted from ckpt28

- Training crashed at epoch 29 validation (Issue 1), restarted, resumed from checkpoint 28

### 2026-04-09 — NCCL crash at epoch 69, restarted from ckpt68

- Training crashed at epoch 69 validation (Issue 2), restarted, resumed from checkpoint 68

### 2026-04-09 — Epoch 68, ckpt68 eval

- Training at epoch 68, val_loss=4.053, val_acc_0=74.7%
- Ran ZClawBench eval (ckpt68) on .17 GPU 4-7 (alongside datagen)

### 2026-04-10 — Epoch 217, ckpt217 eval

- val_loss=3.361, val_acc_0=74.1% — significant improvement over ckpt68 (4.053)
- Ran ZClawBench eval (ckpt217) on .17 GPU 4-7

### 2026-04-13 — Epoch 1018+, ckpt799 final eval

- Training continued to epoch 1018+, val_loss oscillating 3.2–3.7 (best observed: 3.217 at epoch 1015)
- Global progress: 390K+ unique files (~4 datagen rounds)
- ckpt799 selected as eval target (preserved by best-5 checkpoint pruning → one of lowest val_loss across ~950 epochs)
- ZClawBench eval (ckpt799) on .17 GPU 4-7 — **no meaningful improvement over ckpt217**
- Training stopped: model has reached ZClawBench speedup ceiling on this dataset

## Eval Results

### ZClawBench Eval — ckpt68 (2026-04-09)

Training still ongoing (not converged), epoch 68.

#### Simple ZClawBench (116 prompts, MAX_MODEL_LEN=32768)

| Model | Tok/s | Speedup | Acc@0 | Acc Length |
|-------|-------|---------|-------|------------|
| Baseline | 3067.8 | 1.00x | - | - |
| Exp20-3 ckpt68 | 4112.1 | **1.34x** | 52.3% | 1.870 |

#### Full ZClawBench (649 trajectories, bucketed)

| Bucket | #Prompts | Avg Input | Baseline Tok/s | Exp20-3 Tok/s | Speedup | Acc@0 | Acc Length |
|--------|----------|-----------|----------------|---------------|---------|-------|------------|
| short (0-2K) | 210 | 847 | 5388.0 | 4067.6 | 0.75x | 54.7% | 1.892 |
| med (2K-8K) | 320 | 4601 | 2327.4 | 2571.2 | **1.10x** | 57.5% | 1.964 |
| long (8K-32K) | 119 | 12177 | 906.7 | 1051.3 | **1.16x** | 56.8% | 1.956 |

#### Cross-Experiment Comparison (ckpt68)

| Experiment | Dataset | Simple | Short | Med | Long | Acc@0 (simple) |
|-----------|---------|--------|-------|-----|------|----------------|
| Exp19 | novita (166K) | 1.11x | 0.68x | 1.04x | 1.09x | 43.1% |
| Exp20-2 ckpt40 | 中文 (125K) | 1.34x | 0.73x | 1.09x | 1.13x | 48.6% |
| **Exp20-3 ckpt68** | **中文+novita (361K)** | **1.34x** | **0.75x** | **1.10x** | **1.16x** | **52.3%** |

### ZClawBench Eval — ckpt217 (2026-04-10)

Training ongoing, epoch 217, val_loss=3.361 (improved from 4.053 at ckpt68).

#### Simple ZClawBench (116 prompts, MAX_MODEL_LEN=32768)

| Model | Tok/s | Speedup | Acc@0 | Acc Length |
|-------|-------|---------|-------|------------|
| Baseline | 4100.4 | 1.00x | - | - |
| Exp20-3 ckpt217 | 4543.6 | **1.11x** | 53.3% | 1.899 |

*注：simple ZClaw baseline 波动较大（短 prompt avg 270 tokens），本次 4100 vs 上次 3068，speedup 比值不稳定。*

#### Full ZClawBench (649 trajectories, bucketed)

| Bucket | #Prompts | Avg Input | Baseline Tok/s | Exp20-3 Tok/s | Speedup | Acc@0 | Acc Length |
|--------|----------|-----------|----------------|---------------|---------|-------|------------|
| short (0-2K) | 210 | 847 | 5387.0 | 4017.5 | 0.75x | 54.9% | 1.898 |
| med (2K-8K) | 320 | 4601 | 2329.6 | 2610.7 | **1.12x** | 58.4% | 1.989 |
| long (8K-32K) | 119 | 12177 | 907.0 | 1066.8 | **1.18x** | 59.1% | 2.008 |

#### Cross-Experiment Comparison (ckpt217, final)

| Experiment | Dataset | Simple | Short | Med | Long | Acc@0 (long) |
|-----------|---------|--------|-------|-----|------|--------------|
| Exp19 | novita (166K) | 1.11x | 0.68x | 1.04x | 1.09x | ~40% |
| Exp20-2 ckpt40 | 中文 (125K) | 1.34x | 0.73x | 1.09x | 1.13x | 52.6% |
| Exp20-3 ckpt68 | 中文+novita (361K) | 1.34x | 0.75x | 1.10x | 1.16x | 56.8% |
| **Exp20-3 ckpt217** | **中文+novita (361K)** | **1.11x*** | **0.75x** | **1.12x** | **1.18x** | **59.1%** |

*simple ZClaw baseline 本次偏高导致 speedup 比值偏低；绝对 spec throughput 4543 tok/s > ckpt68 的 4112 tok/s

#### Analysis

- **long bucket 持续改善**：1.18x / Acc@0 59.1% / AccLen 2.008，接近 2 token 平均接受长度
- **med 小幅提升**：1.12x (vs ckpt68 1.10x)
- **short 仍 <1x**（0.75x），prefill 开销主导
- **训练仍未收敛**，val_loss 仍在下降（4.053→3.361），继续训练有望进一步改善

### ZClawBench Eval — ckpt799 (2026-04-13)

Training epoch 1018+, val_loss ~3.2 range. ckpt799 is one of the best-5 checkpoints by val_loss across ~950 epochs (69–1018).

#### Simple ZClawBench (116 prompts, MAX_MODEL_LEN=32768)

| Model | Tok/s | Speedup | Acc@0 | Acc Length |
|-------|-------|---------|-------|------------|
| Baseline | 4085.6 | 1.00x | - | - |
| Exp20-3 ckpt799 | 4726.6 | **1.16x** | 53.9% | 1.913 |

#### Full ZClawBench (649 trajectories, bucketed)

| Bucket | #Prompts | Avg Input | Baseline Tok/s | Exp20-3 Tok/s | Speedup | Acc@0 | Acc Length |
|--------|----------|-----------|----------------|---------------|---------|-------|------------|
| short (0-2K) | 210 | 847 | 5380.9 | 4054.3 | 0.75x | 55.6% | 1.924 |
| med (2K-8K) | 320 | 4601 | 2328.9 | 2607.3 | **1.12x** | 58.8% | 2.017 |
| long (8K-32K) | 119 | 12177 | 907.4 | 1065.4 | **1.17x** | 58.4% | 2.001 |

#### Final Cross-Experiment Comparison

| Experiment | Dataset | Simple | Short | Med | Long | Acc@0 (long) |
|-----------|---------|--------|-------|-----|------|--------------|
| Exp19 | novita (166K) | 1.11x | 0.68x | 1.04x | 1.09x | ~40% |
| Exp20-2 ckpt40 | 中文 (125K) | 1.34x | 0.73x | 1.09x | 1.13x | 52.6% |
| Exp20-3 ckpt68 | 中文+novita (361K) | 1.34x | 0.75x | 1.10x | 1.16x | 56.8% |
| Exp20-3 ckpt217 | 中文+novita (361K) | 1.11x* | 0.75x | 1.12x | 1.18x | 59.1% |
| **Exp20-3 ckpt799** | **中文+novita (361K)** | **1.16x** | **0.75x** | **1.12x** | **1.17x** | **58.4%** |

#### Analysis

- **ckpt799 与 ckpt217 性能基本持平**：在训练了 ~580 额外 epoch、经历 ~4 个 datagen round 后，ZClawBench speedup 无显著提升
- **达到当前数据集上限**：med ~1.12x、long ~1.17–1.18x、short 0.75x 为此架构+数据集组合的天花板
- **Val_loss 与 speedup 脱钩**：val_loss 从 3.361 继续下降至 ~3.2，但 ZClawBench speedup 不再提升
- **最佳 checkpoint**：ckpt217 (long 1.18x) 或 ckpt799 (long 1.17x) 性能相当，ckpt217 更早获得
- **结论**：进一步提升需要改变数据分布、架构或训练策略

### ZClawBench Eval — ThoughtWorks Eagle3 比较 (2026-04-13)

模型：`thoughtworks/MiniMax-M2.5-Eagle3`（HuggingFace，Apache 2.0）  
训练数据：英文 coding（HumanEval/SWEBench/Aider），20K 样本  
评测：vLLM + ZClawBench（中文，92% 中文或中英混合），num_speculative_tokens=3

#### ZClawBench 结果

| Bucket | Baseline Tok/s | TW Eagle3 Tok/s | Speedup | Acc@0 | AccLen |
|--------|---------------|----------------|---------|-------|--------|
| Simple (116) | 4103.0 | 3932.1 | 0.96x | 40.1% | 1.789 |
| Short 0-2K (210) | 5392.6 | 3584.8 | 0.66x | 38.9% | 1.702 |
| Med 2K-8K (320) | 2330.3 | 2410.6 | 1.03x | 42.1% | 1.726 |
| Long 8K-32K (119) | 907.6 | 974.1 | **1.07x** | 39.2% | 1.616 |

#### 与 Exp20-3 ckpt799 对比

| Model | Simple | Short | Med | Long | Acc@0 (long) |
|-------|--------|-------|-----|------|--------------|
| ThoughtWorks Eagle3 | 0.96x | 0.66x | 1.03x | 1.07x | 39.2% |
| **Exp20-3 ckpt799** | **1.16x** | **0.75x** | **1.12x** | **1.17x** | **58.4%** |

#### Analysis

- **ThoughtWorks 在 ZClawBench 上明显弱于 Exp20-3**：long 1.07x vs 1.17x，Acc@0 长上下差约 19 个百分点
- **原因**：ThoughtWorks 训练数据为英文 coding（HumanEval/SWEBench），ZClawBench 为 92% 中文内容，分布不匹配
- **ThoughtWorks 发布的数字（2.11x HumanEval）为 SGLang + 8 draft tokens + 英文场景**，与此次测试（vLLM + 3 tokens + 中文）不可比
- **结论**：中文场景下，针对目标分布定制训练（Exp20-3）远优于通用英文 coding 草稿模型

---

## Notes

### Sample 训练频率分析（epoch 263，2026-04-10）

| 指标 | 数值 |
|------|------|
| 训练 epoch 数 | ~263 |
| 每 epoch 步数 | ~400 steps |
| 累计消耗 unique 文件（global progress counter） | 109,167 |
| Datagen Round 1 产出 | 90,349 unique 样本（~46h/round） |
| Buffer 大小 | ~12K 文件（500GB cap，size-based eviction） |

**平均每个 sample 被训练约 0.9–1 次。**

Buffer 机制决定了训练频率分布：
- Buffer 维持 ~12K 文件，每 epoch 消耗 ~400 个、新增 ~460 个
- 一个文件在 buffer 中停留约 26 个 epoch（12,000 / 460）
- 每 epoch 被采样到的概率 ≈ 3.3%（400 / 12,000）
- 期望训练次数 ≈ 26 × 3.3% = **~0.87 次**

Difficulty weighting 导致不均匀分布：高难度样本训练 2–5 次，低难度样本可能 0 次（evict 前从未被采样）。

**当时不到 1 个完整 datagen round**（epoch 263 ≈ 24h，round 1 = 46h），90% 的样本只被生成过一次，模型尚未对全量数据过拟合。这与 val_loss 仍在下降（3.361，未收敛）一致。

**最终状态（epoch 1018+，2026-04-13）**：global progress 390K+ unique files，约 4 datagen rounds 完成。每个样本平均被训练 ~4 次（390K / 90K per round），difficulty weighting 导致高难度样本训练次数更多。尽管如此，ZClawBench speedup 未再提升，说明多轮训练未能突破当前数据集的表达上限。
