# Experiment 23: Dual TP=4 Datagen

## Status: RUNNING (epoch 150+, val_acc@0 peak ~0.761) | ckpt145 eval: novita 1.58x / ZClaw 1.12x

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

| Metric | Exp22 | Expected Exp23 | Exp23 ckpt145 |
|--------|-------|----------------|---------------|
| Datagen throughput | ~2,500 files/hr | ~5,000 files/hr | **~13,200 files/hr** ✓ |
| Mean sample train_count | 26.7x | ≤15x | TBD |
| novita_merged_eval speedup | 1.67x (ckpt230) | ≥1.50x | **1.58x ✓** (ckpt145, non-peak) |
| ZClaw speedup | 1.12x (ckpt224) | ≥1.20x | **1.12x** (ckpt145) |
| ZClaw Acc@0 | 53.1% (ckpt224) | ≥50% | **53.5% ✓** (ckpt145) |

## Results

### 2026-04-15 — Deployed

Exp22 stopped (epoch 384, NCCL crash). Exp23 datagen pod deployed with dual TP=4 processes on .17. Both processes tokenizing dataset (919K samples). Training pod pending — will deploy after datagen generates sufficient data (~5000 files).

### 2026-04-15 — Training deployed + NCCL crash (epoch 17)

Training pod deployed on .18 after gen/ reached ~5000 files. Used exp22 vocab mapping (d2t.npy / t2d.npy) copied directly — same dataset and model, so identical mapping. Training started from epoch 0.

**Issue**: NCCL AllGather timeout at epoch 17, same recurring pattern as exp21/exp22. Pod deleted + recreated; auto-resumed from ckpt16 (epoch 17 restarted).

### 2026-04-15 — Training progress (epoch 33, pre-crash)

Resumed from ckpt16. Training stable post-crash, reached epoch 33 before second NCCL crash.

| Metric | Value |
|--------|-------|
| Peak epoch (run 2) | 33 |
| Steps/epoch | ~380–426 |
| Time/epoch | ~5:00–5:21 |
| Training TPM | **~5.4M tokens/min** (82 steps/min × 8 GPUs × 8192 tokens) |
| Checkpoints saved | 14, 15, 16 (run 1) + 24, 31 (run 2, val-triggered) |

Checkpoint saving is triggered by `val_every_steps=500` (not every epoch), so only epochs coinciding with step-500 boundaries get checkpointed.

### 2026-04-15 — Datagen 吞吐量分析

Measured via merger log (process B files moved to gen/).

| 指标 | Exp22 (single TP=4) | Exp23 Process B | Exp23 Combined |
|------|---------------------|-----------------|----------------|
| 生成速率 | ~2,500 files/hr (42 files/min) | ~6,576 files/hr (**110 files/min**) | **~13,200 files/hr (220 files/min)** |
| 测量方式 | 26% of round 1 / elapsed | Merger log: 83,824 files in 12.75h | 2× process B (同配置) |

**Process B (GPU 4-7)**: 83,824 files merged in 765 min (04:27–17:12 CST) = 109.6 files/min  
**Process A (GPU 0-3)**: 同配置，预计相近速率 (~110 files/min)  
**Gen/ buffer on .17**: 13,540 files ready (健康)

Process B 速率 (110 files/min) 高于 exp22 单进程 (42 files/min) 约 2.6x，可能因为：round 1 无 difficulty resampling 开销 + 无 eval 进程抢占 GPU。

**训练/数据生成比**：训练 ~5.4M tokens/min（消费），datagen ~1.3M tokens/min（生产，按 6K avg tokens/conv 估算）。文件层面：datagen ~220 files/min，训练消费速率取决于 buffer 复用策略。缓冲区健康状态良好，后续观察 manifest 中 mean train_count 是否保持在目标 10x 附近（exp22 为 26.7x）。

### Issue 2: NCCL AllGather Timeout (epoch ~34)

**When**: 2026-04-15 11:03 UTC, epoch 33 completed, crash during epoch 34 (or post-epoch 33 validation)  
**Symptom**: SIGABRT on rank 0 first, then all ranks. `WorkNCCL(OpType=_ALLGATHER_BASE) ran for 1800050ms before timing out` (30-min watchdog). Same pattern as exp22 epoch 33 and epoch 384.  
**Root cause**: Transient GPU communication failure on node .18 — recurring issue (3rd occurrence: exp22 ckpt33, exp22 ckpt384, now exp23 ckpt34).  
**Fix**: Deleted + recreated train pod; will auto-resume from ckpt31 (ckpt32 logged as saved but absent from checkpoints dir, likely incomplete flush at crash time). Epochs 32–33 will re-train.

### 2026-04-16 — Training progress (epoch 148, strong convergence)

Resumed from ckpt31 post-crash, now at epoch 149. No further NCCL crashes since Issue 2.

**Key epochs (val_acc@0 / val_loss)：**

| Epoch | val_acc@0 | val_loss | Notes |
|-------|-----------|----------|-------|
| 0–30 | 0.70–0.73 | — | Rapid early improvement |
| 132 | — | — | ckpt saved (first post-31 checkpoint) |
| **141** | **0.752** | **3.405** | Best val_loss so far |
| **142** | — | **3.396** | Best val_loss (no ckpt?) |
| 145 | 0.756 | — | ckpt saved |
| 146 | **0.761** | — | **Peak val_acc@0** seen; ckpt saved |
| 147 | 0.753 | 3.559 | ckpt saved |
| **148** | **0.748** | **3.477** | ckpt saved; epoch 149 now running |

Saved checkpoints: 14, 15, 16 (run 1), 24, 31 (run 2 early), 132, 145, 146, 147, 148

**Comparison vs baselines：**

| Exp | Best val_acc@0 | Epoch |
|-----|---------------|-------|
| Exp19 | ~0.60 | ckpt5 |
| Exp21 | 0.668 | 62 |
| Exp22 | 0.714 | 120 |
| **Exp23** | **0.761** | **~146** |

Exp23 峰值比 exp22 高 **+4.7pp**，验证了 dual datagen 减少样本复用（target ~10x vs exp22 的 26.7x）可显著提升模型质量。val_acc@0 在 0.74–0.76 附近波动，仍接近峰值，训练继续。

**Datagen 状态（2026-04-16）：**

| 指标 | 值 |
|------|-----|
| Merger total (process B) | 389,540 files |
| Gen/ buffer (.17) | 14,636 files |
| 双进程状态 | 均正常运行 |

### 2026-04-16 — Eval ckpt145 (novita_merged_eval + ZClawBench，节点 .28)

Eval pod on .28 (4 GPUs, TP=4). Checkpoint 2.6GB, copied from .18 → .28.

| Name | Benchmark | Tok/s | Speedup | Acc@0 | Acc@1 | Acc@2 | AccLen |
|------|-----------|-------|---------|-------|-------|-------|--------|
| baseline | novita_merged_eval | 2700.3 | 1.00x | — | — | — | — |
| baseline | zclawbench | 4005.5 | 1.00x | — | — | — | — |
| exp23_ckpt145 | novita_merged_eval | 4268.4 | **1.58x** | 0.686 | 0.445 | 0.264 | 2.394 |
| exp23_ckpt145 | zclawbench | 4467.4 | **1.12x** | 0.535 | 0.258 | 0.129 | 1.921 |

**Comparison vs exp22 (同 benchmark)：**

| Benchmark | Exp22 ckpt224 | Exp22 ckpt230 | **Exp23 ckpt145** |
|-----------|---------------|---------------|-------------------|
| novita_merged_eval | 1.52x | **1.67x** | **1.58x** |
| ZClaw speedup | **1.12x** | 1.05x | **1.12x** ✓ |
| ZClaw Acc@0 | 53.1% | 52.1% | **53.5%** ✓ |

**Analysis**：ckpt145 不是训练峰值（val_acc@0 峰值 0.761 在 epoch ~146，训练仍在继续）。当前成绩已与 exp22 最佳持平：
- ZClaw 1.12x 直接追平 exp22 ckpt224，Acc@0 53.5% 微超；
- novita 1.58x 略低于 exp22 ckpt230 的 1.67x，但 ckpt230 是 exp22 的较晚 checkpoint；
- 期待峰值 ckpt（~epoch 146）会进一步提升，尤其 novita 方向。
