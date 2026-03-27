# Phase 2: Aurora Online Loop (Round-Based)

## 目标

实现 Aurora online loop：datagen → mask 预计算 → 训练 → checkpoint 更新 → 循环。每一轮，draft model 改进后重新计算 mask，反映更新后的接受模式。

## 背景

| | Phase 1 (Exp16) | Phase 1.5 (Exp16.5) | Phase 2 |
|---|---|---|---|
| **Mask 来源** | 动态实时计算 | 静态 Exp15 ckpt67 | 每轮更新（上一轮 ckpt） |
| **数据** | 固定 6K 文件 | 固定 6K 文件 | 每轮新生成 2K 文件 |
| **训练轮次** | 70 epochs | 70 epochs | 每轮 5 epochs × N 轮 |
| **闭环** | 否 | 否 | ✅ 是 |

## 架构

```
ROUND N on .18 (4 GPU):                on .17 (8 GPU, via SSH):
─────────────────────                   ─────────────────────
[1] datagen: 2000 .pt files (TP=4)
[2] mask: precompute_masks (1 GPU)
    using ckpt from round N-1
[3] rsync data + masks ──────────────> data arrives
                                       [4] torchrun train_streaming.py
                                           (5 epochs, Aurora static mask)
                                       [5] checkpoint saved
[6] rsync ckpt back <───────────────── latest ckpt
[7] update ckpt pointer → round N+1
```

## 文件结构

```
scripts/aurora_online_round.sh      — 单轮编排（.18 运行，SSH 到 .17 训练）
k8s/run_minimax_m2.5_aurora_online.sh — K8s 启动脚本，循环调用 aurora_online_round.sh
```

## 关键设计决策

1. **Per-round 目录** — `round_N/gen` + `round_N/masks`，无 stale mask 混合
2. **.17 共享 checkpoints** — `train_streaming.py` 跨轮次从上次 epoch 续训（累计 epoch 计数器）
3. **.18 编排** — .18 驱动循环，SSH 到 .17 训练。.17 被动接受
4. **顺序阶段** — 无 GPU 争用：datagen (TP=4) → masks (1 GPU) → 训练期间空闲
5. **Round 0 种子** — Exp15 ckpt67（最佳标准 KL 模型，63.2% acc@0）

## 实验配置

| 参数 | 值 |
|------|-----|
| Seed Checkpoint | Exp15 ckpt67 |
| Files/Round | 2000 |
| Epochs/Round | 5 |
| Max Rounds | 20 |
| LR | 3e-5 |
| Lambda Discard | 0.1 |
| Discard Top-K | 10 |
| Seq Length | 8192 |
| Datagen TP | 4 (on .18) |
| Training GPUs | 8 (on .17, FSDP) |

## 验证计划

1. **Smoke test**: 1 轮, 50 文件, 1 epoch — 验证完整循环
2. **Accept ratio 趋势**: 从 mask precompute 日志中提取每轮接受率 — 预期上升
3. **Loss 曲线**: 比较 round 0 vs round N 的起始 loss
4. **最终评估**: N 轮后用现有脚本对比 Exp15/16/16.5 基线

## 实验记录

### Smoke Test

| 项目 | 结果 |
|------|------|
| 日期 | |
| 配置 | 1 round, 50 files, 1 epoch |
| 状态 | 🔄 待运行 |

### Full Run

| Round | Files | Accept Ratio | Starting Loss | Notes |
|-------|-------|-------------|---------------|-------|
| 0 | | | | Seed: Exp15 ckpt67 |
| 1 | | | | |
| 2 | | | | |
| ... | | | | |
