# Aurora 复现项目

基于 speculators 框架复现 Aurora 论文的核心训练算法，逐步从离线验证走向在线闭环。

## 分阶段概览

| Phase | 目标 | 状态 | 文档 |
|-------|------|------|------|
| **Phase 1** | 离线 Aurora Loss 验证 | ❌ KL 胜出 | [phase1-dynamic-loss.md](phase1-dynamic-loss.md) |
| **Phase 1.5** | 静态 Mask Aurora Loss | ❌ KL 胜出 | [phase1.5-static-mask.md](phase1.5-static-mask.md) |
| **Phase 2** | 在线闭环（文件系统） | ⛔ 取消（Phase 1/1.5 未达标） | [phase2-online-loop.md](phase2-online-loop.md) |
| **Phase 3** | 优化版在线系统 | ⛔ 取消 | [phase3-optimized.md](phase3-optimized.md) |

## 核心思路

Aurora 将投机解码建模为异步 RL 问题。核心算法改进是将 KL 蒸馏损失拆分为：

- **Acceptance Loss**: 对 draft 与 verifier 一致位置的标准 KL 蒸馏
- **Discard Sampling Loss**: 对不一致位置，用 top-k 过滤后的 KL 蒸馏，以 `lambda_discard` 加权

Phase 1 先验证这个 loss 改进是否有效（离线模式），再决定是否投入 Phase 2-3 的系统工程。

## 已有框架能力

| 能力 | 状态 | 来源 |
|------|------|------|
| EAGLE3 模型架构 | ✅ | `models/eagle3/` |
| KL 蒸馏损失 | ✅ | `eagle3/core.py: loss_function()` |
| Aurora Accept/Discard Loss | ✅ Phase 1 | `eagle3/core.py: aurora_loss_function()` |
| Aurora 静态 Mask 支持 | ✅ Phase 1.5 | `aurora_loss_function(static_accepted_mask=...)` |
| Mask 预计算工具 | ✅ Phase 1.5 | `scripts/precompute_aurora_masks.py` |
| FSDP 分布式训练 | ✅ | `train/trainer.py` |
| 流式训练 (manifest-based) | ✅ | `train_streaming.py` |
| 词表映射 (d2t/t2d) | ✅ | `train/vocab_mapping.py` |

## 相关实验

| 实验 | 说明 | 结果 |
|------|------|------|
| Exp14 | Aurora 架构 + 标准 KL (52K data) | 52.7% Acc@0, 1.14x |
| Exp15 | Aurora 架构 + 标准 KL (114K data) | 63.2% Acc@0, 1.01x |
| **Exp16** | Aurora 架构 + Aurora Loss (动态 mask) | ❌ ZClawBench 31.7% Acc@0 (KL: 39.6%) |
| **Exp16.5** | Aurora 架构 + Aurora Loss (静态 mask) | ❌ ZClawBench 30.9% Acc@0 (KL: 39.6%) |

### Aurora Loss 实验总结

Aurora accept/discard loss 在两种 mask 模式下均不如标准 KL loss（Acc@0 差 ~9pp）。Phase 2/3 取消。标准 KL + 大数据集 (Exp15) 仍为最佳方案。
