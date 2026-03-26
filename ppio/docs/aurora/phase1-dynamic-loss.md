# Phase 1: Dynamic Accept/Discard Loss

## 目标

验证 Aurora 的 accept/discard loss 是否优于标准 KL 蒸馏（纯离线模式，零数据管道改动）。

## 设计决策

### 动态 mask vs 静态 mask

Aurora 论文使用从推理 trace 中收集的 accept/reject 信息。Phase 1 简化为**动态 mask**：在训练时实时比较 draft argmax 与 verifier argmax，无需修改数据管道。

```
accepted = (draft_logits.argmax(-1) == target_logits.argmax(-1))
rejected = ~accepted & loss_mask
```

用 `detach()` 阻断梯度通过 mask，保证 mask 只是路由信号。

### Loss 公式

```
L = L_accept + λ_discard * L_discard

L_accept = KL(draft || verifier) on accepted positions  (标准 KL)
L_discard = KL(draft_topk || verifier_topk) on rejected positions  (top-k 过滤后 KL)
```

- `λ_discard = 0.1` — rejected 位置权重较低，避免对噪声信号过拟合
- `discard_top_k = 10` — 只关注 target 分布中概率最高的 10 个 token

### Top-k 过滤细节

对 rejected 位置：
1. 取 target 分布 top-k，重归一化为概率分布
2. 取 draft logits 在相同 top-k 位置上的值，做 log_softmax 重归一化
3. 计算 KL(log_draft_topk, target_topk)

这确保 discard loss 只关注 verifier 认为重要的 token，而不是整个词表。

### torch.compile 兼容性

初版使用 `if num_accepted > 0` 条件分支导致 `torch.tensor(0.0)` 没有 grad_fn，backward 失败。
修复为**无分支实现**：始终计算两个 loss，通过 mask 乘零处理空集情况。

```python
# 无分支：mask 自动处理空集
accept_kl = (kl.sum(-1) * accept_mask).sum() / (num_accepted + 1e-5)
discard_kl = (kl_topk.sum(-1) * reject_mask).sum() / (num_rejected + 1e-5)
```

## 代码改动

### `src/speculators/models/eagle3/core.py`

| 函数 | 改动 | 说明 |
|------|------|------|
| `aurora_loss_function()` | 新增 ~60 行 | 核心 loss 实现，无分支，torch.compile safe |
| `compute_metrics()` | +15 行 | 新增 `aurora_config` 参数，条件调用 aurora loss |
| `forward()` | +2 行 | 从 kwargs 提取 `aurora_config` 传递给 compute_metrics |
| `get_trainer_kwargs()` | +10 行 | 构建 aurora_config dict，传入 train/val kwargs |

### `scripts/train.py` & `scripts/train_streaming.py`

各增加 3 个 CLI 参数：
- `--aurora-loss` (store_true) — 启用 Aurora loss
- `--lambda-discard` (float, default=0.1) — discard loss 权重
- `--discard-top-k` (int, default=10) — top-k 过滤大小

### 后向兼容

不传 `--aurora-loss` 时行为完全不变，`aurora_config=None` 走原有 `loss_function()` 路径。

## 实验: Exp16

### 配置

| 参数 | 值 |
|------|------|
| 节点 | .17 (8x H200) |
| 数据 | Exp15 数据 (6073 .pt files, 303GB) |
| 架构 | Aurora-like (24 heads, 8192 intermediate, 32K vocab) |
| Loss | Aurora accept/discard (`λ=0.1, top_k=10`) |
| 训练 | 8 GPU FSDP, lr=3e-5, 70 epochs, `train.py` |
| 基线 | Exp15 ckpt67 (标准 KL, 63.2% Acc@0, 1.01x) |

### 关键文件

- `k8s/run_minimax_m2.5_aurora_loss_train.sh` — 训练脚本
- `k8s/k8s-minimax-m2.5-aurora-loss-train.yaml` — K8s pod 配置
- `ppio/docs/experiments/exp16-aurora-loss.md` — 实验日志

### 监控指标

| 指标 | 说明 | 预期范围 |
|------|------|----------|
| `aurora_accept_ratio_0` | 接受比例 (ttt_step 0) | 0.5–0.8 |
| `aurora_accept_loss_0` | 接受位置 KL | 递减 |
| `aurora_discard_loss_0` | 拒绝位置 top-k KL | 递减 |
| `val/loss_epoch` | 总验证 loss | 对比 Exp15 |
| `val/full_acc_0_epoch` | 验证集准确率 | 对比 Exp15 |

### 训练进展

| Epoch | val/loss | accept_ratio_0 | full_acc_0 | Notes |
|-------|---------|----------------|------------|-------|
| 0 | 1.412 | 0.118 | 0.118 | 初始，模型未训练 |
| 5 | 1.259 | 0.442 | 0.442 | 快速提升 |
| 10 | 1.188 | 0.509 | 0.509 | accept_ratio 过 50% |
| 15 | 1.145 | 0.552 | 0.552 | |
| 20 | 1.132 | 0.552 | 0.552 | |
| 23 | 1.116 | 0.563 | 0.563 | 训练中... |

### 问题与修复

1. **SSH rsync 失败**: K8s pod 内 SSH 无法访问 .10。改为手动预置数据。
2. **torch.compile grad 断裂**: `torch.tensor(0.0)` 无 grad_fn。改为无分支实现。
3. **5 个损坏 .pt 文件**: 原始 datagen 时 .28 磁盘满导致。删除后 6073 文件正常。

## 评估计划

训练完成后：

1. 选择 val/loss 最低的 checkpoint
2. 在 Novita eval set (10 prompts × 512 tokens) 上测 Acc@0 / throughput
3. 在 ZClawBench (116 agent prompts) 上测 agent 场景
4. 与 Exp15 ckpt67 (标准 KL) 对比

## 下一步

- 如果 Aurora loss 优于标准 KL → 进入 Phase 2（在线闭环）
- 如果持平或更差 → 调参（λ_discard, top_k）或尝试静态 mask（Phase 1.5）
