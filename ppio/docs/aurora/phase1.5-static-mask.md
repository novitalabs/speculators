# Phase 1.5: Static Mask Aurora Loss Training

## 目标

验证**静态预计算 mask**（来自 Exp15 ckpt67 参考模型）是否优于 Phase 1 的**动态 mask**（每步实时计算）。静态 mask 更接近 Aurora 论文中 mask 来自推理 trace 的设定。

## 背景

| | Phase 1 (Exp16) | Phase 1.5 |
|---|---|---|
| **Mask 来源** | 当前模型实时 argmax 比较 | 预训练参考模型 (Exp15 ckpt67) |
| **Mask 变化** | 随训练改变（每 epoch 不同） | 固定不变（整个训练过程一致） |
| **更接近论文** | 否（在线 self-play） | 是（离线 inference trace） |

## 设计决策

### 独立 mask 文件

- 不修改原始 `.pt` 数据文件，单独保存 `mask_<idx>.pt`
- 每个 mask 文件包含 3 个 key：`accepted_mask_0/1/2`（对应 ttt_step 0/1/2）
- 与现有 collate_fn 兼容（concat → slice/pad → unsqueeze）

### Post-shift 坐标系

- Mask 存储为 `[seq_len-1]` 形状，与 `shift_batch` 后的 `loss_mask` 坐标对齐
- 在 `compute_metrics` 中 `align_for_step` 后再 slice 到对齐长度

### 每步独立 key

- `accepted_mask_0`、`accepted_mask_1`、`accepted_mask_2` 分别存储
- 直接通过 batch dict → model kwargs → forward() 传递，无需修改 collate_fn

## 代码改动

### 新文件

| 文件 | 说明 |
|------|------|
| `scripts/precompute_aurora_masks.py` | 用参考模型预计算所有数据文件的 accept mask |
| `k8s/run_minimax_m2.5_precompute_masks.sh` | 单 GPU 预计算启动脚本 |
| `k8s/run_minimax_m2.5_aurora_static_train.sh` | 8 GPU 静态 mask 训练启动脚本 |

### 修改文件

| 文件 | 改动 | 说明 |
|------|------|------|
| `src/speculators/train/data.py` | `Eagle3SampleFileDataset` 增加 `mask_dir` 参数 | `__getitem__` 在 shift_batch 后加载 mask |
| `src/speculators/models/eagle3/core.py` | `aurora_loss_function` 增加 `static_accepted_mask` | 有静态 mask 时跳过动态计算 |
| `src/speculators/models/eagle3/core.py` | `compute_metrics` 增加 `static_accepted_mask` | slice 到对齐长度后传给 loss |
| `src/speculators/models/eagle3/core.py` | `forward()` 从 kwargs 提取 `accepted_mask_{0,1,2}` | 按 ttt_step 传给 compute_metrics |
| `scripts/train.py` | 新增 `--aurora-static-mask`, `--mask-dir` | 传递 mask_dir 给 dataset |
| `scripts/train_streaming.py` | 同上 | 传递 mask_dir 给 dataset |

### 预计算脚本逻辑 (`precompute_aurora_masks.py`)

1. 加载 draft model + checkpoint (`from_training_args` + `model.pt`)，设为 `eval()` 模式
2. 对每个 `data_<idx>.pt`：
   - `standardize_data_v1` → `shift_batch` → 加 batch dim → GPU
   - 计算 `targets = verifier_lm_head(verifier_norm(verifier_last_hidden_states))`
   - 对每个 ttt_step (0, 1, 2)：
     - 复刻 `forward()` 中的 embed → fc → layers → norm → lm_head 得到 logits
     - `align_for_step` 后比较 `argmax(logits) == argmax(targets)` 得到 mask
     - Pad 到 `seq_len-1` 全长
   - 保存 `mask_<idx>.pt`

### Loss function 改动

```python
# aurora_loss_function 新增参数
def aurora_loss_function(..., static_accepted_mask=None):
    if static_accepted_mask is not None:
        accepted = static_accepted_mask.to(torch.bool)
    else:
        # 原有动态计算
        draft_argmax = logits.detach().argmax(dim=-1)
        target_argmax = targets.detach().argmax(dim=-1)
        accepted = draft_argmax == target_argmax
    # 后续代码不变
```

### 后向兼容

- 不传 `--aurora-static-mask` 时行为完全不变
- `mask_dir=None` 时 dataset 不加载任何 mask
- `static_accepted_mask=None` 时 loss function 走动态路径

## 实验: Exp16.5

### 预计算配置

| 参数 | 值 |
|------|------|
| 参考模型 | Exp15 ckpt67 (标准 KL, 63.2% Acc@0) |
| 数据 | Exp15/16 共享的 6073 .pt files |
| GPU | 单卡 (无 FSDP) |
| 输出 | `masks/mask_*.pt` |

### 训练配置

| 参数 | 值 |
|------|------|
| 节点 | .17 (8x H200) |
| 数据 | Exp15 数据 + 预计算 mask |
| 架构 | Aurora-like (24 heads, 8192 intermediate, 32K vocab) |
| Loss | Aurora accept/discard (`λ=0.1, top_k=10`) + **static mask** |
| 训练 | 8 GPU FSDP, lr=3e-5, 70 epochs |
| 对比基线 | Exp15 (标准 KL), Exp16 (动态 Aurora) |

### 监控指标

| 指标 | 说明 | 预期 |
|------|------|------|
| `aurora_accept_ratio_0` | 接受比例 | **应保持常量**（与 Exp16 不同） |
| `val/loss_epoch` | 验证 loss | 对比 Exp16 |
| `val/full_acc_0_epoch` | 验证准确率 | 对比 Exp16 |

### 验证计划

1. **预计算 sanity check**: 前 5 个文件 — 预计算 mask 与动态 mask 精确匹配
2. **回归测试**: 不加 `--aurora-static-mask` → 与 Exp16 完全一致
3. **静态 mask 验证**: `aurora_accept_ratio` 应跨 epoch 保持不变
4. **A/B 对比**: Static mask vs Dynamic mask (Exp16) vs Standard KL (Exp15)

## 训练记录

**状态**: ✅ 完成 — 70 epochs, ~5h (10:45–15:45 UTC, 2026-03-27)

### 预计算

- 6073 mask 文件，单 GPU 约 75 分钟完成
- 平均 accept_ratio: step0 ~0.25, step1 ~0.13, step2 ~0.07（参考模型 Exp15 ckpt67）

### 训练结果

| Epoch | val/loss | accept_ratio_0 | full_acc_0 | Notes |
|-------|---------|----------------|------------|-------|
| 0 | 7.968 | 0.683 | 0.365 | 初始 |
| 5 | 4.573 | 0.685 | 0.528 | 快速收敛 |
| 9 | 4.254 | 0.684 | 0.552 | |
| **18** | **4.227** | 0.685 | 0.567 | **最佳 val/loss** |
| 23 | 4.300 | 0.692 | 0.575 | |
| 32 | 4.349 | 0.692 | 0.579 | |
| 43 | 4.405 | 0.694 | 0.582 | |
| **53** | 4.404 | 0.695 | **0.587** | **最佳 acc** |
| 60 | 4.536 | 0.683 | 0.573 | 过拟合 |
| 69 | 4.578 | 0.683 | 0.572 | 最终 epoch |

### 关键发现

1. **静态 mask 验证通过** — `accept_ratio_0` 全程恒定 ~0.685（Exp16 动态 mask 从 0.12→0.58）
2. **最佳 acc 58.7%** (epoch 53) — 略优于 Exp16 动态 mask（58.0% at ckpt53）
3. **最佳 val/loss 出现早** — epoch 18 后 loss 持续上升，过拟合比 Exp16 更明显
4. **loss scale 不同** — 静态 mask 的 val/loss ~4.2 vs Exp16 ~1.07（因 accept_ratio 不同导致 loss 分布不同，不可直接比较）
5. **accept_ratio 更高** — 静态 0.685 vs 动态最终 0.58（参考模型 Exp15 ckpt67 更准确）

### A/B 对比（val/full_acc_0 最佳）

| 实验 | 最佳 Acc@0 | 最佳 Epoch | Mask 类型 |
|------|-----------|-----------|----------|
| Exp15 (标准 KL) | 63.2% | ckpt67 | 无 |
| Exp16 (Aurora 动态) | 58.0% | ckpt53 | 动态 |
| **Exp16.5 (Aurora 静态)** | **58.7%** | **ckpt53** | 静态 |

**结论**: Aurora loss（动态/静态）在 val acc 上均不如标准 KL (Exp15)。需要 eval throughput 确认是否有实际加速效果。

## 评估

**状态**: ✅ 完成

### ZClawBench Eval (2026-03-27, .18, TP=4, 116 agent prompts × 512 tokens)

| Model | Tokens/s | Speedup | Acc Len | Acc@0 | Acc@1 | Acc@2 |
|-------|----------|---------|---------|-------|-------|-------|
| baseline (no spec) | 215.3 | 1.00x | — | — | — | — |
| Aurora-Spec-M2.1 | 198.7 | 0.92x | 1.465 | 29.8% | 11.8% | 5.0% |
| Exp16 ckpt53 (dynamic mask) | 569.1 | 0.59x* | 1.431 | 31.7% | 8.8% | 2.6% |
| **Exp16.5 ckpt18 (static mask)** | **205.3** | **0.95x** | **1.432** | **30.9%** | **9.2%** | **3.1%** |
| Exp15 ckpt67 (standard KL) | 225.8 | **1.05x** | 1.589 | **39.6%** | **14.4%** | **5.1%** |

*Exp16 baseline 不同 (960 tok/s)，speedup 不可直接比较

### 结论

**Aurora loss 无论动态还是静态 mask，都不如标准 KL**：

| | Exp15 (KL) | Exp16 (动态) | Exp16.5 (静态) |
|---|-----------|------------|--------------|
| Acc@0 | **39.6%** | 31.7% | 30.9% |
| Acc@1 | **14.4%** | 8.8% | 9.2% |
| Acceptance length | **1.589** | 1.431 | 1.432 |
| 相对 baseline | **1.05x** | — | 0.95x |

- 静态 mask (30.9%) ≈ 动态 mask (31.7%)，两者差异不大
- 两种 Aurora loss 都比标准 KL 低 ~9pp Acc@0
- **Aurora accept/discard loss 实验全线结束，标准 KL loss 仍为最佳训练目标**
