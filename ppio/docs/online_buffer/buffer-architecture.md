# Buffer Architecture: Design & Evolution

Online Eagle3 训练的数据缓冲系统设计文档。记录 ring buffer 从初始设计到多次迭代的完整演进。

## Overview

Online training pipeline 中，datagen 节点持续生成 `.pt` hidden state 文件，training 节点持续消费。
Buffer 系统负责协调两端，确保：
1. Training 节点有足够数据训练
2. 磁盘空间不会无限增长
3. 已训练数据被及时清理
4. 清理不影响正在进行的训练

```
Datagen (.17)                    Training (.18)
┌──────────────┐   rsync    ┌──────────────────────────────┐
│ vLLM prefill │──────────→│  Ring Buffer (1TB cap)        │
│ → .pt files  │           │  ┌─────┐ ┌─────┐ ┌─────┐    │
│ → manifest   │           │  │.pt 1│ │.pt 2│ │.pt N│    │
└──────────────┘           │  └──┬──┘ └──┬──┘ └──┬──┘    │
                           │     ↓       ↓       ↓        │
                           │  train_streaming.py (8 GPU)   │
                           │     ↓                         │
                           │  buffer_cleanup.py (删除已训练)│
                           └──────────────────────────────┘
```

## Components

| 组件 | 文件 | 角色 |
|------|------|------|
| Manifest | `src/speculators/train/manifest.py` | .pt 文件注册表，记录 idx、path、train_count |
| Sync | `scripts/sync_datagen.sh` | rsync .pt 从 datagen → training，更新 manifest |
| Training | `scripts/train_streaming.py` | 读 manifest，训练，更新 train_count |
| Cleanup | `scripts/buffer_cleanup.py` | 删除已训练文件，维持磁盘上限 |

## Evolution

### V1: Basic Ring Buffer (Exp 12)

**设计**: 最简 cleanup — 删除 `train_count >= 2` 的文件，无任何安全限制。

```
cleanup_once():
    for file in manifest.files:
        if file.train_count >= 2:
            delete(file)
    rewrite_manifest(remaining_files)
```

**问题**: 无。Exp 12 数据量较小（38K files），cleanup 和 training 时序上没有冲突。

---

### V2: Epoch Lock + Safety Limits (Exp 13 crash fix)

**触发事件**: Exp 13 Epoch 1 crash — cleanup 一次性删除全部 11045 个已训练文件。

**Root cause**:
```
Epoch 0 结束 → increment_train_count 标记 11045 文件为 train_count=2
    ↓ < 60s
buffer_cleanup 轮询 → 看到 11045 文件可删 → 全部删除
    ↓
DataLoader 尝试加载 Epoch 1 数据 → 全部 FileNotFoundError → crash
```

**Fix (3 层防护)**:

1. **Epoch lock file** (`.epoch_in_progress`):
   - Training 在 epoch 开始时写锁，epoch 结束 + increment_train_count 完成后删锁
   - Cleanup 看到锁就跳过本轮
   - 消除 cleanup/training 的时序竞争

2. **`--max-delete-per-cycle`** (默认 5000):
   - 每轮 cleanup 最多删 N 个文件
   - 即使锁机制失效，也不会一次清空

3. **`--min-retain-count`** (默认 1000):
   - Manifest 中始终保留至少 N 个文件
   - 即使所有文件 train_count 达标也不删到 0

4. **DataLoader fallback** (`data.py`):
   - `__getitem__` 遇到 FileNotFoundError 时尝试最多 5 个替代文件
   - 最后一道防线

**相关文件改动**:
- `scripts/train_streaming.py`: epoch lock 写入/删除
- `scripts/buffer_cleanup.py`: lock 检查 + max_delete + min_retain
- `src/speculators/train/data.py`: fallback retries 1 → 5

---

### V3: Sync ↔ Cleanup Coordination (Exp 13 restart)

**触发事件**: Exp 13 restart 后发现 cleanup 删除的文件被 rsync 重新同步回来，磁盘永远不会缩小。

**Root cause**:
```
Cleanup 删除 data_100.pt (train_count=2) on .18
    ↓ 60s 后
sync_datagen.sh rsync 从 .17 把 data_100.pt 同步回 .18（.17 上仍存在）
    ↓
update_manifest() 把 data_100.pt 加回 manifest，train_count=0（全新副本）
    ↓
Cleanup 不会再删它（需要再训练 2 次才能删）
    → 同一批文件无限循环，磁盘永远不缩小
```

**Fix**: Cleanup 写排除列表，Sync 读排除列表。

1. **`.cleanup_exclude` 文件**:
   - `buffer_cleanup.py` 删除文件后，将文件名 append 到 `<data_dir>/.cleanup_exclude`
   - Phase 1（train_count 删除）和 Phase 2（size cap 删除）都写入

2. **Sync 读排除列表**:
   - `sync_datagen.sh` 在 rsync 前检查 `.cleanup_exclude` 是否存在
   - 存在则传 `--exclude-from=.cleanup_exclude` 给 rsync
   - 被 cleanup 删除的文件不会再被同步回来

3. **Sync size gate** (`--max-sync-size-gb`):
   - Sync 在 rsync 前检查本地目录大小
   - 超过阈值则跳过本轮 sync（仍检查 remote completion 状态）
   - 与 cleanup 的 `--max-size-gb` 使用同一个值（BUFFER_MAX_SIZE_GB）

**数据流（修复后）**:
```
.17 datagen (52K files)
    │
    │ rsync --exclude-from=.cleanup_exclude
    │       (跳过已清理文件)
    ↓
.18 local buffer (≤ 1TB)
    │
    ├── train_streaming.py → 训练 → increment_train_count
    │
    └── buffer_cleanup.py → 删除 train_count≥2 的文件
                          → append 到 .cleanup_exclude
                          → 被删文件不再被 rsync 回来
```

**相关文件改动**:
- `scripts/buffer_cleanup.py`: 两个删除阶段都 append 到 `.cleanup_exclude`
- `scripts/sync_datagen.sh`: `--max-sync-size-gb` 参数 + `--exclude-from` 支持
- `k8s/run_minimax_m2.5_novita_full_train_v2.sh`: 传递 `--max-sync-size-gb`

---

### V4 Design: Prioritized Replay Buffer (Proposed)

**动机**: V3 的 `.cleanup_exclude` 永久禁止文件被 re-sync。但如果目标是 10 global epochs
（52K 文件各训练 10 次），文件在被驱逐后 **必须** 能重新同步回来。永久排除模型仅适用于
single-pass training。

**示例**:
```
52K files, 1TB buffer (~20K files), 目标 10 epochs
- Buffer 装入 20K files，每个训练 2x，驱逐 → 需要再拉回来 4 次
- .cleanup_exclude 阻止 re-sync → 文件只训练了 2x 而不是 10x
- 实际 epoch 数 = 2，远低于目标 10
```

#### 核心概念

**1. Global Epoch Tracking**

一个 "global epoch" = 全部 52K 文件都被训练过一次。需要追踪：
- `total_remote_files`: remote 端总文件数（sync 写入 manifest）
- `files_ever_seen`: 训练过的唯一文件路径集合
- Global epoch count = `files_ever_seen / total_remote_files`（向下取整）

**2. 基于优先级的驱逐策略（替代 train_count 阈值）**

不再用 `min_train_count` 作为驱逐触发条件，而是基于优先级：
- **驱逐优先级**: local train_count 最高的文件优先驱逐（它们已经训练够了）
- **目标**: 为训练次数更少的文件腾出空间，优先保证全局覆盖率
- **安全**: 永远不驱逐正在被使用的文件（epoch lock 机制已有）

```
eviction_priority(file):
    if file.in_current_epoch:
        return -inf  # never evict
    return file.local_train_count  # highest count → first to evict
```

**3. Sync 优先级：未见文件 > 低训练次数 > 已达标**

Sync 拉取文件的优先级：
- Priority 1: 从未被训练过的文件（推进 global progress）
- Priority 2: cumulative train_count 最低的文件（需要更多 passes）
- Priority 3: 跳过已达到目标 train_count 的文件
- 需要跨驱逐周期追踪 per-file train_count（不仅是本地值）

**4. 用 Eviction Ledger 替代 .cleanup_exclude**

`.cleanup_exclude` 是永久黑名单，改为 ledger 机制：

```json
// .eviction_ledger (JSON)
{
    "data_100.pt": {"cumulative_train_count": 4, "evicted_at": "2025-01-15T10:30:00"},
    "data_200.pt": {"cumulative_train_count": 6, "evicted_at": "2025-01-15T11:00:00"},
    "data_300.pt": {"cumulative_train_count": 10, "evicted_at": "2025-01-15T11:30:00"}
}
```

- Sync 读 ledger 决定是否 re-sync：count < target → re-sync，count >= target → skip
- 文件重新进入 buffer 时，train_count 从 ledger 值继续计数，而不是从 0 开始
- `.cleanup_exclude` 的功能被 ledger 完全覆盖（ledger 中 count >= target 的等价于 exclude）

**5. Validation 与 Buffer Epoch 解耦**

当前问题：validation 绑定在 buffer epoch 边界（~5K files），但 buffer epoch ≠ global epoch。

改进：
- Validation 改为每 N training steps 触发（可配置，如 `--val-every-steps 500`）
- 不再依赖 `epoch_end` 事件，由 step counter 驱动
- 更稳定的 validation 频率，不受 buffer 大小波动影响

**6. Difficulty-Based Retention（未来）**

基于训练难度的保留策略：
- 训练时记录每个文件的 average loss
- High-loss 文件 = 难样本 → 保留更久，re-sync 优先级更高
- Low-loss 文件 = 简单样本 → 更早驱逐，re-sync 优先级更低
- 可与 curriculum learning 集成

#### 数据流（V4 完整）

```
.17 datagen (52K files, total_remote_files=52000)
    │
    │ sync: 读 .eviction_ledger
    │       priority 1: 未见文件（不在 ledger 中）
    │       priority 2: ledger 中 count < target 的文件
    │       skip: ledger 中 count >= target 的文件
    ↓
.18 local buffer (≤ 1TB)
    │
    ├── train_streaming.py
    │     → 训练 → increment_train_count（local + cumulative）
    │     → 追踪 files_ever_seen → 计算 global epoch
    │     → 每 N steps → validation
    │
    └── buffer_cleanup.py
          → 驱逐 local_train_count 最高的文件（不是阈值触发）
          → 更新 .eviction_ledger（累计 train_count）
          → 不再写 .cleanup_exclude
```

#### Implementation Phases

**Phase 1 — Sliding Window Sync（立即可做）**:
- Sync 不在 datagen complete 后退出（持续运行直到被 kill）
- Sync 写 `total_remote_files` 到 manifest
- Training 追踪 `files_ever_seen`（已训练的唯一文件路径集合）
- 移除 `min_train_count` 作为驱逐触发条件；改为在 size cap 超限时驱逐
  buffer 中最老的文件（为 sync 拉入的新文件腾位置）
- 用 `.eviction_ledger`（train_count tracker）替代 `.cleanup_exclude`（永久 ban）
- Sync 读 ledger 跳过已达标文件，re-sync 需要更多 passes 的文件

**Phase 2 — Global Epoch + Step-Based Validation**:
- Training 计算 global epochs（`files_ever_seen >= total_remote_files` 即完成一个）
- Validation 改为每 N steps 触发，不再绑定 buffer epoch
- Manifest 追踪 global progress 用于监控

**Phase 3 — Difficulty-Based Retention**:
- 在 manifest 中记录 per-file loss
- 基于 loss 的加权驱逐和 re-sync 优先级
- Curriculum learning 集成

---

## Configuration Reference

| 参数 | 组件 | 默认值 | 说明 |
|------|------|--------|------|
| `--min-train-count` | cleanup | 2 | 文件被训练几次后可删 |
| `--max-size-gb` | cleanup | 0 (无限) | .pt 文件总大小上限 |
| `--max-delete-per-cycle` | cleanup | 5000 | 每轮最多删几个文件 |
| `--min-retain-count` | cleanup | 1000 | manifest 最少保留文件数 |
| `--poll-interval` | cleanup/sync | 60/30 | 轮询间隔（秒） |
| `--max-sync-size-gb` | sync | 0 (无限) | 本地目录超此大小跳过 sync |
| `BUFFER_MAX_SIZE_GB` | k8s env | 1024 | cleanup 和 sync 共用的上限 |

## File Layout

```
<data_dir>/
├── manifest.json          # 文件注册表（atomic write）
├── .epoch_in_progress     # Training 写的锁文件（epoch 期间存在）
├── .cleanup_exclude       # [V3] Cleanup 写的排除列表（sync 读取，V4 由 ledger 替代）
├── .eviction_ledger       # [V4] 驱逐记录（JSON，累计 train_count）
├── data_0.pt              # Hidden state 数据文件
├── data_1.pt
├── ...
└── sample_lengths.json    # 样本长度信息（用于 bin-packing）
```

## Known Limitations

1. **`.cleanup_exclude` 只增不减**: 文件会持续增长（每行一个文件名）。
   对于 52K 文件场景，最终约 1-2MB，不构成问题。
   如果未来数据量达到百万级，需要考虑定期压缩（去重 + 清理不存在于 remote 的条目）。

2. **Datagen 端无自动清理**: .17 上的 datagen 产出一直累积，需要手动清理。
   未来可在 datagen 节点也部署 cleanup，在 training 确认消费后删除。

3. **单向排除**: 当前设计只阻止已清理文件被 re-sync。如果需要"重新训练"某个文件
   （例如发现之前训练有 bug），需要手动编辑 `.cleanup_exclude` 移除对应行。
   → **V4 解决**: eviction ledger 替代永久排除，文件可根据 cumulative train_count 被 re-sync。

4. **Multi-epoch 训练受限**: V3 的 `.cleanup_exclude` 永久禁止文件 re-sync，导致每个文件
   最多只能训练 `min_train_count` 次（通常 2 次）。如果目标 epoch 数 > 2，当前架构无法满足。
   → **V4 解决**: eviction ledger + sync 优先级机制支持任意 epoch 数。
