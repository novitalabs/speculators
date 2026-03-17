# Eagle3 Online Training Design: Datagen-Training Pipeline

## 问题

当前 Eagle3 训练是完全离线的三步串行 pipeline：

```
datagen (数小时) → vocab mapping → training (约1小时)
```

- 5000 样本 datagen 需 **6.4 小时**（TP=2，仅用 2/8 GPU）
- 52K 全量需 **67 小时**（2.8天）
- 每样本 ~262MB，5000 样本 = 1.3TB 磁盘
- datagen 期间其他 GPU 完全空闲

## 目标

**Datagen 和 Training 并行执行**，边生成边训练，消除串行等待。

支持两种部署模式：
1. **单节点**: 2 GPU datagen + 6 GPU training（同一台机器）
2. **多节点**: N 台 datagen 节点 + M 台 training 节点（通过网络传输 .pt 文件）

## Architecture

### 多节点架构（推荐）

```
                    ┌─────────────────────┐
                    │   Controller Node   │
                    │  (scripts/online_   │
                    │   controller.py)    │
                    └──────┬──────────────┘
                           │ SSH / K8s API
              ┌────────────┼────────────────┐
              │            │                │
     ┌────────▼───┐  ┌────▼────────┐  ┌────▼────────┐
     │ Datagen .21│  │ Datagen .XX │  │ Training .18│
     │ 8 GPU TP=2 │  │ 8 GPU TP=2 │  │ 8 GPU FSDP  │
     │ 4 vLLM inst│  │ 4 vLLM inst│  │             │
     │            │  │            │  │             │
     │ writes .pt │  │ writes .pt │  │ reads .pt   │
     │ to local   │  │ to local   │  │ from local  │
     └─────┬──────┘  └─────┬──────┘  └──────▲──────┘
           │               │                 │
           └───── rsync/scp ─────────────────┘
                  (continuous sync)
```

### 数据流

```
1. Datagen nodes: 预处理 → vLLM prefill → capture hidden states → 写 .pt 到本地
2. Sync agent:    rsync 新 .pt 文件到 training node
3. Manifest:      training node 上的 manifest.json 记录已就绪的文件
4. Training:      每 epoch 结束重读 manifest，吸收新文件，重建 DataLoader
```

### 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 跨节点通信 | 文件系统 (.pt + rsync) | 简单可靠，无需共享存储 |
| Vocab mapping | 跳过（用 full target vocab） | 消除 datagen→training 的串行依赖 |
| 数据发现 | manifest.json | 比 os.listdir() 更高效且原子 |
| Epoch 策略 | 每 epoch 结束吸收新文件 | 避免 mid-epoch 一致性问题 |
| Val set | 首次初始化时固定 | 保证跨 epoch 可比 |

## 组件设计

### 1. Manifest Protocol (`src/speculators/train/manifest.py`)

```python
# manifest.json 格式
{
    "status": "generating" | "complete" | "error",
    "files": [
        {"idx": 0, "path": "gen/data_0.pt", "length": 4096},
        {"idx": 1, "path": "gen/data_1.pt", "length": 8192},
        ...
    ],
    "updated_at": "2026-03-10T12:00:00"
}
```

- Datagen/sync agent **原子写入**（写 .tmp → os.rename）
- Training **轮询读取**（每 epoch 间隔或每 30s）
- 每个 .pt 文件附带 length 信息，用于 bin-packing sampler

### 2. Datagen 改造 (`scripts/data_generation_offline.py`)

最小改动：添加 `--manifest-path` 参数

```python
# 在 batch 循环的 save 之后添加 ~10 行：
if args.manifest_path:
    completed_files.append({"idx": idx, "path": f"data_{idx}.pt", "length": seq_len})
    Manifest.write(args.manifest_path, completed_files, "generating")

# 全部完成后：
if args.manifest_path:
    Manifest.write(args.manifest_path, completed_files, "complete")
```

不传 `--manifest-path` 则行为完全不变（向后兼容）。

### 3. Sync Agent (`scripts/sync_datagen.py`)

运行在 controller 或 training 节点上，持续 rsync 新文件：

```python
while True:
    for datagen_node in datagen_nodes:
        # rsync 新 .pt 文件到本地 training data dir
        rsync -az {node}:/data/output/gen/ /data/output/gen/

    # 扫描本地 gen/ 目录，更新 manifest.json
    update_manifest(gen_dir, manifest_path)

    if all_datagen_complete():
        write_manifest_complete()
        break

    sleep(poll_interval)  # 30s
```

或者更简单：用 `inotifywait` / cron + rsync，无需写代码。

### 4. Streaming Training (`scripts/train_streaming.py`)

基于 `scripts/train.py` 的 streaming 版本，核心变化：

```python
def main():
    # ... 模型初始化同 train.py ...

    manifest_path = args.manifest_path
    val_files = None
    val_file_set = set()
    epoch = 0

    while True:
        # 1. 读 manifest 获取当前可用文件
        manifest = Manifest.read(manifest_path)
        all_files = [os.path.join(args.data_path, f["path"]) for f in manifest["files"]]
        all_lengths = [f["length"] for f in manifest["files"]]

        # 2. 首次：固定 val set；后续：只增加 train files
        if val_files is None:
            train_files, val_files = split(all_files, 0.9)
            val_file_set = set(val_files)
            val_loader = setup_dataloader(val_files, ...)
        else:
            train_files = [f for f in all_files if f not in val_file_set]

        # 3. 重建 train DataLoader（新文件 + 重新 bin-pack）
        train_loader = setup_dataloader(train_files, ...)
        trainer.train_loader = train_loader

        # 4. 训练一个 epoch
        trainer.train_epoch(epoch)
        trainer.val_epoch(epoch)
        trainer.save_checkpoint(epoch)
        epoch += 1

        # 5. 检查是否结束
        if manifest["status"] == "complete":
            final_epochs -= 1
            if final_epochs <= 0:
                break
```

**Scheduler 处理**: 使用 `--scheduler-type none` 或固定 lr，因为 total_steps 随数据量增长。

### 5. Orchestrator (`scripts/train_online.py`)

#### 单节点模式
```bash
python scripts/train_online.py \
    --mode single-node \
    --datagen-gpus 0,1 \
    --train-gpus 2,3,4,5,6,7 \
    --target-model-path /data/models/Qwen3-32B \
    --train-data-path /data/novita/novita_v2_sharegpt.jsonl \
    --output-dir /data/output/qwen3_online \
    --min-samples 500 \
    --final-epochs 3
```

#### 多节点模式
```bash
python scripts/train_online.py \
    --mode multi-node \
    --datagen-nodes host-10-83-115-21 \
    --train-node host-10-83-115-18 \
    --target-model-path /data/models/Qwen3-32B \
    --train-data-path /data/novita/novita_v2_sharegpt.jsonl \
    --output-dir /data/output/qwen3_online \
    --min-samples 500 \
    --final-epochs 3 \
    --sync-interval 30
```

多节点 orchestrator 流程：
1. SSH 到各 datagen 节点启动 datagen 进程（K8s pod 或直接 SSH）
2. 启动 sync agent 持续 rsync .pt 文件到 training 节点
3. 等待 min_samples 就绪
4. SSH 到 training 节点启动 torchrun streaming training
5. 监控所有进程，处理错误和优雅退出

## 磁盘空间管理

当数据量达到千万级别时，全量 hidden states 存储不可行（1000万样本 × 262MB ≈ 2.5PB）。需要**环形缓冲区**策略，控制磁盘上 .pt 文件总量。

### Ring Buffer 设计

```
datagen 不断生成 → [  buffer (max 2TB)  ] → training 不断消费
                    ↑ 新文件写入           ↑ 已训练文件删除
```

核心参数：
- `--buffer-max-size`: 磁盘缓冲区大小上限（默认 2TB）
- `--buffer-max-files`: 最大 .pt 文件数（默认 10000）
- `--buffer-cleanup-policy`: 清理策略 — `oldest_trained`（删除已被训练过的最老文件）

### 工作流程

```python
# Datagen 端：写入前检查空间
def before_save(buffer_dir, max_size):
    current_size = get_dir_size(buffer_dir)
    if current_size >= max_size:
        # 等待 training 消费并清理，或阻塞
        wait_for_space(buffer_dir, max_size)

# Training 端：epoch 结束后标记已消费文件
def after_epoch(trained_files, manifest):
    manifest.mark_trained(trained_files)

# Cleanup agent：删除已被训练 N 次的文件
def cleanup(manifest, min_train_count=2):
    for f in manifest.files:
        if f.train_count >= min_train_count:
            os.remove(f.path)
            manifest.remove(f)
```

### Manifest 扩展

```json
{
    "status": "generating",
    "buffer_size_bytes": 1234567890,
    "buffer_max_bytes": 2199023255552,
    "files": [
        {
            "idx": 0,
            "path": "gen/data_0.pt",
            "length": 4096,
            "size_bytes": 262000000,
            "train_count": 0,
            "created_at": "2026-03-10T12:00:00"
        }
    ]
}
```

`train_count` 记录每个文件被训练过几次。当 `train_count >= min_train_count` 时可安全删除。

### 空间预算参考

| buffer_max_size | 可容纳文件数 | 说明 |
|-----------------|-------------|------|
| 500 GB | ~1900 | 适合快速验证 |
| 2 TB | ~7600 | 单节点推荐 |
| 5 TB | ~19000 | 大规模训练 |

### Cleanup ↔ Training 协调

**关键教训 (Exp 13 crash)**: cleanup 后台进程和 training 之间存在 race condition。
cleanup 可能在 DataLoader 正在使用文件时将其删除，导致训练崩溃。

**协调机制**:

```
Training:   [==== Epoch N ====] unlock [==== Epoch N+1 ====] unlock
                 locked                      locked
Cleanup:    skip  skip  skip   DELETE   skip  skip  skip   DELETE
```

1. **Epoch lock**: training 在 epoch 开始时写 `<data_dir>/.epoch_in_progress`，
   epoch 结束且 `increment_train_count` 完成后删除。Cleanup 看到锁文件就跳过本轮。

2. **删除速率限制** (`--max-delete-per-cycle`): 每个 cleanup 周期最多删 N 个文件（默认 5000）。
   即使意外绕过锁，也不会一次性清空所有数据。

3. **最小保留量** (`--min-retain-count`): manifest 中始终保留至少 N 个文件（默认 1000）。
   即使所有文件都 train_count 达标，也不会删到 0。

4. **DataLoader 端容错** (`data.py`): `__getitem__` 遇到 `FileNotFoundError` 时尝试最多 5 个替代文件。
   这是最后一道防线，不应依赖它。

### 对训练质量的影响

- Ring buffer 意味着每个样本只被训练有限次（不像 offline 可以 10 epoch 反复训练同一批数据）
- 但数据量从 5K → 5000万，数据多样性远大于 epoch 重复的收益
- 对于千万级数据，1-2 epoch 通常足够收敛

## 扩展性

### 多 Datagen 节点并行

每个 datagen 节点处理数据的一个 shard：

```
Node .21: samples 0-2499     (shard 0/2)
Node .XX: samples 2500-4999  (shard 1/2)
```

通过 `--shard-id` 和 `--num-shards` 参数：
```bash
# Node .21
python scripts/data_generation_offline.py \
    --shard-id 0 --num-shards 2 \
    --manifest-path /data/output/gen/manifest_shard0.json ...

# Node .XX
python scripts/data_generation_offline.py \
    --shard-id 1 --num-shards 2 \
    --manifest-path /data/output/gen/manifest_shard1.json ...
```

Sync agent 合并多个 shard manifest 到 training 节点的统一 manifest。

### 多 Datagen 实例 / 单节点

每台 8 GPU 节点可跑 4 个 TP=2 vLLM 实例，理论 4x 吞吐：

```
GPU 0,1 → vLLM instance 0 (shard 0/4)
GPU 2,3 → vLLM instance 1 (shard 1/4)
GPU 4,5 → vLLM instance 2 (shard 2/4)
GPU 6,7 → vLLM instance 3 (shard 3/4)
```

## 预期效果

### 时间对比

| 场景 | 离线 (当前) | 在线单节点 | 在线多节点 (2 datagen) |
|------|------------|-----------|----------------------|
| 5000 samples | 6.4h + 1h = **7.4h** | ~**6.4h** | ~**3.2h** |
| 10000 samples | 12.8h + 1h = **13.8h** | ~**12.8h** | ~**6.4h** |
| 52K (全量) | 67h + 2h = **69h** | ~**67h** | ~**17h** (4 nodes) |

> 在线模式主要收益：training 在 datagen 期间完成，省去 training 时间。
> 多节点 datagen 并行是时间的主要缩短来源。

### 磁盘对比

在线模式不直接节省磁盘（仍需存 .pt 文件），但可以：
- 训练完成后立即删除已消费的 .pt 文件（可选的清理策略）
- Datagen 节点只需存放本 shard 的 .pt，sync 到 training 节点后可清理

## 验证计划

1. **Phase 1**: 在 .21 上部署 datagen，用 100 samples 测试 manifest 写入
2. **Phase 2**: 在 .18 上部署 streaming training，验证从 manifest 读取文件并训练
3. **Phase 3**: rsync .pt 从 .21 → .18，验证端到端流程
4. **Phase 4**: 5000 samples 完整实验，对比 offline 训练的 loss/acc

## 实现文件清单

| 文件 | 动作 | 行数 | 说明 |
|------|------|------|------|
| `src/speculators/train/manifest.py` | NEW | ~60 | Manifest 读写 |
| `scripts/train_streaming.py` | NEW | ~200 | Streaming training（基于 train.py） |
| `scripts/train_online.py` | NEW | ~150 | Orchestrator |
| `scripts/sync_datagen.sh` | NEW | ~40 | rsync + manifest 更新脚本 |
| `scripts/data_generation_offline.py` | MODIFY | +15 | 添加 --manifest-path |
| `scripts/buffer_cleanup.py` | NEW | ~50 | Ring buffer 清理 |
| `k8s/k8s-datagen-*.yaml` | NEW | - | Datagen pod specs |
| `k8s/k8s-train-streaming-*.yaml` | NEW | - | Training pod specs |

## 节点规划

| 节点 | 角色 | GPU | 说明 |
|------|------|-----|------|
| .18 | Training | 8x H200 | FSDP training，已有 Qwen3-32B |
| .21 | Datagen | 8x H200 | vLLM 推理，已有 Qwen3-32B |
| .14 | Datagen (备用) | 8x H200 | 当前跑 MiniMax 实验 |
| .23 | Datagen (备用) | 8x H200 | 之前跑 Qwen3 baseline |
