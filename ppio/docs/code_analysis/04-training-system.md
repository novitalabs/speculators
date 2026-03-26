# 训练系统

## 训练系统架构

```
scripts/train.py (入口)
  │
  ├── parse_args()           # 解析命令行参数
  ├── set_seed()             # 设置随机种子
  ├── maybe_setup_distributed()  # 初始化分布式训练
  ├── Model.from_training_args() # 创建模型
  ├── setup_dataloader()     # 创建数据加载器
  │
  └── Trainer.run_training() # 主训练循环
        ├── setup_trainer()   # 恢复训练状态
        ├── setup_model()     # 模型准备 (FSDP)
        ├── setup_optimizer()  # 优化器 + 调度器
        │
        └── for epoch:
              ├── train_epoch()     # 训练一个 epoch
              ├── val_epoch()       # 验证一个 epoch
              └── save_checkpoint() # 保存检查点
```

## 1. Trainer (`train/trainer.py`)

### TrainerConfig

```python
class TrainerConfig(NamedTuple):
    lr: float                          # 学习率
    num_epochs: int                    # 训练 epoch 数
    save_path: str                     # 检查点保存路径
    resume_from_checkpoint: bool       # 是否从检查点恢复
    is_distributed: bool               # 是否分布式训练
    local_rank: int                    # 本地 GPU 编号
    train_call_kwargs: dict            # 模型 forward 额外参数
    val_call_kwargs: dict              # 验证 forward 额外参数
    scheduler_type: str = "none"       # 调度器类型: linear/cosine/none
    scheduler_warmup_steps: int        # 预热步数 (默认总步数的 1%)
    scheduler_total_steps: int         # 总步数
    scheduler_num_cosine_cycles: float # 余弦调度周期数
```

### 训练循环核心

```python
class Trainer:
    def run_training(self):
        self.setup_trainer()     # 恢复或初始化
        self.setup_model()       # FSDP 封装
        self.setup_optimizer()   # AdamW + Scheduler

        for epoch in range(start_epoch, num_epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.val_epoch(epoch)
            self.save_checkpoint(epoch)

    def train_epoch(self, epoch):
        for batch in tqdm(train_loader):
            # 前向传播
            loss, metrics = self.model(**batch, **train_kwargs)

            # 反向传播
            loss.backward()

            # 梯度裁剪 (norm=1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 优化器步进
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # 汇总指标 (分布式 all_reduce)
            aggregated = all_reduce_metrics(metrics)
            logger.log(aggregated)
```

### 优化器配置

- **优化器**: AdamW
- **梯度裁剪**: max_norm=1.0
- **学习率调度器**:
  - `"linear"`: 线性预热 + 线性衰减
  - `"cosine"`: 线性预热 + 余弦退火
  - `"none"`: 恒定学习率

## 2. 数据加载 (`train/data.py`)

### Eagle3SampleFileDataset

每个训练样本是一个 `.pt` 文件，包含一个序列的数据：

```python
class Eagle3SampleFileDataset(Dataset):
    def __init__(self, datapath=None, file_list=None, transforms=None):
        # 发现所有 .pt 文件
        # 计算近似样本长度 (从 sample_lengths.json 或文件大小)

    def __getitem__(self, idx):
        data = torch.load(file, mmap=True)  # 内存映射加载

        # 标准化数据格式
        data = standardize_data(data)

        # 添加 position_ids
        data["position_ids"] = torch.arange(seq_len)

        # 可选: 应用噪声增强
        if self.transforms:
            data = self.transforms(data)

        # Teacher Forcing 对齐 (shift_batch)
        data = shift_batch(data)
        # input_ids 向前移 1 位, hidden_states 向后移 1 位
        # 对齐 (input_token_i, hidden_state_{i-1}) 对

        return data
```

### 数据格式

每个 `.pt` 文件包含：

| 字段 | 形状 | 说明 |
|------|------|------|
| `input_ids` | `[seq_len]` | Token ID 序列 |
| `hidden_states` | `[num_layers, seq_len, hidden_size]` | 各层隐藏状态 |
| `loss_mask` | `[seq_len]` | 损失掩码 (1=assistant token) |

### shift_batch 对齐

```
原始序列:  t0   t1   t2   t3   t4
hidden:    h0   h1   h2   h3   h4

对齐后:
input_ids:     t1   t2   t3   t4     (去掉第一个)
hidden_states: h0   h1   h2   h3     (去掉最后一个)

含义: 给定 h_{i-1}，预测 t_i
```

### 自定义 Collate 函数

```python
def create_collate_fn(max_len):
    def collate_fn(batch):
        # 将多个样本沿序列维度拼接
        # 截断/填充到 max_len
        # 添加 batch 维度: [1, max_len, ...]
        # 记录各样本实际长度用于文档边界掩码
```

这种设计允许多个短序列被打包到一个批次中，最大化 GPU 利用率。

## 3. 分布式批次采样 (`train/distributed_batch_sampler.py`)

### MultipackDistributedBatchSamplerV2

基于 **LPT (Longest Processing Time)** 算法的高效分布式批次打包采样器。

#### 核心算法

```
1. 按样本长度降序排序
2. 初始化 num_replicas 个 bin (每个 GPU 一个)
3. 对每个样本，分配到最空闲的 bin (如果装得下)
4. 使用二分搜索找到最大可打包的批次大小
5. 重复直到数据集耗尽
```

#### 关键参数

- `batch_max_length`: 每个 GPU 每批次的最大 token 数
- `truncate_long_samples`: 截断过长样本 (True) 或丢弃 (False)
- `seed`: 确保每个 epoch 的打乱方式可复现

## 4. 检查点管理 (`train/checkpointer.py`)

### 存储结构

```
{save_path}/
├── 0/                          # Epoch 0
│   ├── model.safetensors       # 模型权重
│   ├── optimizer_state_dict.pt # 优化器状态
│   └── scheduler_state_dict.pt # 调度器状态 (可选)
├── 1/                          # Epoch 1
│   └── ...
└── ...
```

### 两种实现

| 特性 | SingleGPUCheckpointer | DistributedCheckpointer |
|------|----------------------|------------------------|
| 加载方式 | safetensors 直接加载 | FSDP state_dict API |
| 保存方式 | model.save_pretrained() | rank 0 收集并保存 |
| 同步 | 无需 | distributed barrier |
| dtype 转换 | 手动转换 | 通过 StateDictOptions |
| strict 加载 | False (跳过验证器权重) | 通过 FSDP 管理 |

### dtype 管理

- 训练时: `float32` (混合精度的参数副本)
- 保存时: `bfloat16` (节省磁盘空间)
- 加载时: 根据需要转换

## 5. 日志系统 (`train/logger.py`)

### 支持的后端

| 后端 | 处理器类 | 说明 |
|------|---------|------|
| TensorBoard | TensorBoardHandler | 标量指标 + 文本日志 |
| Weights & Biases | WandbHandler | 实验跟踪平台 |
| Trackio | TrackioHandler | W&B 兼容的替代方案 |

### 过滤器链

```
日志消息 → IsMappingFilter (只传递 dict 消息)
         → IsRank0Filter (只传递 rank 0 的消息)
         → FormatDictFilter (格式化 dict 为可读文本)
         → Handler (TensorBoard / W&B / Trackio)
```

### 使用方式

```python
# 设置
setup_root_logger(level="INFO")
setup_metric_logger(loggers=["tensorboard", "wandb"], run_name="exp1", output_dir="./logs")

# 记录指标
logger = logging.getLogger("speculators.metrics")
logger.info({"loss": 0.5, "accuracy": 0.9, "step": 100})
```

## 6. 噪声增强 (`train/noise_transforms.py`)

为隐藏状态添加噪声的数据增强策略：

| 增强方法 | 公式 | 说明 |
|---------|------|------|
| AddGaussianNoise | `tensor + N(0, σ)` | 高斯噪声 |
| AddUniformNoise | `tensor + U(-σ, σ)` | 均匀噪声 |

默认仅对 `hidden_states` 张量应用，`σ = 0.05`。

## 7. 词表映射 (`train/vocab_mapping.py`)

### 构建流程

```
1. save_token_frequency_distribution()
   - 遍历数据集，统计 loss_mask=1 (assistant token) 的频率
   - 保存为 token_freq.pt

2. build_vocab_mappings_from_distribution()
   - 按频率降序排序
   - 取前 draft_vocab_size 个 token
   - 构建 draft_to_target 和 target_to_draft 映射
```

### 映射示例

假设 target vocab = [0..127999], draft vocab = 32000:

```
高频 token: [0, 1, 5, 10, 15, 100, ...]  (按频率排序后取前32K个)
排序后:     [0, 1, 5, 10, 15, 100, ...]  (按 ID 升序)

draft_to_target[0] = 0   (偏移量)
draft_to_target[1] = 0
draft_to_target[2] = 3   (5 - 2 = 3)
...

target_to_draft[0] = True
target_to_draft[1] = True
target_to_draft[2] = False
...
target_to_draft[5] = True
```

## 8. 分布式训练工具 (`train/utils.py`)

### FSDP 设置

```python
def apply_fully_sharded(model):
    """将模型层包装为 FSDP"""
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,    # 参数: bfloat16
        reduce_dtype=torch.float32,     # 梯度归约: float32
    )

    # 逐层封装
    for layer in model.layers:
        fully_shard(layer, mp_policy=mp_policy)

    # 整体封装
    fully_shard(model, mp_policy=mp_policy)
    return model
```

### 分布式初始化

```python
def maybe_setup_distributed():
    """如果通过 torchrun 启动，初始化分布式训练"""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        return (0, 1, 0, False)  # 单 GPU

    torch.accelerator.set_device_index(local_rank)
    dist.init_process_group(backend=...)
    return (local_rank, world_size, rank, True)
```
