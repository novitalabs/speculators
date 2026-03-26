# 数据生成系统

## 数据生成管道概览

```
原始数据集 (ShareGPT / UltraChat / 自定义)
  │
  ├── 1. load_raw_dataset()        # 加载原始对话数据
  ├── 2. build_eagle3_dataset()    # 预处理 + 分词 + 损失掩码
  │       ├── 对话标准化
  │       ├── Chat Template 应用
  │       ├── Tokenization
  │       └── Assistant Response 检测 + Loss Mask
  │
  ├── 3. VllmHiddenStatesGenerator.generate()  # 提取隐藏状态
  │       ├── vLLM 推理引擎初始化
  │       ├── Prefill 阶段执行
  │       ├── 中间层隐藏状态捕获
  │       └── 结果组装
  │
  └── 4. 保存为 .pt 文件 (每序列一个文件)
          ├── input_ids: [seq_len]
          ├── hidden_states: list[[seq_len, hidden_size]] × num_layers
          └── loss_mask: [seq_len]
```

## 1. 数据预处理 (`data_generation/preprocessing.py`)

### 主入口函数

```python
def load_and_preprocess_dataset(
    target_model_path,    # 目标模型路径 (用于获取 tokenizer)
    train_data_path,      # 训练数据路径
    seq_length=2048,      # 最大序列长度
    max_samples=None,     # 最大样本数
    seed=42,              # 随机种子
    turn_dropout=False,   # 对话轮次随机丢弃
    assistant_pattern=None,  # 自定义 assistant 模式
    ...
):
    # 1. 加载 tokenizer (验证 chat_template 支持)
    tokenizer = AutoTokenizer.from_pretrained(target_model_path)

    # 2. 加载原始数据集
    dataset = load_raw_dataset(train_data_path)

    # 3. 打乱并可选采样
    dataset = dataset.shuffle(seed=seed)
    if max_samples:
        dataset = dataset.select(range(max_samples))

    # 4. 构建 EAGLE3 数据集
    processed = build_eagle3_dataset(dataset, tokenizer, seq_length)

    # 5. 计算词频分布 (用于词表映射)
    save_token_frequency_distribution(processed, output_path)

    return processed, tokenizer
```

### 对话标准化

支持多种对话格式的统一处理：

```python
def _normalize_conversation(conv, turn_dropout=False):
    # 角色名映射:
    #   "human"/"user"/"Human" → "user"
    #   "gpt"/"assistant"/"Assistant" → "assistant"
    #   "system"/"System" → "system"

    # SFT 格式: {"role": "user", "content": "..."}
    # ShareGPT 格式: {"from": "human", "value": "..."}
    # → 统一为: {"role": "user", "content": "..."}

    # Turn Dropout: 随机选取 N 个连续轮次 (数据增强)
```

### Assistant Response 检测

检测 tokenized 文本中 assistant 响应的位置，用于构建 loss_mask：

#### 策略 1: HuggingFace 内置掩码 (首选)

```python
def _supports_assistant_mask(tokenizer):
    """检查 tokenizer 是否支持 HF 的 assistant token mask"""
    # 构造测试对话，应用 chat_template
    # 检查返回的 mask 是否标记了 assistant token
```

#### 策略 2: 正则表达式自动检测 (后备)

```python
def _detect_assistant_pattern(tokenizer):
    """自动检测 assistant 响应的正则模式"""
    # 1. 用多轮测试对话生成模板文本
    # 2. 找到 USER_MSG_2 结束和 assistant 消息开始之间的标记
    # 3. 提取角色标记 (如 "<|assistant|>")
    # 4. 找到跨轮次稳定的后缀
    # 5. 构造正则: role_marker(.*?)suffix
```

#### Loss Mask 构建

```python
def _create_loss_mask_from_offsets(text, offsets, assistant_pattern):
    # 1. 使用正则匹配 assistant 响应的字符范围
    # 2. 通过 offset_mapping 将字符范围映射到 token 索引
    # 3. 标记对应 token 为 1 (可训练)
    # 示例:
    # text: "<user>你好</user><assistant>你好！</assistant>"
    # loss_mask: [0, 0, 0, 0, 1, 1, 1, 0]
    #                        ^^^^^^^^^^^
    #                       assistant 部分
```

### 数据集配置注册 (`data_generation/configs.py`)

```python
DATASET_CONFIGS = {
    "sharegpt": DatasetConfig(
        name="sharegpt",
        hf_path="Aeala/ShareGPT_Vicuna_unfiltered",
        split="train",
    ),
    "ultrachat": DatasetConfig(
        name="ultrachat",
        hf_path="HuggingFaceH4/ultrachat_200k",
        split="train_sft",
        normalize_fn=_normalize_ultrachat,
    ),
}
```

## 2. 隐藏状态提取 (`data_generation/vllm_hidden_states_generator.py`)

### VllmHiddenStatesGenerator

核心类，使用 vLLM 推理引擎提取目标模型的中间层隐藏状态。

#### 初始化

```python
class VllmHiddenStatesGenerator:
    CACHE_MEMORY_FRACTION = 0.2   # KV Cache 占 GPU 显存比例
    VLLM_BLOCK_SIZE = 16          # GPU 块大小 (NPU=128)
    MAX_NUM_SEQS = 32             # 最大并发序列数
    MAX_DECODE_TOKENS = 1         # 仅 Prefill 模式

    def __init__(self, model_path, layer_ids=None, max_model_len=2048,
                 gpu_memory_utilization=0.8, tensor_parallel_size=1):
        # 1. 加载 tokenizer
        # 2. 自动选择层 ID (如未指定: [2, 中间层, 倒数第3, 最后])
        # 3. 创建 VllmConfig (自定义 Worker 扩展)
        # 4. 初始化 MultiprocExecutor
        # 5. 通过 RPC 设置隐藏状态捕获 Hook
        # 6. 创建调度器和 KV Cache
```

#### 层选择策略

默认自动选择 4 层：

```python
layer_ids = [
    2,                          # 浅层
    num_layers // 2,            # 中间层
    num_layers - 3,             # 深层 (倒数第3)
    num_layers - 1,             # 最后一层
]
```

#### 生成流程

```python
def generate(self, token_ids):
    """
    输入: batch of token ID 序列 list[list[int]]
    输出: list[dict] 每个包含 input_ids, hidden_states, loss_mask
    """
    # 1. 验证并截断序列 (max_model_len - 1)
    # 2. 为每个序列创建 Request 对象
    # 3. 运行调度器迭代 (仅 Prefill)
    for iteration:
        output = scheduler.schedule()
        # 区分 prefill vs decode token
        # prefill 完成后通过 RPC 捕获隐藏状态
        # 中止请求 (不执行实际解码)

    # 4. 检索捕获的隐藏状态
    # 5. 按原始输入顺序组装结果
    return results
```

#### 关键设计决策

| 决策 | 原因 |
|------|------|
| Prefill-only 模式 | 只需要 prefill 阶段的隐藏状态 |
| 禁用前缀缓存 | 防止状态污染 |
| MAX_DECODE_TOKENS=1 | 确保只执行 prefill |
| 自定义 Worker 扩展 | 在计算图中插入隐藏状态捕获 Hook |
| Eager 模式 | 最大兼容性 |

## 3. 元数据配置 (`data_generation/config_generator.py`)

为每次数据生成运行记录完整的可复现性信息：

```python
@dataclass
class DataGenerationConfig:
    VERSION = "2.0"

    reproducibility: ReproducibilityInfo
      - command: str        # 完整命令行
      - package_versions: PackageVersions (torch, vllm, transformers, speculators)
      - gpu: str            # GPU 信息

    model: ModelConfig
      - target_model_path: str
      - tensor_parallel_size: int
      - gpu_memory_utilization: float
      - hidden_size: int

    data: DataConfig
      - train_data_path: str
      - seq_length: int
      - max_samples: int
      - num_samples: int
      - seed: int

    hidden_states: HiddenStatesConfig
      - layer_ids: list[int]

    format: FormatConfig
      - file_pattern: str   # 如 "data_{idx}.pt"
      - schema: dict        # 各字段的 dtype、shape、描述
```

## 4. 离线数据生成脚本 (`scripts/data_generation_offline.py`)

### 完整流程

```python
def main():
    args = parse_args()

    # 1. 加载并预处理数据集
    dataset, tokenizer = load_and_preprocess_dataset(
        target_model_path=args.target_model,
        train_data_path=args.train_data_path,
        seq_length=args.seq_length,
        max_samples=args.max_samples,
    )

    # 2. 初始化 vLLM 生成器
    generator = VllmHiddenStatesGenerator(
        model_path=args.target_model,
        layer_ids=args.layer_ids,
        max_model_len=args.seq_length,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    # 3. 分批提取隐藏状态
    for batch in batched(dataset, batch_size):
        results = generator.generate(batch["input_ids"])

        # 异步保存到磁盘
        for result in results:
            executor.submit(save_sample_to_disk, result, output_path)

    # 4. 保存元数据配置
    save_config(generator, args, num_samples)
```

### 关键特性

- **断点续传**: 扫描已有文件，从上次中断处继续
- **异步 I/O**: 使用 ThreadPoolExecutor 并行写入磁盘
- **进度跟踪**: 记录每个样本的长度到 sample_lengths.json

## 5. 端到端流程 (`scripts/gen_and_train.py`)

将数据生成、词表映射和训练整合为单一命令：

```
run_e2e()
  │
  ├── 1. 数据生成 (可多个数据集)
  │     for dataset in datasets:
  │       run_script("data_generation_offline.py", dataset_args)
  │
  ├── 2. 词表映射 (可选)
  │     combine_token_frequency_distributions()
  │     build_vocab_mappings_from_distribution()
  │
  └── 3. 训练
        run_script("torchrun", "train.py", train_args)
        # 支持多 GPU 分布式训练
```

### 子进程管理

```python
def run_script(script, args, use_uv=False):
    """运行子脚本，支持中断处理"""
    cmd = ["python", script] + args
    if use_uv:
        cmd = ["uv", "run", "--isolated"] + cmd

    process = subprocess.Popen(cmd)
    try:
        process.wait()
    except KeyboardInterrupt:
        # 优雅终止子进程及其所有子进程
        parent = psutil.Process(process.pid)
        for child in parent.children(recursive=True):
            child.terminate()
        parent.terminate()
```
