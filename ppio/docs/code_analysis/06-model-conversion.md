# 模型转换系统

## 概述

模型转换系统将外部仓库的 EAGLE/EAGLE3 检查点转换为 Speculators 标准格式，确保与 vLLM 和 HuggingFace Hub 兼容。

## 入口 (`convert/entrypoints.py`)

```python
def convert_model(
    model: str,                    # 源模型路径 (本地或 HF Hub)
    algorithm: str = "eagle",      # "eagle" 或 "eagle3"
    output_path: str = "converted", # 输出目录
    verifier: str | None = None,   # 验证器模型路径
    validate_device: str | None = None,  # 验证设备
    **algorithm_kwargs,            # 算法特定参数
):
    if algorithm in ("eagle", "hass"):
        converter = EagleConverter()
    elif algorithm == "eagle3":
        converter = Eagle3Converter()

    converter.convert(model, output_path, verifier, validate_device, **algorithm_kwargs)
```

## CLI 命令

```bash
# EAGLE v1/v2 转换
speculators convert path/to/eagle_model \
    --algorithm eagle \
    --verifier meta-llama/Llama-3-8B \
    --output-path ./converted

# EAGLE3 转换
speculators convert path/to/eagle3_model \
    --algorithm eagle3 \
    --verifier meta-llama/Llama-3-8B \
    --output-path ./converted

# 带验证的转换
speculators convert path/to/model \
    --algorithm eagle \
    --verifier meta-llama/Llama-3-8B \
    --validate-device cuda:0
```

## 1. EAGLE 转换器 (`convert/eagle/eagle_converter.py`)

### 转换流程

```
源检查点 (EAGLE/HASS 格式)
  │
  ├── 1. 获取本地检查点
  │     ensure_checkpoint_is_local(model_path)
  │
  ├── 2. 加载配置和权重
  │     load_checkpoint_config(path)
  │     load_checkpoint_weights(path)
  │
  ├── 3. 特性自动检测
  │     detect_fusion_bias_and_layernorms(weights)
  │     → fusion_bias: bool (HASS 特征)
  │     → layernorms: bool
  │
  ├── 4. 创建 Transformer 配置
  │     _create_transformer_config_from_eagle(eagle_config)
  │     → 从 EAGLE config 提取 LlamaConfig 参数
  │
  ├── 5. 构建 Speculators 配置
  │     _build_eagle_speculator_config(...)
  │     → EagleSpeculatorConfig + SpeculatorsConfig
  │
  ├── 6. 权重重映射
  │     _process_checkpoint_weights(weights)
  │     → 跳过不需要的权重
  │     → 重命名权重键
  │
  ├── 7. 保存转换后的检查点
  │     _save_converted_checkpoint(config, weights, output_path)
  │     → config.json + model.safetensors
  │
  └── 8. 验证 (可选)
        _validate_converted_checkpoint(output_path, verifier, device)
        → 加载模型并执行前向传播
```

### 权重映射规则

| EAGLE 原始键 | Speculators 键 | 说明 |
|-------------|---------------|------|
| `fc.weight` | `fusion_layer.weight` | Fusion 层权重 |
| `fc.bias` | `fusion_layer.bias` | Fusion 层偏置 (HASS) |
| `embed_layernorm.*` | `embed_layernorm.*` | 嵌入层归一化 |
| `post_embedding_layernorm.*` | `post_embedding_layernorm.*` | 嵌入后归一化 |
| `layer.self_attn.*` | `decoder_layer.self_attn.*` | 注意力权重 |
| `layer.mlp.*` | `decoder_layer.mlp.*` | MLP 权重 |
| `layer.input_layernorm.*` | `decoder_layer.input_layernorm.*` | 输入归一化 |
| `layer.post_attention_layernorm.*` | `decoder_layer.post_attention_layernorm.*` | 注意力后归一化 |

### 跳过的权重

- `embed_tokens.*` (从验证器加载)
- `lm_head.*` (从验证器加载)
- `norm.*` (验证器的归一化层)
- `rotary_emb.*` (从验证器加载)

### 特性自动检测

```python
def detect_fusion_bias_and_layernorms(state_dict):
    """根据权重名称自动检测模型特性"""
    has_fusion_bias = "fc.bias" in state_dict
    has_layernorms = any(
        "embed_layernorm" in k or "post_embedding_layernorm" in k
        for k in state_dict
    )
    return has_fusion_bias, has_layernorms
```

## 2. EAGLE3 转换器 (`convert/eagle/eagle3_converter.py`)

### 转换流程

```
源检查点 (EAGLE3 格式)
  │
  ├── 1. 获取本地检查点
  │
  ├── 2. 检测词表大小
  │     从 t2d 张量 或 目标模型 config 中获取
  │
  ├── 3. 创建 Transformer 配置
  │     _create_transformer_config_from_eagle(eagle3_config)
  │     → 从 EAGLE3 config 提取 LlamaConfig
  │
  ├── 4. 构建配置
  │     _build_eagle3_speculator_config(...)
  │     → Eagle3SpeculatorConfig + SpeculatorsConfig
  │     → 包含辅助隐藏状态层 ID
  │
  ├── 5. 决定嵌入层处理
  │     根据检查点是否包含嵌入权重决定:
  │     - 包含 → 保留在检查点中
  │     - 不包含 → 运行时从验证器加载
  │
  ├── 6. 加载并转换权重
  │     → 处理 d2t/t2d 映射缓冲区
  │     → dtype 转换
  │
  ├── 7. 保存
  │     _save_converted_checkpoint(...)
  │
  └── 8. 验证 (可选)
```

### 词表大小检测

```python
# 优先级:
# 1. 从 t2d (target_to_draft) 张量的长度推断
if "t2d" in weights:
    draft_vocab_size = int(weights["t2d"].sum().item())

# 2. 从目标模型 config 获取
elif verifier_path:
    target_config = AutoConfig.from_pretrained(verifier_path)
    target_vocab_size = target_config.vocab_size
```

### EAGLE3 Legacy 模型 (`convert/eagle/eagle3_legacy_model.py`)

用于加载旧格式 EAGLE3 检查点的兼容层：

```python
class Eagle3Speculator(PreTrainedModel):
    def __init__(self, config):
        self.fusion_layer = nn.Linear(3 * hidden_size, hidden_size)
        self.layers = nn.ModuleList([
            Eagle3DecoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])

    def tie_weights(self):
        """覆盖: 防止 HF 自动绑定权重"""
        # EAGLE3 的 embed_tokens (128K) 和 lm_head (32K)
        # 大小不同，不能绑定
        pass
```

关键区别: EAGLE3 的输入嵌入 (128K) 和输出头 (32K) 词表大小不同，必须防止 Transformers 自动绑定权重。

## 3. 工具函数 (`convert/eagle/utils.py`)

### 检查点操作

```python
def ensure_checkpoint_is_local(model_path):
    """确保检查点在本地磁盘"""
    if os.path.isdir(model_path):
        return model_path
    return download_checkpoint_from_hub(model_path)

def load_checkpoint_weights(checkpoint_path):
    """加载权重 (safetensors 优先, 回退 PyTorch)"""
    for safetensors_file in ["model.safetensors", "pytorch_model.bin"]:
        if exists(safetensors_file):
            return load_file(safetensors_file)
    raise FileNotFoundError(...)

def load_checkpoint_config(checkpoint_path):
    """加载 config.json"""
    config_path = checkpoint_path / "config.json"
    return json.loads(config_path.read_text())
```

### 词表大小发现

```python
def find_vocab_size(config):
    """递归搜索嵌套配置中的 vocab_size"""
    if "vocab_size" in config:
        return config["vocab_size"]
    for value in config.values():
        if isinstance(value, dict):
            result = find_vocab_size(value)
            if result is not None:
                return result
    return None
```

## 转换后的输出格式

```
converted/
├── config.json           # SpeculatorModelConfig (Speculators 格式)
│   ├── speculators_model_type: "eagle" | "eagle3"
│   ├── speculators_version: "0.3.0"
│   ├── speculators_config:
│   │   ├── algorithm: "eagle" | "eagle3"
│   │   ├── proposal_methods: [{type: "greedy", ...}]
│   │   └── verifier: {name_or_path: "...", architectures: [...]}
│   └── ... (模型特定配置)
│
└── model.safetensors     # 模型权重 (safetensors 格式)
```

这种格式可以直接通过以下方式加载：

```python
from speculators import SpeculatorModel

model = SpeculatorModel.from_pretrained("converted/")
model.attach_verifier("meta-llama/Llama-3-8B")
```
