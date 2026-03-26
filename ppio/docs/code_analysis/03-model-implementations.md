# 模型实现详解

## 1. EAGLE Speculator (`models/eagle.py`)

### 概述

EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) 是一种基于单层 Transformer 的投机解码方法。核心思想是用一个轻量级模块融合 token embedding 和验证器的隐藏状态来预测下一个 token。

### 配置 (`EagleSpeculatorConfig`)

```python
class EagleSpeculatorConfig:
    speculators_model_type = "eagle"
    transformer_layer_architecture: str  # 自动检测或指定 (如 "LlamaDecoderLayer")
    transformer_layer_config: PretrainedConfig  # Decoder 层配置
    layernorms: bool = False  # 是否包含 LayerNorm
    fusion_bias: bool = False  # Fusion 层是否有 bias (HASS 使用 True)
```

### 架构

```
Input: input_ids + hidden_states (来自验证器)

input_ids → [embed_tokens] → embeddings
                                  │
                          [embed_layernorm] (可选)
                                  │
embeddings ──────────┐            │
hidden_states ───────┤  → [fusion_layer (Linear: 2*H → H)]
                                  │
                          [post_embedding_layernorm] (可选)
                                  │
                      [transformer_decoder_layer]
                                  │
                      [pre_lm_head_layernorm] (可选)
                                  │
                          [lm_head (Linear: H → V)]
                                  │
                              logits
```

### 验证器依赖组件

以下组件在 `attach_verifier()` 时从验证器模型提取：

- `embed_tokens`: Token 嵌入层
- `rotary_emb`: 旋转位置编码
- `lm_head`: 语言模型头

### 关键实现细节

1. **Fusion Layer**: 将 embeddings 和 hidden_states 拼接后投影到 hidden_size
2. **动态组件加载**: `_import_model_classes()` 根据架构名动态导入对应的 Transformer 组件
3. **HASS 变体**: `fusion_bias=True` + `layernorms=True`

---

## 2. EAGLE3 Draft Model (`models/eagle3/`)

### 概述

EAGLE3 是 EAGLE 的第三代版本，引入了多层 Transformer、词表映射和 TTT (Test-Time Training) 机制。这是当前项目的主要训练目标。

### 配置 (`Eagle3SpeculatorConfig`)

```python
class Eagle3SpeculatorConfig:
    speculators_model_type = "eagle3"
    transformer_layer_config: PretrainedConfig  # 默认 LlamaConfig
    draft_vocab_size: int = 32000  # 草稿模型词表大小
    norm_before_residual: bool = False  # 残差连接前是否归一化
    target_hidden_size: int | None = None  # 目标模型 hidden_size
    eagle_aux_hidden_state_layer_ids: list[int] = []  # 辅助隐藏状态层 ID
    embed_requires_grad: bool = False  # 嵌入层是否需要梯度
```

### 架构

```
Input: input_ids + hidden_states (多层) + verifier_last_hidden_states

                    ┌──────────────────────────────┐
input_ids ────────→ │ embed_tokens (128K vocab)     │
                    └──────────┬───────────────────┘
                               │
hidden_states ─────→ [concat along dim=-1] (多层隐藏状态拼接)
                               │
concat(embeddings, hidden_states) → shape: [B, S, 2*H]
                               │
                    ┌──────────┴───────────────────┐
                    │ Eagle3FirstLayer (2*H → H)    │
                    │ - 特殊 Q/K/V 投影 (2*H 输入)  │
                    │ - 分离归一化                    │
                    └──────────┬───────────────────┘
                               │
                    ┌──────────┴───────────────────┐
                    │ Decoder Layers (标准 H → H)    │
                    │ × (num_hidden_layers - 1)      │
                    └──────────┬───────────────────┘
                               │
                    ┌──────────┴───────────────────┐
                    │ norm → lm_head (H → 32K)      │
                    └──────────┬───────────────────┘
                               │
                           draft_logits (32K vocab)
                               │
                    ┌──────────┴───────────────────┐
                    │ 词表映射: d2t (draft → target)  │
                    └──────────────────────────────┘
```

### 词表映射机制

EAGLE3 使用缩减的词表（默认 32K）来降低计算开销：

| 映射 | 形状 | 说明 |
|------|------|------|
| `t2d` (target_to_draft) | `[target_vocab_size]` bool | 目标词表中哪些 token 在草稿词表中 |
| `d2t` (draft_to_target) | `[draft_vocab_size]` int | 草稿词表索引到目标词表 ID 的偏移量 |

映射构建流程：
1. 统计训练数据中各 token 的频率
2. 按频率降序取前 `draft_vocab_size` 个 token
3. 构建双向映射表

### TTT (Test-Time Training) 机制

EAGLE3 的前向传播支持多步 TTT：

```python
def forward(self, input_ids, hidden_states, ...):
    for step in range(ttt_steps):
        # 1. 计算当前步的 logits
        logits = self._forward_step(inputs)

        # 2. 计算损失（KL 散度 vs 验证器 logits）
        loss = loss_function(logits, verifier_logits)

        # 3. 应用损失衰减
        loss *= decay_factor ** step

        # 4. 对齐下一步的输入
        next_input = align_for_step(logits, targets, step)

    return total_loss, metrics
```

### 损失函数

使用 KL 散度衡量草稿模型和验证器的输出分布差异：

```python
def loss_function(draft_logits, verifier_logits, loss_mask):
    # 将验证器 logits 映射到草稿词表空间
    # 计算 KL(verifier || draft)
    # 应用 loss_mask（仅对 assistant token 计算损失）
```

### 自定义注意力 (`attention.py`)

EAGLE3 实现了多种注意力掩码：

1. **因果掩码 (Causal Mask)**: 标准自回归掩码
2. **文档边界掩码 (Document Mask)**: 防止跨文档注意力
3. **对角草稿掩码 (Diagonal Draft Mask)**: 允许草稿 token 的自注意力

使用 PyTorch flex_attention 实现高效的块稀疏注意力。

### 架构适配 (`model_definitions.py`)

`Eagle3FirstLayerMixin` 为不同架构提供统一接口：

```python
class Eagle3FirstLayerMixin:
    def _patch_eagle3_projections(self):
        """修改 Q/K/V 投影以接受 2*hidden_size 输入"""
        # 将原始 hidden_size 的权重扩展为 2*hidden_size

    def forward(self, hidden_states, ...):
        # 1. 分离: embeddings = hidden_states[:, :, :H]
        #          verifier_hs = hidden_states[:, :, H:]
        # 2. 分别归一化
        # 3. 重新拼接
        # 4. 送入注意力层
```

支持的架构实现：
- `LlamaDecoderEagle3FirstLayer`
- `Qwen3DecoderEagle3FirstLayer`

---

## 3. Independent Speculator (`models/independent.py`)

### 概述

最简单的投机器类型——直接包装一个现有的独立模型作为投机器。

```python
class IndependentSpeculatorConfig:
    speculators_model_type = "independent"
    # 保留原始模型的 model_type

    @classmethod
    def from_pretrained_config(cls, config, speculators_config):
        """从现有 PretrainedConfig 创建"""
```

适用场景：用小型 LLM（如 Llama-68M）直接作为 draft model。

---

## 4. MLP Speculator (`models/mlp.py`)

### 概述

基于多层感知机的投机器，目前仅定义了配置（TODO 状态）。

```python
class MLPSpeculatorConfig:
    speculators_model_type = "mlp"
    torch_dtype: str = "bfloat16"
    inputs: list[str]  # 输入来源 (embeddings, hidden_states)
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    num_layers: int  # 层数 = 最大投机 token 数
    tie_weights: bool  # 是否共享层间权重
```

---

## 5. Token 提议方法 (`proposals/greedy.py`)

### GreedyTokenProposalConfig

```python
@TokenProposalConfig.register("greedy")
class GreedyTokenProposalConfig:
    proposal_type = "greedy"
    speculative_tokens: int = 5      # 每轮生成的候选 token 数
    verifier_accept_k: int = 1       # Top-k 接受阈值
    accept_tolerance: float = 0.0    # 接受容差
```

---

## 模型对比

| 特性 | EAGLE | EAGLE3 | Independent | MLP |
|------|-------|--------|-------------|-----|
| Transformer 层数 | 1 | 多层 | 依赖原模型 | 0 |
| 词表大小 | 与验证器相同 | 缩减 (32K) | 与验证器相同 | 自定义 |
| 需要验证器隐藏状态 | 是 | 是 (多层) | 否 | 可选 |
| TTT 支持 | 否 | 是 | 否 | 否 |
| Fusion 机制 | Linear 拼接 | 首层特殊投影 | 无 | MLP |
| 训练实现 | 完整 | 完整 | 无 | 未实现 |
| 实现状态 | 完整 | 完整 | 配置完整 | TODO |
