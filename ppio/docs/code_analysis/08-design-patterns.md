# 设计模式与扩展指南

## 核心设计模式

### 1. 注册表模式 (Registry Pattern)

**用途**: 实现零配置的插件化扩展

**实现链**: `AutoImporterMixin` → `ClassRegistryMixin` → `PydanticClassRegistryMixin`

```python
# 注册一个新模型
@SpeculatorModel.register("my_model")
class MySpeculatorModel(SpeculatorModel):
    config_class = MySpeculatorConfig
    ...

# 注册一个新配置
@SpeculatorModelConfig.register("my_model")
class MySpeculatorConfig(SpeculatorModelConfig):
    speculators_model_type: str = "my_model"
    ...

# 使用: 自动发现，无需手动注册
model = SpeculatorModel.from_pretrained("path")
# → 根据 config.json 中的 speculators_model_type 自动选择子类
```

**自动发现流程**:
1. `auto_import_package_modules()` 遍历 `speculators.models` 中所有 .py 文件
2. 导入每个模块时，`@register` 装饰器被执行
3. 类被添加到 `registry` 字典中
4. 后续通过 `registry[name]` 查找对应类

### 2. 工厂模式 (Factory Pattern)

**用途**: 根据配置创建正确的模型实例

```python
# SpeculatorModel.from_pretrained() 内部逻辑
class SpeculatorModel:
    @classmethod
    def from_pretrained(cls, path, **kwargs):
        config = SpeculatorModelConfig.from_pretrained(path)
        model_cls = cls.registered_model_class_from_config(config)
        # model_cls 是 EagleSpeculator 或 Eagle3DraftModel 等
        return model_cls.from_pretrained(path, config=config, **kwargs)
```

### 3. 策略模式 (Strategy Pattern)

**用途**: 可互换的算法和组件实现

**应用场景**:
- 不同的模型架构 (Eagle, Eagle3, Independent)
- 不同的注意力实现 (standard, flex_attention)
- 不同的检查点策略 (SingleGPU, Distributed)
- 不同的噪声增强 (Gaussian, Uniform)
- 不同的日志后端 (TensorBoard, W&B, Trackio)

### 4. 模板方法模式 (Template Method Pattern)

**用途**: 定义算法骨架，子类实现具体步骤

```python
class SpeculatorModel:
    # 模板方法
    def from_pretrained(cls, path):
        config = cls.load_config(path)
        model = cls.create_model(config)
        model.load_weights(path)
        return model

    # 抽象步骤 (子类实现)
    @abstractmethod
    def forward(self, input_ids, hidden_states, ...): ...

    @abstractmethod
    def from_training_args(cls, ...): ...

    @abstractmethod
    def get_trainer_kwargs(self): ...
```

### 5. 依赖注入模式 (Dependency Injection)

**用途**: 验证器的延迟绑定

```python
# 模型创建时不需要验证器
model = SpeculatorModel.from_pretrained("path")
# 验证器在需要时注入
model.attach_verifier("meta-llama/Llama-3-8B", mode="full")
# 可以分离
model.detach_verifier()
# 可以切换模式
model.attach_verifier("meta-llama/Llama-3-8B", mode="train_only")
```

### 6. 复合配置模式 (Composite Configuration)

**用途**: Pydantic 验证 + HuggingFace 兼容性

```python
class SpeculatorModelConfig(PydanticClassRegistryMixin, PretrainedConfig):
    """同时继承两个配置体系"""
    # Pydantic: 类型验证、序列化、鉴别器
    # PretrainedConfig: from_pretrained(), save_pretrained(), push_to_hub()
```

### 7. 构建器模式 (Builder Pattern)

**用途**: 复杂对象的分步构建

```python
# DataGenerationConfig 的构建
config = DataGenerationConfig.from_generator(
    generator=vllm_generator,
    target_model_path="...",
    train_data_path="...",
    ...
)
# 内部: 自动检测 GPU、收集版本信息、生成 schema 文档
```

---

## 如何添加新的投机解码算法

### 步骤 1: 定义配置

在 `src/speculators/models/` 下创建新文件:

```python
# src/speculators/models/my_algorithm.py
from speculators.config import SpeculatorModelConfig

@SpeculatorModelConfig.register("my_algorithm")
class MyAlgorithmConfig(SpeculatorModelConfig):
    speculators_model_type: str = "my_algorithm"

    # 自定义配置字段
    num_layers: int = 3
    custom_param: float = 0.1
```

### 步骤 2: 实现模型

```python
from speculators.model import SpeculatorModel

@SpeculatorModel.register("my_algorithm")
class MyAlgorithmModel(SpeculatorModel):
    config_class = MyAlgorithmConfig

    def __init__(self, config: MyAlgorithmConfig):
        super().__init__(config)
        # 初始化模型组件
        self.layers = nn.ModuleList([...])

    def forward(self, input_ids, hidden_states, **kwargs):
        """前向传播"""
        # 实现投机解码逻辑
        return loss, metrics

    @classmethod
    def from_training_args(cls, verifier_config, **training_args):
        """从训练参数创建模型"""
        config = MyAlgorithmConfig(...)
        model = cls(config)
        return model

    def get_trainer_kwargs(self):
        """返回训练和验证时的额外参数"""
        return {"train_call_kwargs": {...}, "val_call_kwargs": {...}}

    def attach_verifier(self, verifier, mode="full"):
        """附加验证器模型"""
        super().attach_verifier(verifier, mode)
        # 从验证器提取需要的组件
        self.embed_tokens = verifier.model.embed_tokens
        self.lm_head = verifier.lm_head
```

### 步骤 3: 添加 Token 提议方法 (可选)

```python
# src/speculators/proposals/my_proposal.py
from speculators.config import TokenProposalConfig

@TokenProposalConfig.register("my_proposal")
class MyProposalConfig(TokenProposalConfig):
    proposal_type: str = "my_proposal"
    custom_param: int = 10
```

### 步骤 4: 添加转换器 (可选)

```python
# src/speculators/convert/my_algorithm/converter.py
class MyAlgorithmConverter:
    def convert(self, model_path, output_path, verifier, **kwargs):
        # 加载源检查点
        # 重映射权重
        # 构建 Speculators 配置
        # 保存
```

并在 `convert/entrypoints.py` 中注册:

```python
def convert_model(model, algorithm, ...):
    if algorithm == "my_algorithm":
        converter = MyAlgorithmConverter()
    ...
```

### 步骤 5: 注册自动发现

无需额外操作！只要文件放在 `src/speculators/models/` 目录下，`AutoImporterMixin` 会自动发现并注册。

### 步骤 6: 添加测试

```python
# tests/unit/models/test_my_algorithm.py
def test_my_algorithm_config():
    config = MyAlgorithmConfig(num_layers=3)
    assert config.speculators_model_type == "my_algorithm"

def test_my_algorithm_forward():
    config = MyAlgorithmConfig()
    model = MyAlgorithmModel(config)
    # 测试前向传播
```

---

## 关键架构约束

1. **模型 state_dict 不包含验证器参数**: `SpeculatorModel.state_dict()` 自动排除验证器的参数
2. **配置必须可序列化**: 所有配置字段必须能被 Pydantic 和 JSON 序列化
3. **`speculators_model_type` 必须唯一**: 这是注册表的查找键
4. **训练兼容性**: 如果模型支持训练，必须实现 `from_training_args()` 和 `get_trainer_kwargs()`
5. **HuggingFace Hub 兼容**: 转换后的模型必须能通过 `from_pretrained()` 加载
6. **safetensors 格式**: 推荐使用 safetensors 保存权重

---

## 代码组织原则

| 原则 | 说明 |
|------|------|
| 关注点分离 | 配置、模型、训练、数据生成各自独立 |
| 开放-封闭 | 通过注册表扩展新算法，无需修改核心代码 |
| 依赖倒置 | 核心代码依赖抽象 (SpeculatorModel)，不依赖具体实现 |
| 接口隔离 | 训练、推理、转换各有独立的抽象方法 |
| 单一职责 | 每个模块有明确的职责边界 |

## 技术债务与 TODO

| 位置 | 说明 |
|------|------|
| `models/mlp.py` | MLP 模型仅有配置，实现标记为 TODO |
| `proposals/` | 目前仅有 greedy 提议，缺少 tree-based 等方法 |
| `eagle3/core.py` | `conditional_torch_compile` 暗示编译支持尚不完善 |
| `data.py` | 文件大小估算样本长度的方法不够精确 |
