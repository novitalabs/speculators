# 核心架构

## 架构总览

Speculators 采用**分层架构**，从底层到高层依次为：

```
┌─────────────────────────────────────────────────────────┐
│  CLI 层 (__main__.py)                                   │
│  speculators convert / --version                         │
├─────────────────────────────────────────────────────────┤
│  配置层 (config.py)                                      │
│  SpeculatorModelConfig ← SpeculatorsConfig               │
│                          ├── VerifierConfig               │
│                          └── TokenProposalConfig          │
├─────────────────────────────────────────────────────────┤
│  模型层 (model.py + models/)                             │
│  SpeculatorModel (抽象) → Eagle / Eagle3 / Independent   │
├─────────────────────────────────────────────────────────┤
│  工具层 (utils/)                                         │
│  ClassRegistryMixin + AutoImporterMixin + loading        │
└─────────────────────────────────────────────────────────┘
```

## 1. 自动注册表系统

这是整个架构最核心的设计，实现了**零配置的插件化扩展**。

### AutoImporterMixin (`utils/auto_importer.py`)

负责自动发现并导入指定包中的所有模块：

```python
class AutoImporterMixin:
    auto_package = ...        # 要扫描的包名
    auto_ignore_modules = []  # 忽略的模块
    auto_imported_modules = set()  # 已导入记录

    @classmethod
    def auto_import_package_modules(cls):
        # 使用 pkgutil.walk_packages() 遍历包中所有模块
        # 动态 import 每个模块，触发 @register 装饰器
```

### ClassRegistryMixin (`utils/registry.py`)

基于 AutoImporterMixin 构建的类注册表：

```python
class ClassRegistryMixin(AutoImporterMixin):
    registry: dict = {}       # 名称 → 类的映射
    registry_auto_discovery = True

    @classmethod
    def register(cls, name=None):
        """装饰器：将类注册到 registry"""
        return lambda clazz: cls.register_decorator(clazz, name)

    @classmethod
    def auto_populate_registry(cls):
        """触发自动发现，导入所有模块，填充 registry"""
        cls.auto_import_package_modules()
```

### 工作流程

```
1. 用户 import speculators
2. __init__.py 调用 reload_and_populate_configs()
3. SpeculatorModelConfig.auto_populate_registry()
   → 扫描 speculators.models 包
   → 导入 eagle.py, eagle3/, independent.py, mlp.py
   → @register 装饰器将各 Config 注册到 registry
4. 同样地，TokenProposalConfig 扫描 speculators.proposals
5. 此后 SpeculatorModel.from_pretrained() 可根据 config 自动找到对应的模型类
```

## 2. 配置体系 (`config.py`)

配置体系是 **Pydantic 与 HuggingFace PretrainedConfig 的融合**。

### 配置类层次

```
TokenProposalConfig (PydanticClassRegistryMixin)
  └── GreedyTokenProposalConfig
      - speculative_tokens: int = 5
      - verifier_accept_k: int = 1
      - accept_tolerance: float = 0.0

VerifierConfig (BaseModel)
  - verifier_model_name_or_path: str
  - verifier_architectures: list[str]

SpeculatorsConfig (ReloadableBaseModel)
  - algorithm: str
  - proposal_methods: list[TokenProposalConfig]
  - default_proposal_method: str
  - verifier: VerifierConfig

SpeculatorModelConfig (PydanticClassRegistryMixin + PretrainedConfig)
  ├── speculators_model_type: str  # 鉴别器字段
  ├── speculators_version: str
  ├── speculators_config: SpeculatorsConfig
  ├── EagleSpeculatorConfig ("eagle")
  ├── Eagle3SpeculatorConfig ("eagle3")
  ├── IndependentSpeculatorConfig ("independent")
  └── MLPSpeculatorConfig ("mlp")
```

### 关键设计：双重初始化

`SpeculatorModelConfig` 同时继承了 Pydantic 和 PretrainedConfig，需要特殊的初始化逻辑：

```python
class SpeculatorModelConfig(PydanticClassRegistryMixin, PretrainedConfig):
    def __init__(self, **kwargs):
        # 1. 调用 Pydantic 验证
        PydanticClassRegistryMixin.__init__(self, **kwargs)
        # 2. 调用 HuggingFace PretrainedConfig.__init__
        PretrainedConfig.__init__(self, **self.__dict__)
```

这样配置既能享受 Pydantic 的类型验证和序列化能力，又能与 HuggingFace Hub 的 `from_pretrained` / `push_to_hub` 机制兼容。

### 鉴别器 (Discriminator) 机制

配置使用 `speculators_model_type` 字段作为鉴别器，实现多态反序列化：

```python
# 加载时根据 config.json 中的 speculators_model_type 自动选择正确的子类
config = SpeculatorModelConfig.from_pretrained("model_path")
# → 如果 type = "eagle"，实际返回 EagleSpeculatorConfig
# → 如果 type = "eagle3"，实际返回 Eagle3SpeculatorConfig
```

## 3. 模型抽象层 (`model.py`)

### SpeculatorModel 基类

```python
class SpeculatorModel(ClassRegistryMixin, PreTrainedModel, GenerationMixin):
    auto_package = "speculators.models"  # 自动发现模型

    # 模型加载
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # 1. 加载 config
        # 2. 通过 registry 找到对应的模型子类
        # 3. 调用子类的 from_pretrained

    # 验证器管理
    def attach_verifier(self, verifier, mode="full"):
        """将验证器模型附加到投机器"""
    def detach_verifier(self):
        """分离验证器"""
    def resolve_verifier(self, verifier_model_name_or_path):
        """从路径加载验证器"""

    # 状态字典（排除验证器参数）
    def state_dict(self, **kwargs):
        """只返回投机器自身的参数"""

    # 训练支持（子类实现）
    @abstractmethod
    def from_training_args(cls, ...): ...
    @abstractmethod
    def get_trainer_kwargs(self): ...
    @abstractmethod
    def forward(self, ...): ...
```

### 验证器附加模式

| 模式 | 说明 | 使用场景 |
|------|------|----------|
| `"detached"` | 无验证器 | 模型初始状态/推理时 |
| `"full"` | 完整附加 | 推理生成 |
| `"train_only"` | 仅附加训练组件 | 训练时（节省显存） |

### 模型加载流程

```
SpeculatorModel.from_pretrained("path/to/model")
  │
  ├── 1. 加载 config.json
  ├── 2. 解析 speculators_model_type
  ├── 3. 查找 registry: {"eagle": EagleSpeculator, "eagle3": Eagle3DraftModel, ...}
  ├── 4. 调用对应子类的 from_pretrained()
  └── 5. 返回实例化的模型
```

## 4. 初始化流程详解

```python
# speculators/__init__.py
from speculators.config import (
    SpeculatorModelConfig,
    SpeculatorsConfig,
    TokenProposalConfig,
    VerifierConfig,
)
from speculators.model import SpeculatorModel

def reload_and_populate_configs():
    TokenProposalConfig.auto_populate_registry()
    SpeculatorModelConfig.auto_populate_registry()

def reload_and_populate_models():
    SpeculatorModel.auto_populate_registry()

# 导入时自动执行
reload_and_populate_configs()
reload_and_populate_models()
```

这意味着只要 `import speculators`，所有模型和配置都会被自动发现和注册，用户无需手动 import 具体实现。

## 5. 共享架构组件 (`models/base_components.py`)

为支持多种目标模型架构（Llama, Qwen3），定义了统一的组件接口：

```python
ModelComponents = NamedTuple("ModelComponents", [
    ("first_layer_class", type),    # 自定义首层
    ("decoder_layer_class", type),  # 标准 Decoder 层
    ("norm_class", type),           # 归一化层
    ("rotary_emb_class", type),     # 旋转位置编码
])

# 注册的架构
model_classes = {
    "llama": ModelComponents(LlamaDecoderLayer, LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding),
    "qwen3": ModelComponents(Qwen3DecoderLayer, Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3RotaryEmbedding),
}
```

EAGLE3 通过 `model_definitions.py` 扩展了这些组件，引入了特殊的首层处理（接受 2x hidden_size 输入）。
