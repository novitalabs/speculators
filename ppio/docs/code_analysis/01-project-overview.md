# 项目概览

## 基本信息

- **项目名称**: Speculators
- **版本**: 0.3.0 (通过 setuptools-git-versioning 动态管理)
- **许可证**: Apache 2.0
- **维护方**: Red Hat
- **Python 要求**: >= 3.10
- **CLI 入口**: `speculators` 命令 → `speculators.__main__:app`

## 什么是投机解码 (Speculative Decoding)

投机解码是一种 LLM 推理加速技术。核心思想：

1. 用一个更小的 **Draft Model（草稿模型/投机器）** 快速生成多个候选 token
2. 用一个更大的 **Verifier Model（验证模型）** 并行验证这些 token
3. 如果候选 token 被接受，就跳过了大模型的多步串行推理，从而加速

Speculators 库的目标是**标准化**投机解码算法的创建、训练、存储和加载流程，使其与 vLLM 等推理引擎无缝集成。

## 技术栈

| 类别 | 技术 |
|------|------|
| 深度学习框架 | PyTorch >= 2.9.0 |
| 模型生态 | HuggingFace Transformers >= 4.56.1 |
| 配置管理 | Pydantic >= 2.0.0 + Pydantic Settings |
| CLI 框架 | Typer + Click |
| 推理引擎 (可选) | vLLM >= 0.12.0 (用于数据生成) |
| 模型序列化 | safetensors |
| 日志 | loguru + Python logging + Rich |
| 测试 | pytest + tox |
| 代码质量 | ruff (lint/format) + mypy (类型检查) + mdformat |
| 文档 | MkDocs + Material 主题 |
| CI/CD | GitHub Actions |

## 目录结构

```
speculators/
├── src/speculators/           # 主包源码
│   ├── __init__.py            # 包入口，自动注册模型和配置
│   ├── __main__.py            # CLI 入口 (Typer)
│   ├── config.py              # 核心配置类
│   ├── model.py               # SpeculatorModel 基类
│   ├── models/                # 具体模型实现
│   │   ├── eagle.py           # EAGLE v1/v2 模型
│   │   ├── eagle3/            # EAGLE v3 子包
│   │   │   ├── config.py      # EAGLE3 配置
│   │   │   ├── core.py        # EAGLE3 核心实现
│   │   │   ├── model_definitions.py  # 架构适配层
│   │   │   └── attention.py   # 自定义注意力机制
│   │   ├── independent.py     # 独立模型配置
│   │   ├── mlp.py             # MLP 模型配置
│   │   └── base_components.py # 共享架构组件
│   ├── train/                 # 训练基础设施
│   │   ├── trainer.py         # 训练循环
│   │   ├── data.py            # 数据集和数据加载
│   │   ├── checkpointer.py    # 检查点管理
│   │   ├── logger.py          # 日志系统 (TB/W&B/Trackio)
│   │   ├── utils.py           # FSDP 分布式工具
│   │   ├── noise_transforms.py # 噪声增强
│   │   ├── vocab_mapping.py   # 词表映射
│   │   └── distributed_batch_sampler.py # 分布式批次采样
│   ├── data_generation/       # 训练数据生成
│   │   ├── vllm_hidden_states_generator.py # vLLM 隐藏状态提取
│   │   ├── configs.py         # 数据集配置注册
│   │   ├── preprocessing.py   # 数据预处理管道
│   │   ├── config_generator.py # 元数据配置生成
│   │   └── custom_worker.py   # vLLM 自定义 Worker
│   ├── proposals/             # Token 提议方法
│   │   └── greedy.py          # 贪心提议配置
│   ├── convert/               # 模型转换工具
│   │   ├── entrypoints.py     # 转换入口
│   │   └── eagle/             # EAGLE 系列转换器
│   │       ├── eagle_converter.py
│   │       ├── eagle3_converter.py
│   │       ├── eagle3_legacy_model.py
│   │       └── utils.py
│   └── utils/                 # 通用工具
│       ├── registry.py        # 类注册表 Mixin
│       ├── auto_importer.py   # 自动模块发现
│       ├── loading.py         # safetensors 加载
│       ├── pydantic_utils.py  # Pydantic 工具
│       └── util.py            # 通用工具函数
├── scripts/                   # 脚本入口
│   ├── train.py               # 训练脚本
│   ├── data_generation_offline.py # 离线数据生成
│   ├── gen_and_train.py       # 端到端生成+训练
│   └── build_vocab_mapping.py # 词表映射构建
├── tests/                     # 测试套件
│   ├── unit/                  # 单元测试
│   ├── integration/           # 集成测试
│   ├── e2e/                   # 端到端测试
│   └── datagen/               # 数据生成测试
├── examples/                  # 使用示例
├── docs/                      # MkDocs 文档源码
├── pyproject.toml             # 项目配置
├── setup.py                   # 版本管理
├── tox.ini                    # 测试自动化
├── Makefile                   # 开发任务自动化
└── mkdocs.yml                 # 文档站点配置
```

## 核心依赖关系图

```
speculators.__init__
  ├── config.py (SpeculatorModelConfig, SpeculatorsConfig, ...)
  │     ├── utils/registry.py (PydanticClassRegistryMixin)
  │     └── utils/auto_importer.py (AutoImporterMixin)
  ├── model.py (SpeculatorModel)
  │     ├── config.py
  │     └── utils/registry.py (ClassRegistryMixin)
  └── models/ (各具体模型实现)
        ├── eagle.py → base_components.py
        ├── eagle3/ → model_definitions.py, attention.py
        ├── independent.py
        └── mlp.py
```

## 支持的模型架构

| 目标模型 | 参数量 |
|----------|--------|
| Llama 3/3.1 | 8B, 70B |
| Qwen3 | 8B, 14B, 32B, MoE 30B, MoE 235B, VL 235B |
| GPT-OSS | 20B, 120B |
| Mistral 3 Large | 675B |

## 支持的投机解码算法

| 算法 | 说明 |
|------|------|
| EAGLE (v1/v2) | 基于单层 Transformer + 特征融合的投机器 |
| EAGLE3 (v3) | 多层 Transformer + 词表映射 + TTT 的投机器 |
| HASS | EAGLE 变体，fusion layer 带 bias |
| Independent | 独立模型包装器 |
| MLP | 基于多层感知机的投机器（配置阶段） |
