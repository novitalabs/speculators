# Speculators 项目分析文档

本文档目录包含对 [Speculators](https://github.com/redhat-et/speculators) 项目的全面技术分析。

## 文档索引

| 文档 | 内容 |
|------|------|
| [项目概览](01-project-overview.md) | 项目简介、技术栈、目录结构、依赖关系 |
| [核心架构](02-core-architecture.md) | 注册表系统、配置体系、模型抽象层 |
| [模型实现](03-model-implementations.md) | EAGLE、EAGLE3、Independent、MLP 各模型详解 |
| [训练系统](04-training-system.md) | Trainer、数据加载、分布式训练、Checkpoint |
| [数据生成](05-data-generation.md) | vLLM 隐藏状态提取、预处理、词表映射 |
| [模型转换](06-model-conversion.md) | EAGLE/EAGLE3 转换器、权重映射 |
| [脚本与工作流](07-scripts-and-workflows.md) | 端到端训练流程、CLI 命令、CI/CD |
| [设计模式与扩展](08-design-patterns.md) | 关键设计模式、如何添加新算法 |

## 项目一句话总结

Speculators 是一个统一的投机解码 (Speculative Decoding) 库，用于创建、表示和存储适用于 vLLM 等 LLM 推理引擎的投机解码算法和模型。
