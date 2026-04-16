# novita_merged/eval.jsonl 数据集分析报告

## 基本信息

| 项目 | 值 |
|------|-----|
| 总条数 | 5,920 |
| 文件大小 | 116 MB |
| 平均总轮次 | 23.2 轮 |
| 最大轮次 | 1,413 轮 |
| 最小轮次 | 3 轮 |
| 含 `<think>` 标签 | 1,030 条（17.4%） |

## 数据结构

- **多轮对话**：与 nemotron 不同，本数据集全部为多轮（system + user/assistant 交替）
- **角色类型**：`system`、`user`、`assistant`（无独立 `tool` 角色）
- **工具调用形式**：工具调用以自然语言描述嵌入 assistant 消息中，不使用结构化 JSON/XML 格式
- **无 system prompt** 的条目：13 条

---

## 场景分类

### 1. GitHub Issue 修复智能体（2,335 条）

**System prompt 特征**：  
> "You are an expert AI software engineering agent. Your primary goal is to resolve a given GitHub issue..."

**任务描述**：  
给定真实 GitHub issue，模型需要浏览代码库、定位根因、实现修复并确保改动安全。

**Issue 类型分布**（按关键词粗略分类）：
| 类型 | 条数 |
|------|------|
| Bug 修复 | 302 |
| 功能新增 | 214 |
| 重构 / 清理 | 30 |
| 内存 / Crash | 27 |
| 性能优化 | 4 |
| 其他 | 1,584 |

**典型 Issue 示例**：
- `Memory-leak in fetch API`
- `db.Not method when used with multiple Where conditions has changed unexpectedly`
- `Custom asymmetric expect matchers aren't able to print Symbol arguments`
- `Uploading files from Apple Photos fails`
- `Implement Chargebacks API resource`

---

### 2. 通用软件工程智能体（1,982 条）

**System prompt 特征**：  
> "You are an expert AI software engineering agent."（不含"resolve a GitHub issue"明确指令）

与类别 1 类似，但任务描述更宽泛，包含直接操作 Python testbed 代码仓库的任务（约占全数据集 56%，即 3,353 条含 `testbed` 或 `I have access to a python code repository`）。

---

### 3. PR 测试环境配置智能体（483 条）

**System prompt 特征**：  
> "You are a repository maintainer responsible for ensuring that new pull requests can be properly tested."

**任务**：
- 给定 PR 的 commit SHA 和目标测试文件，配置环境使测试能够运行
- 代表性仓库：`django/django`、其他开源项目

**示例任务**：
```
Target repository name: django/django
Commit SHA: 91e3d1215b081289f435e6ae1967ad85db2a9f52
Version: 1.7
Target test files: tests/forms_tests/tests/test_forms.py
```

---

### 4. OpenClaw 个人助理（313 条）

**System prompt 特征**：  
> "You are a personal assistant running inside OpenClaw."

**可用工具**：`read`、`write` 等文件操作工具

**任务特点**：
- 定时任务（cron job）执行，如 `[cron:xxxx editorial-review] Process the editorial queue.`
- 文件管理与传输（发送 dashboard 文件给用户）
- 数据库备份监控
- 多语言（法语、中文混用）
- 上下文压缩（含 `<summary>` 标签的历史摘要）
- 心跳检查协议（`HEARTBEAT.md`）

---

### 5. 其他编程智能体品牌（~320 条合计）

| 品牌 | 条数 | 描述 |
|------|------|------|
| Kilo Code | 159 | 多框架软件工程智能体 |
| opencode | 80 | 交互式 CLI 编程助手 |
| Blackbox | 29 | 软件工程专用 CLI 智能体 |
| Roo | 29 | 软件工程智能体 |
| Cline | 12 | 软件工程智能体 |
| Deep Agent | 12 | 通用工具调用智能体 |

---

### 6. 通用编程智能体（163 条）

**System prompt 特征**：  
> "You are an AI coding agent. You operate in a workspace with a provided codebase."

任务以代码修改和功能实现为主，如：
- "During parsing, show the exact status that comes from the backend instead of hardcoding date variables."

---

### 7. 客服智能体（20 条）

**System prompt 特征**：  
> "You are a customer service agent that helps the user according to the `<policy>` provided below."

通过 policy 文档驱动，回答用户询问或发起操作（如退款、改期等）。

---

### 8. 化学 / 生物学智能体（8 条）

**任务**：给定产物，预测合成反应物（逆合成分析）。

---

### 9. 创意写作智能体（7 条）

品牌：Spicy Writer，定向生成特定风格的创意写作内容。

---

### 10. OpenHands / SUPERVISOR 等（约 40 条）

- **OpenHands**：通用计算机交互型智能体，涵盖代码修改、命令执行、网页查询等
- **SUPERVISOR**：多子智能体编排，VSCode 本地模式

---

## 与 nemotron-v2 数据集的对比

| 维度 | nemotron-v2-jsonl | novita_merged/eval.jsonl |
|------|-------------------|--------------------------|
| 总规模 | ~340 万条 | 5,920 条 |
| 对话类型 | 单轮（全部） | 多轮（全部） |
| System prompt | 全空 | 丰富多样（603 种） |
| 工具调用 | 无 | 有（自然语言形式） |
| `<think>` 覆盖 | 约 80%+ | 17.4% |
| 主要用途 | SFT 预训练数据 | 智能体轨迹 / Eval 数据 |
| 场景多样性 | 分类清晰（5 大类） | 高度多样（10+ 场景） |

## 总结

`novita_merged/eval.jsonl` 是一个**多智能体场景的评估轨迹数据集**，核心用途是训练或评估能够在真实工程环境中自主行动的 AI 智能体，主要包括：

1. **代码仓库修复**（主体，约 73%）：SWE-bench 风格的 GitHub issue 修复
2. **CI/CD 环境配置**（8%）：PR 测试环境搭建
3. **个人工作区助理**（5%）：文件管理、定时任务、多语言交互
4. **专业领域问答**（1%）：化学逆合成等垂直场景
