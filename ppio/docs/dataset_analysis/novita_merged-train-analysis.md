# novita_merged/train.jsonl 数据集分析报告

## 基本信息

| 项目 | 值 |
|------|-----|
| 总条数 | 724,196 |
| 文件大小 | 11 GB |
| 平均 assistant 轮次 | 14.1 轮 |
| 含 `<think>` 标签 | ~10.6%（~76,765 条） |
| 分析方法 | 每 72 条取样 1 条，共抽样 10,000 条 |

> 与 eval.jsonl 的关系：train 规模为 eval 的 **122 倍**，场景分类结构相似。

---

## 数据结构

- **多轮对话**：全部为多轮，角色为 `system` / `user` / `assistant`
- **无独立 `tool` 角色**：工具调用以自然语言描述或代码块形式嵌入 assistant 消息
- **用户消息格式**主要有三类：
  - `<issue>` 标签：GitHub issue 描述
  - `<task>` 标签：软件工程任务描述
  - 无标签：OpenClaw 对话、PR 测试、其他

---

## 场景分类与规模估算

基于 10k 样本推算全量分布：

| 场景 | 样本数 | 估算全量 | 占比 |
|------|--------|----------|------|
| GitHub Issue 修复智能体 | 4,576 | ~331,000 | 46% |
| 通用软件工程智能体（`<task>`）| 4,180 | ~303,000 | 42% |
| PR 测试环境配置 | 378 | ~27,000 | 4% |
| OpenClaw 个人助理 | 343 | ~25,000 | 3% |
| Kilo Code | 120 | ~8,700 | 1% |
| opencode | 84 | ~6,100 | 1% |
| Blackbox | 34 | ~2,500 | <1% |
| Roo | 24 | ~1,700 | <1% |
| Cline | 15 | ~1,100 | <1% |
| 其他（杂项）| 213 | ~15,000 | 2% |

---

## 各场景详细说明

### 1. GitHub Issue 修复智能体（~331,000 条，46%）

**System prompt 特征**：
> "You are an expert AI software engineering agent. Your primary goal is to resolve a given GitHub issue..."

**用户消息格式**：`<issue>Please solve following issue: {title}\n{body}</issue>`

**Issue 类型分布**（样本估算）：

| 类型 | 占比（issue 样本内） |
|------|---------------------|
| Bug 修复 | 13% |
| 功能新增 | 13% |
| testbed 仓库任务 | 10% |
| 重构 / 清理 | 2% |
| 文档 / 拼写 | 2% |
| 性能优化 | <1% |
| 其他 | 60% |

**平均 assistant 轮次**：12.2（中位数 9，最大 658）

---

### 2. 通用软件工程智能体 - `<task>` 类型（~303,000 条，42%）

**System prompt 特征**：
> "You are an expert AI software engineering agent."（含 thinking 格式要求）

与类别 1 不同，任务通过 `<task>` 标签描述，更侧重**主动式工程任务**而非 issue 修复。

**任务类型细分**：

| 任务类型 | 样本数 | 描述 |
|----------|--------|------|
| 功能新增 | 1,367 | 实现新功能、API、接口 |
| 代码理解 | 1,087 | 解释模块、梳理设计、审计行为覆盖 |
| 重构 / 清理 | 788 | 去重、现代化、可读性提升 |
| 测试编写 | 354 | 单元测试、覆盖率提升 |
| Bug 排查 | 52 | 分析 CI 失败、复现问题 |
| 风格 / 格式化 | 47 | PEP8、flake8 等格式整理 |
| 自动化脚本 | 30 | 发布流程、部署脚本 |
| 文档 | 26 | README、注释维护 |
| 安全审计 | 5 | 格式注入检查、漏洞扫描 |

**常见 testbed 仓库**（按出现频次）：

| 仓库 | 频次 | 仓库 | 频次 |
|------|------|------|------|
| astropy | 701 | xarray | 230 |
| seaborn | 356 | flask | 169 |
| matplotlib | 305 | pytest | 148 |
| django | 279 | sympy | 112 |
| requests | 236 | sphinx | 90 |

**平均 assistant 轮次**：9.5（中位数 6，最大 244）

---

### 3. PR 测试环境配置智能体（~27,000 条，4%）

**System prompt 特征**：
> "You are a repository maintainer responsible for ensuring that new pull requests can be properly tested."

**任务**：给定仓库名、commit SHA 和测试文件路径，配置可运行的测试环境

**代表仓库**：`django/django`、`astral-sh/ruff`、`gohugoio/hugo` 等

**平均 assistant 轮次**：10.8（中位数 9，最大 52）

---

### 4. OpenClaw 个人助理（~25,000 条，3%）

**特点**：
- 定时 cron 任务执行（如 `editorial-review`、`database-backup-hourly`）
- 文件读写与传输
- 上下文压缩（含 `<summary>` 标签历史摘要）
- 心跳检查协议（`HEARTBEAT.md`）
- 多语言混用（法语/中文/英语）
- 最长对话达 1,406 assistant 轮次

---

### 5. IDE 编程智能体品牌（~22,000 条合计）

| 品牌 | 估算条数 | 特点 |
|------|----------|------|
| Kilo Code | ~8,700 | 多框架 SWE 智能体，带 Markdown 规则 |
| opencode | ~6,100 | 交互式 CLI 编程助手 |
| Blackbox | ~2,500 | Blackbox 开发的 CLI 编程智能体 |
| Roo | ~1,700 | 软件工程智能体 |
| Cline | ~1,100 | 软件工程智能体 |

平均 assistant 轮次：32.0（对话较长，中位数 15）

---

### 6. 杂项场景（~15,000 条，2%）

包含多种特殊用途智能体：

| 场景 | 描述 |
|------|------|
| 日志 / Dockerfile 分析 | 验证测试是否正确执行，构建 Dockerfile 环境 |
| 多 Agent 编排（Builder Agent）| 设计 agentic 工作流、Blueprint 实现 |
| 神经网络电路解释器 | 分析神经网络归因图（`cir` CLI 工具） |
| 多跳问答校验 | 验证推理链是否正确 |
| Android Java 代码分析 | 逐类分析 Android 源码 |
| 量化金融助手（中文）| 量化策略、多空建议、因子投资（中文） |
| Norma B2B 助理（法语）| Google Workspace 管理、B2B 开发 |
| nanobot（Android 终端）| Termux 环境下的通用助手 |
| Red team 漏洞验证 | 确认漏洞是否可利用（需明确授权语境） |
| 创意写作 Spicy Writer | 角色扮演写作 |
| GitHub Copilot 风格助手 | VS Code 内编程助手 |
| Cursor AI 助手 | Cursor IDE 内 minimax 模型对话 |
| Claude Code 风格助手 | CLI 交互式编程（极少量） |
| 客服智能体 | 基于 policy 文档的客服应答 |
| 化学 / 生物学 | 逆合成反应预测 |

---

## 与 eval.jsonl 的对比

| 维度 | train.jsonl | eval.jsonl |
|------|-------------|------------|
| 总条数 | 724,196 | 5,920 |
| 文件大小 | 11 GB | 116 MB |
| 平均 assistant 轮次 | 14.1 | 17.9 |
| 含 `<think>` 覆盖率 | 10.6% | 17.4% |
| 主要差异 | 含大量 `<task>` 格式任务 | 以 issue 修复为主 |
| `<task>` 类型 | ~303,000 条（42%） | 较少 |
| 对话最大长度 | 1,406 轮 | 1,413 轮 |

---

## 总结

`novita_merged/train.jsonl` 是一个**大规模多场景智能体训练轨迹数据集**，核心构成：

1. **GitHub Issue 修复**（46%）：SWE-bench 风格的真实 issue 修复轨迹
2. **主动式工程任务**（42%）：代码理解、功能新增、重构、测试等，涵盖 astropy/django/matplotlib 等主流开源仓库
3. **PR 测试配置**（4%）：自动化 CI 环境搭建
4. **个人助理 / 工作区助理**（3%）：OpenClaw 定时任务、文件管理等长上下文场景
5. **杂项特殊场景**（2%）：量化金融、多语言、Android 分析等垂直领域

---

## 场景样例索引

各场景的完整对话样例见 → **[novita_merged-train-samples.md](./novita_merged-train-samples.md)**

| # | 场景 | 锚点 |
|---|------|------|
| 1 | GitHub Issue 修复智能体 | [§1](./novita_merged-train-samples.md#1-github-issue-修复智能体) |
| 2 | 通用工程任务 — 功能新增 | [§2](./novita_merged-train-samples.md#2-通用工程任务--功能新增) |
| 3 | 通用工程任务 — 代码理解 | [§3](./novita_merged-train-samples.md#3-通用工程任务--代码理解) |
| 4 | 通用工程任务 — 重构 / 清理 | [§4](./novita_merged-train-samples.md#4-通用工程任务--重构--清理) |
| 5 | 通用工程任务 — 测试编写 | [§5](./novita_merged-train-samples.md#5-通用工程任务--测试编写) |
| 6 | PR 测试环境配置 | [§6](./novita_merged-train-samples.md#6-pr-测试环境配置) |
| 7 | OpenClaw 个人助理 | [§7](./novita_merged-train-samples.md#7-openclaw-个人助理) |
| 8 | Kilo Code 编程智能体 | [§8](./novita_merged-train-samples.md#8-kilo-code-编程智能体) |
| 9 | opencode CLI 助手 | [§9](./novita_merged-train-samples.md#9-opencode-cli-助手) |
| 10 | Blackbox 编程智能体 | [§10](./novita_merged-train-samples.md#10-blackbox-编程智能体) |
| 11 | 日志 / Dockerfile 分析智能体 | [§11](./novita_merged-train-samples.md#11-日志--dockerfile-分析智能体) |
| 12 | 多 Agent 编排（Builder Agent）| [§12](./novita_merged-train-samples.md#12-多-agent-编排builder-agent) |
| 13 | 神经网络电路解释器 | [§13](./novita_merged-train-samples.md#13-神经网络电路解释器) |
| 14 | 量化金融助手（中文）| [§14](./novita_merged-train-samples.md#14-量化金融助手中文) |
| 15 | 客服智能体 | [§15](./novita_merged-train-samples.md#15-客服智能体) |
| 16 | 化学 / 生物逆合成智能体 | [§16](./novita_merged-train-samples.md#16-化学--生物逆合成智能体) |
