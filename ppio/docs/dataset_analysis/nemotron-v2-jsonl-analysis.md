# Nemotron-v2-jsonl 数据集分析报告

## 文件列表

```
nemotron-v2-jsonl/
├── chat.jsonl
├── code.jsonl
├── math.jsonl
├── multilingual.jsonl
├── multilingual_de.jsonl
├── multilingual_es.jsonl
├── multilingual_fr.jsonl
├── multilingual_it.jsonl
├── multilingual_ja.jsonl
└── stem.jsonl
```

## 数据规模统计

| 文件 | 总条数 | 含 `<think>` | 覆盖率 |
|------|--------|--------------|--------|
| `chat.jsonl` | 627,720 | 627,532 | ~100% |
| `code.jsonl` | 175,000 | 175,000 | 100% |
| `math.jsonl` | 239,467 | 239,467 | 100% |
| `stem.jsonl` | 355,000 | 288,000 | 81% |
| `multilingual_de.jsonl` | 1,015,314 | 1,015,314 | 100% |
| `multilingual_es.jsonl` | 935,704 | — | — |
| `multilingual_fr.jsonl` | 1,001,504 | — | — |
| `multilingual_it.jsonl` | 1,016,503 | — | — |
| `multilingual_ja.jsonl` | 975,202 | 975,202 | 100% |
| `multilingual.jsonl` | 100 | — | — |

## 数据结构特点

- **全部为单轮对话**：turn distribution 均为 `{1: N}`，无多轮对话
- **无非空 system prompt**：所有条目的 system 字段均为空字符串
- **统一格式**：每条数据结构为 `{"conversations": [{"role": "system", "content": ""}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`
- **CoT 推理标签**：大多数 assistant 回答包含 `<think>...</think>` 推理过程（chain-of-thought）

---

## 各文件使用场景

### chat.jsonl — 通用开放域对话

- 创意写作：故事、童谣、角色设计与融合
- 知识问答：历史、地理、时事等通用问答
- 写作辅助：邮件撰写、社交媒体文案、产品描述
- 法律 / HR 咨询
- 网页与游戏代码生成（HTML/CSS/JS、Lua、Bevy 等）
- AI 图像提示词生成（Midjourney prompt）
- 角色扮演场景

> 注：~188 条无 `<think>` 标签，其余全部含推理过程（内容可为空）

---

### math.jsonl — 数学推理

- 定积分、微积分
- 函数方程（functional equations）
- 数论与组合数学
- 答案统一格式：`\boxed{答案}`
- 所有回答均含完整 CoT 推理过程

---

### stem.jsonl — STEM 多选题

覆盖学科：
- **医学**：外科术式选择、睡眠障碍、抗药性机制
- **物理**：弹簧-质量振动系统、电容器、Schottky 效应、主动振动控制
- **化学**：缓冲液原理、催化裂化、光学活性
- **生物**：细菌耐药机制
- **法学**：法律解释原则（stare decisis）
- **信息安全**：网络安全准备策略、数据隐私保护
- **经济**：油价影响因素分析

> 约 81% 的条目含 `<think>` 推理过程，约 19% 直接给出解释（无推理标签）

---

### code.jsonl — 算法竞赛编程

题目风格类似 Codeforces / LeetCode 竞赛题，涵盖：
- 数组操作与子数组问题
- 字符串处理
- 图论与树结构（Cartesian Tree 等）
- 博弈论问题
- 二进制 / 位运算
- 贪心、动态规划

所有回答含完整 CoT 推理过程。

---

### multilingual_*.jsonl — 多语言版本

各语言内容侧重：

| 语言 | 文件 | 主要场景 |
|------|------|----------|
| 德语 | `multilingual_de.jsonl` | 数学题 + 代码（memoization 等）+ STEM 计算 |
| 西班牙语 | `multilingual_es.jsonl` | 数学 + Python 编程 + STEM（化学计量学等） |
| 法语 | `multilingual_fr.jsonl` | 以数学为主 + 代码 |
| 意大利语 | `multilingual_it.jsonl` | 代码（异常处理、排序）+ 数学 + 算法题 |
| 日语 | `multilingual_ja.jsonl` | 数学 + STEM（扩散系数等物理计算） |

### multilingual.jsonl — 多语言小样本

仅 100 条，内容与各语言文件相同，为小规模验证样本。

---

## 适用训练目标

该数据集整体适合训练：

1. **指令跟随**（Instruction Following）：覆盖通用对话、代码生成、问答等多类任务
2. **链式思考推理**（Chain-of-Thought, CoT）：绝大多数数据含 `<think>` 显式推理过程
3. **多语言能力**：德/西/法/意/日五种语言各超 90 万条
4. **数学与科学推理**：数学、STEM、算法题合计约 77 万条

---

## 场景样例索引

各场景的代表性对话样例见 → **[nemotron-v2-jsonl-samples.md](./nemotron-v2-jsonl-samples.md)**

| # | 场景 | 来源文件 | 锚点 |
|---|------|----------|------|
| 1 | chat — 创意写作 | `chat.jsonl` | [§1](./nemotron-v2-jsonl-samples.md#1-chatjsonl--创意写作) |
| 2 | chat — 知识问答 | `chat.jsonl` | [§2](./nemotron-v2-jsonl-samples.md#2-chatjsonl--知识问答) |
| 3 | chat — 写作辅助 | `chat.jsonl` | [§3](./nemotron-v2-jsonl-samples.md#3-chatjsonl--写作辅助) |
| 4 | chat — 法律 / HR 咨询 | `chat.jsonl` | [§4](./nemotron-v2-jsonl-samples.md#4-chatjsonl--法律--hr-咨询) |
| 5 | chat — 代码生成 | `chat.jsonl` | [§5](./nemotron-v2-jsonl-samples.md#5-chatjsonl--代码生成) |
| 6 | chat — AI 图像提示词 | `chat.jsonl` | [§6](./nemotron-v2-jsonl-samples.md#6-chatjsonl--ai-图像提示词midjourney) |
| 7 | chat — 角色扮演 | `chat.jsonl` | [§7](./nemotron-v2-jsonl-samples.md#7-chatjsonl--角色扮演) |
| 8 | 数学推理 | `math.jsonl` | [§8](./nemotron-v2-jsonl-samples.md#8-mathjsonl--数学推理) |
| 9 | STEM 多选题（含 CoT） | `stem.jsonl` | [§9](./nemotron-v2-jsonl-samples.md#9-stemjsonl--stem-多选题含-cot) |
| 10 | STEM 多选题（直接回答） | `stem.jsonl` | [§10](./nemotron-v2-jsonl-samples.md#10-stemjsonl--stem-多选题直接回答) |
| 11 | 算法竞赛编程 | `code.jsonl` | [§11](./nemotron-v2-jsonl-samples.md#11-codejsonl--算法竞赛编程) |
| 12 | 多语言 — 德语 | `multilingual_de.jsonl` | [§12](./nemotron-v2-jsonl-samples.md#12-multilingual_dejsonl--德语) |
| 13 | 多语言 — 西班牙语 | `multilingual_es.jsonl` | [§13](./nemotron-v2-jsonl-samples.md#13-multilingual_esjsonl--西班牙语) |
| 14 | 多语言 — 法语 | `multilingual_fr.jsonl` | [§14](./nemotron-v2-jsonl-samples.md#14-multilingual_frjsonl--法语) |
| 15 | 多语言 — 意大利语 | `multilingual_it.jsonl` | [§15](./nemotron-v2-jsonl-samples.md#15-multilingual_itjsonl--意大利语) |
| 16 | 多语言 — 日语 | `multilingual_ja.jsonl` | [§16](./nemotron-v2-jsonl-samples.md#16-multilingual_jajsonl--日语) |
| 17 | 多语言混合小样本 | `multilingual.jsonl` | [§17](./nemotron-v2-jsonl-samples.md#17-multilingualJsonl--多语言混合小样本) |
