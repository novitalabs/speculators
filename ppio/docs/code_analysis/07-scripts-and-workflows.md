# 脚本与工作流

## 1. 训练脚本 (`scripts/train.py`)

### 使用方式

```bash
# 单 GPU 训练
python scripts/train.py \
    --verifier-model meta-llama/Llama-3.1-8B-Instruct \
    --datapath ./data/hidden_states/ \
    --save-path ./checkpoints/ \
    --lr 5e-4 \
    --num-epochs 3 \
    --batch-max-length 8192

# 多 GPU 分布式训练
torchrun --nproc_per_node=4 scripts/train.py \
    --verifier-model meta-llama/Llama-3.1-8B-Instruct \
    --datapath ./data/hidden_states/ \
    --save-path ./checkpoints/ \
    --lr 5e-4 \
    --num-epochs 3
```

### 完整参数列表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--verifier-model` | 必需 | 验证器模型路径 |
| `--datapath` | 必需 | 训练数据目录 |
| `--save-path` | 必需 | 检查点保存路径 |
| `--lr` | 5e-4 | 学习率 |
| `--num-epochs` | 3 | 训练 epoch 数 |
| `--batch-max-length` | 8192 | 每批次最大 token 数 |
| `--seed` | 42 | 随机种子 |
| `--resume-from-checkpoint` | False | 从检查点恢复 |
| `--ttt-steps` | 5 | TTT 步数 |
| `--ttt-decay` | 0.8 | TTT 损失衰减 |
| `--noise-type` | None | 噪声类型 (gaussian/uniform) |
| `--noise-std` | 0.05 | 噪声标准差 |
| `--turn-dropout` | False | 对话轮次丢弃 |
| `--scheduler-type` | "none" | 调度器 (linear/cosine/none) |
| `--scheduler-warmup-steps` | 自动 | 预热步数 |
| `--loggers` | "" | 日志后端 (tensorboard,wandb,trackio) |
| `--run-name` | None | 实验名称 |
| `--num-workers` | 4 | DataLoader 工作线程 |
| `--vocab-mapping-path` | None | 词表映射路径 |
| `--off-policy-tokens` | False | 使用 off-policy token |

## 2. 离线数据生成脚本 (`scripts/data_generation_offline.py`)

### 使用方式

```bash
python scripts/data_generation_offline.py \
    --target-model meta-llama/Llama-3.1-8B-Instruct \
    --train-data-path sharegpt \
    --output-dir ./data/hidden_states/ \
    --seq-length 2048 \
    --max-samples 5000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8
```

### 完整参数列表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--target-model` | 必需 | 目标模型路径 |
| `--train-data-path` | 必需 | 训练数据 (sharegpt/ultrachat/本地文件) |
| `--output-dir` | 必需 | 输出目录 |
| `--seq-length` | 2048 | 最大序列长度 |
| `--max-samples` | None | 最大样本数 |
| `--tensor-parallel-size` | 1 | 张量并行度 |
| `--gpu-memory-utilization` | 0.8 | GPU 显存利用率 |
| `--layer-ids` | 自动 | 提取的层 ID |
| `--batch-size` | 32 | 批次大小 |
| `--seed` | 42 | 随机种子 |
| `--turn-dropout` | False | 对话轮次丢弃 |
| `--cache-dir` | None | 预处理缓存目录 |

## 3. 端到端脚本 (`scripts/gen_and_train.py`)

### 使用方式

```bash
python scripts/gen_and_train.py \
    --target-model meta-llama/Llama-3.1-8B-Instruct \
    --datasets sharegpt \
    --output-dir ./output/ \
    --seq-length 2048 \
    --max-samples 5000 \
    --lr 5e-4 \
    --num-epochs 3
```

### 流程

```
gen_and_train.py
  │
  ├── Phase 1: 数据生成
  │   for each dataset:
  │     python data_generation_offline.py --target-model X --train-data-path Y
  │
  ├── Phase 2: 词表映射 (如果指定)
  │   combine_token_frequency_distributions(all_token_freq_files)
  │   build_vocab_mappings_from_distribution(combined, draft_vocab_size)
  │
  └── Phase 3: 训练
      torchrun --nproc_per_node=N train.py \
          --verifier-model X \
          --datapath output/hidden_states/ \
          --save-path output/checkpoints/
```

## 4. 词表映射构建脚本 (`scripts/build_vocab_mapping.py`)

```bash
python scripts/build_vocab_mapping.py \
    --token-freq-path ./data/token_freq.pt \
    --draft-vocab-size 32000 \
    --target-vocab-size 128000 \
    --output-path ./vocab_mapping/
```

## 5. CLI 命令 (`speculators`)

### 转换命令

```bash
# 基本转换
speculators convert path/to/model --algorithm eagle

# 带验证器的转换
speculators convert path/to/model \
    --algorithm eagle3 \
    --verifier meta-llama/Llama-3-8B

# 带验证的转换
speculators convert path/to/model \
    --algorithm eagle \
    --verifier meta-llama/Llama-3-8B \
    --validate-device cuda:0

# 查看版本
speculators --version
```

## 6. CI/CD 工作流

### 开发工作流 (`development.yml`) — PR 触发

```
PR 提交
  │
  ├── Link Checks           # 检查文档和仓库链接
  │
  ├── Quality Checks         # Python 3.10, 3.13
  │   ├── ruff check         # 代码风格
  │   ├── mdformat --check   # Markdown 格式
  │   └── mypy               # 类型检查
  │
  ├── Unit Tests             # Python 3.10, 3.13 (GPU)
  │   └── tox run test-unit
  │
  ├── Integration Tests      # Python 3.10, 3.13 (GPU)
  │   └── tox run test-integration
  │
  ├── DataGen Tests          # Python 3.10 (GPU)
  │   └── tox run test-datagen
  │
  └── Build                  # 构建包并上传 artifact
      └── python -m build
```

### 主分支工作流 (`main.yml`) — Push 触发

与开发工作流类似，但不包含 Build 步骤。

### 硬件环境

- GPU Runner: `ibm-wdc-k8s-vllm-h100-solo` (H100)
- 环境变量: `UV_TORCH_BACKEND: auto`

## 7. 测试框架

### 测试命令

```bash
# 单元测试
tox run -e test-unit

# 集成测试
tox run -e test-integration

# 数据生成测试
tox run -e test-datagen

# 端到端测试
tox run -e test-e2e

# 代码质量
make quality
```

### 测试标记

```python
@pytest.mark.smoke      # 快速基本功能测试
@pytest.mark.sanity     # 主要功能详细测试
@pytest.mark.regression # 防止回归的测试
```

## 8. 开发工具

### Makefile 命令

```bash
make quality    # 运行所有质量检查 (ruff + mdformat + mypy)
make style      # 自动修复代码风格问题
```

### Ruff 配置摘要

- 行长度: 88
- 规则集: E, W, A, C, COM, ERA, I, ICN, N, NPY, PD, PT, PTH, Q, TCH, RUF022
- 测试文件豁免: 允许更宽松的命名和断言

### MyPy 配置

- Python 目标: 3.10
- 忽略的第三方库: datasets, transformers, setuptools, vllm

## 9. 示例工作流

### 完整训练示例 (Llama3 8B)

```python
# examples/data_generation_and_training/llama3_8b_sharegpt_5k.py
from speculators.scripts import gen_and_train

gen_and_train.run_e2e(
    target_model="meta-llama/Llama-3.1-8B-Instruct",
    datasets=["sharegpt"],
    max_samples=5000,
    seq_length=2048,
    lr=5e-4,
    num_epochs=3,
)
```

### 评估工作流

```bash
# examples/evaluate/eval-guidellm/
# 使用 GuideLLM 进行推理性能评估
# 包含日志解析脚本
python scripts/parse_logs.py --log-dir ./logs/
```
