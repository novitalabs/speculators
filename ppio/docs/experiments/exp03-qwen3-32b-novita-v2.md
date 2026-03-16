# Experiment 3: Qwen3-32B + Novita Online Data (v2 - HuggingFace dataset)

Using larger dataset from `weilan55/novita20260309` (HuggingFace).

- **Node**: .18 (8x H200 143GB)
- **Image**: speculators:v0.16.0
- **Data**: HF dataset `weilan55/novita20260309`
  - Source: 1.5GB tar.gz → 8.6GB JSON (799622 log records, 56016 with request_body)
  - Conversion: `k8s/convert_novita_logs.py` → 52323 conversations → `novita_v2_sharegpt.jsonl`
  - Sampled 5000 for training (consistent with baseline experiments)
- **Config**:
  - target_vocab_size=151936, draft_vocab_size=32000
  - seq_len=8192, lr=3e-5, epochs=10, 8 GPUs FSDP
  - Data gen: TP=2
- **K8s**: `k8s/k8s-qwen3-32b-eagle3-novita.yaml` (pod: `qwen3-32b-eagle3-novita-v2`)
- **Status**: IN PROGRESS — data generation (~28%, 1440/5000)

## Data Generation Performance Analysis

当前数据生成是整个 pipeline 的瓶颈。以 Qwen3-32B + TP=2 为基准：

| 指标 | 值 |
|------|-----|
| 当前配置 | TP=2 (2 GPUs), batch_size=8, seq_len=8192 |
| 生成速度 | ~13 samples/min |
| 每样本输出大小 | 平均 262 MB（含 4 层 hidden states, bf16） |
| 空闲 GPU | 6/8（TP=2 仅用 2 张卡） |

**时间估算：**

| 数据量 | TP=2 (当前) | TP=2 x4 并行 (理论) |
|--------|-------------|---------------------|
| 5,000  | **6.4 小时** | 1.6 小时 |
| 10,000 | 12.8 小时 | 3.2 小时 |
| 52,323 (全量) | **66.8 小时 (~2.8天)** | 16.7 小时 |

**磁盘估算：**

| 数据量 | 生成数据 | 训练 checkpoint (10 epochs) | 合计 |
|--------|----------|----------------------------|------|
| 5,000  | ~1.3 TB  | ~26 GB | ~1.3 TB |
| 10,000 | ~2.6 TB  | ~26 GB | ~2.6 TB |
| 52,323 (全量) | **~13.7 TB** | ~26 GB | **~13.7 TB** |
| .18 可用 | - | - | 5.6 TB |

> 全量 52323 样本的数据生成将 **超出 .18 磁盘容量**（需 13.7 TB > 可用 5.6 TB）。

**加速方案：**

1. **多实例并行 datagen**：当前 TP=2 只用 2 张 GPU，剩余 6 张空闲。可以启动 4 个 TP=2 的 vLLM 实例，每个处理数据的 1/4，理论提速 4x。需要修改 `data_generation_offline.py` 支持 `--shard-id` / `--num-shards` 参数，或手动切分输入数据后并行运行。
2. **提高 TP**：TP=4 或 TP=8 可能不会提速（受限于 prefill 计算，且 TP=8 有 segfault 问题），但可以增大 `--gpu-memory-utilization` 和 `--batch-size` 来提高吞吐。
3. **减少 hidden states 层数**：当前采集 4 层 ([2, 32, 61, 63])，减到 2 层可以减少 ~50% 磁盘和 I/O。
4. **降低 seq_length**：从 8192 降到 4096 可以大约减半时间和磁盘（但影响训练质量）。
5. **跨节点并行**：在 .23 或其他节点上同时运行 datagen（需要模型和数据都在对应节点上）。

**建议策略：**
- 5000 样本实验：当前配置即可（~6.4h），磁盘 1.3 TB 可接受
- 全量实验：必须使用多实例并行 + 可能需要跨节点，或者限制在 ~20000 样本以内（~5.2 TB）以适应磁盘
