# Production-Serving Online Buffer: Design Document

## Motivation

当前的 speculative decoding 训练流程是**离线批处理**：

```
[离线] datagen 节点跑 vLLM prefill → 生成 .pt 文件 → rsync 到训练节点 → 训练
```

这有两个核心浪费：
1. **算力浪费**：生产集群每天处理数百万请求，prefill 阶段本就会产生 hidden_states，但这些中间结果被丢弃
2. **数据时效性**：离线 datagen 使用固定数据集，无法反映真实请求分布的变化

理想状态是生产集群在正常服务的同时，将 prefill 产生的 hidden_states 作为训练数据旁路输出，形成持续的在线学习闭环：

```
[在线] 生产 vLLM 服务请求 → 旁路捕获 hidden_states → 写入 buffer → 训练节点消费
```

## Current System Review

现有系统已经具备的能力：

| 组件 | 文件 | 功能 |
|------|------|------|
| Hidden states 捕获 | `custom_worker.py` | 通过 monkey-patch vLLM worker 的 forward，在 prefill 时捕获指定层的 hidden_states |
| Manifest 协议 | `manifest.py` | 原子性的文件注册表，支持 train_count 追踪 |
| 流式训练 | `train_streaming.py` | 按 manifest 发现新文件，epoch 级增量加载 |
| Buffer 清理 | `buffer_cleanup.py` | 按 train_count 淘汰，epoch lock 防竞态 |
| 数据同步 | `sync_datagen.sh` | rsync + size gate + manifest 实时更新 |

**关键限制**：
- `VllmHiddenStatesGenerator` 是独立进程，不能嵌入 serving engine
- 当前 .pt 文件格式是单样本 `{input_ids, hidden_states, loss_mask}`，没有批量打包
- Manifest 只支持本地文件发现，不支持远程 push
- 没有数据质量过滤（生产请求可能包含低质量 prompt）

## Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────┐
│                  Production Cluster                      │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ vLLM Node A  │  │ vLLM Node B  │  │ vLLM Node C  │  │
│  │              │  │              │  │              │  │
│  │  Serving     │  │  Serving     │  │  Serving     │  │
│  │  + Emitter   │  │  + Emitter   │  │  + Emitter   │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │ .pt files       │                  │          │
│         ▼                 ▼                  ▼          │
│  ┌─────────────────────────────────────────────────┐    │
│  │           Local Buffer (per-node)                │    │
│  │  /data/buffer/prod_{node}_{idx}.pt               │    │
│  │  manifest.json (local)                           │    │
│  │  MAX_SIZE: configurable per node                 │    │
│  └──────────────────────┬──────────────────────────┘    │
└─────────────────────────┼───────────────────────────────┘
                          │ rsync / object store
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  Training Cluster                        │
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │          Aggregated Buffer                       │    │
│  │  Merges manifests from all production nodes      │    │
│  │  + offline datagen output                        │    │
│  │  Unified manifest.json                           │    │
│  └──────────────────────┬──────────────────────────┘    │
│                         ▼                                │
│  ┌──────────────────────────────────────────────────┐   │
│  │  train_streaming.py (8 GPU FSDP)                  │   │
│  │  Consumes from unified manifest                   │   │
│  │  Increments train_count → triggers cleanup        │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Component Design

#### 1. Hidden States Emitter (生产节点侧)

在 vLLM serving engine 中嵌入 hidden states 捕获，作为 prefill 的旁路输出。

**核心思路**：复用现有 `HiddenStatesWorkerExtension`，但不再 abort 请求，而是在 prefill 完成后异步写出 hidden_states，正常继续 decode。

```python
class ServingEmitter:
    """Async hidden states emitter for production vLLM serving."""

    def __init__(self, buffer_dir, layer_ids, max_buffer_gb=100,
                 flush_interval=30, min_seq_len=128):
        self.buffer_dir = Path(buffer_dir)
        self.layer_ids = layer_ids
        self.max_buffer_gb = max_buffer_gb
        self.flush_interval = flush_interval
        self.min_seq_len = min_seq_len  # skip short prompts

        self.pending = queue.Queue(maxsize=1000)
        self.file_idx = find_last_checkpoint(buffer_dir)
        self.writer_thread = Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()

    def emit(self, input_ids, hidden_states, loss_mask):
        """Non-blocking: enqueue for async write. Drop if queue full."""
        if len(input_ids) < self.min_seq_len:
            return  # skip short requests
        try:
            self.pending.put_nowait({
                "input_ids": input_ids,
                "hidden_states": hidden_states,
                "loss_mask": loss_mask,
            })
        except queue.Full:
            pass  # backpressure: drop sample, don't block serving

    def _writer_loop(self):
        """Background thread: batch samples and write .pt files."""
        while True:
            sample = self.pending.get()
            output_path = self.buffer_dir / f"data_{self.file_idx}.pt"
            torch.save(sample, output_path)
            self.file_idx += 1

            # Check buffer size limit
            if get_dir_size_gb(self.buffer_dir) >= self.max_buffer_gb:
                self._wait_for_space()
```

**关键设计决策**：
- **异步非阻塞**：emit() 只入队，不阻塞 serving 请求。队列满时丢弃样本（宁丢数据不影响延迟）
- **最小序列长度过滤**：短 prompt 的训练价值低，跳过以减少 I/O
- **Per-node buffer 限制**：每个生产节点独立控制本地 buffer 大小

#### 2. Loss Mask Generation (生产环境)

离线 datagen 通过 chat template 预计算 loss_mask。生产环境需要实时生成。

**方案**：在 tokenizer 阶段计算。大多数 chat template 会在 assistant response 前后插入特殊 token，可用 regex 或 token ID 匹配。

```python
def generate_loss_mask_online(input_ids, tokenizer):
    """Generate loss mask from tokenized input using chat template markers."""
    # Decode and find assistant response boundaries
    text = tokenizer.decode(input_ids)
    mask = torch.zeros(len(input_ids), dtype=torch.bool)

    # Use same regex detection as preprocessing.py
    pattern = detect_assistant_pattern(tokenizer)
    for match in pattern.finditer(text):
        start_token = len(tokenizer.encode(text[:match.start()]))
        end_token = len(tokenizer.encode(text[:match.end()]))
        mask[start_token:end_token] = 1

    return mask
```

#### 3. Aggregated Buffer (训练节点侧)

训练节点需要从多个数据源消费：

```
数据源:
  ├── 生产节点 A → rsync prod_A_*.pt
  ├── 生产节点 B → rsync prod_B_*.pt
  ├── 生产节点 C → rsync prod_C_*.pt
  └── 离线 datagen → rsync data_*.pt (可选)
```

**扩展 sync_datagen.sh**：支持多源 `--datagen-nodes`（已支持），每个源可以是生产节点或 datagen 节点。Manifest updater 统一扫描所有 .pt 文件，无论来源。

#### 4. Data Quality & Filtering

生产请求质量参差不齐，需要过滤：

| 过滤规则 | 原因 |
|---------|------|
| `seq_len < 128` | 太短，训练价值低 |
| `seq_len > max_seq_len` | 超长序列 OOM 风险 |
| `assistant_ratio < 0.1` | 几乎没有 assistant response |
| `repeat_ratio > 0.5` | 重复内容过多 |

过滤在 emitter 侧完成，减少无用数据的 I/O 和传输。

### Data Flow Timeline

```
t=0   User request arrives at vLLM Node A
t=1   Tokenizer encodes input, generates loss_mask
t=2   vLLM prefill: computes hidden_states for all layers
t=2+  Emitter copies selected layer hidden_states to CPU (async)
t=3   vLLM decode continues normally (不受影响)
t=3+  Writer thread saves .pt to local buffer
t=30  Manifest updater refreshes local manifest
t=60  rsync syncs new .pt files to training node
t=60  Training node manifest updater picks up new files
t=??  train_streaming.py includes new files in next epoch
```

**端到端延迟**：从请求到训练数据可用 ~1-2 分钟（取决于 rsync 周期）。

## Comparison: Current vs Proposed

| 维度 | 当前 (离线 datagen) | 提议 (生产旁路) |
|------|-------------------|----------------|
| GPU 利用率 | datagen 专用 8 GPU | 复用生产 serving GPU，零额外 GPU |
| 数据时效性 | 固定数据集，无法跟踪分布变化 | 实时反映线上请求分布 |
| 数据量 | 受 datagen 速度限制 (~1.5 it/s) | 受生产 QPS 限制 (远高于 datagen) |
| 数据质量 | 可控（预处理过滤） | 需要在线过滤 |
| 延迟影响 | 无（独立进程） | 极小（async copy，~0.1ms per request） |
| 运维复杂度 | 简单（独立 pod） | 需要嵌入 serving engine |

## Implementation Phases

### Phase 1: 验证 Serving 旁路捕获可行性
- 在单个 vLLM serving 节点上启用 `HiddenStatesWorkerExtension`
- 验证 prefill 时捕获 hidden_states 对 decode latency 的影响
- 预期结果：P99 延迟增加 < 1%

### Phase 2: Emitter + Local Buffer
- 实现 `ServingEmitter` 异步写入
- 实现在线 loss_mask 生成
- 本地 buffer 大小控制
- 验证 .pt 文件格式与 train_streaming.py 兼容

### Phase 3: 多源 Aggregated Buffer
- 扩展 sync 支持多个生产节点
- 统一 manifest 管理
- 混合离线 + 在线数据训练

### Phase 4: 生产部署
- 灰度：1 个生产节点启用 emitter
- 监控 serving latency 和 buffer 增长
- 全量推广

## Open Questions

1. **Hidden states copy 开销**：prefill 产生的 hidden_states 在 GPU 显存中，copy 到 CPU 的延迟是否可接受？可能需要在 decode 阶段异步 copy。
2. **Serving 和 datagen 用不同 TP**：生产节点可能用 TP=8，datagen 用 TP=4，hidden_states 的 shape 是否一致？（应该一致，因为 hidden_states 是 all-gather 后的完整张量）
3. **数据分布偏移**：生产请求分布可能与评测分布差异大，需要考虑混合训练时的采样策略。
4. **模型版本对齐**：draft model 更新后，生产节点的 target model 版本需要与训练保持一致。

## Related Documents

- [buffer-architecture.md](./buffer-architecture.md) — Buffer 系统设计演进 (V1-V4)
- [online-training-design.md](./online-training-design.md) — 离线 datagen + 流式训练设计
