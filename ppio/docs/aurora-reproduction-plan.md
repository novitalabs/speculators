# Aurora Reproduction Plan

基于 speculators 现有框架，使用 vLLM serving（非 sglang）实现 Aurora 论文的核心系统。

## 背景

Aurora 将投机解码建模为异步 RL 问题：推理端执行投机解码并收集 trace → 训练端从 trace 学习 → 新权重推送回推理端。核心价值在于**在线闭环**：draft model 持续从真实推理请求中学习，无需离线数据生成。

### 当前框架已有能力

| 能力 | 状态 | 来源 |
|------|------|------|
| EAGLE3 模型架构 | ✅ | `models/eagle3/` |
| KL 蒸馏损失 | ✅ | `eagle3/core.py: loss_function()` |
| FSDP 分布式训练 | ✅ | `train/trainer.py` |
| 流式训练 (manifest-based) | ✅ | `train_streaming.py`, `manifest.py` |
| vLLM Hidden States 提取 | ✅ | `vllm_hidden_states_generator.py` |
| 词表映射 (d2t/t2d) | ✅ | `train/vocab_mapping.py` |
| Checkpoint 管理 | ✅ | `train/checkpointer.py` |
| flex_attention 基础 | ✅ | `eagle3/attention.py` |
| Buffer cleanup + sync | ✅ | `buffer_cleanup.py`, `sync_datagen.sh` |

### 需要新建/改造的能力

| 能力 | 复杂度 | 说明 |
|------|--------|------|
| vLLM Serving + Trace 收集 | 🔴 高 | 在投机解码中收集 hidden states + accept/reject 信息 |
| Discard Sampling Loss | 🟢 低 | 在现有 KL loss 上增加 rejected token 加权项 |
| Tree Attention Mask | 🟡 中 | 扩展 flex_attention 支持树形分支掩码 |
| Hot-swap 权重同步 | 🟡 中 | 推理端定期加载新 checkpoint |
| 在线数据格式 | 🟢 低 | .pt 文件增加 logits + reject_info 字段 |

---

## 分阶段实施计划

### Phase 1: 离线模拟验证（1-2 周）

**目标**: 用离线数据验证 Aurora 的训练算法（acceptance loss + discard sampling loss）是否优于纯 KL 蒸馏。

**原理**: 论文 Section A.5 明确指出 Aurora 可退化为离线训练模式。核心思路：用 vLLM 离线执行投机解码（而非仅 prefill），收集完整的 accepted/rejected trace，用于离线训练。

#### Step 1.1: 扩展离线数据生成 — 投机解码 Trace 收集

**新建**: `scripts/data_generation_spec_decode.py`

当前 `data_generation_offline.py` 仅执行 prefill 提取 hidden states。新脚本需要：

1. 加载 verifier（MiniMax-M2.5）+ draft model（Exp15 ckpt67 或 Aurora-Spec）
2. 对每个输入 prompt 执行**完整的投机解码**（draft → verify → accept/reject）
3. 收集每个位置的：
   - `hidden_states`: verifier 多层隐藏状态 `[T, num_layers, H]`
   - `target_logits`: verifier 输出 logits `[T, V]`（或 top-k logits 节省内存）
   - `accepted_mask`: 布尔掩码，标记哪些 draft token 被接受
   - `rejected_tokens`: 被拒绝位置的 draft token 和 verifier token
   - `input_ids`: 最终接受的 token 序列

**实现方案**: 利用 vLLM 的 `LLM` API 执行投机解码推理，通过扩展 `HiddenStatesWorkerExtension` 在验证步骤中捕获额外信息。

```python
# 伪代码
class SpecDecodeTraceGenerator:
    def __init__(self, verifier_path, draft_path, tp_size):
        self.llm = LLM(
            model=verifier_path,
            speculative_config={
                "model": draft_path,
                "num_speculative_tokens": 3,
                "method": "eagle3",
            },
            tensor_parallel_size=tp_size,
        )

    def generate_traces(self, prompts):
        # 执行投机解码，收集 trace
        outputs = self.llm.generate(prompts, sampling_params)
        # 提取 hidden states + accept/reject 信息
        traces = self.extract_traces(outputs)
        return traces
```

**难点**: vLLM 的投机解码内部不直接暴露 accept/reject trace。需要：
- 方案 A: Monkey-patch vLLM 的 spec decode scorer，在验证时 hook 出 accepted/rejected 信息（参考 `custom_worker.py` 的 hook 模式）
- 方案 B: 自行实现一个轻量级投机解码循环：用 vLLM 分别执行 draft forward 和 verify forward，手动实现 rejection sampling

**推荐方案 B**：更可控，不依赖 vLLM 内部实现细节。

```python
# 方案 B: 手动投机解码循环
def speculative_decode_with_traces(verifier_llm, draft_model, input_ids):
    traces = []
    position = 0

    while position < max_length:
        # 1. Draft: 生成 K 个候选 token
        draft_tokens, draft_logits = draft_model.forward(input_ids[:position])

        # 2. Verify: 用 verifier 一次性验证所有候选
        #    同时提取 hidden_states 和 target_logits
        verify_result = verifier_llm.forward_with_hidden_states(
            input_ids[:position] + draft_tokens
        )

        # 3. Accept/Reject: 按 rejection sampling 规则决定
        accepted, rejected = rejection_sample(
            draft_logits, verify_result.logits
        )

        # 4. 记录 trace
        traces.append({
            "hidden_states": verify_result.hidden_states,
            "target_logits": verify_result.logits,
            "accepted_mask": accepted,
            "rejected_info": rejected,
        })

        position += len(accepted)

    return traces
```

**输出格式**: 扩展现有 .pt 文件格式

```python
{
    "input_ids": [seq_len],
    "hidden_states": [num_layers, seq_len, hidden_size],
    "loss_mask": [seq_len],           # 现有
    "target_logits": [seq_len, V],    # 新增：verifier 输出 logits
    "accepted_mask": [seq_len],       # 新增：1=accepted, 0=rejected
    "reject_info": {                  # 新增：拒绝位置的详细信息
        "positions": [num_rejected],
        "draft_tokens": [num_rejected],
        "target_tokens": [num_rejected],
    }
}
```

#### Step 1.2: 实现 Discard Sampling Loss

**改造**: `src/speculators/models/eagle3/core.py`

在现有 `loss_function()` 基础上增加 discard sampling loss：

```python
def aurora_loss_function(
    draft_logits,      # draft model 预测
    target_logits,     # verifier 输出 (ground truth)
    accepted_mask,     # 哪些 token 被接受
    loss_mask,         # 现有 loss mask
    lambda_discard=0.1,
    discard_top_k=10,
):
    # 1. Acceptance Loss: 对 accepted tokens 的 KL 蒸馏
    # （与现有 loss_function 基本一致）
    accept_loss = kl_divergence(
        target_logits[accepted_mask],
        draft_logits[accepted_mask]
    )

    # 2. Discard Sampling Loss: 对 rejected tokens 的 KL
    # 仅对 draft model 给出高概率但被 verifier 拒绝的 token 计算
    rejected_mask = ~accepted_mask & loss_mask
    if rejected_mask.any():
        # Top-k 过滤：只对 draft 认为高概率的错误 token 计算
        draft_probs = draft_logits[rejected_mask].softmax(-1)
        topk_mask = draft_probs >= draft_probs.topk(discard_top_k).values[..., -1:]
        discard_loss = kl_divergence(
            target_logits[rejected_mask],
            draft_logits[rejected_mask],
            token_mask=topk_mask,
        )
    else:
        discard_loss = 0.0

    return accept_loss + lambda_discard * discard_loss
```

**工作量**: ~100 行代码，在现有 `loss_function` 上扩展。

#### Step 1.3: 实现 Tree Attention Mask（可选，Phase 1 可跳过）

**改造**: `src/speculators/models/eagle3/attention.py`

论文的 Tree Attention 允许在一次前向传播中同时处理 accepted 和 rejected 分支。当前 `create_combined_mask_mod()` 已有 causal + document + diagonal draft 掩码。

Phase 1 可先用简单的分开处理（accepted 和 rejected 分别前向），Phase 2 再实现真正的 Tree Attention。

#### Step 1.4: 训练实验

**改造**: `scripts/train.py` 或新建 `scripts/train_aurora.py`

```bash
# 对比实验
# A: 标准 KL 蒸馏 (现有 baseline)
torchrun --nproc=8 scripts/train.py \
    --data-path ./data/spec_decode_traces/ \
    --loss-type kl

# B: Aurora Loss (acceptance + discard)
torchrun --nproc=8 scripts/train_aurora.py \
    --data-path ./data/spec_decode_traces/ \
    --loss-type aurora \
    --lambda-discard 0.1 \
    --discard-top-k 10
```

**验证指标**: 对比 A 和 B 在相同 val set 上的 acceptance rate、throughput。

#### Phase 1 文件清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `scripts/data_generation_spec_decode.py` | 新建 | 投机解码 trace 收集 |
| `src/speculators/models/eagle3/core.py` | 改造 | 增加 `aurora_loss_function()` |
| `src/speculators/train/data.py` | 改造 | 扩展数据加载支持新字段 |
| `scripts/train_aurora.py` | 新建 | Aurora 训练脚本 |
| `k8s/run_aurora_phase1_datagen.sh` | 新建 | K8s 部署脚本 |
| `k8s/run_aurora_phase1_train.sh` | 新建 | K8s 训练脚本 |

---

### Phase 2: 简化版在线闭环（2-3 周）

**目标**: 实现基于文件系统的"穷人版"在线系统 — 推理端产生 trace 文件，训练端消费 trace 文件，新权重通过 checkpoint 回传。

#### 系统架构

```
┌─────────────────────────────┐     ┌─────────────────────────────┐
│  vLLM Serving (推理节点)      │     │  Training Server (训练节点)   │
│                             │     │                             │
│  vllm serve MiniMax-M2.5    │     │  train_streaming.py         │
│  + Eagle3 Draft (ckpt67)    │     │  (FSDP, 8 GPU)             │
│                             │     │                             │
│  ┌───────────────────────┐  │     │  ┌───────────────────────┐  │
│  │ Trace Collector Hook  │──┼─────┼──│ Streaming DataLoader  │  │
│  │ (hidden_states,       │  │ rsync│  │ (manifest-based)      │  │
│  │  logits, accept/rej)  │  │     │  └───────────────────────┘  │
│  └───────────────────────┘  │     │                             │
│                             │     │  checkpoint/latest/ ────────┤
│  ┌───────────────────────┐  │     │                             │
│  │ Weight Reloader       │──┼─────┼── (poll & reload)           │
│  │ (poll checkpoint dir) │  │     │                             │
│  └───────────────────────┘  │     └─────────────────────────────┘
└─────────────────────────────┘
         │
    rsync traces/*.pt  ──→  训练节点 gen/ 目录
    rsync checkpoint/  ←──  训练节点 checkpoint/
```

#### Step 2.1: vLLM Serving + Trace 收集

**新建**: `scripts/serve_with_traces.py`

使用 vLLM 的 OpenAI-compatible server 作为推理服务，同时在后台收集 trace。

**方案**: 运行两个进程：
1. **vLLM Serving 进程**: `vllm serve` 正常处理请求
2. **Trace 收集进程**: 定期用相同的 prompt 执行 prefill + spec decode 提取 trace

```python
# serve_with_traces.py

import subprocess
import threading
from speculators.data_generation import VllmHiddenStatesGenerator

class TracingServer:
    def __init__(self, model_path, draft_path, output_dir):
        self.output_dir = output_dir
        self.serving_proc = None
        self.trace_thread = None

    def start_serving(self):
        """启动 vLLM OpenAI server"""
        self.serving_proc = subprocess.Popen([
            "vllm", "serve", self.model_path,
            "--speculative-config", json.dumps({
                "model": self.draft_path,
                "num_speculative_tokens": 3,
                "method": "eagle3",
            }),
            "--tensor-parallel-size", "4",
            "--port", "8000",
        ])

    def start_trace_collection(self):
        """后台线程：收集推理请求的 trace"""
        # 监听 vLLM 的请求日志，获取实际推理 prompt
        # 对每个 prompt 执行离线 trace 收集
        # 保存到 output_dir/*.pt
        pass

    def run(self):
        self.start_serving()
        self.start_trace_collection()
```

**更实际的方案**: 不修改 vLLM serving，而是在 vLLM 旁边跑一个独立的 trace 收集进程：

1. vLLM server 正常 serve 请求（带 Eagle3 投机解码）
2. 收集进程通过日志/API 获取请求的 prompt
3. 收集进程用独立的 vLLM 实例（prefill-only）生成 hidden states
4. 利用 vLLM server 的 `/v1/completions` API 获取实际的 acceptance 信息

**最简实现**（推荐先做这个）:

```
vLLM Serving (GPU 0-3, TP=4)
  │
  ├── 正常处理推理请求
  ├── 请求日志 → prompt 收集
  │
Trace Generator (GPU 4-7, TP=4)
  │
  ├── 读取 prompt 队列
  ├── 执行 prefill 提取 hidden states
  ├── 通过 rejection sampling 模拟 accept/reject
  └── 保存 trace 到 .pt 文件
```

这种方案**完全复用现有 `VllmHiddenStatesGenerator`**，只需增加 rejection sampling 逻辑。

#### Step 2.2: Hot-swap 权重加载

**新建**: `scripts/weight_reloader.py`

推理端定期检查训练端输出的 checkpoint，加载新权重。

```python
# weight_reloader.py

class WeightReloader:
    """定期 poll checkpoint 目录，检测到新 checkpoint 时重新加载 draft model"""

    def __init__(self, checkpoint_dir, poll_interval=60):
        self.checkpoint_dir = checkpoint_dir
        self.poll_interval = poll_interval
        self.current_version = None

    def poll_and_reload(self, serving_process):
        while True:
            latest = self.find_latest_checkpoint()
            if latest != self.current_version:
                self.reload_draft_model(serving_process, latest)
                self.current_version = latest
            time.sleep(self.poll_interval)

    def reload_draft_model(self, proc, ckpt_path):
        """方案 A: 重启 vLLM 进程（简单但有停机时间）"""
        proc.terminate()
        proc = start_vllm_serve(draft_model=ckpt_path)

        """方案 B: 使用 vLLM 的 LoRA adapter API 热加载（如果支持）"""
        # requests.post("http://localhost:8000/v1/load_weights", ...)
```

**Phase 2 先用方案 A**（重启），因为：
- 实现最简单
- 对于 MiniMax-M2.5，模型加载 + torch.compile 约需 30-40 分钟
- 可以设置较长的同步间隔（如每 2000 请求或每小时同步一次）

#### Step 2.3: 训练端适配

**复用**: 现有 `train_streaming.py` + `sync_datagen.sh` + `manifest.py`

训练端几乎不需要改造：
- trace 文件与现有 .pt 格式兼容（增加新字段）
- `sync_datagen.sh` 已支持从远端 rsync 数据
- `train_streaming.py` 已支持 manifest-based 流式训练
- 唯一变化：训练完成后，额外 rsync checkpoint 回推理节点

```bash
# 在 train_streaming.py 完成 checkpoint 保存后，增加：
rsync -avz $CHECKPOINT_DIR/ $SERVING_NODE:$CHECKPOINT_DIR/
```

#### Step 2.4: 端到端编排

**新建**: `scripts/aurora_online.py`

```python
# aurora_online.py — 编排 serving + trace + training + sync

def main():
    # 1. 启动 vLLM serving (推理节点)
    serving = start_serving(model, draft_model, node=SERVING_NODE)

    # 2. 启动 trace 收集 (推理节点)
    tracer = start_trace_collection(
        model, draft_model,
        output_dir=TRACE_DIR,
        node=SERVING_NODE,
    )

    # 3. 启动数据同步 (训练节点)
    syncer = start_sync(
        remote_dir=f"{SERVING_NODE}:{TRACE_DIR}",
        local_dir=LOCAL_TRACE_DIR,
    )

    # 4. 启动流式训练 (训练节点)
    trainer = start_training(
        data_dir=LOCAL_TRACE_DIR,
        save_path=CHECKPOINT_DIR,
    )

    # 5. 启动权重回传 + 热加载
    reloader = start_weight_sync(
        checkpoint_dir=CHECKPOINT_DIR,
        serving_node=SERVING_NODE,
        interval=3600,  # 每小时同步
    )
```

#### Phase 2 文件清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `scripts/serve_with_traces.py` | 新建 | vLLM serving + trace 收集 |
| `scripts/weight_reloader.py` | 新建 | Checkpoint poll + reload |
| `scripts/aurora_online.py` | 新建 | 端到端编排 |
| `scripts/sync_checkpoint.sh` | 新建 | Checkpoint 回传脚本 |
| `k8s/run_aurora_serving.sh` | 新建 | K8s 推理节点部署 |
| `k8s/run_aurora_training.sh` | 新建 | K8s 训练节点部署 |
| `k8s/k8s-aurora-serving.yaml` | 新建 | 推理 Pod |
| `k8s/k8s-aurora-training.yaml` | 新建 | 训练 Pod |

---

### Phase 3: 优化版在线系统（2-4 周）

**目标**: 在 Phase 2 基础上优化性能，接近论文描述的完整系统。

#### Step 3.1: vLLM 内部 Trace Hook

替换 Phase 2 的"旁路 trace 收集"为 vLLM 内部 hook，直接在投机解码验证步骤中收集 trace。

**改造**: 扩展 `custom_worker.py` 的 hook 机制

```python
class SpecDecodeTraceHook(HiddenStatesWorkerExtension):
    """在 vLLM 投机解码验证步骤中捕获 trace"""

    def on_verify_step(self, draft_tokens, target_logits, accepted_mask):
        """验证步骤的 hook — 收集 accepted/rejected 信息"""
        trace = {
            "hidden_states": self.captured_hidden_states,
            "target_logits": target_logits,
            "accepted_mask": accepted_mask,
            "draft_tokens": draft_tokens,
        }
        self.trace_buffer.append(trace)
```

**优势**: 零额外推理开销（复用已有的 verify forward pass）。
**难度**: 需要深入 vLLM spec decode 内部，修改 `spec_decode_worker.py`。

#### Step 3.2: GPU 内存 Buffer

替换文件系统为 GPU 内存 buffer，减少 I/O 延迟。

```python
class GPUTraceBuffer:
    """GPU 内存中的 trace 缓冲区"""

    def __init__(self, max_size_gb=8):
        self.buffer = []
        self.max_size = max_size_gb * 1024**3
        self.lock = threading.Lock()

    def push(self, trace):
        with self.lock:
            self.buffer.append(trace)
            self._evict_if_full()

    def sample_batch(self, batch_size):
        with self.lock:
            return random.sample(self.buffer, batch_size)
```

**通信**: 使用 `torch.distributed.rpc` + TensorPipe 实现跨节点 GPU-to-GPU 传输。

#### Step 3.3: 真正的 Hot-swap

使用 vLLM 的模型加载 API（如果支持）或直接修改 worker 的 model state_dict。

```python
def hot_swap_draft_weights(vllm_engine, new_state_dict):
    """在不重启 vLLM 的情况下更新 draft model 权重"""
    for worker in vllm_engine.workers:
        worker.model.speculator.load_state_dict(new_state_dict)
```

#### Step 3.4: Lazy Sync 策略

实现论文的 lazy sync 策略（每 N 个请求同步一次）。

```python
class LazySyncPolicy:
    def __init__(self, sync_interval_requests=80):
        self.request_count = 0
        self.sync_interval = sync_interval_requests

    def should_sync(self):
        self.request_count += 1
        if self.request_count >= self.sync_interval:
            self.request_count = 0
            return True
        return False
```

---

## 节点规划

基于当前集群状态（.10, .26, .28 有空闲 GPU）：

### Phase 1 (离线模拟)

| 角色 | 节点 | GPU | 说明 |
|------|------|-----|------|
| Trace 数据生成 | .28 | 8x H200, TP=4 | 执行投机解码 + 收集 trace |
| 训练 | .10 | 8x H200, FSDP | 用 Aurora loss 训练 |

### Phase 2 (在线闭环)

| 角色 | 节点 | GPU | 说明 |
|------|------|-----|------|
| vLLM Serving | .28 | GPU 0-3, TP=4 | 处理推理请求 |
| Trace 收集 | .28 | GPU 4-7, TP=4 | 旁路 trace 生成 |
| 训练 | .10 | 8x H200, FSDP | 流式训练 |

---

## 实验计划

### Exp-A1: 离线 Aurora Loss vs KL 蒸馏

| 配置 | 说明 |
|------|------|
| Verifier | MiniMax-M2.5 |
| Draft 架构 | Aurora-like (24 heads, 8192 intermediate, 32K vocab) |
| 数据 | 投机解码 trace (from Exp15 ckpt67 as draft) |
| 对比组 A | 标准 KL 蒸馏 (现有 loss) |
| 对比组 B | Acceptance Loss only |
| 对比组 C | Acceptance + Discard Sampling Loss |
| 评估 | Acc@0, throughput (tok/s), val loss |

### Exp-A2: 在线闭环 vs 离线训练

| 配置 | 说明 |
|------|------|
| 对比组 D | 离线训练 (Exp15 最佳) |
| 对比组 E | Phase 2 在线闭环 (文件系统) |
| 评估 | Acc@0 收敛速度, 最终 throughput |

---

## 风险与缓解

| 风险 | 等级 | 缓解措施 |
|------|------|----------|
| vLLM spec decode 内部 hook 困难 | 🔴 高 | Phase 1-2 用旁路方案（独立 prefill），Phase 3 再深入 vLLM |
| GPU 内存不足 (serving + trace) | 🟡 中 | 分 GPU 组（TP=4 serving + TP=4 trace），或用 CPU offload logits |
| 文件系统 I/O 瓶颈 (Phase 2) | 🟡 中 | 使用 NVMe SSD，控制 trace 文件大小 (不保存完整 logits，只保存 top-k) |
| torch.compile 重编译 (hot-swap) | 🟡 中 | Phase 2 用重启方案；Phase 3 用 state_dict 替换（不触发重编译） |
| Discard loss 训练不稳定 | 🟢 低 | lambda_discard 从小值 (0.01) 开始调，top-k 过滤保守 |

---

## 优先级建议

**强烈建议从 Phase 1 开始**。理由：

1. **最小改动量**: Phase 1 主要是新增 ~300 行代码（trace 收集 + loss 扩展），不涉及系统架构变更
2. **验证核心假设**: 如果 Aurora loss 在离线数据上不优于纯 KL 蒸馏，则无需投入 Phase 2-3 的系统工程
3. **复用现有基础**: 训练端完全复用 `train_streaming.py`，数据端复用 `VllmHiddenStatesGenerator`
4. **论文支撑**: Section 5 消融实验表明简单 RKL on-policy fine-tuning 已捕获大部分收益
5. **快速迭代**: 1-2 周可出结果，指导后续投入方向
