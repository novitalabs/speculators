# Eagle3 Speculative Decoding Experiments

## Overview

Training Eagle3 speculative decoding draft models for Qwen3-32B using different data sources,
targeting vLLM inference acceleration.

**Pipeline**: data_generation_offline.py → build_vocab_mapping.py → train.py (FSDP)

---

## Experiment 1: Qwen3-32B + ShareGPT/UltraChat (Baseline)

- **Node**: .23 (8x H200 143GB)
- **Image**: speculators:v0.16.0
- **Data**: ShareGPT (5000 samples) + UltraChat (5000 samples)
- **Config**:
  - target_vocab_size=151936, draft_vocab_size=32000
  - seq_len=8192, lr=3e-5, epochs=10, batch via FSDP 8 GPUs
- **Result**:
  - val_loss = 9.556
  - top1_acc = 52%
- **Status**: COMPLETED

---

## Experiment 2: Qwen3-32B + Novita Online Data (v1 - local logs)

Using production API logs from MiniMax-M2.5 endpoint (coding agent conversations).

- **Node**: .18 (8x H200 143GB)
- **Image**: speculators:v0.16.0
- **Data**: 812 conversations extracted from Novita API gateway logs
  - Source: `/data/novita/minimax-m2.5/` tar.gz export
  - Conversion: `k8s/convert_novita_logs.py` → `novita_sharegpt.jsonl`
  - Characteristics: coding agent dialogues with tool calls, median 85 turns/conversation
  - Roles: system, user, assistant, tool (tool → mapped to user)
- **Config**:
  - target_vocab_size=151936, draft_vocab_size=32000
  - seq_len=8192, lr=3e-5, epochs=10, 8 GPUs FSDP
  - Data gen: TP=2 (segfault workaround)
- **K8s**: `k8s/k8s-qwen3-32b-eagle3-novita.yaml`
- **Script**: `k8s/run_qwen3_32b_eagle3_novita.sh`

### Training Results

| Epoch | Val Loss | Top-1 Acc (cond_acc_0) |
|-------|----------|------------------------|
| 0     | 1.895    | 15.9%                  |
| 1     | 1.621    | 18.6%                  |
| 2     | 1.538    | 19.4%                  |
| 3     | 1.345    | 19.2%                  |
| 4     | 1.175    | 17.6%                  |
| 5     | 1.214    | 17.9%                  |
| 6     | 1.351    | 20.2%                  |
| 7     | 1.214    | 19.9%                  |
| 8     | **1.117**| 19.8%                  |
| 9     | 1.213    | 19.9%                  |

- **Best loss**: epoch 8 (1.117)
- **Best acc**: epoch 6 (20.2%)
- **Checkpoints**: `/data/output/qwen3_32b_eagle3_novita/checkpoints/` on .18 (263GB total)

### Analysis

- Loss is much lower than baseline (1.117 vs 9.556) but top-1 accuracy is also much lower (20% vs 52%)
- Note: loss metrics are not directly comparable since the data distributions differ significantly
- The novita data is dominated by coding agent conversations (tool use, code analysis), a narrow domain
- Only 812 samples vs 10000 in baseline — likely underfitting
- The loss curves show some oscillation after epoch 4, suggesting the small dataset causes instability

---

## Experiment 3: Qwen3-32B + Novita Online Data (v2 - HuggingFace dataset)

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

### Data Generation Performance Analysis

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

---

## Experiment 4: MiniMax-M2.5 + ShareGPT/UltraChat

Training Eagle3 draft model for MiniMax-M2.5 (MoE architecture: 256 experts / 8 active).

- **Node**: .14 (8x H200 143GB)
- **Image**: speculators:v0.17.0 (vLLM 0.17.0, required for MiniMax-M2.5 support)
- **Data**: ShareGPT (4965 samples) + UltraChat (4945 samples), ~208GB total
  - 43 files removed due to NaN hidden states from data generation
- **Config**:
  - vocab_size=200064 (no vocab mapping, full target vocab)
  - hidden_size=3072, 62 layers, MoE
  - seq_len=4096, lr=3e-5, epochs=10, cosine scheduler, 200-step warmup
  - 8 GPUs FSDP (param_dtype=bf16, reduce_dtype=fp32)
  - num_workers=4, prefetch_factor=2, shm=256Gi
- **Data gen**: TP=4 (TP=8 segfaults for MiniMax-M2.5)
- **K8s**: `k8s/k8s-minimax-m2.5-eagle3-full.yaml`
- **Script**: `k8s/run_minimax_m2.5_eagle3_train_only.sh`

### Training Results

| Epoch | Val Loss | Top-1 Acc | Cond Acc 1 | Cond Acc 2 |
|-------|----------|-----------|------------|------------|
| 0     | 15.199   | 28.9%     | 11.6%      | 7.9%       |
| 1     | 12.043   | 42.5%     | 27.2%      | 19.5%      |
| 2     | 10.793   | 48.1%     | 32.9%      | 25.1%      |
| 3     | 10.063   | 51.7%     | 36.0%      | 28.1%      |
| 4     | 9.723    | 53.2%     | 37.9%      | 30.3%      |
| 5     | 9.416    | 54.5%     | 39.0%      | 31.3%      |
| 6     | 9.244    | 55.4%     | 39.6%      | 32.0%      |
| 7     | 9.187    | 55.8%     | 40.0%      | 32.4%      |
| **8** | **9.118**| **56.0%** | **40.3%**  | **32.9%**  |
| 9     | 9.127    | 55.9%     | 40.3%      | 32.8%      |

- **Best checkpoint**: epoch 8 (val_loss=9.118, top1_acc=56.0%)
- **Training time**: ~70 min (10 epochs, 8x H200)
- **Checkpoints**: `/data/output/minimax_m2.5_eagle3/checkpoints/` on .14
- **Status**: COMPLETED

### Issues Encountered

1. **vLLM version**: v0.16.0 does not support MiniMax-M2.5 → upgraded to v0.17.0
2. **trust_remote_code**: MiniMax-M2.5 custom model code requires `trust_remote_code=True` in all `AutoConfig.from_pretrained` calls (train.py x2, eagle3/core.py x1)
3. **NaN hidden states**: MiniMax-M2.5 produces NaN hidden states for ~0.4% of inputs during data generation (43/9957 files). These poison the entire training via gradient updates. Fix: removed corrupt files + added `nan_to_num` safety check in data loader
4. **shm OOM**: MoE model requires larger shared memory → increased to 256Gi

### Analysis

- MiniMax-M2.5 achieves **56% top-1 accuracy**, outperforming Qwen3-32B baseline (52%)
- Model converges by epoch 8, with epoch 9 showing no improvement (slight regression)
- Cosine schedule with warmup helped stabilize early training (no NaN from optimizer)
- Conditional accuracy for 2nd/3rd draft tokens: 40.3% / 32.9%, decent for speculative decoding

---

## Experiment 5: MiniMax-M2.5 Online/Streaming Training (Quicktest)

Validating the online training pipeline: datagen and training run on separate nodes,
coordinated via manifest.json + rsync.

- **Datagen node**: .21 (8x H200 143GB)
- **Training node**: .18 (8x H200 143GB)
- **Image**: speculators:v0.17.0
- **Data**: ShareGPT, 100 samples
- **Config**:
  - Datagen: TP=4, batch_size=4, seq_len=8192
  - Training: 8 GPUs FSDP, lr=3e-5, FINAL_EPOCHS=3, MIN_SAMPLES=50
  - Scheduler: constant LR (no cosine — total steps unknown in streaming mode)
- **Architecture**: datagen on .21 writes .pt files + manifest.json → rsync to .18 → streaming training on .18 reads manifest, trains on arriving data

### Issues Encountered

1. **DeepGEMM warmup on new node (.21)**: First-time kernel compilation for MoE model (~3300 kernels) takes ~4 minutes. Looks like a hang (shm_broadcast timeout messages) but is actually normal. **Fix**: ran a debug pod first to warm up the kernel cache, or just wait.

2. **Training needs full model weights**: Eagle3 training calls `load_model_layers` to load embedding weights from the target model. Initially only copied config files to .18. **Fix**: copied full 215GB MiniMax-M2.5 model to .18.

3. **`KeyError: None` in LlamaAttention.forward**: `train_streaming.py` was missing `transformer_layer_config._attn_implementation = "simple_flex_attention"` (present in `train.py` but not copied to the streaming variant). Without this, the LlamaConfig's `_attn_implementation` defaults to `None`, which isn't registered in transformers' `ALL_ATTENTION_FUNCTIONS`. **Fix**: added the missing line to `create_transformer_layer_config()` in `train_streaming.py`.

4. **rsync between two remotes**: Can't rsync directly between two remote hosts. **Fix**: SSH to source node and rsync from there to destination.

### Training Results

| Epoch | Val Loss | Top-1 Acc |
|-------|----------|-----------|
| 0     | 27.526   | 1.7%      |
| 2     | 23.459   | ~2%       |

- 3 epochs completed in ~15 minutes
- 90 train / 10 val file split (from 100 total)
- Loss improving but accuracy very low (expected with only 100 samples)
- Manifest train_count correctly tracked: 90 train files at count=10, 10 val files at count=0
- **Status**: COMPLETED (pipeline validation successful)

---

## Experiment 6: MiniMax-M2.5 Online/Streaming Training (5K)

Full-scale online training: 5000 samples datagen + streaming training.

- **Datagen node**: .21 (8x H200 143GB)
- **Training node**: .18 (8x H200 143GB)
- **Image**: speculators:v0.17.0
- **Data**: ShareGPT, 5000 samples
- **Config**:
  - Datagen: TP=4, batch_size=4, seq_len=8192
  - Training: 8 GPUs FSDP, lr=3e-5, FINAL_EPOCHS=10, MIN_SAMPLES=500
  - Continuous rsync (.21 → .18) every 30s
- **Output**: `/data/output/minimax_m2.5_eagle3_online_5k/`
- **Comparison target**: Experiment 4 (offline training, val_loss=9.118, top1_acc=56.0%)
- **Status**: COMPLETED

### Issues Encountered

1. **Race condition: manifest synced before .pt files**: rsync copies manifest.json (small) before all .pt files arrive. Training reads manifest, tries to load missing files → `FileNotFoundError`. **Fix**: `resolve_file_paths()` now filters to only files that exist on disk.

2. **NCCL collective timeout (transient)**: First run with 1460 files (partial data, datagen still running) hit NCCL `_ALLGATHER_BASE` timeout after 20 training steps. Root cause unclear (possibly OOM or FSDP sync issue). Restart with full 5000 files resolved the issue.

### Training Results

| Epoch | Val Loss | Top-1 Acc | Cond Acc 1 | Cond Acc 2 |
|-------|----------|-----------|------------|------------|
| 0     | 17.940   | 19.7%     | 3.3%       | 2.2%       |
| 1     | 15.222   | 28.8%     | 10.9%      | 7.9%       |
| 2     | 13.613   | 36.6%     | 21.8%      | 16.7%      |
| 3     | 12.574   | 41.1%     | 27.0%      | 20.5%      |
| 4     | 11.855   | 44.6%     | 31.1%      | 24.8%      |
| 5     | 11.158   | 47.6%     | 34.3%      | 27.8%      |
| 6     | 10.788   | 49.5%     | 36.5%      | 30.1%      |
| 7     | 10.575   | 50.7%     | 37.1%      | 30.7%      |
| 8     | 10.260   | 51.8%     | 39.2%      | 33.2%      |
| **9** | **10.166**| **52.9%** | **39.0%**  | **32.7%**  |

- **Best checkpoint**: epoch 9 (val_loss=10.166, top1_acc=52.9%)
- **Training time**: ~55 min (10 epochs, 8x H200), ~5.5 min/epoch
- **Datagen time**: ~10 min (5000 samples, TP=4, .21)
- **Checkpoints**: `/data/output/minimax_m2.5_eagle3_online_5k/checkpoints/` on .18

### Comparison with Offline Training (Experiment 4)

| Metric | Offline (Exp 4) | Online 5K (Exp 6) | Delta |
|--------|-----------------|-------------------|-------|
| Data | sharegpt+ultrachat (9957) | sharegpt only (5000) | -50% data |
| Val Loss (best) | **9.118** | 10.166 | +11.5% |
| Top-1 Acc (best) | **56.0%** | 52.9% | -3.1pp |
| Cond Acc 1 | **40.3%** | 39.0% | -1.3pp |
| Cond Acc 2 | **32.9%** | 32.7% | -0.2pp |
| Scheduler | cosine (warmup) | constant LR | - |
| Seq Length | 4096 | 8192 | 2x |
| Training time | ~70 min | ~55 min | -21% |

**Analysis**: Online training with 5K samples (sharegpt only) achieves 94.5% of offline accuracy (52.9% vs 56.0%) despite using only half the data and no ultrachat. The constant LR scheduler performs comparably to cosine. Loss is ~11.5% higher, likely due to less data rather than the streaming approach itself. The pipeline overhead (manifest coordination, rsync) is negligible. Model is still converging at epoch 9 — more epochs or data would likely close the gap further.

---

## Experiment 7: MiniMax-M2.5 Online/Streaming Training (Novita 812)

Training with production Novita API logs (coding agent conversations).

- **Node**: .14 (8x H200 143GB), datagen + training sequential
- **Image**: speculators:v0.17.0
- **Data**: `/data/novita/minimax-m2.5/novita_sharegpt.jsonl` (812 conversations)
  - Source: MiniMax-M2.5 production API gateway logs (coding agent)
  - Median ~85 turns/conversation, roles: system, user, assistant, tool
- **Config**:
  - Datagen: TP=4, batch_size=4, seq_len=8192
  - Training: 8 GPUs FSDP, lr=3e-5, FINAL_EPOCHS=10, MIN_SAMPLES=50
  - 730 train / 82 val split
- **Output**: `/data/output/minimax_m2.5_eagle3_novita/`
- **Status**: COMPLETED

### Training Results

| Epoch | Val Loss | Top-1 Acc | Cond Acc 1 | Cond Acc 2 |
|-------|----------|-----------|------------|------------|
| 0     | 5.155    | 64.6%     | 73.8%      | 81.0%      |
| 1     | 4.071    | 69.9%     | 76.3%      | 82.1%      |
| 2     | 3.371    | 73.2%     | 78.5%      | 82.7%      |
| 3     | 3.009    | 75.4%     | 79.0%      | 82.6%      |
| 4     | 2.640    | 77.5%     | 80.6%      | 83.5%      |
| 5     | 2.324    | 78.9%     | 81.8%      | 84.2%      |
| 6     | 2.289    | 80.3%     | 83.6%      | 86.2%      |
| 7     | 2.152    | 80.5%     | 83.0%      | 85.7%      |
| 8     | 2.149    | 80.6%     | 83.2%      | 86.0%      |
| **9** | **1.952**| **81.5%** | **83.8%**  | **86.2%**  |

- **Best checkpoint**: epoch 9 (val_loss=1.952, top1_acc=81.5%)
- **Training time**: ~50 min (10 epochs, 8x H200)
- **Datagen time**: ~11 min (812 samples, TP=4, includes DeepGEMM warmup)
- **Checkpoints**: `/data/output/minimax_m2.5_eagle3_novita/checkpoints/` on .14

### Comparison

| Metric | ShareGPT 5K (Exp 6) | Novita 812 (Exp 7) | Offline ShareGPT+UC (Exp 4) |
|--------|--------------------|--------------------|----------------------------|
| Data | sharegpt (5000) | novita (812) | sharegpt+ultrachat (9957) |
| Val Loss | 10.166 | **1.952** | 9.118 |
| Top-1 Acc | 52.9% | **81.5%** | 56.0% |
| Cond Acc 1 | 39.0% | **83.8%** | 40.3% |
| Cond Acc 2 | 32.7% | **86.2%** | 32.9% |

**Analysis**: Novita data dramatically outperforms ShareGPT despite having only 16% of the samples (812 vs 5000). This is because:
1. **Domain match**: Novita data is from production MiniMax-M2.5 API usage (coding agent), so the draft model learns the actual distribution it will encounter
2. **Val/train correlation**: Both train and val are from the same narrow domain, so in-domain accuracy is very high
3. **Caveat**: The 81.5% accuracy is domain-specific — performance on general tasks may be lower. Need real inference benchmarks to validate

---

## Experiment 8: MiniMax-M2.5 Online/Streaming Training (50K ShareGPT)

Large-scale online training: 50000 samples with true streaming (datagen and training concurrent).

- **Datagen node**: .21 (8x H200 143GB)
- **Training node**: .18 (8x H200 143GB)
- **Image**: speculators:v0.17.0
- **Data**: ShareGPT, 50000 samples (45000 train / 5000 val)
- **Config**:
  - Datagen: TP=4, batch_size=4, seq_len=8192
  - Training: 8 GPUs FSDP, lr=3e-5, FINAL_EPOCHS=10, MIN_SAMPLES=5000
  - NCCL_TIMEOUT=1800 (30 min, increased from default 600s)
- **Output**: `/data/output/minimax_m2.5_eagle3_online_50k/`
- **Status**: IN PROGRESS (epoch 1+ training)

### Issues Encountered

1. **NCCL `_ALLGATHER_BASE` timeout at epoch boundary**: Training steps complete (303-612 steps/epoch) but crashes during validation/checkpoint phase. Root cause: default NCCL timeout (600s / 10 min) is too short for FSDP validation with ~5000 val files on a large MoE model. The `dist.reduce` calls in `val_epoch` accumulate latency.
   - **Failed 4 times** before fix:
     - Run 1: 1460 files, hung at work 402 after 20 steps
     - Run 2: 542 files, hung at work 142 after 28 steps
     - Run 3: 20276 files, hung at work 6062 after epoch 0 (303 steps)
     - Run 4: 40896 files, hung at work 12242 after epoch 0 (612 steps)
   - **Fix**: Set `NCCL_TIMEOUT=1800` env var → read in `init_process_group(timeout=timedelta(seconds=1800))`
   - Also removed redundant `torch.distributed.barrier()` calls between train/val/checkpoint
   - Added `torch.cuda.empty_cache()` before validation to reduce memory pressure

2. **Code not synced to training node**: Pod on .18 mounts hostPath `/root/develop/speculators` from .18's filesystem. Code edits on the dev machine don't auto-propagate. **Fix**: `rsync` code to .18 before pod restart.

### Training Results (partial)

| Epoch | Val Loss | Top-1 Acc | Cond Acc 1 | Cond Acc 2 |
|-------|----------|-----------|------------|------------|
| 0     | 9.636    | 54.9%     | 41.3%      | 34.8%      |

- Epoch 0 already outperforms the 5K experiment (52.9% final acc) with only 1 epoch
- Training time per epoch: ~50 min (45000 files, 8x H200)
- Datagen: 50000 samples completed on .21

---

## Experiment 9: MiniMax-M2.5 Eagle3 Inference Evaluation (vLLM Spec Decode)

Evaluating Eagle3 speculative decoding speedup for MiniMax-M2.5 with different draft models.

- **Node**: .14 (8x H200 143GB, using 4 GPUs via TP=4)
- **Image**: speculators:v0.17.0 (vLLM 0.17.0)
- **Base model**: MiniMax-M2.5 (FP8 quantized)
- **Draft models tested**:
  1. **Aurora-Spec**: `togethercomputer/Aurora-Spec-Minimax-M2.1` (1.6GB, `LlamaForCausalLMEagle3`, draft_vocab=32000)
  2. **Novita-trained**: Our Exp 7 model converted to vLLM format (2.7GB, `LlamaForCausalLMEagle3`, draft_vocab=200064)
- **Config**:
  - TP=4, max_model_len=8192, gpu_memory_utilization=0.85
  - 10 prompts (coding + general), 512 tokens each, temperature=0.6, ignore_eos=True
  - num_speculative_tokens=3 for both draft models
- **Eval script**: `k8s/run_minimax_m2.5_eval.sh`
- **K8s**: `k8s/k8s-minimax-m2.5-eval.yaml`
- **Converter**: `k8s/convert_speculators_to_vllm_eagle3.py`

### vLLM Patching Required

vLLM 0.17.0 does not natively support Eagle3 speculative decoding for `minimax_m2` model type. Created `k8s/patch_vllm_minimax_eagle3.py` which applies two patches at container startup:

1. **Whitelist patch** (`vllm/config/speculative.py`): Adds `minimax_m2` to Eagle3 supported model types
2. **Interface patch** (`vllm/model_executor/models/minimax_m2.py`): Adds `SupportsEagle3` protocol to `MiniMaxM2ForCausalLM`:
   - `supports_eagle3 = True` class variable
   - `has_own_lm_head = False` and `has_own_embed_tokens = False` (required by `SupportsEagleBase` protocol)
   - `set_aux_hidden_state_layers()` and `get_eagle3_aux_hidden_state_layers()` methods
   - Modified `MiniMaxM2Model.forward()` to collect auxiliary hidden states at specified layers
   - Auxiliary layers: `(2, num_layers // 2, num_layers - 3)` = `(2, 31, 59)` for 62-layer model

### Speculators → vLLM Model Conversion

Created `k8s/convert_speculators_to_vllm_eagle3.py` to convert speculators `Eagle3DraftModel` format to vLLM `Eagle3LlamaForCausalLM` format:

1. **Config**: Translates speculators `transformer_layer_config` → standard `LlamaConfig` with `architectures: ["LlamaForCausalLMEagle3"]`
2. **Weight key renaming**: `layers.0.*` → `midlayer.*` (vLLM's `load_weights` reverses this back to `layers.0.*`)
3. **Vocab mapping**: Adds identity `d2t` (draft-to-target) mapping when `draft_vocab_size == target_vocab_size`; vLLM expects weight name `d2t` not `draft_id_to_target_id`
4. **Dtype**: Converts fp32 → bfloat16

Weight verification confirmed all weights match exactly after conversion (max_diff=0).

### Issues Encountered and Fixed

1. **vLLM forces `spawn` multiprocessing**: When CUDA is initialized, vLLM overrides `VLLM_WORKER_MULTIPROC_METHOD=fork` to `spawn`. All patches must be on-disk `.py` files, not runtime monkey-patches.
   - **Fix**: `PYTHONDONTWRITEBYTECODE=1` + clearing `__pycache__` ensures spawned workers use patched files

2. **`@runtime_checkable Protocol` isinstance check**: `isinstance(model, SupportsEagle3)` checks ALL protocol attributes including inherited `has_own_lm_head` and `has_own_embed_tokens` from `SupportsEagleBase`. Missing attributes caused the check to fail silently.
   - **Fix**: Added `has_own_lm_head = False` and `has_own_embed_tokens = False` class variables to `MiniMaxM2ForCausalLM`

3. **Speculators format incompatible with vLLM**: Speculators saves as `Eagle3DraftModel` with `layers.0.*` weight prefix. vLLM expects `LlamaForCausalLMEagle3`/`Eagle3LlamaForCausalLM` architecture with `midlayer.*` prefix and standard LlamaConfig.
   - **Fix**: Created conversion script `k8s/convert_speculators_to_vllm_eagle3.py`

4. **torch.compile shape mismatch with non-Aurora draft models**: `AssertionError: expected size 2048==1280` during `profile_run`. The Aurora model has `num_attention_heads=24` (QKV per TP shard = 1280), but our Novita model has `num_attention_heads=48` (QKV per TP shard = 2048). The compiled graph from one draft model cannot be reused for another with different attention configs. Per-eval `TORCHINDUCTOR_CACHE_DIR` isolation did not fix it — the issue is in vLLM's own compile cache, not torchinductor.
   - **Workaround**: Use `enforce_eager=True` for the Novita model to disable torch.compile entirely. This makes the comparison unfair (eager mode is slower), so throughput numbers are not directly comparable.
   - **Root cause**: Likely vLLM's compile cache at `~/.cache/vllm/` persists across subprocesses in the same pod, and the cache key doesn't differentiate draft model attention configs

5. **Aux hidden state capture timing**: Initial patch captured hidden states BEFORE the layer executed (`hidden_states + residual` before `layer()`), but speculators data generation captures AFTER (`layer()` then `hidden_states + residual`). Fixed to capture after layer execution, but this did not improve acceptance rates — the off-by-one had minimal impact.

### Results

| Model | Tokens/s | Speedup | Acceptance Length | Acc@0 | Acc@1 | Acc@2 |
|-------|----------|---------|-------------------|-------|-------|-------|
| **Baseline** (no spec) | 436.9 | 1.00x | - | - | - | - |
| **Aurora-Spec** | 509.6 | **1.17x** | 1.803 | 52.15% | 24.44% | 12.0% |
| **Novita-trained** (eager) | 108.6 | 0.25x | 1.255 | 24.24% | 1.2% | 0.02% |

- **Aurora-Spec**: 17% throughput improvement with mean acceptance length of 1.80 tokens
  - Acceptance rates: ~52% for 1st draft token, ~24% for 2nd, ~12% for 3rd
  - Decent performance considering it was trained for MiniMax-M2.1, not M2.5
- **Novita-trained**: Very poor acceptance rates (24% at position 0)
  - Throughput is 0.25x baseline (slower due to `enforce_eager` + draft overhead)
  - The model had 81.5% top-1 accuracy on Novita val data, but only 24% on general prompts
  - Cause: severe overfitting on 812 Novita samples (coding agent conversations)
  - The model learned to predict tokens in multi-turn coding agent conversations, but doesn't generalize to single-turn general/coding prompts

### Analysis

1. **Aurora-Spec is the better choice** for general MiniMax-M2.5 speculative decoding. Despite being trained for M2.1, it generalizes well to M2.5 and provides a real 17% speedup.

2. **Our Novita-trained model needs more diverse training data**. Training on only 812 domain-specific samples caused severe overfitting. To match or beat Aurora-Spec, we need:
   - 5000+ diverse samples (sharegpt + ultrachat + novita)
   - The Experiment 8 (50K online data) model may perform much better

3. **torch.compile compatibility**: Draft models with different `num_attention_heads` than Aurora (24) cause compile cache conflicts. This is a vLLM limitation that requires `enforce_eager=True` as a workaround, which significantly hurts throughput.

4. **MiniMax-M2.5 Eagle3 architecture notes**:
   - `hidden_size=3072`, `num_attention_heads=48`, `head_dim=128` → Q dimension (6144) = 2× hidden_size
   - Aurora uses `num_attention_heads=24` making Q = hidden_size (standard)
   - Eagle3 draft model's QKV input is always `2 × hidden_size` (embeds + hidden concatenated)
   - `fc` layer: `3 × hidden_size → hidden_size` (3 aux hidden states from layers 2, 31, 59)

- **Status**: COMPLETED

---

## Infrastructure Notes

- **CUDA compat lib conflict**: Must set `LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:...` in pod env
- **Data gen segfault on TP=8**: Use TP=2 (Qwen3-32B) or TP=4 (MiniMax-M2.5)
- **CNI bridge conflict on .18**: Fixed by deleting stale cni0 (`ip link delete cni0`)
- **No direct network in pods**: Pre-download data, set `HF_HUB_OFFLINE=1`
- **Proxy on .18**: `https_proxy=http://127.0.0.1:1083` for HuggingFace access
- **MiniMax-M2.5 requires vLLM >= 0.17.0**: v0.16.0 has no MiniMaxM2 model support
- **trust_remote_code for custom models**: Must add `trust_remote_code=True` to all `AutoConfig.from_pretrained` calls
- **NaN data from MoE models**: MiniMax-M2.5 occasionally produces NaN hidden states (~0.4%). Always validate generated data before training
- **shm sizing for MoE**: Large MoE models need 256Gi+ shared memory to avoid OOM with DataLoader workers
- **DeepGEMM warmup**: MoE models need ~4 min first-time kernel compilation on new nodes. Looks like a hang but is normal
- **Online training `_attn_implementation`**: `train_streaming.py` must set `transformer_layer_config._attn_implementation = "simple_flex_attention"` — this is a `classmethod` registration on transformers' global `AttentionInterface._global_mapping`
- **Multi-node rsync**: Can't rsync between two remote hosts directly; SSH to source and rsync from there
- **Eagle3 requires full model weights on training node**: Not just config — the embedding layer is loaded from target model safetensors
- **NCCL timeout for large-scale FSDP training**: Default 600s timeout is insufficient for validation/checkpoint with large val sets on MoE models. Set `NCCL_TIMEOUT=1800` env var (read by `utils.py:maybe_setup_distributed`)
- **Code sync to pod nodes**: K8s pods with hostPath mounts use the node's local filesystem. Always `rsync` code changes to the target node before restarting pods
