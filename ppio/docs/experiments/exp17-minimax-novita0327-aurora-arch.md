# Experiment 17: MiniMax-M2.5 Eagle3 Aurora-Arch with novita20260327

Same Aurora architecture as Exp14/15, retrained on the newer weilan55/novita20260327 dataset.

- **Nodes**: .17 (datagen, 8x H200), .18 (training, 8x H200)
- **Image**: speculators:v0.17.0
- **Dataset**: weilan55/novita20260327
- **Architecture**: Aurora-like (same as Exp14/15)
  - num_attention_heads: 24
  - intermediate_size: 8192
  - rope_theta: 5000000
  - draft_vocab_size: 32000
  - hidden_size: 3072, num_kv_heads: 8, head_dim: 128
- **Config**:
  - Training: 8 GPU FSDP, lr=3e-5, streaming with buffer cleanup
  - Buffer max: 500GB (conservative, learned from Exp15 disk-full crashes)
  - Vocab mapping: d2t/t2d from token_freq.pt
  - Architecture overrides: `--override-num-attention-heads 24 --override-intermediate-size 8192 --override-rope-theta 5000000`
- **Output**: /data/output/minimax_m2.5_eagle3_novita0327/
- **Baseline**: Exp15 ckpt67 (63.2% Acc@0)
- **Status**: STOPPED (2026-04-01, epoch 141, replaced by Exp18)
- **Key Files**:
  - `k8s/run_minimax_m2.5_novita0327_datagen.sh` -- datagen script
  - `k8s/k8s-minimax-m2.5-novita0327-datagen.yaml` -- datagen pod (.12)
  - `k8s/run_minimax_m2.5_novita0327_train.sh` -- training script
  - `k8s/k8s-minimax-m2.5-novita0327-train.yaml` -- training pod (.23)

## Progress

- 2026-03-30: Restarted experiment on .17/.18 (previous .12/.23 nodes were reinstalled, all prior data lost)
- 2026-03-30: Datagen downloading dataset from HF, train pod installing deps and waiting for data
- 2026-03-30: Datagen stalled at 2446GB output (exceeded 1024GB limit). Cleaned old experiment data on .17 (aurora_loss 479G + aurora_static 176G + aurora_online 120G) and evicted 52K old .pt files. Train pod crashed (NCCL timeout), recreated. Added background cleanup to datagen and enabled buffer_cleanup on train side.
- 2026-03-31: Both pods stable overnight. Datagen at 57K/186K (30.8%), training epoch 51. Metrics: full_acc_0=48.7%, loss=0.043.
- 2026-03-31: Datagen at 74K/186K (39.8%), training epoch 75. Metrics improving: full_acc_0=70.2%, full_acc_1=63.0%, full_acc_2=56.8%.
- 2026-04-01: Datagen at 104K/186K (55.7%), training epoch 123. Loss=0.043, acc fluctuating with new data.
- 2026-04-01: Datagen at 113K/186K (60.3%), training epoch 141. Loss=0.107, full_acc_0=46.7%. Metrics plateaued — no improvement since epoch 75 despite 2x more data. Root cause identified: session-level duplication (825K convs from 63K unique tasks). **Experiment stopped, replaced by Exp18** (session-dedup + multi-dataset merge).

### 2026-04-01 — Eval ckpt86 (Inference Throughput)

Selected ckpt86 as best by val_acc0=56.2% (top across all checkpoints). Eval ran on .17 GPU 4-7 (alongside datagen on GPU 0-3) using vLLM V1, TP=4.

**Novita eval (10 prompts, 512 tokens):**

| Metric | Exp17 ckpt86 | Exp15 ckpt67 (baseline) |
|--------|-------------|------------------------|
| Acc@0 | 43.1% | 63.2% |
| Acc@1 | 16.1% | — |
| Acc@2 | 3.3% | — |
| Acceptance Length | 1.624 | — |
| Throughput | 440.1 tok/s | ~434 tok/s |

**ZClawBench eval (116 agent prompts, 512 tokens):**

| Metric | Exp17 ckpt86 |
|--------|-------------|
| Acc@0 | 23.4% |
| Acc@1 | 5.6% |
| Acc@2 | 1.6% |
| Acceptance Length | 1.305 |
| Throughput | 2480.1 tok/s |

**Analysis**: Inference Acc@0 (43.1%) significantly below Exp15 baseline (63.2%) despite val_acc0 looking reasonable (56.2%). See root cause analysis below.

### 2026-04-01 — Root Cause Analysis: Dataset Diversity

Investigated why Exp17 ckpt86 underperforms Exp15 ckpt67 despite larger dataset and more training epochs.

**Dataset diversity comparison:**

| | Exp15 (novita20260320) | Exp17 (novita20260327) |
|--|----------------------|----------------------|
| Total conversations | 221K | 825K |
| Unique system prompts | **1,268** | **188** |
| Top-1 system prompt share | 66.2% (147K) | 60.2% (497K) |
| Top-2 cumulative share | 78.0% | **99.4%** |
| Non-coding-agent data | ~22% (49K) | **<1%** (4K) |

**novita20260327 is almost entirely a single task type**: 99.4% of conversations share the same two system prompt variants — "You are an expert AI software engineering agent" — solving GitHub issues (`<issue>`) or tasks (`<task>`). Only 188 unique system prompt patterns vs Exp15's 1,268.

**Exp15's novita20260320 was significantly more diverse**: 22% of data came from non-coding-agent sources (OpenClaw personal assistant, code review bots, general AI assistants, customer service, etc.), providing broader token distribution coverage.

**Data ordering in training:**
- Preprocessing shuffles conversations (`raw_dataset.shuffle(seed=0)`)
- `MultipackDistributedBatchSamplerV2` re-shuffles indices each epoch with `rng.permutation(seed + epoch)`, then bin-packs by length (LPT algorithm)
- So within-epoch order is randomized — but **the data pool itself lacks diversity**, making shuffle ineffective at reducing inter-batch correlation

**Deeper analysis — session-level duplication** (the real problem):

Coding agent being the dominant category (~90%) is expected and reflects production distribution. The actual issue is **same-session repetition**: turn_dropout expands each conversation into multiple training samples with identical prefixes.

| Dataset | Total convs | Unique tasks* | Unique % | >10x repeated |
|---------|------------|--------------|----------|---------------|
| novita20260309 | 52K | 7.3K | 14.0% | 50.8% |
| novita20260320 | 222K | 30.6K | 13.8% | 54.5% |
| novita20260327 | 825K | 63.4K | **7.7%** | **74.1%** |

*Unique task = unique first user message (500 char fingerprint)

Most duplicated tasks in novita20260327 appear **500-700 times** each. The model sees the same issue description + system prompt hundreds of times with slightly different truncation points, causing overfitting to specific token sequences rather than learning general draft prediction.

**Conclusion**: The performance gap is caused by **session-level data duplication**, not lack of category diversity:
1. 825K conversations contain only 63K unique tasks — 74% are >10x duplicates
2. Turn dropout creates many samples from one conversation, but all share identical prefixes
3. Model overfits to repeated prefixes → poor generalization to unseen prompts
4. Exp15's novita20260320 had the same issue but to a lesser degree (54.5% >10x vs 74.1%)

**Recommendation → Exp18**: Merge all three datasets with **session-level dedup + per-task sampling cap** (max 5 samples per unique task). This preserves turn_dropout diversity without excessive repetition. See [Exp18](exp18-minimax-merged-diversity.md).

## Issue Log

- 2026-03-30: Original nodes (.12/.23) were reinstalled, experiment data completely lost. Redeployed from scratch on .17/.18.
- 2026-03-30: Datagen output dir hit 2446GB, stalling generation. Root cause: no cleanup on datagen side — training node's buffer_cleanup only cleans local copy, not remote. Fix: added `_start_output_cleanup_thread()` to `data_generation_offline.py` (auto-evicts oldest files at 90% capacity). Also enabled `buffer_cleanup.py` on train side (was hardcoded disabled for .23 which had 4.9TB disk, but .18 only has 7TB total).
- 2026-03-30: Train pod crashed with `FileNotFoundError: data_27287.pt`. Root cause: datagen cleanup deleted files on .17 that were referenced in manifest but not yet synced to .18. Fix: recreated train pod; sync_datagen.sh manifest updater (every 30s) keeps manifest in sync with actual local files.
- 2026-04-01: Eval on GPU 4-7 failed with OOM (1.24 GiB free). Root cause: previous eval's spawn'd worker processes leaked ~140GB/GPU. Fix: manually killed leaked PIDs, re-ran eval.
