# Experiment 18: MiniMax-M2.5 Eagle3 Aurora-Arch with Merged Novita (Session-Dedup)

Combines novita20260309 + novita20260320 + novita20260327 with session-level deduplication to address Exp17's overfitting problem. Same Aurora architecture as Exp14/15/17.

- **Nodes**: .17 (datagen, 8x H200), .18 (training, 8x H200)
- **Image**: speculators:v0.17.0
- **Dataset**: Merged novita (session-dedup, max 5 samples per unique task)
  - Source: novita20260309 (52K), novita20260320 (222K), novita20260327 (825K)
  - After merge: 165,771 conversations from ~80K unique tasks (max 5 per task)
- **Architecture**: Aurora-like (same as Exp14/15/17)
  - num_attention_heads: 24, intermediate_size: 8192, rope_theta: 5000000
  - draft_vocab_size: 32000, hidden_size: 3072, num_kv_heads: 8, head_dim: 128
- **Config**:
  - Training: 8 GPU FSDP, lr=3e-5, streaming with buffer cleanup
  - Buffer max: 500GB, cleanup enabled on both datagen and train sides
  - Vocab mapping: d2t/t2d from token_freq.pt
- **Output**: /data/output/minimax_m2.5_eagle3_exp18/
- **Baseline**: Exp15 ckpt67 (63.2% Acc@0 Novita, 39.6% Acc@0 ZClawBench)
- **Status**: STOPPED (2026-04-03, epoch 44, replaced by Exp19)
- **Result**: **ckpt42 matches Exp15 on Novita (60.4% vs 63.2%) and beats it on ZClawBench (41.3% vs 39.6%)**
- **Key Files**:
  - `k8s/merge_novita_datasets.py` — session-dedup merge script
  - `k8s/run_minimax_m2.5_exp18_datagen.sh` — datagen script
  - `k8s/k8s-minimax-m2.5-exp18-datagen.yaml` — datagen pod (.17)
  - `k8s/run_minimax_m2.5_exp18_train.sh` — training script
  - `k8s/k8s-minimax-m2.5-exp18-train.yaml` — training pod (.18)

## Motivation

Exp17 trained on novita20260327 (825K conversations) achieved only 43.1% Acc@0, far below Exp15's 63.2%. Root cause analysis revealed **session-level data duplication**:

| Dataset | Total convs | Unique tasks | >10x repeated |
|---------|------------|-------------|---------------|
| novita20260309 | 52K | 7.3K (14%) | 50.8% |
| novita20260320 | 222K | 30.6K (14%) | 54.5% |
| novita20260327 | 825K | 63.4K (7.7%) | **74.1%** |

Turn dropout expands each conversation into multiple training samples with identical prefixes. When the same GitHub issue appears 500-700 times, the model overfits to those specific token sequences rather than learning generalizable draft prediction.

## Approach

**Session-level dedup + per-task sampling cap** (`k8s/merge_novita_datasets.py`):

1. Fingerprint each conversation by hash of first user message (500 chars)
2. Group by fingerprint across all 3 datasets
3. Cap each group to 5 samples, preferring longer conversations (more complete turns)
4. This preserves turn_dropout diversity (different truncation points) without >5x repetition
5. Natural task distribution is preserved (coding agent still ~90%)

## Known Limitations

### seq_length=8192 truncation

Current datagen uses `--seq-length 8192`, truncating all conversations to 8K tokens. Conversation length distribution in merged dataset:

| Percentile | Est. tokens | Impact |
|-----------|------------|--------|
| p50 | ~1.7K | Within limit |
| p75 | ~2.2K | Within limit |
| p90 | ~7.7K | Near limit |
| p95 | ~13K | **Truncated** |
| p99 | ~42K | **Severely truncated** |

9.4% of conversations (~15.6K) exceed 8K tokens. These long conversations (multi-turn tool calls, debugging sessions, code modifications) are truncated to only their opening portion — model only learns the beginning of these interactions.

This is an inherited limitation from Exp14/15/17. Increasing to 32K would cover p99+ but requires more KV cache memory. Current setup (TP=4, H200 144GB, gpu_memory_utilization=0.85) uses ~75GB/GPU with ~47GB available for KV cache — likely sufficient for 32K with reduced batch_size.

**Future improvement (Exp19+)**: Test `--seq-length 32768` with `--batch-size 1` or `--batch-size 2` to capture full-length coding agent conversations.

### Datagen single-pass limitation

Current datagen runs one pass over the dataset and exits. With buffer cleanup active, early-generated samples get evicted before training sees them enough times. In Exp18, 90% of generated data (148K/165K samples) was permanently lost to cleanup — training was left cycling over a shrinking 17K-sample pool.

**Root cause**: datagen speed (~1.2s/batch) << training speed (minutes per epoch on buffer). Buffer fills up, cleanup evicts old files, datagen eventually finishes and exits. The remaining files get overfit.

**Future improvement (Exp19+)**: Datagen should run in continuous loop mode:
1. **Round 1 (uniform)**: Generate all samples with equal weight — ensure full dataset coverage
2. **Round 2+ (difficulty-weighted)**: Re-generate samples based on training difficulty feedback
   - Training records per-sample loss → writes difficulty scores to a shared file
   - Datagen reads difficulty scores → prioritizes high-loss (hard) samples for regeneration
   - Low-loss (easy) samples get regenerated less frequently → implicit curriculum learning
3. **Buffer becomes prioritized replay**: eviction prefers low-difficulty, high-train-count files
4. **Datagen never exits**: continuously loops over the dataset, adapting sampling weights each round

## Eval Results (Inference)

### ckpt42 (val_acc0=60.6%, best at time of eval)

Evaluated on .17 with TP=4, vLLM V1, gpu_memory_utilization=0.95.

**Novita eval (10 prompts, 512 tokens):**

| Model | Tokens/s | Acc@0 | Acc@1 | Acc@2 | Acc Len |
|-------|----------|-------|-------|-------|---------|
| Exp15 ckpt67 (baseline) | 538.5 | 63.2% | 34.8% | 15.5% | 2.135 |
| Exp17 ckpt86 (no dedup) | 440.1 | 43.1% | 16.1% | 3.3% | 1.624 |
| **Exp18 ckpt42 (session dedup)** | **506.8** | **60.4%** | **32.3%** | **16.4%** | **2.091** |

**ZClawBench eval (116 agent prompts, 512 tokens):**

| Model | Tokens/s | Acc@0 | Acc@1 | Acc@2 | Acc Len |
|-------|----------|-------|-------|-------|---------|
| Exp15 ckpt67 (baseline) | 225.8 | 39.6% | 14.2% | 5.1% | 1.589 |
| Exp17 ckpt86 (no dedup) | 2480.1 | 23.4% | 5.6% | 1.6% | 1.305 |
| **Exp18 ckpt42 (session dedup)** | **3402.9** | **41.3%** | **15.6%** | **5.9%** | **1.628** |

**Key findings:**
- **Session dedup validates**: Exp18 recovers from Exp17's 43.1% to 60.4% Acc@0 on Novita (+17.3pp), confirming session-level duplication was the root cause
- **ZClawBench: new best**: 41.3% Acc@0 beats Exp15's 39.6% (+1.7pp) — better generalization to agent tasks despite less total data
- **Novita: near Exp15**: 60.4% vs 63.2% (−2.8pp), with training still ongoing (epoch 43, may improve)
- **Acc@2 improves**: 16.4% vs 15.5% (Novita) and 5.9% vs 5.1% (ZClawBench) — deeper speculation is more accurate
- **Throughput note**: Novita/ZClawBench throughput numbers not directly comparable across experiments (different nodes, batch conditions)

## Training Metrics

| Epoch | train/loss | train/full_acc_0 | Notes |
|-------|-----------|-----------------|-------|
| 6 | 5.5 | 58.1% | Resumed from ckpt5 |
| 10 | 3.9 | 66.3% | |
| 13 | 2.4 | 71.0% | |
| 43 | 0.94 | 91.0% | Best val_acc0=61.2% |

Val acc0 top checkpoints: ckpt43 (61.2%), ckpt42 (60.6%), ckpt13 (60.3%).

## Progress

- 2026-04-01: Experiment created. Exp17 stopped (epoch 141, datagen 60.3%).
- 2026-04-01: Datasets merged: 1.1M → 165,771 conversations (session-dedup, max 5/task). Datagen and train pods deployed on .17/.18.
- 2026-04-01: Datagen at 4960/41443 batches (12%), train in torch.compile phase.
- 2026-04-01: Datagen cleanup stalled (old file-count-based logic). Fixed with size-based eviction (`_evict_to_budget`). Restarted both pods.
- 2026-04-02: Train crashed (FileNotFoundError) — buffer_cleanup deleted files during epoch transition. Root cause: epoch lock released before DataLoader rebuilt. Fixed: lock now held until after DataLoader rebuild.
- 2026-04-02: Train crashed again (NCCL timeout). Restarted, resumed from ckpt5.
- 2026-04-02: Datagen completed (35004 batches, 165771 samples). Training stable at epoch 43.
- 2026-04-03: Eval ckpt42: Novita 60.4% Acc@0, ZClawBench 41.3% Acc@0 — matches/beats Exp15 baseline.
- 2026-04-03: Training reached epoch 44. Data pool severely degraded: 90% of generated samples lost to cleanup (165K→17K). Remaining files overtrained (avg train_count=11.6). **Experiment stopped, replaced by Exp19** (continuous datagen + difficulty-weighted resampling).

## Issue Log

- 2026-04-01: .17 initially blocked by dynamo-system prod pod (8 GPU). Resolved after pod moved off. .18 had stuck Terminating dynamo pod — force deleted to free GPUs.
- 2026-04-01: Train pod failed with "No such file" — .18 had stale code. Fixed by rsync'ing k8s/ and scripts/ to .18.
- 2026-04-01: Datagen cleanup thread used file-count estimate (avg 35MB) but actual files were 55MB avg → cleanup never triggered, datagen stalled at 1024GB. Fix: rewrote `_start_output_cleanup_thread` to use size-based eviction with high/low watermarks (85%/70% of budget).
- 2026-04-02: Train crash (FileNotFoundError data_13832.pt). Root cause: buffer_cleanup on .18 deleted 5000 files at 04:57, training tried to read them at 04:58. Epoch lock was released after `increment_train_count` but before DataLoader rebuild — 46-second race window. Fix: epoch lock now held through DataLoader rebuild; released only temporarily during cleanup+sync wait, then re-acquired before reading manifest and building DataLoader.
- 2026-04-02: NCCL timeout (SeqNum=6573, `_ALLGATHER_BASE`, 30min). Likely intermittent .18 GPU/network issue. Restarted pod, resumed from ckpt.
