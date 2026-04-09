# Experiment 20-2: Chinese Subset of Nemotron-v2

## Status: COMPLETE (converged ~epoch 30, final eval ckpt40, 2026-04-08)

## Motivation

Exp20 trained on full nemotron-v2 (6.3M samples, 10 categories) achieving Novita 2.02x / ZClawBench 1.18x speedup. However, ZClawBench performance is still limited — analysis shows 92% of ZClawBench trajectories contain Chinese or mixed Chinese+English content. Training on all 6.3M samples dilutes Chinese representation (~20% of chat split = ~125K Chinese entries out of 6.3M total = ~2%).

**Hypothesis**: Training on Chinese-only subset will improve ZClawBench acceptance rate by better matching the eval distribution, at the cost of Novita (English) performance.

## Dataset

### Source
nemotron-v2-jsonl (`/data/tengwan/datasets/nemotron-v2-jsonl/`, 94GB, 6.3M samples)

### Extraction
Chinese detection: `re.compile(r'[\u4e00-\u9fff]')` on user/assistant message content. Excludes `multilingual_ja.jsonl` (Japanese uses kanji but is not Chinese).

| Source File | Total | Chinese | % |
|------------|-------|---------|---|
| chat.jsonl | 627K | 121K | 19.3% |
| multilingual_fr.jsonl | 1.0M | 20.6K | 2.1% |
| multilingual_es.jsonl | 936K | 16.8K | 1.8% |
| multilingual_it.jsonl | 1.0M | 17.1K | 1.7% |
| multilingual_de.jsonl | 1.0M | 15.2K | 1.5% |
| stem.jsonl | 355K | 4.5K | 1.3% |
| Others (code/math/multilingual) | 415K | 281 | <0.1% |
| **Total** | **5.4M** | **195,624** | **3.6%** |

### Output
`/data/tengwan/datasets/nemotron-v2-chinese/conversations.jsonl` (195,624 conversations, 2.9GB)

Extraction script: `scripts/extract_chinese_data.py`

## Architecture

Same as Exp19/20: Aurora (24 attention heads, 8192 intermediate, 32K draft vocab, RoPE theta 5M)

## Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | nemotron-v2-chinese (~125K samples) |
| Datagen node | .17 (10.83.115.17) |
| Training node | .18 (10.83.115.18) |
| seq_length | 8192 |
| lr | 3e-5 |
| GPUs | 8 (datagen) + 8 (training) |
| Pipeline | Continuous datagen + streaming training |
| Buffer | 500GB max |

## Expected Outcome

- **ZClawBench**: Higher Acc@0 and speedup (training distribution matches eval)
- **Novita**: Possibly lower (less English data coverage)
- **Convergence**: Faster (125K vs 6.3M dataset, similar to Exp19 scale)

## Comparison Baselines

| Experiment | Dataset | ZClaw Speedup | Novita Speedup |
|-----------|---------|---------------|----------------|
| Exp19 | novita_merged (166K) | 1.11x | 1.53x |
| Exp20 | nemotron-v2 (6.3M) | 1.18x | 2.02x |
| **Exp20-2** | **nemotron-v2 Chinese (~125K)** | **1.34x** | not evaluated |

## Eval Results

### Training Progress

- Converged ~epoch 30 (val_loss plateau: 3.723 at epoch 30 → 3.643 at epoch 41)
- Final eval: epoch 40 checkpoint
- Peak Val Acc@0: 79.1%, Val Loss: 3.643

### Simple ZClawBench (116 prompts, MAX_MODEL_LEN=32768)

| Model | Tok/s | Speedup | Acc@0 | Acc Length |
|-------|-------|---------|-------|------------|
| Baseline (no spec) | 3020.9 | 1.00x | - | - |
| Exp20-2 ckpt29 | 4071.8 | 1.35x | 48.5% | 1.754 |
| **Exp20-2 ckpt40** | **4040.5** | **1.34x** | **48.6%** | **1.756** |

### Full ZClawBench (649 trajectories, bucketed by input length, ckpt40)

| Bucket | #Prompts | Avg Input | Baseline Tok/s | Exp20-2 Tok/s | Speedup | Acc@0 | Acc Length |
|--------|----------|-----------|----------------|---------------|---------|-------|------------|
| short (0-2K) | 210 | 847 | 5394.6 | 3912.4 | 0.73x | 50.9% | 1.794 |
| med (2K-8K) | 320 | 4601 | 2322.5 | 2531.9 | **1.09x** | 53.7% | 1.865 |
| long (8K-32K) | 119 | 12177 | 905.5 | 1024.2 | **1.13x** | 52.6% | 1.820 |

### Cross-Experiment Comparison (Final)

| Experiment | Dataset | Simple ZClaw | Full Short | Full Med | Full Long | Acc@0 (simple) |
|-----------|---------|-------------|------------|----------|-----------|----------------|
| Exp19 | novita_merged (166K) | 1.11x | 0.68x | 1.04x | 1.09x | 43.1% |
| Exp20 | nemotron-v2 (6.3M) | 1.18x | - | - | - | - |
| **Exp20-2** | **nemotron-v2 Chinese (125K)** | **1.34x** | **0.73x** | **1.09x** | **1.13x** | **48.6%** |

### Analysis

1. **Chinese subset hypothesis validated**: ZClawBench simple speedup 1.34x vs Exp19 1.11x / Exp20 1.18x, confirming distribution match matters
2. **Acc@0 ~51-54%** across all buckets, a major improvement over Exp19's ~31-43%
3. **Short bucket still <1x** (0.73x): prefill dominates for short inputs, spec decode overhead outweighs the benefit; adding novita data (Exp20-3) may help
4. **Med/long buckets improved** over Exp19: med 1.09x (vs 1.04x), long 1.13x (vs 1.09x)
5. **Fully converged at epoch 30**: ckpt29 (1.35x) and ckpt40 (1.34x) are essentially identical

