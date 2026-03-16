# Experiment 8: MiniMax-M2.5 Online/Streaming Training (50K ShareGPT)

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
- **Status**: COMPLETED

## Issues Encountered

1. **NCCL `_ALLGATHER_BASE` timeout at epoch boundary**: Training steps complete (303-677 steps/epoch) but crashes during validation/checkpoint phase. Root cause: default NCCL timeout (600s / 10 min) is too short for FSDP validation with ~5000 val files on a large MoE model. The `dist.reduce` calls in `val_epoch` accumulate latency.
   - **Failed 4 times** before fix:
     - Run 1: 1460 files, hung at work 402 after 20 steps
     - Run 2: 542 files, hung at work 142 after 28 steps
     - Run 3: 20276 files, hung at work 6062 after epoch 0 (303 steps)
     - Run 4: 40896 files, hung at work 12242 after epoch 0 (612 steps)
   - **Fix**: Set `NCCL_TIMEOUT=1800` env var → read in `init_process_group(timeout=timedelta(seconds=1800))`
   - Also removed redundant `torch.distributed.barrier()` calls between train/val/checkpoint
   - Added `torch.cuda.empty_cache()` before validation to reduce memory pressure

2. **Code not synced to training node**: Pod on .18 mounts hostPath `/root/develop/speculators` from .18's filesystem. Code edits on the dev machine don't auto-propagate. **Fix**: `rsync` code to .18 before pod restart.

## Training Results

| Epoch | Val Loss | Top-1 Acc | Cond Acc 1 | Cond Acc 2 |
|-------|----------|-----------|------------|------------|
| 0     | 9.636    | 54.9%     | 41.3%      | 34.8%      |
| 1     | 7.851    | 63.2%     | 50.2%      | 43.9%      |
| 2     | 6.817    | 68.2%     | 56.2%      | 51.4%      |
| 3     | 6.181    | 70.8%     | 59.7%      | 56.2%      |
| 4     | 5.803    | 72.2%     | 61.6%      | 58.8%      |
| 5     | 5.556    | 73.1%     | 62.8%      | 60.5%      |
| 6     | 5.377    | 73.7%     | 63.6%      | 61.7%      |
| 7     | 5.235    | 74.2%     | 64.2%      | 62.5%      |
| 8     | 5.131    | 74.6%     | 64.7%      | 62.9%      |
| **9** | **5.043**| **74.9%** | **65.3%**  | **63.8%**  |

- **Best checkpoint**: epoch 9 (val_loss=5.043, top1_acc=74.9%)
- **Training time**: ~7.5 hours (10 epochs, 8x H200), ~45 min/epoch
- **Datagen time**: ~6 hours (50000 samples, TP=4, .21)
- **Checkpoints**: `/data/output/minimax_m2.5_eagle3_online_50k/checkpoints/` on .18

## Comparison with Previous Experiments

| Metric | Offline 10K (Exp 4) | Online 5K (Exp 6) | Novita 812 (Exp 7) | **Online 50K (Exp 8)** |
|--------|--------------------|--------------------|--------------------|-----------------------|
| Data | sharegpt+UC (9957) | sharegpt (5000) | novita (812) | **sharegpt (50000)** |
| Val Loss | 9.118 | 10.166 | **1.952** | 5.043 |
| Top-1 Acc | 56.0% | 52.9% | 81.5%* | **74.9%** |
| Cond Acc 1 | 40.3% | 39.0% | 83.8%* | **65.3%** |
| Cond Acc 2 | 32.9% | 32.7% | 86.2%* | **63.8%** |
| Training | ~70 min | ~55 min | ~50 min | ~7.5 hours |

*Novita results are domain-specific (coding agent conversations) and not directly comparable.

## Analysis

- 50K ShareGPT achieves **74.9% top-1 accuracy**, a massive improvement over offline 10K (56.0%) and online 5K (52.9%)
- More data consistently helps: 5K→50K provides +22pp accuracy boost (52.9%→74.9%)
- Conditional accuracy is excellent: 65.3% / 63.8% for 2nd/3rd draft tokens
- Loss is still improving at epoch 9 — more epochs could yield further gains
- The model has not yet converged (epoch 8→9 still improves), suggesting 15-20 epochs may be optimal for 50K data
- The NCCL timeout issue was the main blocker; once resolved, all 10 epoch boundaries crossed without issues
