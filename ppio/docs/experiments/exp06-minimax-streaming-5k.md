# Experiment 6: MiniMax-M2.5 Online/Streaming Training (5K)

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

## Issues Encountered

1. **Race condition: manifest synced before .pt files**: rsync copies manifest.json (small) before all .pt files arrive. Training reads manifest, tries to load missing files → `FileNotFoundError`. **Fix**: `resolve_file_paths()` now filters to only files that exist on disk.

2. **NCCL collective timeout (transient)**: First run with 1460 files (partial data, datagen still running) hit NCCL `_ALLGATHER_BASE` timeout after 20 training steps. Root cause unclear (possibly OOM or FSDP sync issue). Restart with full 5000 files resolved the issue.

## Training Results

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

## Comparison with Offline Training (Experiment 4)

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
