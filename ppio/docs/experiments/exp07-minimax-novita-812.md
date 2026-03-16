# Experiment 7: MiniMax-M2.5 Online/Streaming Training (Novita 812)

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

## Training Results

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

## Comparison

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
