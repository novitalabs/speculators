# Experiment 10: MiniMax-M2.5 Eagle3 Training (Novita 5K)

Training with 5000 samples from Novita production API logs (weilan55/novita20260309).

- **Datagen node**: .23 (8x H200 143GB)
- **Training node**: .14 (8x H200 143GB)
- **Image**: speculators:v0.17.0
- **Data**: `weilan55/novita20260309` → 10K conversations preprocessed from 800K API log records
  - Source: Novita production MiniMax-M2.5 API gateway logs (coding agent)
  - Preprocessed: `k8s/preprocess_novita_logs.py` → 10,000 conversations (avg 62.2 turns)
  - Datagen: 5000 samples, TP=4 on .23
- **Config**:
  - Training: 8 GPUs FSDP, lr=3e-5, 10 epochs, seq_len=8192
  - Constant LR scheduler
- **Output**: `/data/output/minimax_m2.5_eagle3_novita2/`
- **Status**: COMPLETED

## Training Results

| Epoch | Val Loss | Top-1 Acc | Cond Acc 1 | Cond Acc 2 |
|-------|----------|-----------|------------|------------|
| 0     | 2.254    | 57.3%     | 38.3%      | 30.3%      |
| 1     | 1.581    | 63.1%     | 44.9%      | 36.6%      |
| 2     | 1.230    | 66.7%     | 48.5%      | 40.0%      |
| 3     | 1.053    | 68.8%     | 50.8%      | 42.3%      |
| 4     | 0.942    | 70.1%     | 52.5%      | 44.0%      |
| 5     | 0.872    | 71.0%     | 53.7%      | 45.2%      |
| 6     | 0.822    | 71.7%     | 54.5%      | 46.1%      |
| 7     | 0.790    | 72.1%     | 55.3%      | 46.8%      |
| **8** | **0.776**| **72.5%** | **57.0%**  | **46.7%**  |
| 9     | 0.782    | 72.4%     | 55.7%      | 47.0%      |

- **Best checkpoint**: epoch 8 (val_loss=0.776, top1_acc=72.5%)
- **vLLM conversion**: `k8s/convert_speculators_to_vllm_eagle3.py` → `/data/output/minimax_m2.5_eagle3_novita2_vllm/`

## Analysis

- 5K Novita samples (val_loss=0.776) dramatically lower loss than 812 Novita (1.952) and 5K ShareGPT (10.166)
- Top-1 accuracy 72.5% vs 81.5% (812 Novita) reflects less overfitting with 6× more data
- Still converging at epoch 8 — more data or epochs could improve further
