# Experiment 4: MiniMax-M2.5 + ShareGPT/UltraChat

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

## Training Results

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

## Issues Encountered

1. **vLLM version**: v0.16.0 does not support MiniMax-M2.5 → upgraded to v0.17.0
2. **trust_remote_code**: MiniMax-M2.5 custom model code requires `trust_remote_code=True` in all `AutoConfig.from_pretrained` calls (train.py x2, eagle3/core.py x1)
3. **NaN hidden states**: MiniMax-M2.5 produces NaN hidden states for ~0.4% of inputs during data generation (43/9957 files). These poison the entire training via gradient updates. Fix: removed corrupt files + added `nan_to_num` safety check in data loader
4. **shm OOM**: MoE model requires larger shared memory → increased to 256Gi

## Analysis

- MiniMax-M2.5 achieves **56% top-1 accuracy**, outperforming Qwen3-32B baseline (52%)
- Model converges by epoch 8, with epoch 9 showing no improvement (slight regression)
- Cosine schedule with warmup helped stabilize early training (no NaN from optimizer)
- Conditional accuracy for 2nd/3rd draft tokens: 40.3% / 32.9%, decent for speculative decoding
