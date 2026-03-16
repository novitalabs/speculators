# Experiment 1: Qwen3-32B + ShareGPT/UltraChat (Baseline)

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
