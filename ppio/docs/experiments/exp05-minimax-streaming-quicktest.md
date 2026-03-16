# Experiment 5: MiniMax-M2.5 Online/Streaming Training (Quicktest)

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

## Issues Encountered

1. **DeepGEMM warmup on new node (.21)**: First-time kernel compilation for MoE model (~3300 kernels) takes ~4 minutes. Looks like a hang (shm_broadcast timeout messages) but is actually normal. **Fix**: ran a debug pod first to warm up the kernel cache, or just wait.

2. **Training needs full model weights**: Eagle3 training calls `load_model_layers` to load embedding weights from the target model. Initially only copied config files to .18. **Fix**: copied full 215GB MiniMax-M2.5 model to .18.

3. **`KeyError: None` in LlamaAttention.forward**: `train_streaming.py` was missing `transformer_layer_config._attn_implementation = "simple_flex_attention"` (present in `train.py` but not copied to the streaming variant). Without this, the LlamaConfig's `_attn_implementation` defaults to `None`, which isn't registered in transformers' `ALL_ATTENTION_FUNCTIONS`. **Fix**: added the missing line to `create_transformer_layer_config()` in `train_streaming.py`.

4. **rsync between two remotes**: Can't rsync directly between two remote hosts. **Fix**: SSH to source node and rsync from there to destination.

## Training Results

| Epoch | Val Loss | Top-1 Acc |
|-------|----------|-----------|
| 0     | 27.526   | 1.7%      |
| 2     | 23.459   | ~2%       |

- 3 epochs completed in ~15 minutes
- 90 train / 10 val file split (from 100 total)
- Loss improving but accuracy very low (expected with only 100 samples)
- Manifest train_count correctly tracked: 90 train files at count=10, 10 val files at count=0
- **Status**: COMPLETED (pipeline validation successful)
