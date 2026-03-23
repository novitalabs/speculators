# Experiment 15: MiniMax-M2.5 Eagle3 Aurora-Arch with novita20260320 (Larger Dataset)

Scales up Exp14's Aurora-architecture Eagle3 draft model with a new, larger dataset `weilan55/novita20260320`. Same architecture as Exp14, only the dataset changes.

- **Nodes**: .22 (datagen, 8x H200), .23 (training, 8x H200)
- **Image**: speculators:v0.17.0
- **Dataset**: `weilan55/novita20260320` (downloaded from HF, preprocessed with `--min-turns 2`)
- **Architecture**: Aurora-like (same as Exp14):

  | Parameter | Value |
  |-----------|-------|
  | num_attention_heads | 24 |
  | intermediate_size | 8192 |
  | draft_vocab_size | 32000 |
  | rope_theta | 5000000 |
  | hidden_size | 3072 |
  | num_kv_heads | 8 |
  | head_dim | 128 |

- **Config**:
  - Datagen: TP=4, batch_size=4, seq_length=8192 on .22
  - Training: 8 GPU FSDP, lr=3e-5, streaming with buffer cleanup on .23
  - Vocab mapping: `build_vocab_mapping.py` with `token_freq.pt` from datagen output (new dataset distribution)
  - Data synced from .22 to .23 via rsync
- **Output**: `/data/output/minimax_m2.5_eagle3_novita0320/`
- **Key Files**:
  - `k8s/run_minimax_m2.5_novita0320_datagen.sh` — datagen script (downloads HF dataset + generates)
  - `k8s/k8s-minimax-m2.5-novita0320-datagen.yaml` — datagen pod (.22)
  - `k8s/run_minimax_m2.5_novita0320_train.sh` — training script
  - `k8s/k8s-minimax-m2.5-novita0320-train.yaml` — training pod (.23)
- **Changes from Exp 14**:
  - New dataset: `weilan55/novita20260320` (larger than Exp14's 52K novita20260309)
  - Separate datagen on .22 (Exp14 reused Exp13's datagen data)
  - Training on .23 (Exp14 used .18)
  - Vocab mapping uses new dataset's token_freq.pt (fallback to Exp14's if unavailable)
- **Status**: TRAINING COMPLETE (epoch 68 on .10), EVAL IN PROGRESS on .10
- **Expected Outcome**: Better or comparable acceptance rate to Exp14 with more diverse training data from the larger dataset

## Progress

- **2026-03-22**: Training completed to epoch 68 on .10. Training pod exited (Error state, deleted).
- **2026-03-23**: Inference eval pod `minimax-m2-5-eval-exp15` deployed on .10, currently loading baseline model.

## Issue Log

_(No issues yet)_
