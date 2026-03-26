# Experiment 16: Aurora Accept/Discard Loss

A/B comparison against Exp15 (standard KL). Same architecture, same data, only the loss function changes.

- **Node**: .17 (8x H200)
- **Image**: speculators:v0.17.0
- **Dataset**: Same as Exp15 — 6K .pt files rsynced from .10 (originally from novita20260320, 114K conversations)
- **Architecture**: Same as Exp14/15 (Aurora-like):

  | Parameter | Value |
  |-----------|-------|
  | num_attention_heads | 24 |
  | intermediate_size | 8192 |
  | draft_vocab_size | 32000 |
  | rope_theta | 5000000 |

- **Loss**: Aurora accept/discard (Phase 1 — dynamic mask)
  - `--aurora-loss` enabled
  - `--lambda-discard 0.1`
  - `--discard-top-k 10`
- **Training**: 8 GPU FSDP, lr=3e-5, 70 epochs, `train.py` (non-streaming, all data local)
- **Output**: `/data/output/minimax_m2.5_eagle3_aurora_loss/`
- **Key Files**:
  - `k8s/run_minimax_m2.5_aurora_loss_train.sh` — training script
  - `k8s/k8s-minimax-m2.5-aurora-loss-train.yaml` — K8s pod (.17)
- **Baseline**: Exp15 ckpt67 (standard KL, 63.2% Acc@0, 1.01x speedup)
- **Status**: TRAINING

## Key Metrics to Watch

- `aurora_accept_ratio_0`: Should be 0.5–0.8 (fraction of positions where draft argmax == verifier argmax)
- `aurora_accept_loss_0`: Standard KL on accepted positions
- `aurora_discard_loss_0`: Top-k filtered KL on rejected positions
- `loss_0`, `full_acc_0`, `cond_acc_0`: Same as Exp15 for comparison

## Training Metrics

| Epoch | train/loss | val/loss | aurora_accept_ratio_0 | val/full_acc_0 | Notes |
|-------|-----------|----------|----------------------|----------------|-------|
| 0 | 1.53 | 1.412 | 0.12 | 0.10 | Initial |

## Progress

- **2026-03-26 07:56**: Experiment created. First pod failed (rsync SSH issue). Pre-staged data manually (rsync 6078 .pt files, 303GB from .10).
- **2026-03-26 08:01**: Second pod failed — `torch.tensor(0.0)` has no grad_fn under torch.compile. Fixed aurora_loss_function to be branch-free.
- **2026-03-26 08:03**: Third pod failed — 5 corrupt .pt files from original datagen (disk full on .28). Removed them. 6073 clean files remain.
- **2026-03-26 08:06**: Fourth pod deployed. Training stable, epoch 0 completed in 3:35. ~4 hours estimated for 70 epochs.
