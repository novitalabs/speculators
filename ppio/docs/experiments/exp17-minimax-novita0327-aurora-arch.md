# Experiment 17: MiniMax-M2.5 Eagle3 Aurora-Arch with novita20260327

Same Aurora architecture as Exp14/15, retrained on the newer weilan55/novita20260327 dataset.

- **Nodes**: .17 (datagen, 8x H200), .18 (training, 8x H200)
- **Image**: speculators:v0.17.0
- **Dataset**: weilan55/novita20260327
- **Architecture**: Aurora-like (same as Exp14/15)
  - num_attention_heads: 24
  - intermediate_size: 8192
  - rope_theta: 5000000
  - draft_vocab_size: 32000
  - hidden_size: 3072, num_kv_heads: 8, head_dim: 128
- **Config**:
  - Training: 8 GPU FSDP, lr=3e-5, streaming with buffer cleanup
  - Buffer max: 500GB (conservative, learned from Exp15 disk-full crashes)
  - Vocab mapping: d2t/t2d from token_freq.pt
  - Architecture overrides: `--override-num-attention-heads 24 --override-intermediate-size 8192 --override-rope-theta 5000000`
- **Output**: /data/output/minimax_m2.5_eagle3_novita0327/
- **Baseline**: Exp15 ckpt67 (63.2% Acc@0)
- **Status**: RUNNING (restarted 2026-03-30 on .17/.18 after node reinstall)
- **Key Files**:
  - `k8s/run_minimax_m2.5_novita0327_datagen.sh` -- datagen script
  - `k8s/k8s-minimax-m2.5-novita0327-datagen.yaml` -- datagen pod (.12)
  - `k8s/run_minimax_m2.5_novita0327_train.sh` -- training script
  - `k8s/k8s-minimax-m2.5-novita0327-train.yaml` -- training pod (.23)

## Progress

- 2026-03-30: Restarted experiment on .17/.18 (previous .12/.23 nodes were reinstalled, all prior data lost)
- 2026-03-30: Datagen downloading dataset from HF, train pod installing deps and waiting for data

## Issue Log

- 2026-03-30: Original nodes (.12/.23) were reinstalled, experiment data completely lost. Redeployed from scratch on .17/.18.
