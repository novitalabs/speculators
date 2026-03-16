# Experiment 12: MiniMax-M2.5 + Full Novita 38K (Online Streaming)

Full-scale online training with all available Novita API production logs (~38K conversations).

- **Nodes**: .21 (datagen, 8x H200) + .18 (training, 8x H200)
- **Image**: speculators:v0.17.0
- **Data**: weilan55/novita20260309 full dataset
  - Raw: 799,622 API log records → 56,012 with messages → 38,071 conversations (after --min-turns 4 filter)
  - Average 66.5 turns per conversation
  - Preprocessed with `preprocess_novita_logs.py --max-samples 0 --min-turns 4`
- **Config**:
  - Datagen: TP=4, batch_size=4, seq_len=8192, gpu_memory_utilization=0.85
  - Training: 8 GPU FSDP, lr=3e-5, final_epochs=10, max_val_files=200
  - Ring buffer: 1TB cap on .18, buffer_cleanup.py with min_train_count=2
  - NCCL_TIMEOUT=3600
- **Pipeline**: Online streaming — datagen on .21 → rsync → train_streaming.py on .18
- **Datagen Progress**: 36248/38071 files (95%), 2.1TB on .21
- **Training Progress**: Epoch 0 (partial, ~40 min of training before stop)
  - 11678 train files, 200 val files (capped)
  - Latest train metrics (Epoch 0, not converged):
    - train/loss_0 ≈ 1.0-2.0, train/full_acc_0 ≈ 52-73%
    - train/loss = 4.5-6.8 (combined 3-head)
    - train/cond_acc_1 ≈ 47-51%, train/cond_acc_2 ≈ 49-58%
  - No validation or checkpoint completed (stopped mid-epoch)
- **Issues Encountered**:
  1. `--max-samples 0` interpreted as "0 samples" not "unlimited" → fixed by omitting the flag
  2. rsync not installed in vllm container → `apt-get install rsync openssh-client`
  3. Manifest not synced (rsync excluded manifest.json) → added manifest.json to rsync include
  4. `ls *.pt` fails with 30K+ files (argument list too long) → use `find -name '*.pt' | wc -l`
  5. NCCL _ALLGATHER_BASE timeout during validation with 502 val files → capped val to 200 files, increased NCCL_TIMEOUT to 3600s
- **Key Files**:
  - `k8s/run_minimax_m2.5_novita_full_datagen.sh` — datagen script
  - `k8s/run_minimax_m2.5_novita_full_train.sh` — training + rsync + cleanup
  - `k8s/k8s-minimax-m2.5-novita-full-datagen.yaml` — datagen pod (.21)
  - `k8s/k8s-minimax-m2.5-novita-full-train.yaml` — training pod (.18)
  - `scripts/buffer_cleanup.py` — ring buffer cleanup with --max-size-gb
- **Status**: STOPPED (mid Epoch 0, user requested stop)
