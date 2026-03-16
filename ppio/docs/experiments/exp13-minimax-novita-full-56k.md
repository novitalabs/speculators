# Experiment 13: MiniMax-M2.5 + Full Novita ~56K (No Turn Filter, Online Streaming)

Full-scale online training with ALL Novita API production logs (~56K conversations, no turn filter).
Exp 12 used `--min-turns 4` which reduced dataset from ~56K to 38K. This experiment uses `--min-turns 2` (default) to keep all conversations with at least 1 user + 1 assistant message.

- **Nodes**: .21 (datagen, 8x H200) + .22 (training, 8x H200)
- **Image**: speculators:v0.17.0
- **Data**: weilan55/novita20260309 full dataset (no turn filter)
  - Raw: 799,622 API log records → 56,012 with messages → ~56K conversations (--min-turns 2)
  - Preprocessed: `preprocess_novita_logs.py --max-samples 0 --min-turns 2`
  - JSONL: `conversations_full_no_filter.jsonl`
- **Config**:
  - Datagen: TP=4, batch_size=4, seq_len=8192, gpu_memory_utilization=0.85
  - Training: 8 GPU FSDP, lr=3e-5, final_epochs=10, max_val_files=200
  - Ring buffer: 1TB cap on .22, buffer_cleanup.py with min_train_count=2
  - NCCL_TIMEOUT=3600
- **Pipeline**: Online streaming — datagen on .21 → rsync → train_streaming.py on .22
- **Output**: `/data/output/minimax_m2.5_eagle3_novita_full_v2/`
- **Key Files**:
  - `k8s/run_minimax_m2.5_novita_full_datagen_v2.sh` — datagen script
  - `k8s/run_minimax_m2.5_novita_full_train_v2.sh` — training + rsync + cleanup
  - `k8s/k8s-minimax-m2.5-novita-full-datagen-v2.yaml` — datagen pod (.21)
  - `k8s/k8s-minimax-m2.5-novita-full-train-v2.yaml` — training pod (.22)
- **Changes from Exp 12**:
  - `--min-turns 4` → `--min-turns 2` (no turn filter, ~56K vs 38K conversations)
  - Training node: .22 (instead of .18)
  - New output path: `minimax_m2.5_eagle3_novita_full_v2`
- **Status**: PENDING
