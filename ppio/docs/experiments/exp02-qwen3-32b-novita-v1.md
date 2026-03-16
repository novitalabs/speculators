# Experiment 2: Qwen3-32B + Novita Online Data (v1 - local logs)

Using production API logs from MiniMax-M2.5 endpoint (coding agent conversations).

- **Node**: .18 (8x H200 143GB)
- **Image**: speculators:v0.16.0
- **Data**: 812 conversations extracted from Novita API gateway logs
  - Source: `/data/novita/minimax-m2.5/` tar.gz export
  - Conversion: `k8s/convert_novita_logs.py` → `novita_sharegpt.jsonl`
  - Characteristics: coding agent dialogues with tool calls, median 85 turns/conversation
  - Roles: system, user, assistant, tool (tool → mapped to user)
- **Config**:
  - target_vocab_size=151936, draft_vocab_size=32000
  - seq_len=8192, lr=3e-5, epochs=10, 8 GPUs FSDP
  - Data gen: TP=2 (segfault workaround)
- **K8s**: `k8s/k8s-qwen3-32b-eagle3-novita.yaml`
- **Script**: `k8s/run_qwen3_32b_eagle3_novita.sh`

## Training Results

| Epoch | Val Loss | Top-1 Acc (cond_acc_0) |
|-------|----------|------------------------|
| 0     | 1.895    | 15.9%                  |
| 1     | 1.621    | 18.6%                  |
| 2     | 1.538    | 19.4%                  |
| 3     | 1.345    | 19.2%                  |
| 4     | 1.175    | 17.6%                  |
| 5     | 1.214    | 17.9%                  |
| 6     | 1.351    | 20.2%                  |
| 7     | 1.214    | 19.9%                  |
| 8     | **1.117**| 19.8%                  |
| 9     | 1.213    | 19.9%                  |

- **Best loss**: epoch 8 (1.117)
- **Best acc**: epoch 6 (20.2%)
- **Checkpoints**: `/data/output/qwen3_32b_eagle3_novita/checkpoints/` on .18 (263GB total)

## Analysis

- Loss is much lower than baseline (1.117 vs 9.556) but top-1 accuracy is also much lower (20% vs 52%)
- Note: loss metrics are not directly comparable since the data distributions differ significantly
- The novita data is dominated by coding agent conversations (tool use, code analysis), a narrow domain
- Only 812 samples vs 10000 in baseline — likely underfitting
- The loss curves show some oscillation after epoch 4, suggesting the small dataset causes instability
