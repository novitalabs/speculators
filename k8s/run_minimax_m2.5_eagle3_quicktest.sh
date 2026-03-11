#!/bin/bash
set -euo pipefail

###############################################################################
# MiniMax-M2.5 Eagle3 - Quick Validation Run
# Minimal config: 100 samples, 1 dataset, 1 epoch
###############################################################################

# Workaround: use TP=4 for data gen (MoE model needs more VRAM per shard)
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
DATAGEN_TP=4

# Ensure 'python' is available
if ! command -v python &>/dev/null; then
    ln -s "$(command -v python3)" /usr/local/bin/python
fi

VERIFIER_NAME_OR_PATH="${MODEL_PATH:-/data/models/MiniMax-M2.5}"
OUTPUT_PATH="${OUTPUT_PATH:-/data/output/minimax_m2.5_eagle3_quicktest}"
NUM_GPUS="${NUM_GPUS:-8}"
SEQ_LENGTH=2048          # shorter seq for quick test
MAX_SAMPLES=100

echo "========== Quick Validation Run =========="
echo " Model:  $VERIFIER_NAME_OR_PATH"
echo " Output: $OUTPUT_PATH"
echo " Samples: $MAX_SAMPLES | SeqLen: $SEQ_LENGTH | Epochs: 1"
echo "=========================================="

mkdir -p "$OUTPUT_PATH"/{gen,checkpoints,logs}
cd /workspace/speculators

# Step 1: Data Generation (sharegpt only, 100 samples)
# No vocab mapping needed for MiniMax-M2.5 (use full target vocab)
echo "[1/2] Data generation..."
python scripts/data_generation_offline.py \
    --target-model-path "$VERIFIER_NAME_OR_PATH" \
    --train-data-path sharegpt \
    --output-dir "$OUTPUT_PATH/gen/sharegpt" \
    --seq-length "$SEQ_LENGTH" \
    --max-samples "$MAX_SAMPLES" \
    --tensor-parallel-size "$DATAGEN_TP" \
    --gpu-memory-utilization 0.85 \
    --batch-size 4

# Step 2: Training (1 epoch, no vocab mapping = full target vocab)
echo "[2/2] Training (1 epoch)..."
export LOCAL_TRAIN_ENV=1
torchrun --standalone --nproc_per_node="$NUM_GPUS" \
    scripts/train.py \
    --verifier-name-or-path "$VERIFIER_NAME_OR_PATH" \
    --data-path "$OUTPUT_PATH/gen" \
    --save-path "$OUTPUT_PATH/checkpoints" \
    --log-dir "$OUTPUT_PATH/logs" \
    --lr 3e-5 \
    --total-seq-len "$SEQ_LENGTH" \
    --epochs 1 \
    --run-name "minimax_m2.5_eagle3_quicktest"

echo "========== Quick validation PASSED =========="
echo " Checkpoints: $OUTPUT_PATH/checkpoints"
