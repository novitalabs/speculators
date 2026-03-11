#!/bin/bash
set -euo pipefail

###############################################################################
# MiniMax-M2.5 Eagle3 Training Pipeline
# Steps: Data Generation -> Training (no vocab mapping needed)
###############################################################################

# Workaround: use TP=4 for data gen (MoE model needs more VRAM per shard)
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
DATAGEN_TP=4

# Ensure 'python' is available (vllm-openai image only has python3)
if ! command -v python &>/dev/null; then
    ln -s "$(command -v python3)" /usr/local/bin/python
fi

# -- Config --
VERIFIER_NAME_OR_PATH="${MODEL_PATH:-/data/models/MiniMax-M2.5}"
OUTPUT_PATH="${OUTPUT_PATH:-/data/output/minimax_m2.5_eagle3}"
MAX_SAMPLES="${MAX_SAMPLES:-5000}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
NUM_GPUS="${NUM_GPUS:-8}"
LR="${LR:-3e-5}"
EPOCHS="${EPOCHS:-10}"

echo "============================================="
echo " MiniMax-M2.5 Eagle3 Training Pipeline"
echo " Model:       $VERIFIER_NAME_OR_PATH"
echo " Output:      $OUTPUT_PATH"
echo " Max Samples: $MAX_SAMPLES"
echo " Seq Length:  $SEQ_LENGTH"
echo " GPUs:        $NUM_GPUS"
echo " Epochs:      $EPOCHS"
echo "============================================="

mkdir -p "$OUTPUT_PATH"/{gen,checkpoints,logs}

cd /workspace/speculators

# ========================
# Step 1: Data Generation
# ========================
echo "[Step 1/3] Generating training data (ShareGPT)..."
python scripts/data_generation_offline.py \
    --target-model-path "$VERIFIER_NAME_OR_PATH" \
    --train-data-path sharegpt \
    --output-dir "$OUTPUT_PATH/gen/sharegpt" \
    --seq-length "$SEQ_LENGTH" \
    --max-samples "$MAX_SAMPLES" \
    --tensor-parallel-size "$DATAGEN_TP" \
    --gpu-memory-utilization 0.85 \
    --turn-dropout \
    --batch-size 4

echo "[Step 2/3] Generating training data (UltraChat)..."
python scripts/data_generation_offline.py \
    --target-model-path "$VERIFIER_NAME_OR_PATH" \
    --train-data-path ultrachat \
    --output-dir "$OUTPUT_PATH/gen/ultrachat" \
    --seq-length "$SEQ_LENGTH" \
    --max-samples "$MAX_SAMPLES" \
    --tensor-parallel-size "$DATAGEN_TP" \
    --gpu-memory-utilization 0.85 \
    --turn-dropout \
    --batch-size 4

# ========================
# Step 2: Training
# ========================
# No vocab mapping needed for MiniMax-M2.5 (use full target vocab, vocab_size=200064)
echo "[Step 3/3] Starting Eagle3 training with $NUM_GPUS GPUs..."
export LOCAL_TRAIN_ENV=1
torchrun \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    scripts/train.py \
    --verifier-name-or-path "$VERIFIER_NAME_OR_PATH" \
    --data-path "$OUTPUT_PATH/gen" \
    --save-path "$OUTPUT_PATH/checkpoints" \
    --log-dir "$OUTPUT_PATH/logs" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --epochs "$EPOCHS" \
    --num-workers 4 \
    --prefetch-factor 2 \
    --run-name "minimax_m2.5_eagle3"

echo "============================================="
echo " Training complete!"
echo " Checkpoints: $OUTPUT_PATH/checkpoints"
echo "============================================="
