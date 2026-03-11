#!/bin/bash
set -euo pipefail

###############################################################################
# MiniMax-M2.5 Eagle3 Training - Train Only
# Data already generated (ShareGPT + UltraChat), just run training
# Lower LR (1e-5) + warmup to prevent NaN with MoE config
###############################################################################

if ! command -v python &>/dev/null; then
    ln -s "$(command -v python3)" /usr/local/bin/python
fi

VERIFIER_NAME_OR_PATH="${MODEL_PATH:-/data/models/MiniMax-M2.5}"
OUTPUT_PATH="${OUTPUT_PATH:-/data/output/minimax_m2.5_eagle3}"
NUM_GPUS="${NUM_GPUS:-8}"
LR="${LR:-3e-5}"
EPOCHS="${EPOCHS:-10}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"

echo "============================================="
echo " MiniMax-M2.5 Eagle3 - Train Only"
echo " Model:       $VERIFIER_NAME_OR_PATH"
echo " Output:      $OUTPUT_PATH"
echo " LR:          $LR"
echo " Epochs:      $EPOCHS"
echo "============================================="

cd /workspace/speculators

# Clear previous failed checkpoints
rm -rf "$OUTPUT_PATH/checkpoints"/*

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
    --scheduler-type cosine \
    --scheduler-warmup-steps 200 \
    --no-resume-from-checkpoint \
    --run-name "minimax_m2.5_eagle3"

echo "============================================="
echo " Training complete!"
echo " Checkpoints: $OUTPUT_PATH/checkpoints"
echo "============================================="
