#!/bin/bash
set -euo pipefail

###############################################################################
# MiniMax-M2.5 Eagle3 Training - Debug run
# Using same config as successful quicktest but with more epochs
###############################################################################

if ! command -v python &>/dev/null; then
    ln -s "$(command -v python3)" /usr/local/bin/python
fi

VERIFIER_NAME_OR_PATH="${MODEL_PATH:-/data/models/MiniMax-M2.5}"
OUTPUT_PATH="${OUTPUT_PATH:-/data/output/minimax_m2.5_eagle3}"
NUM_GPUS="${NUM_GPUS:-8}"

echo "============================================="
echo " MiniMax-M2.5 Eagle3 - Debug Run"
echo " Using quicktest data only (ShareGPT)"
echo "============================================="

cd /workspace/speculators

# Use ONLY ShareGPT data, seq_len=2048, same as quicktest
rm -rf "$OUTPUT_PATH/checkpoints"/*

export LOCAL_TRAIN_ENV=1
torchrun \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    scripts/train.py \
    --verifier-name-or-path "$VERIFIER_NAME_OR_PATH" \
    --data-path /data/output/minimax_m2.5_eagle3_quicktest/gen \
    --save-path "$OUTPUT_PATH/checkpoints" \
    --log-dir "$OUTPUT_PATH/logs" \
    --lr 3e-5 \
    --total-seq-len 2048 \
    --epochs 3 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --no-resume-from-checkpoint \
    --run-name "minimax_m2.5_eagle3_debug"

echo "Done."
