#!/bin/bash
set -euo pipefail

###############################################################################
# MiniMax-M2.5 Online Streaming Training
# Reads manifest.json, trains as data arrives, absorbs new files each epoch
###############################################################################

# Ensure 'python' is available
if ! command -v python &>/dev/null; then
    ln -s "$(command -v python3)" /usr/local/bin/python
fi

# -- Config --
VERIFIER_NAME_OR_PATH="${MODEL_PATH:-/data/models/MiniMax-M2.5}"
OUTPUT_PATH="${OUTPUT_PATH:-/data/output/minimax_m2.5_eagle3_online}"
NUM_GPUS="${NUM_GPUS:-8}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
LR="${LR:-3e-5}"
MIN_SAMPLES="${MIN_SAMPLES:-50}"
FINAL_EPOCHS="${FINAL_EPOCHS:-3}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"

GEN_DIR="$OUTPUT_PATH/gen"
MANIFEST_PATH="$GEN_DIR/manifest.json"

echo "============================================="
echo " MiniMax-M2.5 Online Streaming Training"
echo " Model:        $VERIFIER_NAME_OR_PATH"
echo " Data:         $GEN_DIR"
echo " Manifest:     $MANIFEST_PATH"
echo " GPUs:         $NUM_GPUS"
echo " Min Samples:  $MIN_SAMPLES"
echo " Final Epochs: $FINAL_EPOCHS"
echo "============================================="

mkdir -p "$OUTPUT_PATH"/{checkpoints,logs}

cd /workspace/speculators

export LOCAL_TRAIN_ENV=1
torchrun \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    scripts/train_streaming.py \
    --verifier-name-or-path "$VERIFIER_NAME_OR_PATH" \
    --manifest-path "$MANIFEST_PATH" \
    --data-path "$GEN_DIR" \
    --save-path "$OUTPUT_PATH/checkpoints" \
    --log-dir "$OUTPUT_PATH/logs" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --min-samples "$MIN_SAMPLES" \
    --final-epochs "$FINAL_EPOCHS" \
    --poll-interval "$POLL_INTERVAL" \
    --num-workers 4 \
    --prefetch-factor 2 \
    --run-name "minimax_m2.5_eagle3_online"

echo "============================================="
echo " Streaming training complete!"
echo " Checkpoints: $OUTPUT_PATH/checkpoints"
echo "============================================="
