#!/bin/bash
set -euo pipefail

###############################################################################
# MiniMax-M2.5 Eagle3 Training - Novita dataset
# Runs on .14 node, trains draft model from generated data
###############################################################################

if ! command -v python &>/dev/null; then
    ln -s "$(command -v python3)" /usr/local/bin/python
fi

# -- Config --
VERIFIER_NAME_OR_PATH="${MODEL_PATH:-/data/models/MiniMax-M2.5}"
OUTPUT_PATH="${OUTPUT_PATH:-/data/output/minimax_m2.5_eagle3_novita2}"
NUM_GPUS="${NUM_GPUS:-8}"
LR="${LR:-3e-5}"
EPOCHS="${EPOCHS:-10}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"

echo "============================================="
echo " MiniMax-M2.5 Eagle3 Training (Novita)"
echo " Model:       $VERIFIER_NAME_OR_PATH"
echo " Data:        $OUTPUT_PATH/gen"
echo " Output:      $OUTPUT_PATH/checkpoints"
echo " GPUs:        $NUM_GPUS"
echo " LR:          $LR"
echo " Epochs:      $EPOCHS"
echo " Seq Length:  $SEQ_LENGTH"
echo "============================================="

# Verify data exists
DATA_DIR="$OUTPUT_PATH/gen"
if [ ! -d "$DATA_DIR" ] || [ -z "$(find "$DATA_DIR" -name '*.pt' -print -quit 2>/dev/null)" ]; then
    echo "[ERROR] No training data found in $DATA_DIR"
    echo "        Run datagen first (run_minimax_m2.5_novita_datagen.sh)"
    exit 1
fi

echo "Training data:"
for subdir in "$DATA_DIR"/*/; do
    if [ -d "$subdir" ]; then
        count=$(find "$subdir" -name "*.pt" | wc -l)
        size=$(du -sh "$subdir" | cut -f1)
        echo "  $(basename "$subdir"): $count files, $size"
    fi
done

cd /workspace/speculators

mkdir -p "$OUTPUT_PATH"/{checkpoints,logs}

export LOCAL_TRAIN_ENV=1
torchrun \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    scripts/train.py \
    --verifier-name-or-path "$VERIFIER_NAME_OR_PATH" \
    --data-path "$DATA_DIR" \
    --save-path "$OUTPUT_PATH/checkpoints" \
    --log-dir "$OUTPUT_PATH/logs" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --epochs "$EPOCHS" \
    --num-workers 4 \
    --prefetch-factor 2 \
    --run-name "minimax_m2.5_eagle3_novita2"

echo "============================================="
echo " Training complete!"
echo " Checkpoints: $OUTPUT_PATH/checkpoints"
echo "============================================="
