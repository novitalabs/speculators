#!/bin/bash
set -euo pipefail

###############################################################################
# Qwen3-32B Eagle3 - Training Only (data already generated)
# Resume from previously generated data in OUTPUT_PATH/gen/
###############################################################################

# Ensure 'python' is available (vllm-openai image only has python3)
if ! command -v python &>/dev/null; then
    ln -s "$(command -v python3)" /usr/local/bin/python
fi

# -- Config --
VERIFIER_NAME_OR_PATH="${MODEL_PATH:-/data/models/Qwen3-32B}"
OUTPUT_PATH="${OUTPUT_PATH:-/data/output/qwen3_32b_eagle3}"
NUM_GPUS="${NUM_GPUS:-8}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
LR="${LR:-3e-5}"
EPOCHS="${EPOCHS:-10}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH="${PREFETCH:-2}"

echo "============================================="
echo " Qwen3-32B Eagle3 Training (resume)"
echo " Model:       $VERIFIER_NAME_OR_PATH"
echo " Data:        $OUTPUT_PATH/gen"
echo " Seq Length:  $SEQ_LENGTH"
echo " GPUs:        $NUM_GPUS"
echo " Workers:     $NUM_WORKERS  Prefetch: $PREFETCH"
echo "============================================="

cd /workspace/speculators

# Verify data exists
if [ ! -d "$OUTPUT_PATH/gen/sharegpt" ] || [ ! -d "$OUTPUT_PATH/gen/ultrachat" ]; then
    echo "ERROR: Generated data not found in $OUTPUT_PATH/gen/"
    exit 1
fi

echo "[1/1] Starting Eagle3 training with $NUM_GPUS GPUs..."
export LOCAL_TRAIN_ENV=1
torchrun \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    scripts/train.py \
    --verifier-name-or-path "$VERIFIER_NAME_OR_PATH" \
    --data-path "$OUTPUT_PATH/gen" \
    --save-path "$OUTPUT_PATH/checkpoints" \
    --log-dir "$OUTPUT_PATH/logs" \
    --d2t-path "$OUTPUT_PATH/vocab_mapping/d2t.npy" \
    --t2d-path "$OUTPUT_PATH/vocab_mapping/t2d.npy" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --epochs "$EPOCHS" \
    --num-workers "$NUM_WORKERS" \
    --prefetch-factor "$PREFETCH" \
    --run-name "qwen3_32b_eagle3"

echo "============================================="
echo " Training complete!"
echo " Checkpoints: $OUTPUT_PATH/checkpoints"
echo "============================================="
