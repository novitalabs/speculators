#!/bin/bash
set -euo pipefail

###############################################################################
# Qwen3-32B Eagle3 - Quick Validation Run
# Minimal config: 100 samples, 1 dataset, 1 epoch
###############################################################################

# Workaround: use TP=2 for data gen to reduce multiproc workers (segfault on 8)
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
DATAGEN_TP=2

# Ensure 'python' is available
if ! command -v python &>/dev/null; then
    ln -s "$(command -v python3)" /usr/local/bin/python
fi

VERIFIER_NAME_OR_PATH="${MODEL_PATH:-/data/models/Qwen3-32B}"
OUTPUT_PATH="${OUTPUT_PATH:-/data/output/qwen3_32b_eagle3_quicktest}"
NUM_GPUS="${NUM_GPUS:-8}"
SEQ_LENGTH=2048          # shorter seq for quick test
MAX_SAMPLES=100
DRAFT_VOCAB_SIZE=32000
TARGET_VOCAB_SIZE=151936

# Proxy
if [ -n "${HTTPS_PROXY:-}" ]; then
    export https_proxy="$HTTPS_PROXY"
    export http_proxy="$HTTPS_PROXY"
    echo "[INFO] Using proxy: $HTTPS_PROXY"
fi

echo "========== Quick Validation Run =========="
echo " Model:  $VERIFIER_NAME_OR_PATH"
echo " Output: $OUTPUT_PATH"
echo " Samples: $MAX_SAMPLES | SeqLen: $SEQ_LENGTH | Epochs: 1"
echo "=========================================="

mkdir -p "$OUTPUT_PATH"/{gen,vocab_mapping,checkpoints,logs}
cd /workspace/speculators

# Step 1: Data Generation (sharegpt only, 100 samples)
echo "[1/3] Data generation..."
python scripts/data_generation_offline.py \
    --target-model-path "$VERIFIER_NAME_OR_PATH" \
    --train-data-path sharegpt \
    --output-dir "$OUTPUT_PATH/gen/sharegpt" \
    --seq-length "$SEQ_LENGTH" \
    --max-samples "$MAX_SAMPLES" \
    --tensor-parallel-size "$DATAGEN_TP" \
    --gpu-memory-utilization 0.70 \
    --token-freq-path "$OUTPUT_PATH/vocab_mapping/token_freq_sharegpt.pt" \
    --batch-size 8

# Step 2: Vocab Mapping
echo "[2/3] Vocab mapping..."
python scripts/build_vocab_mapping.py \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --target-vocab-size "$TARGET_VOCAB_SIZE" \
    --token-freq-path "$OUTPUT_PATH/vocab_mapping/token_freq_sharegpt.pt" \
    --output-path "$OUTPUT_PATH/vocab_mapping"

# Step 3: Training (1 epoch)
echo "[3/3] Training (1 epoch)..."
export LOCAL_TRAIN_ENV=1
torchrun --standalone --nproc_per_node="$NUM_GPUS" \
    scripts/train.py \
    --verifier-name-or-path "$VERIFIER_NAME_OR_PATH" \
    --data-path "$OUTPUT_PATH/gen" \
    --save-path "$OUTPUT_PATH/checkpoints" \
    --log-dir "$OUTPUT_PATH/logs" \
    --d2t-path "$OUTPUT_PATH/vocab_mapping/d2t.npy" \
    --t2d-path "$OUTPUT_PATH/vocab_mapping/t2d.npy" \
    --lr 3e-5 \
    --total-seq-len "$SEQ_LENGTH" \
    --epochs 1 \
    --run-name "qwen3_32b_eagle3_quicktest"

echo "========== Quick validation PASSED =========="
echo " Checkpoints: $OUTPUT_PATH/checkpoints"
