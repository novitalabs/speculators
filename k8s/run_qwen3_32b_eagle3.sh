#!/bin/bash
set -euo pipefail

###############################################################################
# Qwen3-32B Eagle3 Training Pipeline
# Steps: Data Generation -> Vocab Mapping -> Training
###############################################################################

# Workaround: use TP=2 for data gen to reduce multiproc workers (segfault on 8)
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
DATAGEN_TP=2

# Ensure 'python' is available (vllm-openai image only has python3)
if ! command -v python &>/dev/null; then
    ln -s "$(command -v python3)" /usr/local/bin/python
fi

# -- Config --
VERIFIER_NAME_OR_PATH="${MODEL_PATH:-/data/models/Qwen3-32B}"
OUTPUT_PATH="${OUTPUT_PATH:-/data/output/qwen3_32b_eagle3}"
MAX_SAMPLES="${MAX_SAMPLES:-5000}"       # Use 5000 for initial run; remove for full
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
NUM_GPUS="${NUM_GPUS:-8}"
DRAFT_VOCAB_SIZE=32000
TARGET_VOCAB_SIZE=151936                 # Qwen3 family vocab size
LR="${LR:-3e-5}"
EPOCHS="${EPOCHS:-10}"

# -- Proxy (if needed for HF downloads) --
if [ -n "${HTTPS_PROXY:-}" ]; then
    export https_proxy="$HTTPS_PROXY"
    export http_proxy="$HTTPS_PROXY"
    echo "[INFO] Using proxy: $HTTPS_PROXY"
fi

echo "============================================="
echo " Qwen3-32B Eagle3 Training Pipeline"
echo " Model:       $VERIFIER_NAME_OR_PATH"
echo " Output:      $OUTPUT_PATH"
echo " Max Samples: $MAX_SAMPLES"
echo " Seq Length:  $SEQ_LENGTH"
echo " GPUs:        $NUM_GPUS"
echo "============================================="

mkdir -p "$OUTPUT_PATH"/{gen,vocab_mapping,checkpoints,logs}

cd /workspace/speculators

# ========================
# Step 1: Data Generation
# ========================
echo "[Step 1/4] Generating training data (ShareGPT)..."
python scripts/data_generation_offline.py \
    --target-model-path "$VERIFIER_NAME_OR_PATH" \
    --train-data-path sharegpt \
    --output-dir "$OUTPUT_PATH/gen/sharegpt" \
    --seq-length "$SEQ_LENGTH" \
    --max-samples "$MAX_SAMPLES" \
    --tensor-parallel-size "$DATAGEN_TP" \
    --gpu-memory-utilization 0.70 \
    --token-freq-path "$OUTPUT_PATH/vocab_mapping/token_freq_sharegpt.pt" \
    --turn-dropout \
    --batch-size 8

echo "[Step 2/4] Generating training data (UltraChat)..."
python scripts/data_generation_offline.py \
    --target-model-path "$VERIFIER_NAME_OR_PATH" \
    --train-data-path ultrachat \
    --output-dir "$OUTPUT_PATH/gen/ultrachat" \
    --seq-length "$SEQ_LENGTH" \
    --max-samples "$MAX_SAMPLES" \
    --tensor-parallel-size "$DATAGEN_TP" \
    --gpu-memory-utilization 0.70 \
    --token-freq-path "$OUTPUT_PATH/vocab_mapping/token_freq_ultrachat.pt" \
    --turn-dropout \
    --batch-size 8

# Combine token frequency distributions
python -c "
from speculators.train.vocab_mapping import combine_token_frequency_distributions
combine_token_frequency_distributions(
    ['$OUTPUT_PATH/vocab_mapping/token_freq_sharegpt.pt',
     '$OUTPUT_PATH/vocab_mapping/token_freq_ultrachat.pt'],
    '$OUTPUT_PATH/vocab_mapping/token_freq_combined.pt'
)
print('[INFO] Combined token frequency distributions.')
"

# ========================
# Step 2: Vocab Mapping
# ========================
echo "[Step 3/4] Building vocab mapping..."
python scripts/build_vocab_mapping.py \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --target-vocab-size "$TARGET_VOCAB_SIZE" \
    --token-freq-path "$OUTPUT_PATH/vocab_mapping/token_freq_combined.pt" \
    --output-path "$OUTPUT_PATH/vocab_mapping"

# ========================
# Step 3: Training
# ========================
echo "[Step 4/4] Starting Eagle3 training with $NUM_GPUS GPUs..."
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
    --run-name "qwen3_32b_eagle3"

echo "============================================="
echo " Training complete!"
echo " Checkpoints: $OUTPUT_PATH/checkpoints"
echo "============================================="
