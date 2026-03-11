#!/bin/bash
set -euo pipefail

###############################################################################
# Qwen3-32B Eagle3 Training Pipeline - Novita Online Data
# Steps: Data Generation (from novita logs) -> Vocab Mapping -> Training
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
OUTPUT_PATH="${OUTPUT_PATH:-/data/output/qwen3_32b_eagle3_novita}"
DATA_JSONL="${DATA_JSONL:-/data/novita/minimax-m2.5/novita_sharegpt.jsonl}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
NUM_GPUS="${NUM_GPUS:-8}"
DRAFT_VOCAB_SIZE=32000
TARGET_VOCAB_SIZE=151936                 # Qwen3 family vocab size
LR="${LR:-3e-5}"
EPOCHS="${EPOCHS:-10}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

echo "============================================="
echo " Qwen3-32B Eagle3 - Novita Data Pipeline"
echo " Model:       $VERIFIER_NAME_OR_PATH"
echo " Output:      $OUTPUT_PATH"
echo " Data:        $DATA_JSONL"
echo " Seq Length:  $SEQ_LENGTH"
echo " Max Samples: ${MAX_SAMPLES:-all}"
echo " GPUs:        $NUM_GPUS"
echo "============================================="

mkdir -p "$OUTPUT_PATH"/{gen,vocab_mapping,checkpoints,logs}

cd /workspace/speculators

# ========================
# Step 1: Data Generation
# ========================
echo "[Step 1/3] Generating training data from novita logs..."
DATAGEN_ARGS=(
    --target-model-path "$VERIFIER_NAME_OR_PATH"
    --train-data-path "$DATA_JSONL"
    --output-dir "$OUTPUT_PATH/gen/novita"
    --seq-length "$SEQ_LENGTH"
    --tensor-parallel-size "$DATAGEN_TP"
    --gpu-memory-utilization 0.70
    --token-freq-path "$OUTPUT_PATH/vocab_mapping/token_freq_novita.pt"
    --turn-dropout
    --batch-size 8
)
if [ -n "$MAX_SAMPLES" ]; then
    DATAGEN_ARGS+=(--max-samples "$MAX_SAMPLES")
fi
python scripts/data_generation_offline.py "${DATAGEN_ARGS[@]}"

# ========================
# Step 2: Vocab Mapping
# ========================
echo "[Step 2/3] Building vocab mapping..."
python scripts/build_vocab_mapping.py \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --target-vocab-size "$TARGET_VOCAB_SIZE" \
    --token-freq-path "$OUTPUT_PATH/vocab_mapping/token_freq_novita.pt" \
    --output-path "$OUTPUT_PATH/vocab_mapping"

# ========================
# Step 3: Training
# ========================
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
    --d2t-path "$OUTPUT_PATH/vocab_mapping/d2t.npy" \
    --t2d-path "$OUTPUT_PATH/vocab_mapping/t2d.npy" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --epochs "$EPOCHS" \
    --run-name "qwen3_32b_eagle3_novita_v2"

echo "============================================="
echo " Training complete!"
echo " Checkpoints: $OUTPUT_PATH/checkpoints"
echo "============================================="
