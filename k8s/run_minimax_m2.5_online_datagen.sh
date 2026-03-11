#!/bin/bash
set -euo pipefail

###############################################################################
# MiniMax-M2.5 Online Datagen (writes manifest for streaming training)
###############################################################################

export VLLM_ATTENTION_BACKEND=FLASH_ATTN

DATAGEN_TP="${DATAGEN_TP:-4}"
SHARD_ID="${SHARD_ID:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"

# Ensure 'python' is available
if ! command -v python &>/dev/null; then
    ln -s "$(command -v python3)" /usr/local/bin/python
fi

# -- Config --
VERIFIER_NAME_OR_PATH="${MODEL_PATH:-/data/models/MiniMax-M2.5}"
OUTPUT_PATH="${OUTPUT_PATH:-/data/output/minimax_m2.5_eagle3_online}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"

GEN_DIR="$OUTPUT_PATH/gen"
MANIFEST_PATH="$GEN_DIR/manifest.json"

echo "============================================="
echo " MiniMax-M2.5 Online Datagen"
echo " Model:       $VERIFIER_NAME_OR_PATH"
echo " Output:      $GEN_DIR"
echo " Max Samples: $MAX_SAMPLES"
echo " Seq Length:  $SEQ_LENGTH"
echo " TP:          $DATAGEN_TP"
echo " Shard:       $SHARD_ID / $NUM_SHARDS"
echo "============================================="

mkdir -p "$GEN_DIR"

cd /workspace/speculators

echo "[Datagen] Generating training data (ShareGPT)..."
python scripts/data_generation_offline.py \
    --target-model-path "$VERIFIER_NAME_OR_PATH" \
    --train-data-path sharegpt \
    --output-dir "$GEN_DIR" \
    --manifest-path "$MANIFEST_PATH" \
    --seq-length "$SEQ_LENGTH" \
    --max-samples "$MAX_SAMPLES" \
    --tensor-parallel-size "$DATAGEN_TP" \
    --gpu-memory-utilization 0.85 \
    --turn-dropout \
    --batch-size 4 \
    --shard-id "$SHARD_ID" \
    --num-shards "$NUM_SHARDS"

echo "============================================="
echo " Datagen complete!"
echo " Manifest: $MANIFEST_PATH"
echo "============================================="
