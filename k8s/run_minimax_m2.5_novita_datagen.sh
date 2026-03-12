#!/bin/bash
set -euo pipefail

###############################################################################
# MiniMax-M2.5 Eagle3 Data Generation - Novita dataset
# Runs on .23 node, generates training data from Novita API logs
###############################################################################

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
DATAGEN_TP=4

if ! command -v python &>/dev/null; then
    ln -s "$(command -v python3)" /usr/local/bin/python
fi

# -- Config --
VERIFIER_NAME_OR_PATH="${MODEL_PATH:-/data/models/MiniMax-M2.5}"
OUTPUT_PATH="${OUTPUT_PATH:-/data/output/minimax_m2.5_eagle3_novita2}"
NOVITA_JSONL="${NOVITA_JSONL:-/data/datasets/novita20260309/conversations.jsonl}"
MAX_SAMPLES="${MAX_SAMPLES:-5000}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"

echo "============================================="
echo " MiniMax-M2.5 Eagle3 Datagen (Novita)"
echo " Model:       $VERIFIER_NAME_OR_PATH"
echo " Output:      $OUTPUT_PATH"
echo " Data:        $NOVITA_JSONL"
echo " Max Samples: $MAX_SAMPLES"
echo " Seq Length:  $SEQ_LENGTH"
echo " TP:          $DATAGEN_TP"
echo "============================================="

mkdir -p "$OUTPUT_PATH"/gen

cd /workspace/speculators

###############################################################################
# Step 1: Preprocess Novita logs (if conversations.jsonl doesn't exist yet)
###############################################################################
if [ ! -f "$NOVITA_JSONL" ]; then
    echo "[Step 0] Preprocessing Novita API logs..."
    RAW_JSON=$(find /data/datasets/novita20260309 -name "*.json" | head -1)
    if [ -z "$RAW_JSON" ]; then
        echo "[ERROR] No raw JSON found in /data/datasets/novita20260309"
        exit 1
    fi
    python3 /workspace/speculators/k8s/preprocess_novita_logs.py \
        --input "$RAW_JSON" \
        --output "$NOVITA_JSONL" \
        --max-samples 10000 \
        --min-turns 4
fi

###############################################################################
# Step 2: Generate training data from Novita conversations
###############################################################################
echo "[Step 1] Generating training data from Novita conversations..."
python scripts/data_generation_offline.py \
    --target-model-path "$VERIFIER_NAME_OR_PATH" \
    --train-data-path "$NOVITA_JSONL" \
    --output-dir "$OUTPUT_PATH/gen/novita" \
    --seq-length "$SEQ_LENGTH" \
    --max-samples "$MAX_SAMPLES" \
    --tensor-parallel-size "$DATAGEN_TP" \
    --gpu-memory-utilization 0.85 \
    --turn-dropout \
    --batch-size 4

echo "============================================="
echo " Data generation complete!"
echo " Output: $OUTPUT_PATH/gen/novita"
echo "============================================="
# Count generated files
echo " Files: $(find "$OUTPUT_PATH/gen/novita" -name "*.pt" | wc -l)"
echo " Size:  $(du -sh "$OUTPUT_PATH/gen/novita" | cut -f1)"
