#!/bin/bash
set -euo pipefail

###############################################################################
# MiniMax-M2.5 Novita Full Dataset Datagen v2 (no turn filter)
# Uses ALL ~56K Novita API log conversations from weilan55/novita20260309
# --min-turns 2 (default) keeps any conversation with at least 1 user + 1 assistant message
###############################################################################

export VLLM_ATTENTION_BACKEND=FLASH_ATTN

DATAGEN_TP="${DATAGEN_TP:-4}"

# Ensure 'python' is available
if ! command -v python &>/dev/null; then
    ln -s "$(command -v python3)" /usr/local/bin/python
fi

# -- Config --
VERIFIER_NAME_OR_PATH="${MODEL_PATH:-/data/models/MiniMax-M2.5}"
OUTPUT_PATH="${OUTPUT_PATH:-/data/output/minimax_m2.5_eagle3_novita_full_v2}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
NOVITA_JSONL="${NOVITA_JSONL:-/data/datasets/novita20260309/conversations_full_no_filter.jsonl}"

GEN_DIR="$OUTPUT_PATH/gen"
MANIFEST_PATH="$GEN_DIR/manifest.json"

echo "============================================="
echo " MiniMax-M2.5 Novita Full Datagen v2 (no turn filter)"
echo " Model:       $VERIFIER_NAME_OR_PATH"
echo " Output:      $GEN_DIR"
echo " Data:        $NOVITA_JSONL"
echo " Max Samples: ${MAX_SAMPLES:-unlimited}"
echo " Seq Length:  $SEQ_LENGTH"
echo " TP:          $DATAGEN_TP"
echo "============================================="

mkdir -p "$GEN_DIR"

cd /workspace/speculators

###############################################################################
# Step 0: Preprocess Novita logs (no turn filter — min-turns 2)
###############################################################################
if [ ! -f "$NOVITA_JSONL" ]; then
    echo "[Step 0] Preprocessing Novita API logs (full dataset, no turn filter)..."
    RAW_JSON=$(find /data/datasets/novita20260309 -name "*.json" | head -1)
    if [ -z "$RAW_JSON" ]; then
        echo "[ERROR] No raw JSON found in /data/datasets/novita20260309"
        exit 1
    fi
    python3 /workspace/speculators/k8s/preprocess_novita_logs.py \
        --input "$RAW_JSON" \
        --output "$NOVITA_JSONL" \
        --max-samples 0 \
        --min-turns 2
    echo "[Step 0] Done. Conversations: $(wc -l < "$NOVITA_JSONL")"
fi

echo "[Step 1] Total conversations: $(wc -l < "$NOVITA_JSONL")"

###############################################################################
# Step 1: Generate training data
###############################################################################
echo "[Step 1] Generating training data from Novita conversations..."
DATAGEN_CMD="python scripts/data_generation_offline.py \
    --target-model-path $VERIFIER_NAME_OR_PATH \
    --train-data-path $NOVITA_JSONL \
    --output-dir $GEN_DIR \
    --manifest-path $MANIFEST_PATH \
    --seq-length $SEQ_LENGTH \
    --tensor-parallel-size $DATAGEN_TP \
    --gpu-memory-utilization 0.85 \
    --turn-dropout \
    --batch-size 4"

# Only add --max-samples if > 0 (0 means unlimited)
if [ "$MAX_SAMPLES" -gt 0 ] 2>/dev/null; then
    DATAGEN_CMD="$DATAGEN_CMD --max-samples $MAX_SAMPLES"
fi

eval $DATAGEN_CMD

echo "============================================="
echo " Datagen complete!"
echo " Manifest: $MANIFEST_PATH"
echo " Files: $(find "$GEN_DIR" -name "*.pt" | wc -l)"
echo " Size:  $(du -sh "$GEN_DIR" | cut -f1)"
echo "============================================="
