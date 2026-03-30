#!/bin/bash
set -euo pipefail

###############################################################################
# Exp 17: MiniMax-M2.5 Novita 0327 Datagen
# Downloads weilan55/novita20260327 from HF, preprocesses, and generates
# training data with Aurora-arch compatible settings.
###############################################################################

export VLLM_ATTENTION_BACKEND=FLASH_ATTN

DATAGEN_TP="${DATAGEN_TP:-4}"

# Ensure 'python' is available
if ! command -v python &>/dev/null; then
    ln -s "$(command -v python3)" /usr/local/bin/python
fi

# -- Config --
VERIFIER_NAME_OR_PATH="${MODEL_PATH:-/data/models/MiniMax-M2.5}"
OUTPUT_PATH="${OUTPUT_PATH:-/data/output/minimax_m2.5_eagle3_novita0327}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
DATASET_DIR="${DATASET_DIR:-/data/datasets/novita20260327}"
NOVITA_JSONL="${NOVITA_JSONL:-$DATASET_DIR/conversations.jsonl}"

GEN_DIR="$OUTPUT_PATH/gen"
MANIFEST_PATH="$GEN_DIR/manifest.json"

echo "============================================="
echo " Exp 17: MiniMax-M2.5 Novita 0327 Datagen"
echo " Model:       $VERIFIER_NAME_OR_PATH"
echo " Output:      $GEN_DIR"
echo " Dataset:     $DATASET_DIR"
echo " Data:        $NOVITA_JSONL"
echo " Max Samples: ${MAX_SAMPLES:-unlimited}"
echo " Seq Length:  $SEQ_LENGTH"
echo " TP:          $DATAGEN_TP"
echo "============================================="

mkdir -p "$GEN_DIR" "$DATASET_DIR"

cd /workspace/speculators

###############################################################################
# Step 0a: Download dataset from HuggingFace
###############################################################################
if [ -z "$(find "$DATASET_DIR" -name "*.json" -o -name "*.jsonl" 2>/dev/null | head -1)" ]; then
    echo "[Step 0a] Downloading weilan55/novita20260327 from HuggingFace..."

    # Install hf_transfer for fast downloads if not present
    export http_proxy=http://127.0.0.1:1083 https_proxy=http://127.0.0.1:1083
    pip install -q hf_transfer 2>/dev/null || true

    export HF_HUB_ENABLE_HF_TRANSFER=1
    huggingface-cli download weilan55/novita20260327 \
        --repo-type dataset \
        --local-dir "$DATASET_DIR"

    unset http_proxy https_proxy
    echo "[Step 0a] Download complete: $(ls "$DATASET_DIR" | wc -l) files"

    # Extract any .tar.gz files (Novita exports are tar.gz archives containing JSON)
    for tarball in "$DATASET_DIR"/*.tar.gz; do
        [ -f "$tarball" ] || continue
        echo "[Step 0a] Extracting $tarball..."
        tar xzf "$tarball" -C "$DATASET_DIR"
        echo "[Step 0a] Extracted."
    done
else
    echo "[Step 0a] Dataset already exists at $DATASET_DIR"
fi

###############################################################################
# Step 0b: Preprocess Novita logs (min-turns 2)
###############################################################################
if [ ! -f "$NOVITA_JSONL" ]; then
    echo "[Step 0b] Preprocessing Novita API logs (min-turns 2)..."
    # Search recursively for .json files (may be inside extracted subdirectories)
    RAW_JSON=$(find "$DATASET_DIR" -name "*.json" | head -1)
    if [ -z "$RAW_JSON" ]; then
        echo "[ERROR] No raw JSON found in $DATASET_DIR"
        exit 1
    fi
    python3 /workspace/speculators/k8s/preprocess_novita_logs.py \
        --input "$RAW_JSON" \
        --output "$NOVITA_JSONL" \
        --max-samples 0 \
        --min-turns 2
    echo "[Step 0b] Done. Conversations: $(wc -l < "$NOVITA_JSONL")"
else
    echo "[Step 0b] Preprocessed data already exists: $(wc -l < "$NOVITA_JSONL") conversations"
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
