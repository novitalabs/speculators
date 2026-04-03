#!/bin/bash
set -euo pipefail

###############################################################################
# Exp 21: Continuous Datagen with Difficulty-Weighted Resampling
# Uses expanded novita_merged dataset (747K conversations, 6 sources merged)
# Same architecture as exp19, larger + deduplicated training data
###############################################################################

export VLLM_ATTENTION_BACKEND=FLASH_ATTN

DATAGEN_TP="${DATAGEN_TP:-4}"

if ! command -v python &>/dev/null; then
    ln -s "$(command -v python3)" /usr/local/bin/python
fi

VERIFIER_NAME_OR_PATH="${MODEL_PATH:-/data/models/MiniMax-M2.5}"
OUTPUT_PATH="${OUTPUT_PATH:-/data/output/minimax_m2.5_eagle3_exp21}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
MERGED_JSONL="${MERGED_JSONL:-/data/datasets/novita_merged/train.jsonl}"

GEN_DIR="$OUTPUT_PATH/gen"
MANIFEST_PATH="$GEN_DIR/manifest.json"
DIFFICULTY_SCORES="$GEN_DIR/difficulty_scores.json"

echo "============================================="
echo " Exp 21: Continuous Datagen (difficulty-weighted)"
echo " Model:        $VERIFIER_NAME_OR_PATH"
echo " Output:       $GEN_DIR"
echo " Data:         $MERGED_JSONL"
echo " Seq Length:   $SEQ_LENGTH"
echo " TP:           $DATAGEN_TP"
echo " Mode:         CONTINUOUS"
echo "============================================="

mkdir -p "$GEN_DIR"

cd /workspace/speculators

###############################################################################
# Generate training data (continuous loop — never exits)
###############################################################################
echo "[Step 1] Starting continuous datagen..."
DATAGEN_CMD="python scripts/data_generation_offline.py \
    --target-model-path $VERIFIER_NAME_OR_PATH \
    --train-data-path $MERGED_JSONL \
    --output-dir $GEN_DIR \
    --manifest-path $MANIFEST_PATH \
    --seq-length $SEQ_LENGTH \
    --tensor-parallel-size $DATAGEN_TP \
    --gpu-memory-utilization 0.85 \
    --turn-dropout \
    --batch-size 4 \
    --max-output-size-gb ${MAX_OUTPUT_SIZE_GB:-1024} \
    --continuous \
    --difficulty-scores-path $DIFFICULTY_SCORES"

if [ "$MAX_SAMPLES" -gt 0 ] 2>/dev/null; then
    DATAGEN_CMD="$DATAGEN_CMD --max-samples $MAX_SAMPLES"
fi

eval $DATAGEN_CMD

# Should never reach here in continuous mode
echo "[WARN] Continuous datagen exited unexpectedly"
