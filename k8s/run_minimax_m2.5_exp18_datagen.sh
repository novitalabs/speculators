#!/bin/bash
set -euo pipefail

###############################################################################
# Exp 18: MiniMax-M2.5 Merged Novita Datagen (session-dedup)
# Uses merged dataset from novita20260309 + novita20260320 + novita20260327
# with session-level dedup (max 5 samples per unique task).
###############################################################################

export VLLM_ATTENTION_BACKEND=FLASH_ATTN

DATAGEN_TP="${DATAGEN_TP:-4}"

if ! command -v python &>/dev/null; then
    ln -s "$(command -v python3)" /usr/local/bin/python
fi

# -- Config --
VERIFIER_NAME_OR_PATH="${MODEL_PATH:-/data/models/MiniMax-M2.5}"
OUTPUT_PATH="${OUTPUT_PATH:-/data/output/minimax_m2.5_eagle3_exp18}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
MERGED_JSONL="${MERGED_JSONL:-/data/datasets/novita_merged_exp18/conversations.jsonl}"
MAX_PER_TASK="${MAX_PER_TASK:-5}"

GEN_DIR="$OUTPUT_PATH/gen"
MANIFEST_PATH="$GEN_DIR/manifest.json"

echo "============================================="
echo " Exp 18: MiniMax-M2.5 Merged Novita Datagen"
echo " Model:        $VERIFIER_NAME_OR_PATH"
echo " Output:       $GEN_DIR"
echo " Merged Data:  $MERGED_JSONL"
echo " Max/Task:     $MAX_PER_TASK"
echo " Max Samples:  ${MAX_SAMPLES:-unlimited}"
echo " Seq Length:   $SEQ_LENGTH"
echo " TP:           $DATAGEN_TP"
echo "============================================="

mkdir -p "$GEN_DIR" "$(dirname "$MERGED_JSONL")"

cd /workspace/speculators

###############################################################################
# Step 0: Merge datasets (if not already done)
###############################################################################
if [ ! -f "$MERGED_JSONL" ]; then
    echo "[Step 0] Merging novita datasets with session-level dedup..."

    INPUTS=""
    for ds in /data/datasets/novita20260309/conversations_full_no_filter.jsonl \
              /data/datasets/novita20260320/conversations.jsonl \
              /data/datasets/novita20260327/conversations.jsonl; do
        if [ -f "$ds" ]; then
            INPUTS="$INPUTS $ds"
            echo "  [+] $ds"
        else
            echo "  [-] $ds (not found, skipping)"
        fi
    done

    python3 k8s/merge_novita_datasets.py \
        --inputs $INPUTS \
        --output "$MERGED_JSONL" \
        --max-per-task "$MAX_PER_TASK" \
        --min-turns 2 \
        --seed 42

    echo "[Step 0] Merge complete: $(wc -l < "$MERGED_JSONL") conversations"
else
    echo "[Step 0] Merged dataset already exists: $(wc -l < "$MERGED_JSONL") conversations"
fi

###############################################################################
# Step 1: Generate training data
# NOTE: data_generation_offline.py automatically runs a background cleanup
# thread when --max-output-size-gb is set, evicting oldest files to stay
# within the budget. No shell-level cleanup needed.
###############################################################################
echo "[Step 1] Generating training data..."
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
    --max-output-size-gb ${MAX_OUTPUT_SIZE_GB:-1024}"

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
