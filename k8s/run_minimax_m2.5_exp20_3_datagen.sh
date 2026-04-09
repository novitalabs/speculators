#!/bin/bash
set -euo pipefail

###############################################################################
# Exp 20-3: Continuous Datagen with Chinese + Novita Dataset
# Uses merged nemotron-v2-chinese (195K) + novita_merged_exp18 (166K) = 361K samples.
# Runs in continuous loop mode:
# Round 1: uniform sampling (full coverage)
# Round 2+: difficulty-weighted (focus on hard samples)
###############################################################################

export VLLM_ATTENTION_BACKEND=FLASH_ATTN

DATAGEN_TP="${DATAGEN_TP:-4}"

if ! command -v python &>/dev/null; then
    ln -s "$(command -v python3)" /usr/local/bin/python
fi

VERIFIER_NAME_OR_PATH="${MODEL_PATH:-/data/models/MiniMax-M2.5}"
OUTPUT_PATH="${OUTPUT_PATH:-/data/output/minimax_m2.5_eagle3_exp20_3}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
TRAIN_DATA="${TRAIN_DATA:-/data/tengwan/datasets/exp20_3_merged/}"

GEN_DIR="$OUTPUT_PATH/gen"
MANIFEST_PATH="$GEN_DIR/manifest.json"
DIFFICULTY_SCORES="$GEN_DIR/difficulty_scores.json"

echo "============================================="
echo " Exp 20-3: Continuous Datagen (chinese+novita)"
echo " Model:        $VERIFIER_NAME_OR_PATH"
echo " Output:       $GEN_DIR"
echo " Data:         $TRAIN_DATA"
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
    --train-data-path $TRAIN_DATA \
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
