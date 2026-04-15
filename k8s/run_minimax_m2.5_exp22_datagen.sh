#!/bin/bash
set -euo pipefail

###############################################################################
# Exp 22: Continuous Datagen — Exp21 Novita + Exp20-2 Chinese Combined
# Training data: novita_merged/train.jsonl (724K) + nemotron-v2-chinese (195K)
# Hypothesis: Combined English+Chinese training improves both Novita and ZClaw
# Node: .17 (10.83.115.17)
###############################################################################

export VLLM_ATTENTION_BACKEND=FLASH_ATTN

DATAGEN_TP="${DATAGEN_TP:-4}"

if ! command -v python &>/dev/null; then
    ln -s "$(command -v python3)" /usr/local/bin/python
fi

VERIFIER_NAME_OR_PATH="${MODEL_PATH:-/data/models/MiniMax-M2.5}"
OUTPUT_PATH="${OUTPUT_PATH:-/data/output/minimax_m2.5_eagle3_exp22}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
TRAIN_DATA="${TRAIN_DATA:-/data/datasets/exp22_merged}"

GEN_DIR="$OUTPUT_PATH/gen"
MANIFEST_PATH="$GEN_DIR/manifest.json"
DIFFICULTY_SCORES="$GEN_DIR/difficulty_scores.json"

echo "============================================="
echo " Exp 22: Continuous Datagen (novita + nemotron-v2 Chinese)"
echo " Model:        $VERIFIER_NAME_OR_PATH"
echo " Output:       $GEN_DIR"
echo " Data:         $TRAIN_DATA"
echo " Seq Length:   $SEQ_LENGTH"
echo " TP:           $DATAGEN_TP"
echo " Mode:         CONTINUOUS"
echo "============================================="

# Create merged dataset directory if not already set up
if [ ! -d "$TRAIN_DATA" ]; then
    echo "[setup] Creating merged dataset directory at $TRAIN_DATA..."
    mkdir -p "$TRAIN_DATA"
    ln -sf /data/datasets/novita_merged/train.jsonl "$TRAIN_DATA/novita_merged_train.jsonl"
    ln -sf /data/tengwan/datasets/nemotron-v2-chinese/conversations.jsonl "$TRAIN_DATA/nemotron_chinese.jsonl"
    echo "[setup] Linked:"
    echo "  novita_merged_train.jsonl -> /data/datasets/novita_merged/train.jsonl"
    echo "  nemotron_chinese.jsonl    -> /data/tengwan/datasets/nemotron-v2-chinese/conversations.jsonl"
else
    echo "[setup] Using existing merged dataset at $TRAIN_DATA"
fi

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
