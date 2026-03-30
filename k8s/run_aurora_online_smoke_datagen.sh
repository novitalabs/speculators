#!/bin/bash
set -euo pipefail

###############################################################################
# Aurora Online Smoke Test — Datagen + Mask Precompute (runs on .18)
# After this completes, rsync data to .17 and launch training pod.
###############################################################################

export HF_HUB_OFFLINE=1
export HF_HOME=/data/hf_cache
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

VERIFIER_NAME_OR_PATH="${MODEL_PATH:-/data/models/MiniMax-M2.5}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/data/datasets/novita20260320/conversations.jsonl}"
OUTPUT_DIR="/data/output/minimax_m2.5_eagle3_aurora_online_smoke"
SEED_CKPT="/data/output/minimax_m2.5_eagle3_novita0320/checkpoints/67"
FILES_PER_ROUND=50
SEQ_LENGTH=8192

# Architecture overrides
OVERRIDE_NUM_ATTENTION_HEADS=24
OVERRIDE_INTERMEDIATE_SIZE=8192
OVERRIDE_ROPE_THETA=5000000

ROUND_DIR="$OUTPUT_DIR/round_0"
GEN_DIR="$ROUND_DIR/gen"
MASK_DIR="$ROUND_DIR/masks"
VOCAB_DIR="$OUTPUT_DIR/vocab_mapping"

echo "============================================="
echo " Aurora Online Smoke — Datagen + Masks"
echo " Model:     $VERIFIER_NAME_OR_PATH"
echo " Data:      $TRAIN_DATA_PATH"
echo " Output:    $ROUND_DIR"
echo " Seed Ckpt: $SEED_CKPT"
echo " Files:     $FILES_PER_ROUND"
echo "============================================="

# Ensure 'python' is available
if ! command -v python &>/dev/null; then
    ln -s "$(command -v python3)" /usr/local/bin/python
fi

if ! getent hosts "$(hostname)" &>/dev/null; then
    echo "127.0.0.1 $(hostname)" >> /etc/hosts
fi

cd /workspace/speculators
mkdir -p "$GEN_DIR" "$MASK_DIR" "$VOCAB_DIR"

###############################################################################
# Step 0: Vocab mapping
###############################################################################
D2T_PATH="$VOCAB_DIR/d2t.npy"
T2D_PATH="$VOCAB_DIR/t2d.npy"

if [[ ! -f "$D2T_PATH" ]] || [[ ! -f "$T2D_PATH" ]]; then
    echo "[ERROR] Vocab mapping not found at $VOCAB_DIR. Pre-stage before launching."
    echo "  rsync from .17: rsync -az 10.83.115.17:$VOCAB_DIR/ $VOCAB_DIR/"
    exit 1
fi
echo "[Step 0] Vocab mapping ready: $VOCAB_DIR"

###############################################################################
# Step 1: Datagen (TP=4)
###############################################################################
echo "[Step 1] Datagen ($FILES_PER_ROUND files, TP=4)"
python scripts/data_generation_offline.py \
    --target-model-path "$VERIFIER_NAME_OR_PATH" \
    --train-data-path "$TRAIN_DATA_PATH" \
    --output-dir "$GEN_DIR" \
    --start-idx 0 \
    --max-samples "$FILES_PER_ROUND" \
    --manifest-path "$GEN_DIR/manifest.json" \
    --tensor-parallel-size 4 \
    --seq-length "$SEQ_LENGTH" \
    2>&1 | tee "$ROUND_DIR/datagen.log"

PT_COUNT=$(find "$GEN_DIR" -name "data_*.pt" | wc -l)
echo "[Step 1] Datagen complete: $PT_COUNT files"

###############################################################################
# Step 2: Mask precompute (1 GPU)
###############################################################################
echo "[Step 2] Mask precompute (1 GPU)"
CUDA_VISIBLE_DEVICES=0 python scripts/precompute_aurora_masks.py \
    --data-dir "$GEN_DIR" \
    --mask-output-dir "$MASK_DIR" \
    --checkpoint-path "$SEED_CKPT" \
    --verifier-name-or-path "$VERIFIER_NAME_OR_PATH" \
    --d2t-path "$D2T_PATH" \
    --t2d-path "$T2D_PATH" \
    --override-num-attention-heads "$OVERRIDE_NUM_ATTENTION_HEADS" \
    --override-intermediate-size "$OVERRIDE_INTERMEDIATE_SIZE" \
    --override-rope-theta "$OVERRIDE_ROPE_THETA" \
    2>&1 | tee "$ROUND_DIR/masks.log"

MASK_COUNT=$(find "$MASK_DIR" -name "mask_*.pt" | wc -l)
echo "[Step 2] Mask precompute complete: $MASK_COUNT masks"

echo "============================================="
echo " Datagen + Mask complete!"
echo " Data:  $GEN_DIR ($PT_COUNT files)"
echo " Masks: $MASK_DIR ($MASK_COUNT masks)"
echo ""
echo " Next: rsync to .17 and launch training pod"
echo "   rsync -az $ROUND_DIR/ 10.83.115.17:$ROUND_DIR/"
echo "   rsync -az $VOCAB_DIR/ 10.83.115.17:$VOCAB_DIR/"
echo "============================================="
