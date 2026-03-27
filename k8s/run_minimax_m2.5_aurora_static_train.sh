#!/bin/bash
set -euo pipefail

###############################################################################
# Exp 16.5: Aurora Static Mask Training
# Same as Exp16 (Aurora loss) but with precomputed static masks from Exp15
# ckpt67 instead of dynamic accept/reject masks computed on-the-fly.
#
# Data: Exp15's 6K .pt files + precomputed masks
# Node: .17 (8x H200)
###############################################################################

export HF_HUB_OFFLINE=1
export HF_HOME=/data/hf_cache
export LOCAL_TRAIN_ENV=1

VERIFIER_NAME_OR_PATH="${MODEL_PATH:-/data/models/MiniMax-M2.5}"
OUTPUT_PATH="${OUTPUT_PATH:-/data/output/minimax_m2.5_eagle3_aurora_static}"
MASK_DIR="${MASK_DIR:-/data/output/minimax_m2.5_eagle3_aurora_loss/masks}"
NUM_GPUS="${NUM_GPUS:-8}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
LR="${LR:-3e-5}"
EPOCHS="${EPOCHS:-70}"

# Aurora loss hyperparameters
LAMBDA_DISCARD="${LAMBDA_DISCARD:-0.1}"
DISCARD_TOP_K="${DISCARD_TOP_K:-10}"

# Aurora architecture overrides (same as Exp14/15/16)
OVERRIDE_NUM_ATTENTION_HEADS="${OVERRIDE_NUM_ATTENTION_HEADS:-24}"
OVERRIDE_INTERMEDIATE_SIZE="${OVERRIDE_INTERMEDIATE_SIZE:-8192}"
OVERRIDE_ROPE_THETA="${OVERRIDE_ROPE_THETA:-5000000}"
DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-32000}"

GEN_DIR="$OUTPUT_PATH/gen"
VOCAB_DIR="$OUTPUT_PATH/vocab_mapping"

echo "============================================="
echo " Exp 16.5: Aurora Static Mask Training"
echo " Model:          $VERIFIER_NAME_OR_PATH"
echo " Data:           $GEN_DIR"
echo " Mask Dir:       $MASK_DIR"
echo " GPUs:           $NUM_GPUS"
echo " Epochs:         $EPOCHS"
echo " LR:             $LR"
echo " --- Aurora Loss ---"
echo " Lambda Discard: $LAMBDA_DISCARD"
echo " Discard Top-K:  $DISCARD_TOP_K"
echo " Static Mask:    YES"
echo " --- Architecture ---"
echo " Attention Heads: $OVERRIDE_NUM_ATTENTION_HEADS"
echo " Intermediate:    $OVERRIDE_INTERMEDIATE_SIZE"
echo " Rope Theta:      $OVERRIDE_ROPE_THETA"
echo " Draft Vocab:     $DRAFT_VOCAB_SIZE"
echo "============================================="

mkdir -p "$OUTPUT_PATH"/{checkpoints,logs} "$GEN_DIR" "$VOCAB_DIR"

# Ensure 'python' is available
if ! command -v python &>/dev/null; then
    ln -s "$(command -v python3)" /usr/local/bin/python
fi

# Ensure hostname resolves (needed for torchrun with hostNetwork)
if ! getent hosts "$(hostname)" &>/dev/null; then
    echo "127.0.0.1 $(hostname)" >> /etc/hosts
fi

cd /workspace/speculators

###############################################################################
# Step 0: Verify pre-staged data and masks
###############################################################################
D2T_PATH="$VOCAB_DIR/d2t.npy"
T2D_PATH="$VOCAB_DIR/t2d.npy"

PT_COUNT=$(ls "$GEN_DIR"/*.pt 2>/dev/null | wc -l)
echo "[data] Found $PT_COUNT .pt files in $GEN_DIR"
if [ "$PT_COUNT" -lt 1000 ]; then
    echo "[ERROR] Not enough data files. Pre-stage data with rsync before launching."
    exit 1
fi

MASK_COUNT=$(ls "$MASK_DIR"/mask_*.pt 2>/dev/null | wc -l)
echo "[masks] Found $MASK_COUNT mask files in $MASK_DIR"
if [ "$MASK_COUNT" -lt 1000 ]; then
    echo "[ERROR] Not enough mask files. Run precompute_masks.sh first."
    exit 1
fi

if [ ! -f "$D2T_PATH" ] || [ ! -f "$T2D_PATH" ]; then
    echo "[ERROR] Vocab mapping not found at $VOCAB_DIR. Pre-stage before launching."
    exit 1
fi
echo "[vocab] Using vocab mapping at $VOCAB_DIR"

###############################################################################
# Step 1: Training with Aurora static mask loss
###############################################################################
TRAIN_LOG="$OUTPUT_PATH/logs/train.log"
torchrun \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    scripts/train.py \
    --verifier-name-or-path "$VERIFIER_NAME_OR_PATH" \
    --data-path "$GEN_DIR" \
    --save-path "$OUTPUT_PATH/checkpoints" \
    --log-dir "$OUTPUT_PATH/logs" \
    --lr "$LR" \
    --epochs "$EPOCHS" \
    --total-seq-len "$SEQ_LENGTH" \
    --num-workers 4 \
    --prefetch-factor 2 \
    --d2t-path "$D2T_PATH" \
    --t2d-path "$T2D_PATH" \
    --override-num-attention-heads "$OVERRIDE_NUM_ATTENTION_HEADS" \
    --override-intermediate-size "$OVERRIDE_INTERMEDIATE_SIZE" \
    --override-rope-theta "$OVERRIDE_ROPE_THETA" \
    --aurora-loss \
    --lambda-discard "$LAMBDA_DISCARD" \
    --discard-top-k "$DISCARD_TOP_K" \
    --aurora-static-mask \
    --mask-dir "$MASK_DIR" \
    --run-name "minimax_m2.5_eagle3_aurora_static" \
    2>&1 | tee "$TRAIN_LOG"

echo "============================================="
echo " Exp 16.5 Aurora static mask training complete!"
echo " Checkpoints: $OUTPUT_PATH/checkpoints"
echo "============================================="
