#!/bin/bash
set -euo pipefail

###############################################################################
# Precompute static Aurora masks from Exp15 ckpt67 reference model
# Single GPU — runs draft model on all .pt files and saves per-step accept masks
#
# Data: Exp15's 6K .pt files
# Node: .17 (single GPU)
###############################################################################

export HF_HUB_OFFLINE=1
export HF_HOME=/data/hf_cache
export LOCAL_TRAIN_ENV=1

VERIFIER_NAME_OR_PATH="${MODEL_PATH:-/data/models/MiniMax-M2.5}"
DATA_DIR="${DATA_DIR:-/data/output/minimax_m2.5_eagle3_aurora_loss/gen}"
MASK_OUTPUT_DIR="${MASK_OUTPUT_DIR:-/data/output/minimax_m2.5_eagle3_aurora_loss/masks}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/data/output/minimax_m2.5_eagle3/checkpoints/ckpt_67}"
VOCAB_DIR="${VOCAB_DIR:-/data/output/minimax_m2.5_eagle3_aurora_loss/vocab_mapping}"

# Architecture overrides (same as Exp14/15)
OVERRIDE_NUM_ATTENTION_HEADS="${OVERRIDE_NUM_ATTENTION_HEADS:-24}"
OVERRIDE_INTERMEDIATE_SIZE="${OVERRIDE_INTERMEDIATE_SIZE:-8192}"
OVERRIDE_ROPE_THETA="${OVERRIDE_ROPE_THETA:-5000000}"

echo "============================================="
echo " Precompute Static Aurora Masks"
echo " Model:      $VERIFIER_NAME_OR_PATH"
echo " Data:       $DATA_DIR"
echo " Output:     $MASK_OUTPUT_DIR"
echo " Checkpoint: $CHECKPOINT_PATH"
echo "============================================="

mkdir -p "$MASK_OUTPUT_DIR"

# Ensure 'python' is available
if ! command -v python &>/dev/null; then
    ln -s "$(command -v python3)" /usr/local/bin/python
fi

# Ensure hostname resolves
if ! getent hosts "$(hostname)" &>/dev/null; then
    echo "127.0.0.1 $(hostname)" >> /etc/hosts
fi

cd /workspace/speculators

D2T_PATH="$VOCAB_DIR/d2t.npy"
T2D_PATH="$VOCAB_DIR/t2d.npy"

python scripts/precompute_aurora_masks.py \
    --data-dir "$DATA_DIR" \
    --mask-output-dir "$MASK_OUTPUT_DIR" \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --verifier-name-or-path "$VERIFIER_NAME_OR_PATH" \
    --d2t-path "$D2T_PATH" \
    --t2d-path "$T2D_PATH" \
    --override-num-attention-heads "$OVERRIDE_NUM_ATTENTION_HEADS" \
    --override-intermediate-size "$OVERRIDE_INTERMEDIATE_SIZE" \
    --override-rope-theta "$OVERRIDE_ROPE_THETA" \
    2>&1 | tee "$MASK_OUTPUT_DIR/precompute.log"

echo "============================================="
echo " Mask precomputation complete!"
echo " Masks: $MASK_OUTPUT_DIR"
echo "============================================="
