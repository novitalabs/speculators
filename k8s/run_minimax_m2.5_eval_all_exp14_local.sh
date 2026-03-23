#!/bin/bash
set -euo pipefail

###############################################################################
# Exp 14: Evaluate ALL checkpoints with eval_checkpoints.py (offline val eval)
# Runs on .17 — data and checkpoints already synced locally.
###############################################################################

export HF_HUB_OFFLINE=1
export HF_HOME=/data/hf_cache

VERIFIER="/data/models/MiniMax-M2.5"
LOCAL_CKPT_DIR="/data/output/minimax_m2.5_eagle3_aurora_arch/checkpoints"
LOCAL_GEN_DIR="/data/output/minimax_m2.5_eagle3_aurora_arch/gen"
NUM_GPUS="${NUM_GPUS:-8}"

# Aurora architecture overrides
OVERRIDE_NUM_ATTENTION_HEADS=24
OVERRIDE_INTERMEDIATE_SIZE=8192
OVERRIDE_ROPE_THETA=5000000
DRAFT_VOCAB_SIZE=32000
VOCAB_DIR="/data/output/minimax_m2.5_eagle3_aurora_arch/vocab_mapping"

if ! command -v python &>/dev/null; then
    ln -s "$(command -v python3)" /usr/local/bin/python
fi

if ! getent hosts "$(hostname)" &>/dev/null; then
    echo "127.0.0.1 $(hostname)" >> /etc/hosts
fi

cd /workspace/speculators

# Build list of all available checkpoint numbers
CKPT_NUMS=$(ls "$LOCAL_CKPT_DIR" | sort -n | tr '\n' ' ')
NUM_FILES=$(find "$LOCAL_GEN_DIR" -name "*.pt" | wc -l)
echo "[eval] Available checkpoints: $CKPT_NUMS"
echo "[eval] Gen data files: $NUM_FILES"
echo "[eval] Manifest: $LOCAL_GEN_DIR/manifest.json"

torchrun \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    scripts/eval_checkpoints.py \
    --verifier-name-or-path "$VERIFIER" \
    --manifest-path "$LOCAL_GEN_DIR/manifest.json" \
    --data-path "$LOCAL_GEN_DIR" \
    --checkpoint-dir "$LOCAL_CKPT_DIR" \
    --checkpoints $CKPT_NUMS \
    --total-seq-len 8192 \
    --max-val-files 200 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --override-num-attention-heads "$OVERRIDE_NUM_ATTENTION_HEADS" \
    --override-intermediate-size "$OVERRIDE_INTERMEDIATE_SIZE" \
    --override-rope-theta "$OVERRIDE_ROPE_THETA" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --d2t-path "$VOCAB_DIR/d2t.npy" \
    --t2d-path "$VOCAB_DIR/t2d.npy"

echo ""
echo "============================================="
echo " Exp 14: All checkpoint evaluation complete!"
echo " Results: $LOCAL_CKPT_DIR/eval_results.json"
echo "============================================="
