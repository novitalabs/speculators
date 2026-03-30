#!/bin/bash
set -euo pipefail

###############################################################################
# Aurora Online Loop — Datagen + Mask Precompute (runs on .18 in k8s pod)
# Parameterized by env vars: AURORA_ROUND, AURORA_OUTPUT_DIR, etc.
###############################################################################

export HF_HUB_OFFLINE=1
export HF_HOME=/data/hf_cache
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Required env vars
ROUND="${AURORA_ROUND:?AURORA_ROUND required}"
OUTPUT_DIR="${AURORA_OUTPUT_DIR:?AURORA_OUTPUT_DIR required}"

# Config with defaults
VERIFIER_NAME_OR_PATH="${MODEL_PATH:-/data/models/MiniMax-M2.5}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/data/datasets/novita20260320/conversations.jsonl}"
SEED_CKPT="${AURORA_SEED_CKPT:-/data/output/minimax_m2.5_eagle3_novita0320/checkpoints/67}"
FILES_PER_ROUND="${AURORA_FILES_PER_ROUND:-2000}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
START_IDX=$(( ROUND * FILES_PER_ROUND ))

# Architecture overrides
OVERRIDE_NUM_ATTENTION_HEADS="${OVERRIDE_NUM_ATTENTION_HEADS:-24}"
OVERRIDE_INTERMEDIATE_SIZE="${OVERRIDE_INTERMEDIATE_SIZE:-8192}"
OVERRIDE_ROPE_THETA="${OVERRIDE_ROPE_THETA:-5000000}"

ROUND_DIR="$OUTPUT_DIR/round_${ROUND}"
GEN_DIR="$ROUND_DIR/gen"
MASK_DIR="$ROUND_DIR/masks"
VOCAB_DIR="$OUTPUT_DIR/vocab_mapping"
D2T_PATH="$VOCAB_DIR/d2t.npy"
T2D_PATH="$VOCAB_DIR/t2d.npy"

# Checkpoint: round 0 uses seed, round N>0 uses latest_ckpt
if [[ "$ROUND" -eq 0 ]]; then
    CKPT_PATH="$SEED_CKPT"
else
    CKPT_PATH="$OUTPUT_DIR/latest_ckpt"
    if [[ ! -d "$CKPT_PATH" ]] || [[ ! -f "$CKPT_PATH/model.safetensors" ]]; then
        echo "ERROR: latest_ckpt not found at $CKPT_PATH (expected from round $((ROUND - 1)))"
        exit 1
    fi
fi

echo "============================================="
echo " Aurora Online — Round $ROUND Datagen + Masks"
echo " Output:     $ROUND_DIR"
echo " Checkpoint: $CKPT_PATH"
echo " Files:      $FILES_PER_ROUND (start_idx=$START_IDX)"
echo "============================================="

# Ensure 'python' is available
if ! command -v python &>/dev/null; then
    ln -s "$(command -v python3)" /usr/local/bin/python
fi
if ! getent hosts "$(hostname)" &>/dev/null; then
    echo "127.0.0.1 $(hostname)" >> /etc/hosts
fi

cd /workspace/speculators
mkdir -p "$GEN_DIR" "$MASK_DIR"

# Verify vocab mapping
if [[ ! -f "$D2T_PATH" ]] || [[ ! -f "$T2D_PATH" ]]; then
    echo "ERROR: Vocab mapping not found at $VOCAB_DIR"
    exit 1
fi

###############################################################################
# Stage 1: Datagen (TP=4)
###############################################################################
echo "[round $ROUND] Stage 1: Datagen ($FILES_PER_ROUND files, TP=4)"
python scripts/data_generation_offline.py \
    --target-model-path "$VERIFIER_NAME_OR_PATH" \
    --train-data-path "$TRAIN_DATA_PATH" \
    --output-dir "$GEN_DIR" \
    --start-idx "$START_IDX" \
    --max-samples "$FILES_PER_ROUND" \
    --manifest-path "$GEN_DIR/manifest.json" \
    --tensor-parallel-size 4 \
    --seq-length "$SEQ_LENGTH" \
    2>&1 | tee "$ROUND_DIR/datagen.log"

PT_COUNT=$(find "$GEN_DIR" -name "data_*.pt" | wc -l)
echo "[round $ROUND] Datagen complete: $PT_COUNT files"

###############################################################################
# Stage 2: Mask precompute (1 GPU)
###############################################################################
echo "[round $ROUND] Stage 2: Mask precompute (1 GPU)"
CUDA_VISIBLE_DEVICES=0 python scripts/precompute_aurora_masks.py \
    --data-dir "$GEN_DIR" \
    --mask-output-dir "$MASK_DIR" \
    --checkpoint-path "$CKPT_PATH" \
    --verifier-name-or-path "$VERIFIER_NAME_OR_PATH" \
    --d2t-path "$D2T_PATH" \
    --t2d-path "$T2D_PATH" \
    --override-num-attention-heads "$OVERRIDE_NUM_ATTENTION_HEADS" \
    --override-intermediate-size "$OVERRIDE_INTERMEDIATE_SIZE" \
    --override-rope-theta "$OVERRIDE_ROPE_THETA" \
    2>&1 | tee "$ROUND_DIR/masks.log"

MASK_COUNT=$(find "$MASK_DIR" -name "mask_*.pt" | wc -l)
echo "[round $ROUND] Mask precompute complete: $MASK_COUNT masks"

echo "============================================="
echo " Round $ROUND datagen + masks complete!"
echo " Data:  $GEN_DIR ($PT_COUNT files)"
echo " Masks: $MASK_DIR ($MASK_COUNT masks)"
echo "============================================="
