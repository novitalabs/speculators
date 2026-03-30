#!/bin/bash
set -euo pipefail

###############################################################################
# Aurora Online Loop — Training (runs on .17 in k8s pod)
# Parameterized by env vars: AURORA_ROUND, AURORA_OUTPUT_DIR, etc.
###############################################################################

export HF_HUB_OFFLINE=1
export HF_HOME=/data/hf_cache
export LOCAL_TRAIN_ENV=1

# Required env vars
ROUND="${AURORA_ROUND:?AURORA_ROUND required}"
OUTPUT_DIR="${AURORA_OUTPUT_DIR:?AURORA_OUTPUT_DIR required}"

# Config with defaults
VERIFIER_NAME_OR_PATH="${MODEL_PATH:-/data/models/MiniMax-M2.5}"
EPOCHS_PER_ROUND="${AURORA_EPOCHS_PER_ROUND:-5}"
FILES_PER_ROUND="${AURORA_FILES_PER_ROUND:-2000}"
NUM_GPUS="${NUM_GPUS:-8}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
LR="${LR:-3e-5}"

# Aurora loss hyperparameters
LAMBDA_DISCARD="${LAMBDA_DISCARD:-0.1}"
DISCARD_TOP_K="${DISCARD_TOP_K:-10}"

# Architecture overrides
OVERRIDE_NUM_ATTENTION_HEADS="${OVERRIDE_NUM_ATTENTION_HEADS:-24}"
OVERRIDE_INTERMEDIATE_SIZE="${OVERRIDE_INTERMEDIATE_SIZE:-8192}"
OVERRIDE_ROPE_THETA="${OVERRIDE_ROPE_THETA:-5000000}"

ROUND_DIR="$OUTPUT_DIR/round_${ROUND}"
GEN_DIR="$ROUND_DIR/gen"
MASK_DIR="$ROUND_DIR/masks"
MANIFEST_PATH="$GEN_DIR/manifest.json"
SAVE_PATH="$OUTPUT_DIR/checkpoints"
LOG_DIR="$OUTPUT_DIR/logs"
VOCAB_DIR="$OUTPUT_DIR/vocab_mapping"
D2T_PATH="$VOCAB_DIR/d2t.npy"
T2D_PATH="$VOCAB_DIR/t2d.npy"

echo "============================================="
echo " Aurora Online — Round $ROUND Training"
echo " Data:     $GEN_DIR"
echo " Masks:    $MASK_DIR"
echo " Epochs:   $EPOCHS_PER_ROUND"
echo " GPUs:     $NUM_GPUS"
echo " Save:     $SAVE_PATH"
echo "============================================="

# Ensure 'python' is available
if ! command -v python &>/dev/null; then
    ln -s "$(command -v python3)" /usr/local/bin/python
fi
if ! getent hosts "$(hostname)" &>/dev/null; then
    echo "127.0.0.1 $(hostname)" >> /etc/hosts
fi

cd /workspace/speculators
mkdir -p "$SAVE_PATH" "$LOG_DIR"

# Verify data
PT_COUNT=$(find "$GEN_DIR" -name "data_*.pt" 2>/dev/null | wc -l)
MASK_COUNT=$(find "$MASK_DIR" -name "mask_*.pt" 2>/dev/null | wc -l)
echo "[verify] Data: $PT_COUNT files, Masks: $MASK_COUNT files"

if [[ "$PT_COUNT" -lt 100 ]]; then
    echo "ERROR: Not enough data files ($PT_COUNT). Rsync from .18 first."
    exit 1
fi
if [[ ! -f "$D2T_PATH" ]] || [[ ! -f "$T2D_PATH" ]]; then
    echo "ERROR: Vocab mapping not found at $VOCAB_DIR"
    exit 1
fi

# Ensure manifest is marked complete
python -c "
from speculators.train.manifest import read, write
m = read('$MANIFEST_PATH')
if m['status'] != 'complete':
    write('$MANIFEST_PATH', m['files'], status='complete')
    print(f'Marked complete: {len(m[\"files\"])} files')
else:
    print(f'Already complete: {len(m[\"files\"])} files')
"

# Training
MIN_SAMPLES=$(( FILES_PER_ROUND / 5 ))
echo "[train] Starting ($EPOCHS_PER_ROUND epochs, $NUM_GPUS GPUs, min_samples=$MIN_SAMPLES)"
torchrun \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    scripts/train_streaming.py \
    --verifier-name-or-path "$VERIFIER_NAME_OR_PATH" \
    --data-path "$GEN_DIR" \
    --manifest-path "$MANIFEST_PATH" \
    --mask-dir "$MASK_DIR" \
    --save-path "$SAVE_PATH" \
    --log-dir "$LOG_DIR" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --aurora-loss \
    --lambda-discard "$LAMBDA_DISCARD" \
    --discard-top-k "$DISCARD_TOP_K" \
    --aurora-static-mask \
    --final-epochs "$EPOCHS_PER_ROUND" \
    --min-samples "$MIN_SAMPLES" \
    --d2t-path "$D2T_PATH" \
    --t2d-path "$T2D_PATH" \
    --override-num-attention-heads "$OVERRIDE_NUM_ATTENTION_HEADS" \
    --override-intermediate-size "$OVERRIDE_INTERMEDIATE_SIZE" \
    --override-rope-theta "$OVERRIDE_ROPE_THETA" \
    --run-name "aurora_online_round_${ROUND}" \
    2>&1 | tee "$LOG_DIR/train_round_${ROUND}.log"

echo "============================================="
echo " Round $ROUND training complete!"
echo " Checkpoints: $SAVE_PATH"
echo "============================================="
