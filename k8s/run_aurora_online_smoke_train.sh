#!/bin/bash
set -euo pipefail

###############################################################################
# Aurora Online Smoke Test — Training (runs on .17)
# Expects data + masks already synced to /data on this node.
###############################################################################

export HF_HUB_OFFLINE=1
export HF_HOME=/data/hf_cache
export LOCAL_TRAIN_ENV=1

VERIFIER_NAME_OR_PATH="${MODEL_PATH:-/data/models/MiniMax-M2.5}"
OUTPUT_DIR="/data/output/minimax_m2.5_eagle3_aurora_online_smoke"
EPOCHS_PER_ROUND=1
NUM_GPUS=8
SEQ_LENGTH=8192
LR=3e-5

# Aurora loss hyperparameters
LAMBDA_DISCARD=0.1
DISCARD_TOP_K=10

# Architecture overrides
OVERRIDE_NUM_ATTENTION_HEADS=24
OVERRIDE_INTERMEDIATE_SIZE=8192
OVERRIDE_ROPE_THETA=5000000

ROUND_DIR="$OUTPUT_DIR/round_0"
GEN_DIR="$ROUND_DIR/gen"
MASK_DIR="$ROUND_DIR/masks"
MANIFEST_PATH="$GEN_DIR/manifest.json"
SAVE_PATH="$OUTPUT_DIR/checkpoints"
LOG_DIR="$OUTPUT_DIR/logs"
VOCAB_DIR="$OUTPUT_DIR/vocab_mapping"

D2T_PATH="$VOCAB_DIR/d2t.npy"
T2D_PATH="$VOCAB_DIR/t2d.npy"

echo "============================================="
echo " Aurora Online Smoke — Training"
echo " Model:      $VERIFIER_NAME_OR_PATH"
echo " Data:       $GEN_DIR"
echo " Masks:      $MASK_DIR"
echo " Manifest:   $MANIFEST_PATH"
echo " Epochs:     $EPOCHS_PER_ROUND"
echo " GPUs:       $NUM_GPUS"
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

###############################################################################
# Verify data is present
###############################################################################
PT_COUNT=$(find "$GEN_DIR" -name "data_*.pt" 2>/dev/null | wc -l)
MASK_COUNT=$(find "$MASK_DIR" -name "mask_*.pt" 2>/dev/null | wc -l)
echo "[verify] Data: $PT_COUNT files, Masks: $MASK_COUNT files"

if [[ "$PT_COUNT" -lt 10 ]]; then
    echo "[ERROR] Not enough data files. Did you rsync from .18?"
    echo "  rsync -az 10.83.115.18:$ROUND_DIR/ $ROUND_DIR/"
    exit 1
fi

if [[ ! -f "$D2T_PATH" ]] || [[ ! -f "$T2D_PATH" ]]; then
    echo "[ERROR] Vocab mapping not found at $VOCAB_DIR"
    echo "  rsync -az 10.83.115.18:$VOCAB_DIR/ $VOCAB_DIR/"
    exit 1
fi

###############################################################################
# Ensure manifest is marked complete (triggers --final-epochs)
###############################################################################
echo "[setup] Marking manifest complete..."
python -c "
from speculators.train.manifest import read, write
m = read('$MANIFEST_PATH')
if m['status'] != 'complete':
    write('$MANIFEST_PATH', m['files'], status='complete')
    print(f'  Marked complete: {len(m[\"files\"])} files')
else:
    print(f'  Already complete: {len(m[\"files\"])} files')
"

###############################################################################
# Training
###############################################################################
echo "[train] Starting training ($EPOCHS_PER_ROUND epochs, $NUM_GPUS GPUs)"
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
    --min-samples 10 \
    --d2t-path "$D2T_PATH" \
    --t2d-path "$T2D_PATH" \
    --override-num-attention-heads "$OVERRIDE_NUM_ATTENTION_HEADS" \
    --override-intermediate-size "$OVERRIDE_INTERMEDIATE_SIZE" \
    --override-rope-theta "$OVERRIDE_ROPE_THETA" \
    --run-name "aurora_online_smoke_round_0" \
    2>&1 | tee "$LOG_DIR/train_smoke_round_0.log"

echo "============================================="
echo " Smoke test training complete!"
echo " Checkpoints: $SAVE_PATH"
echo "============================================="
