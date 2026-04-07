#!/bin/bash
set -euo pipefail

###############################################################################
# Exp 20: Evaluate all available checkpoints
# Uses same Aurora architecture overrides + vocab mapping as training
###############################################################################

export HF_HUB_OFFLINE=1
export HF_HOME=/data/hf_cache

VERIFIER_NAME_OR_PATH="${MODEL_PATH:-/data/models/MiniMax-M2.5}"
OUTPUT_PATH="${OUTPUT_PATH:-/data/output/minimax_m2.5_eagle3_exp20}"
NUM_GPUS="${NUM_GPUS:-8}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"

CHECKPOINT_DIR="$OUTPUT_PATH/checkpoints"
GEN_DIR="$OUTPUT_PATH/gen"
MANIFEST_PATH="$GEN_DIR/manifest.json"
VOCAB_DIR="$OUTPUT_PATH/vocab_mapping"
D2T_PATH="$VOCAB_DIR/d2t.npy"
T2D_PATH="$VOCAB_DIR/t2d.npy"
LOGFILE="$OUTPUT_PATH/logs/eval.log"

# Discover all checkpoint epochs
CHECKPOINTS=$(ls -d "$CHECKPOINT_DIR"/*/ 2>/dev/null | xargs -I{} basename {} | sort -n | tr '\n' ' ')

echo "============================================="
echo " Exp 20: Checkpoint Evaluation"
echo " Model:       $VERIFIER_NAME_OR_PATH"
echo " Data:        $GEN_DIR"
echo " Checkpoints: $CHECKPOINTS"
echo " GPUs:        $NUM_GPUS"
echo "============================================="

mkdir -p "$OUTPUT_PATH/logs"

torchrun \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    scripts/eval_checkpoints.py \
    --verifier-name-or-path "$VERIFIER_NAME_OR_PATH" \
    --manifest-path "$MANIFEST_PATH" \
    --data-path "$GEN_DIR" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --checkpoints $CHECKPOINTS \
    --total-seq-len "$SEQ_LENGTH" \
    --max-val-files 200 \
    --d2t-path "$D2T_PATH" \
    --t2d-path "$T2D_PATH" \
    --override-num-attention-heads 24 \
    --override-intermediate-size 8192 \
    --override-rope-theta 5000000 \
    2>&1 | tee "$LOGFILE"

echo "============================================="
echo " Eval complete! Results: $CHECKPOINT_DIR/eval_results.json"
echo "============================================="
