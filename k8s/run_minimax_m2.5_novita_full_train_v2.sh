#!/bin/bash
set -euo pipefail

###############################################################################
# MiniMax-M2.5 Novita Full Streaming Training v2 (no turn filter)
# Runs: (1) rsync from datagen node, (2) buffer cleanup, (3) streaming training
###############################################################################

export HF_HUB_OFFLINE=1
export HF_HOME=/data/hf_cache
export LOCAL_TRAIN_ENV=1

VERIFIER_NAME_OR_PATH="${MODEL_PATH:-/data/models/MiniMax-M2.5}"
OUTPUT_PATH="${OUTPUT_PATH:-/data/output/minimax_m2.5_eagle3_novita_full_v2}"
NUM_GPUS="${NUM_GPUS:-8}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
LR="${LR:-3e-5}"
MIN_SAMPLES="${MIN_SAMPLES:-5000}"
FINAL_EPOCHS="${FINAL_EPOCHS:-10}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
DATAGEN_NODE="${DATAGEN_NODE:-10.83.115.21}"
BUFFER_MAX_SIZE_GB="${BUFFER_MAX_SIZE_GB:-1024}"
TARGET_TRAIN_COUNT="${TARGET_TRAIN_COUNT:-10}"
TARGET_GLOBAL_EPOCHS="${TARGET_GLOBAL_EPOCHS:-10}"
VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-500}"

GEN_DIR="$OUTPUT_PATH/gen"
MANIFEST_PATH="$GEN_DIR/manifest.json"
REMOTE_GEN_DIR="$OUTPUT_PATH/gen"

echo "============================================="
echo " MiniMax-M2.5 Novita Full Streaming Training v2 (no turn filter)"
echo " Model:        $VERIFIER_NAME_OR_PATH"
echo " Data:         $GEN_DIR"
echo " Manifest:     $MANIFEST_PATH"
echo " GPUs:         $NUM_GPUS"
echo " Min Samples:  $MIN_SAMPLES"
echo " Final Epochs: $FINAL_EPOCHS"
echo " Datagen Node: $DATAGEN_NODE"
echo " Buffer Max:   ${BUFFER_MAX_SIZE_GB}GB"
echo " Target TC:    $TARGET_TRAIN_COUNT"
echo " Target Epochs:$TARGET_GLOBAL_EPOCHS"
echo " Val Steps:    $VAL_EVERY_STEPS"
echo "============================================="

mkdir -p "$OUTPUT_PATH"/{checkpoints,logs} "$GEN_DIR"

# Ensure 'python' is available
if ! command -v python &>/dev/null; then
    ln -s "$(command -v python3)" /usr/local/bin/python
fi

# Ensure hostname resolves (needed for torchrun with hostNetwork)
if ! getent hosts "$(hostname)" &>/dev/null; then
    echo "127.0.0.1 $(hostname)" >> /etc/hosts
fi

# Install rsync and ssh client (not in vllm base image)
if ! command -v rsync &>/dev/null; then
    echo "[setup] Installing rsync and openssh-client..."
    export http_proxy=http://127.0.0.1:1083 https_proxy=http://127.0.0.1:1083
    apt-get update -qq && apt-get install -y -qq rsync openssh-client 2>/dev/null
    unset http_proxy https_proxy
fi

cd /workspace/speculators

###############################################################################
# Background: rsync from datagen node
###############################################################################
echo "[sync] Starting rsync loop from $DATAGEN_NODE..."
bash scripts/sync_datagen.sh \
    --datagen-nodes "$DATAGEN_NODE" \
    --remote-dir "$REMOTE_GEN_DIR" \
    --local-dir "$GEN_DIR" \
    --manifest-path "$MANIFEST_PATH" \
    --poll-interval "$POLL_INTERVAL" \
    --max-sync-size-gb "$BUFFER_MAX_SIZE_GB" \
    --target-train-count "$TARGET_TRAIN_COUNT" \
    > "$OUTPUT_PATH/logs/sync.log" 2>&1 &
SYNC_PID=$!
echo "[sync] PID=$SYNC_PID"

###############################################################################
# Background: buffer cleanup (with epoch lock + safety limits)
###############################################################################
echo "[cleanup] Starting buffer cleanup (max_delete=5000, min_retain=1000)..."
python scripts/buffer_cleanup.py \
    --manifest-path "$MANIFEST_PATH" \
    --data-dir "$GEN_DIR" \
    --max-size-gb "$BUFFER_MAX_SIZE_GB" \
    --max-delete-per-cycle 5000 \
    --min-retain-count 1000 \
    --poll-interval 60 \
    > "$OUTPUT_PATH/logs/cleanup.log" 2>&1 &
CLEANUP_PID=$!
echo "[cleanup] PID=$CLEANUP_PID"

###############################################################################
# Foreground: streaming training
###############################################################################
TRAIN_LOG="$OUTPUT_PATH/logs/train.log"
torchrun \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    scripts/train_streaming.py \
    --verifier-name-or-path "$VERIFIER_NAME_OR_PATH" \
    --manifest-path "$MANIFEST_PATH" \
    --data-path "$GEN_DIR" \
    --save-path "$OUTPUT_PATH/checkpoints" \
    --log-dir "$OUTPUT_PATH/logs" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --min-samples "$MIN_SAMPLES" \
    --final-epochs "$FINAL_EPOCHS" \
    --poll-interval "$POLL_INTERVAL" \
    --num-workers 4 \
    --prefetch-factor 2 \
    --max-val-files 200 \
    --val-every-steps "$VAL_EVERY_STEPS" \
    --target-global-epochs "$TARGET_GLOBAL_EPOCHS" \
    --run-name "minimax_m2.5_eagle3_novita_full_v2" \
    2>&1 | tee "$TRAIN_LOG"

# Kill background processes
kill $SYNC_PID ${CLEANUP_PID:+$CLEANUP_PID} 2>/dev/null || true

echo "============================================="
echo " Streaming training complete!"
echo " Checkpoints: $OUTPUT_PATH/checkpoints"
echo "============================================="
