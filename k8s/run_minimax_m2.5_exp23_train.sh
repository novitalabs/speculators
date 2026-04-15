#!/bin/bash
set -euo pipefail

###############################################################################
# Exp 23: Training — Same dataset as exp22 (919K), dual TP=4 datagen on .17
# Aurora architecture. Data synced from datagen node (.17) via rsync.
# Training node: .18 (10.83.115.18)
###############################################################################

export HF_HUB_OFFLINE=1
export HF_HOME=/data/hf_cache
export LOCAL_TRAIN_ENV=1

VERIFIER_NAME_OR_PATH="${MODEL_PATH:-/data/models/MiniMax-M2.5}"
OUTPUT_PATH="${OUTPUT_PATH:-/data/output/minimax_m2.5_eagle3_exp23}"
NUM_GPUS="${NUM_GPUS:-8}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
LR="${LR:-3e-5}"
MIN_SAMPLES="${MIN_SAMPLES:-5000}"
FINAL_EPOCHS="${FINAL_EPOCHS:-10}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
DATAGEN_NODE="${DATAGEN_NODE:-10.83.115.17}"
BUFFER_MAX_SIZE_GB="${BUFFER_MAX_SIZE_GB:-500}"
TARGET_TRAIN_COUNT="${TARGET_TRAIN_COUNT:-10}"
TARGET_GLOBAL_EPOCHS="${TARGET_GLOBAL_EPOCHS:-0}"
VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-500}"

OVERRIDE_NUM_ATTENTION_HEADS="${OVERRIDE_NUM_ATTENTION_HEADS:-24}"
OVERRIDE_INTERMEDIATE_SIZE="${OVERRIDE_INTERMEDIATE_SIZE:-8192}"
OVERRIDE_ROPE_THETA="${OVERRIDE_ROPE_THETA:-5000000}"
DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-32000}"

GEN_DIR="$OUTPUT_PATH/gen"
MANIFEST_PATH="$GEN_DIR/manifest.json"
REMOTE_GEN_DIR="/data/output/minimax_m2.5_eagle3_exp23/gen"
VOCAB_DIR="$OUTPUT_PATH/vocab_mapping"

echo "============================================="
echo " Exp 23: Dual Datagen Training"
echo " Model:        $VERIFIER_NAME_OR_PATH"
echo " Data:         $GEN_DIR"
echo " GPUs:         $NUM_GPUS"
echo " Buffer Max:   ${BUFFER_MAX_SIZE_GB}GB"
echo " Datagen Node: $DATAGEN_NODE"
echo "============================================="

mkdir -p "$OUTPUT_PATH"/{checkpoints,logs} "$GEN_DIR" "$VOCAB_DIR"

if ! command -v python &>/dev/null; then
    ln -s "$(command -v python3)" /usr/local/bin/python
fi

if ! getent hosts "$(hostname)" &>/dev/null; then
    echo "127.0.0.1 $(hostname)" >> /etc/hosts
fi

if ! command -v rsync &>/dev/null; then
    echo "[setup] Installing rsync and openssh-client..."
    export http_proxy=http://127.0.0.1:1083 https_proxy=http://127.0.0.1:1083
    apt-get update -qq && apt-get install -y -qq rsync openssh-client 2>/dev/null
    unset http_proxy https_proxy
fi

cd /workspace/speculators

###############################################################################
# Step 0: Vocab mapping
###############################################################################
D2T_PATH="$VOCAB_DIR/d2t.npy"
T2D_PATH="$VOCAB_DIR/t2d.npy"

if [ ! -f "$D2T_PATH" ] || [ ! -f "$T2D_PATH" ]; then
    TOKEN_FREQ_PATH="$GEN_DIR/token_freq.pt"

    if [ ! -f "$TOKEN_FREQ_PATH" ]; then
        echo "[vocab] Waiting for token_freq.pt from datagen node ($DATAGEN_NODE)..."
        for i in $(seq 1 60); do
            if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "$DATAGEN_NODE" \
                "test -f $REMOTE_GEN_DIR/token_freq.pt" 2>/dev/null; then
                echo "[vocab] Found token_freq.pt on $DATAGEN_NODE, copying..."
                scp -o StrictHostKeyChecking=no "$DATAGEN_NODE:$REMOTE_GEN_DIR/token_freq.pt" "$TOKEN_FREQ_PATH"
                break
            fi
            echo "[vocab] Attempt $i/60: token_freq.pt not ready, waiting 30s..."
            sleep 30
        done
    fi

    if [ ! -f "$TOKEN_FREQ_PATH" ]; then
        echo "[vocab] WARNING: Using local fallback token_freq.pt"
        if [ -f "token_freq.pt" ]; then
            cp token_freq.pt "$TOKEN_FREQ_PATH"
        else
            echo "[ERROR] No token_freq.pt available."
            exit 1
        fi
    fi

    echo "[vocab] Generating d2t/t2d vocab mapping..."
    python scripts/build_vocab_mapping.py \
        --token-freq-path "$TOKEN_FREQ_PATH" \
        --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
        --target-vocab-size 200064 \
        --output-path "$VOCAB_DIR"
else
    echo "[vocab] Reusing existing vocab mapping at $VOCAB_DIR"
fi

###############################################################################
# Background: rsync from datagen node
# gen/ on .17 contains BOTH process A's files and merger's moved files
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

# Reverse sync: push difficulty_scores.json to datagen node periodically
(
    while true; do
        sleep 120
        if [ -f "$GEN_DIR/difficulty_scores.json" ]; then
            scp -o StrictHostKeyChecking=no -q \
                "$GEN_DIR/difficulty_scores.json" \
                "$DATAGEN_NODE:$REMOTE_GEN_DIR/difficulty_scores.json" 2>/dev/null || true
        fi
    done
) &
DIFFICULTY_SYNC_PID=$!
echo "[difficulty-sync] PID=$DIFFICULTY_SYNC_PID"

###############################################################################
# Background: buffer cleanup (prioritized eviction by difficulty)
###############################################################################
echo "[cleanup] Starting buffer cleanup..."
python scripts/buffer_cleanup.py \
    --manifest-path "$MANIFEST_PATH" \
    --data-dir "$GEN_DIR" \
    --max-size-gb "$BUFFER_MAX_SIZE_GB" \
    --poll-interval 120 \
    --max-delete-per-cycle 5000 \
    --min-retain-count 1000 \
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
    --d2t-path "$D2T_PATH" \
    --t2d-path "$T2D_PATH" \
    --override-num-attention-heads "$OVERRIDE_NUM_ATTENTION_HEADS" \
    --override-intermediate-size "$OVERRIDE_INTERMEDIATE_SIZE" \
    --override-rope-theta "$OVERRIDE_ROPE_THETA" \
    --run-name "minimax_m2.5_eagle3_exp23" \
    2>&1 | tee "$TRAIN_LOG"

kill $SYNC_PID $CLEANUP_PID $DIFFICULTY_SYNC_PID 2>/dev/null || true

echo "============================================="
echo " Exp 23 training complete!"
echo " Checkpoints: $OUTPUT_PATH/checkpoints"
echo "============================================="
