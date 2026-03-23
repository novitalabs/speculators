#!/bin/bash
set -euo pipefail

###############################################################################
# Exp 14: Evaluate ALL checkpoints with eval_checkpoints.py (offline val eval)
# Runs on .17 — first rsyncs data+checkpoints from .18, then evaluates.
###############################################################################

export HF_HUB_OFFLINE=1
export HF_HOME=/data/hf_cache

VERIFIER="/data/models/MiniMax-M2.5"
SOURCE_NODE="10.83.115.18"
REMOTE_CKPT_DIR="/data/output/minimax_m2.5_eagle3_aurora_arch/checkpoints"
REMOTE_GEN_DIR="/data/output/minimax_m2.5_eagle3_aurora_arch/gen"
LOCAL_CKPT_DIR="/data/output/minimax_m2.5_eagle3_aurora_arch/checkpoints"
LOCAL_GEN_DIR="/data/output/minimax_m2.5_eagle3_aurora_arch/gen"
NUM_GPUS="${NUM_GPUS:-8}"

# Aurora architecture overrides
OVERRIDE_NUM_ATTENTION_HEADS=24
OVERRIDE_INTERMEDIATE_SIZE=8192
OVERRIDE_ROPE_THETA=5000000
DRAFT_VOCAB_SIZE=32000

if ! command -v python &>/dev/null; then
    ln -s "$(command -v python3)" /usr/local/bin/python
fi

# Ensure hostname resolves
if ! getent hosts "$(hostname)" &>/dev/null; then
    echo "127.0.0.1 $(hostname)" >> /etc/hosts
fi

# Install rsync and ssh
if ! command -v rsync &>/dev/null; then
    echo "[setup] Installing rsync and openssh-client..."
    export http_proxy=http://127.0.0.1:1083 https_proxy=http://127.0.0.1:1083
    apt-get update -qq && apt-get install -y -qq rsync openssh-client 2>/dev/null
    unset http_proxy https_proxy
fi

cd /workspace/speculators

###############################################################################
# Step 1: Rsync checkpoints from .18 (only model files, skip optimizer)
###############################################################################
echo "[sync] Syncing checkpoints from $SOURCE_NODE..."
mkdir -p "$LOCAL_CKPT_DIR"

# Retry wrapper for unstable SSH
sync_with_retry() {
    local src="$1" dst="$2" extra="${3:-}"
    for attempt in 1 2 3 4 5; do
        if rsync -av --timeout=60 -e "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10" $extra "$src" "$dst"; then
            return 0
        fi
        echo "  [retry] Attempt $attempt failed, waiting 5s..."
        sleep 5
    done
    echo "  [ERROR] Failed after 5 attempts: $src"
    return 1
}

# Sync all checkpoint dirs (model files only)
for ckpt_dir in $(ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o BatchMode=yes "$SOURCE_NODE" "ls -d $REMOTE_CKPT_DIR/*/"); do
    ckpt_name=$(basename "$ckpt_dir")
    if [ -f "$LOCAL_CKPT_DIR/$ckpt_name/model.safetensors" ]; then
        echo "  [skip] Checkpoint $ckpt_name already exists locally"
        continue
    fi
    echo "  [sync] Checkpoint $ckpt_name..."
    mkdir -p "$LOCAL_CKPT_DIR/$ckpt_name"
    sync_with_retry \
        "$SOURCE_NODE:$REMOTE_CKPT_DIR/$ckpt_name/" \
        "$LOCAL_CKPT_DIR/$ckpt_name/" \
        "--exclude=optimizer_state_dict.pt" || true
done

echo "[sync] Checkpoint sync complete. Local checkpoints:"
ls "$LOCAL_CKPT_DIR" | sort -n | tr '\n' ' '
echo ""

###############################################################################
# Step 2: Rsync gen data (manifest + .pt files) from .18
###############################################################################
echo "[sync] Syncing gen data from $SOURCE_NODE..."
mkdir -p "$LOCAL_GEN_DIR"

# First sync manifest
sync_with_retry "$SOURCE_NODE:$REMOTE_GEN_DIR/manifest.json" "$LOCAL_GEN_DIR/manifest.json"

# Then sync .pt files in batches
sync_with_retry "$SOURCE_NODE:$REMOTE_GEN_DIR/" "$LOCAL_GEN_DIR/" "--include=*.pt --exclude=*"

echo "[sync] Gen data sync complete. Files:"
ls "$LOCAL_GEN_DIR"/*.pt 2>/dev/null | wc -l
echo " .pt files synced"

###############################################################################
# Step 3: Evaluate all checkpoints
###############################################################################
# Build list of all available checkpoint numbers
CKPT_NUMS=$(ls "$LOCAL_CKPT_DIR" | sort -n | tr '\n' ' ')
echo "[eval] Evaluating checkpoints: $CKPT_NUMS"

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
    --draft-vocab-size "$DRAFT_VOCAB_SIZE"

echo ""
echo "============================================="
echo " Exp 14: All checkpoint evaluation complete!"
echo " Results: $LOCAL_CKPT_DIR/eval_results.json"
echo "============================================="
