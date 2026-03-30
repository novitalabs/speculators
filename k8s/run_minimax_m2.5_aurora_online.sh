#!/bin/bash
set -euo pipefail

###############################################################################
# Phase 2: Aurora Online Loop — Multi-Round Driver
#
# Runs on .18 (datagen node, 4 GPU TP=4). Loops over rounds, calling
# aurora_online_round.sh for each. .17 (8 GPU) handles training via SSH.
#
# Seed: Exp15 ckpt67 (best standard KL model, 63.2% acc@0)
###############################################################################

export HF_HUB_OFFLINE=1
export HF_HOME=/data/hf_cache
export LOCAL_TRAIN_ENV=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Ensure 'python' is available
if ! command -v python &>/dev/null; then
    ln -s "$(command -v python3)" /usr/local/bin/python
fi

# Ensure hostname resolves (needed for torchrun with hostNetwork)
if ! getent hosts "$(hostname)" &>/dev/null; then
    echo "127.0.0.1 $(hostname)" >> /etc/hosts
fi

# -- Config --
VERIFIER_NAME_OR_PATH="${MODEL_PATH:-/data/models/MiniMax-M2.5}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/output/minimax_m2.5_eagle3_aurora_online}"
SEED_CKPT="${SEED_CKPT:-/data/output/minimax_m2.5_eagle3_novita0320/checkpoints/67}"
TRAIN_NODE="${TRAIN_NODE:-10.83.115.17}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/data/datasets/novita20260320/conversations.jsonl}"

# Loop parameters
FILES_PER_ROUND="${FILES_PER_ROUND:-2000}"
EPOCHS_PER_ROUND="${EPOCHS_PER_ROUND:-5}"
MAX_ROUNDS="${MAX_ROUNDS:-20}"
START_ROUND="${START_ROUND:-0}"

# Architecture overrides (same as Exp14/15/16)
OVERRIDE_NUM_ATTENTION_HEADS="${OVERRIDE_NUM_ATTENTION_HEADS:-24}"
OVERRIDE_INTERMEDIATE_SIZE="${OVERRIDE_INTERMEDIATE_SIZE:-8192}"
OVERRIDE_ROPE_THETA="${OVERRIDE_ROPE_THETA:-5000000}"

# Training hyperparameters
LR="${LR:-3e-5}"
LAMBDA_DISCARD="${LAMBDA_DISCARD:-0.1}"
DISCARD_TOP_K="${DISCARD_TOP_K:-10}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"

VOCAB_DIR="$OUTPUT_DIR/vocab_mapping"

echo "============================================="
echo " Phase 2: Aurora Online Loop"
echo " Verifier:   $VERIFIER_NAME_OR_PATH"
echo " Output:     $OUTPUT_DIR"
echo " Seed Ckpt:  $SEED_CKPT"
echo " Train Node: $TRAIN_NODE"
echo " --- Loop ---"
echo " Rounds:     $START_ROUND → $((MAX_ROUNDS - 1))"
echo " Files/Round:  $FILES_PER_ROUND"
echo " Epochs/Round: $EPOCHS_PER_ROUND"
echo " --- Architecture ---"
echo " Attention Heads: $OVERRIDE_NUM_ATTENTION_HEADS"
echo " Intermediate:    $OVERRIDE_INTERMEDIATE_SIZE"
echo " Rope Theta:      $OVERRIDE_ROPE_THETA"
echo "============================================="

mkdir -p "$OUTPUT_DIR"/{checkpoints,logs} "$VOCAB_DIR"

SPECULATORS_DIR="${SPECULATORS_DIR:-/workspace/speculators}"
cd "$SPECULATORS_DIR"

###############################################################################
# Step 0: Ensure vocab mapping exists on both nodes
###############################################################################
D2T_PATH="$VOCAB_DIR/d2t.npy"
T2D_PATH="$VOCAB_DIR/t2d.npy"

if [[ ! -f "$D2T_PATH" ]] || [[ ! -f "$T2D_PATH" ]]; then
    echo "[setup] Vocab mapping not found at $VOCAB_DIR"
    echo "[setup] Generating vocab mapping..."
    python scripts/build_vocab_mapping.py \
        --token-freq-path ./token_freq.pt \
        --target-model-path "$VERIFIER_NAME_OR_PATH" \
        --draft-vocab-size 32000 \
        --output-path "$VOCAB_DIR"
fi
echo "[setup] Vocab mapping: $VOCAB_DIR"

# Sync vocab mapping to training node
REMOTE_VOCAB="$OUTPUT_DIR/vocab_mapping"
ssh "$TRAIN_NODE" "mkdir -p $REMOTE_VOCAB"
rsync -az "$VOCAB_DIR/" "$TRAIN_NODE:$REMOTE_VOCAB/"
echo "[setup] Synced vocab mapping to $TRAIN_NODE"

###############################################################################
# Main loop
###############################################################################
for (( R=START_ROUND; R<MAX_ROUNDS; R++ )); do
    echo ""
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo "  Starting Round $R / $((MAX_ROUNDS - 1))"
    echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    echo ""

    bash scripts/aurora_online_round.sh \
        --round "$R" \
        --output-dir "$OUTPUT_DIR" \
        --verifier-name-or-path "$VERIFIER_NAME_OR_PATH" \
        --seed-ckpt "$SEED_CKPT" \
        --train-node "$TRAIN_NODE" \
        --files-per-round "$FILES_PER_ROUND" \
        --epochs-per-round "$EPOCHS_PER_ROUND" \
        --train-data-path "$TRAIN_DATA_PATH" \
        --vocab-dir "$VOCAB_DIR" \
        --seq-length "$SEQ_LENGTH" \
        --override-num-attention-heads "$OVERRIDE_NUM_ATTENTION_HEADS" \
        --override-intermediate-size "$OVERRIDE_INTERMEDIATE_SIZE" \
        --override-rope-theta "$OVERRIDE_ROPE_THETA" \
        --lr "$LR" \
        --lambda-discard "$LAMBDA_DISCARD" \
        --discard-top-k "$DISCARD_TOP_K"

    echo "[loop] Round $R complete. Checkpoint at $OUTPUT_DIR/latest_ckpt/"
done

echo "============================================="
echo " Aurora Online Loop complete!"
echo " Rounds: $START_ROUND → $((MAX_ROUNDS - 1))"
echo " Final ckpt: $OUTPUT_DIR/latest_ckpt/"
echo "============================================="
