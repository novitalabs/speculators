#!/bin/bash
set -euo pipefail

###############################################################################
# Aurora Online Loop — Single Round Orchestrator
#
# Runs on .18 (datagen node, 4 GPU TP=4). SSHes to .17 for training (8 GPU).
# Each round: datagen → mask precompute → rsync → train → rsync ckpt back.
###############################################################################

usage() {
    echo "Usage: $0 --round N --output-dir DIR --verifier-name-or-path PATH --seed-ckpt PATH --train-node HOST [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  --round N                     Round number (0-indexed)"
    echo "  --output-dir DIR              Base output directory"
    echo "  --verifier-name-or-path PATH  Verifier model path"
    echo "  --seed-ckpt PATH              Seed checkpoint (Exp15 ckpt67)"
    echo "  --train-node HOST             Training node IP (e.g., 10.83.115.17)"
    echo ""
    echo "Optional:"
    echo "  --files-per-round N           Files per round (default: 2000)"
    echo "  --epochs-per-round N          Epochs per round (default: 5)"
    echo "  --train-data-path PATH        Training data path (default: /data/datasets/novita20260320/conversations.jsonl)"
    echo "  --vocab-dir DIR               Vocab mapping dir (default: OUTPUT_DIR/vocab_mapping)"
    echo "  --remote-base DIR             Remote base dir on train node (default: same as output-dir)"
    echo "  --seq-length N                Sequence length (default: 8192)"
    echo "  --override-num-attention-heads N   (default: 24)"
    echo "  --override-intermediate-size N     (default: 8192)"
    echo "  --override-rope-theta N            (default: 5000000)"
    echo "  --lr FLOAT                    Learning rate (default: 3e-5)"
    echo "  --lambda-discard FLOAT        Aurora discard weight (default: 0.1)"
    echo "  --discard-top-k N             Aurora discard top-k (default: 10)"
    exit 1
}

# Defaults
ROUND=""
OUTPUT_DIR=""
VERIFIER_NAME_OR_PATH=""
SEED_CKPT=""
TRAIN_NODE=""
FILES_PER_ROUND=2000
EPOCHS_PER_ROUND=5
TRAIN_DATA_PATH="/data/datasets/novita20260320/conversations.jsonl"
VOCAB_DIR=""
REMOTE_BASE=""
SEQ_LENGTH=8192
OVERRIDE_NUM_ATTENTION_HEADS=24
OVERRIDE_INTERMEDIATE_SIZE=8192
OVERRIDE_ROPE_THETA=5000000
LR=3e-5
LAMBDA_DISCARD=0.1
DISCARD_TOP_K=10

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --round) ROUND="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --verifier-name-or-path) VERIFIER_NAME_OR_PATH="$2"; shift 2 ;;
        --seed-ckpt) SEED_CKPT="$2"; shift 2 ;;
        --train-node) TRAIN_NODE="$2"; shift 2 ;;
        --files-per-round) FILES_PER_ROUND="$2"; shift 2 ;;
        --epochs-per-round) EPOCHS_PER_ROUND="$2"; shift 2 ;;
        --train-data-path) TRAIN_DATA_PATH="$2"; shift 2 ;;
        --vocab-dir) VOCAB_DIR="$2"; shift 2 ;;
        --remote-base) REMOTE_BASE="$2"; shift 2 ;;
        --seq-length) SEQ_LENGTH="$2"; shift 2 ;;
        --override-num-attention-heads) OVERRIDE_NUM_ATTENTION_HEADS="$2"; shift 2 ;;
        --override-intermediate-size) OVERRIDE_INTERMEDIATE_SIZE="$2"; shift 2 ;;
        --override-rope-theta) OVERRIDE_ROPE_THETA="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --lambda-discard) LAMBDA_DISCARD="$2"; shift 2 ;;
        --discard-top-k) DISCARD_TOP_K="$2"; shift 2 ;;
        --help) usage ;;
        *) echo "Unknown arg: $1"; usage ;;
    esac
done

# Validate required args
[[ -z "$ROUND" ]] && echo "ERROR: --round required" && exit 1
[[ -z "$OUTPUT_DIR" ]] && echo "ERROR: --output-dir required" && exit 1
[[ -z "$VERIFIER_NAME_OR_PATH" ]] && echo "ERROR: --verifier-name-or-path required" && exit 1
[[ -z "$SEED_CKPT" ]] && echo "ERROR: --seed-ckpt required" && exit 1
[[ -z "$TRAIN_NODE" ]] && echo "ERROR: --train-node required" && exit 1

# Derived paths
VOCAB_DIR="${VOCAB_DIR:-$OUTPUT_DIR/vocab_mapping}"
REMOTE_BASE="${REMOTE_BASE:-$OUTPUT_DIR}"
ROUND_DIR="$OUTPUT_DIR/round_${ROUND}"
GEN_DIR="$ROUND_DIR/gen"
MASK_DIR="$ROUND_DIR/masks"
MANIFEST_PATH="$GEN_DIR/manifest.json"
START_IDX=$(( ROUND * FILES_PER_ROUND ))

# Checkpoint selection: round 0 uses seed, round N>0 uses latest
if [[ "$ROUND" -eq 0 ]]; then
    CKPT_PATH="$SEED_CKPT"
else
    CKPT_PATH="$OUTPUT_DIR/latest_ckpt"
    if [[ ! -d "$CKPT_PATH" ]]; then
        echo "ERROR: latest_ckpt not found at $CKPT_PATH (expected from round $((ROUND - 1)))"
        exit 1
    fi
fi

D2T_PATH="$VOCAB_DIR/d2t.npy"
T2D_PATH="$VOCAB_DIR/t2d.npy"

echo "============================================="
echo " Aurora Online Loop — Round $ROUND"
echo " Output:       $ROUND_DIR"
echo " Checkpoint:   $CKPT_PATH"
echo " Files:        $FILES_PER_ROUND (start_idx=$START_IDX)"
echo " Epochs:       $EPOCHS_PER_ROUND"
echo " Train Node:   $TRAIN_NODE"
echo "============================================="

mkdir -p "$GEN_DIR" "$MASK_DIR"

###############################################################################
# Stage 1: Datagen (TP=4 on .18)
###############################################################################
echo "[round $ROUND] Stage 1: Datagen ($FILES_PER_ROUND files, TP=4)"

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
python scripts/data_generation_offline.py \
    --target-model-path "$VERIFIER_NAME_OR_PATH" \
    --train-data-path "$TRAIN_DATA_PATH" \
    --output-dir "$GEN_DIR" \
    --start-idx "$START_IDX" \
    --max-samples "$FILES_PER_ROUND" \
    --manifest-path "$MANIFEST_PATH" \
    --tensor-parallel-size 4 \
    --seq-length "$SEQ_LENGTH" \
    2>&1 | tee "$ROUND_DIR/datagen.log"

PT_COUNT=$(ls "$GEN_DIR"/data_*.pt 2>/dev/null | wc -l)
echo "[round $ROUND] Datagen complete: $PT_COUNT files"

###############################################################################
# Stage 2: Mask precompute (1 GPU on .18)
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

MASK_COUNT=$(ls "$MASK_DIR"/mask_*.pt 2>/dev/null | wc -l)
echo "[round $ROUND] Mask precompute complete: $MASK_COUNT masks"

###############################################################################
# Stage 3: Sync data + masks to training node
###############################################################################
REMOTE_ROUND="$REMOTE_BASE/round_${ROUND}"
REMOTE_GEN="$REMOTE_ROUND/gen"
REMOTE_MASKS="$REMOTE_ROUND/masks"

echo "[round $ROUND] Stage 3: Syncing to $TRAIN_NODE"
ssh "$TRAIN_NODE" "mkdir -p $REMOTE_GEN $REMOTE_MASKS"
rsync -az "$GEN_DIR/" "$TRAIN_NODE:$REMOTE_GEN/"
rsync -az "$MASK_DIR/" "$TRAIN_NODE:$REMOTE_MASKS/"
echo "[round $ROUND] Sync complete"

###############################################################################
# Stage 4: Mark manifest complete on remote (triggers --final-epochs)
###############################################################################
REMOTE_MANIFEST="$REMOTE_GEN/manifest.json"
echo "[round $ROUND] Stage 4: Marking manifest complete on $TRAIN_NODE"
ssh "$TRAIN_NODE" "python -c \"
from speculators.train.manifest import read, write
m = read('$REMOTE_MANIFEST')
write('$REMOTE_MANIFEST', m['files'], status='complete')
\""

###############################################################################
# Stage 5: Train on .17 (8 GPU FSDP)
###############################################################################
REMOTE_SAVE="$REMOTE_BASE/checkpoints"
REMOTE_LOGS="$REMOTE_BASE/logs"
REMOTE_VOCAB="$REMOTE_BASE/vocab_mapping"

echo "[round $ROUND] Stage 5: Training on $TRAIN_NODE ($EPOCHS_PER_ROUND epochs)"
ssh "$TRAIN_NODE" "
    export HF_HUB_OFFLINE=1
    export HF_HOME=/data/hf_cache
    export LOCAL_TRAIN_ENV=1
    mkdir -p $REMOTE_SAVE $REMOTE_LOGS
    cd /workspace/speculators
    torchrun --standalone --nproc_per_node=8 \
        scripts/train_streaming.py \
        --verifier-name-or-path $VERIFIER_NAME_OR_PATH \
        --data-path $REMOTE_GEN \
        --manifest-path $REMOTE_MANIFEST \
        --mask-dir $REMOTE_MASKS \
        --save-path $REMOTE_SAVE \
        --log-dir $REMOTE_LOGS \
        --lr $LR \
        --total-seq-len $SEQ_LENGTH \
        --aurora-loss \
        --lambda-discard $LAMBDA_DISCARD \
        --discard-top-k $DISCARD_TOP_K \
        --aurora-static-mask \
        --final-epochs $EPOCHS_PER_ROUND \
        --d2t-path $REMOTE_VOCAB/d2t.npy \
        --t2d-path $REMOTE_VOCAB/t2d.npy \
        --override-num-attention-heads $OVERRIDE_NUM_ATTENTION_HEADS \
        --override-intermediate-size $OVERRIDE_INTERMEDIATE_SIZE \
        --override-rope-theta $OVERRIDE_ROPE_THETA \
        --run-name aurora_online_round_${ROUND} \
        2>&1 | tee $REMOTE_LOGS/train_round_${ROUND}.log
"

###############################################################################
# Stage 6: Sync latest checkpoint back to .18
###############################################################################
echo "[round $ROUND] Stage 6: Syncing checkpoint back"

# Find latest checkpoint on remote
LATEST_CKPT=$(ssh "$TRAIN_NODE" "ls -1d $REMOTE_SAVE/ckpt_* 2>/dev/null | sort -t_ -k2 -n | tail -1")
if [[ -z "$LATEST_CKPT" ]]; then
    echo "ERROR: No checkpoint found on $TRAIN_NODE at $REMOTE_SAVE"
    exit 1
fi

mkdir -p "$OUTPUT_DIR/latest_ckpt"
rsync -az "$TRAIN_NODE:$LATEST_CKPT/" "$OUTPUT_DIR/latest_ckpt/"
echo "[round $ROUND] Synced checkpoint: $LATEST_CKPT → $OUTPUT_DIR/latest_ckpt/"

echo "============================================="
echo " Round $ROUND complete!"
echo " Data:   $PT_COUNT files"
echo " Masks:  $MASK_COUNT masks"
echo " Ckpt:   $OUTPUT_DIR/latest_ckpt/"
echo "============================================="
