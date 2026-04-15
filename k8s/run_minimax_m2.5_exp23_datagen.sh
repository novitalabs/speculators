#!/bin/bash
set -euo pipefail

###############################################################################
# Exp 23: Dual TP=4 Continuous Datagen
# Same dataset as exp22 (novita_merged 724K + nemotron-v2-chinese 195K = 919K)
# Two parallel TP=4 datagen processes on .17 to double throughput.
#
# Process A: GPU 0-3, writes to gen/
# Process B: GPU 4-7, writes to gen_b/
# Merger:    Moves gen_b/*.pt -> gen/ with index offset 5000000
###############################################################################

export VLLM_ATTENTION_BACKEND=FLASH_ATTN

if ! command -v python &>/dev/null; then
    ln -s "$(command -v python3)" /usr/local/bin/python
fi

VERIFIER_NAME_OR_PATH="${MODEL_PATH:-/data/models/MiniMax-M2.5}"
OUTPUT_PATH="${OUTPUT_PATH:-/data/output/minimax_m2.5_eagle3_exp23}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
TRAIN_DATA="${TRAIN_DATA:-/data/datasets/exp22_merged}"

GEN_DIR="$OUTPUT_PATH/gen"
GEN_DIR_B="$OUTPUT_PATH/gen_b"
DIFFICULTY_SCORES="$GEN_DIR/difficulty_scores.json"
INDEX_OFFSET=5000000

echo "============================================="
echo " Exp 23: Dual TP=4 Datagen"
echo " Model:        $VERIFIER_NAME_OR_PATH"
echo " Output A:     $GEN_DIR"
echo " Output B:     $GEN_DIR_B"
echo " Data:         $TRAIN_DATA"
echo " Seq Length:   $SEQ_LENGTH"
echo " Index Offset: $INDEX_OFFSET"
echo " Mode:         DUAL CONTINUOUS"
echo "============================================="

# Create merged dataset directory if not already set up
if [ ! -d "$TRAIN_DATA" ]; then
    echo "[setup] Creating merged dataset directory at $TRAIN_DATA..."
    mkdir -p "$TRAIN_DATA"
    ln -sf /data/datasets/novita_merged/train.jsonl "$TRAIN_DATA/novita_merged_train.jsonl"
    ln -sf /data/tengwan/datasets/nemotron-v2-chinese/conversations.jsonl "$TRAIN_DATA/nemotron_chinese.jsonl"
    echo "[setup] Linked:"
    echo "  novita_merged_train.jsonl -> /data/datasets/novita_merged/train.jsonl"
    echo "  nemotron_chinese.jsonl    -> /data/tengwan/datasets/nemotron-v2-chinese/conversations.jsonl"
else
    echo "[setup] Using existing merged dataset at $TRAIN_DATA"
fi

mkdir -p "$GEN_DIR" "$GEN_DIR_B"

cd /workspace/speculators

###############################################################################
# Symlink difficulty scores so process B reads from the same file
###############################################################################
ln -sf "$DIFFICULTY_SCORES" "$GEN_DIR_B/difficulty_scores.json"

###############################################################################
# Process A: GPU 0-3, writes to gen/
###############################################################################
echo "[datagen-A] Starting on GPU 0-3..."
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/data_generation_offline.py \
    --target-model-path "$VERIFIER_NAME_OR_PATH" \
    --train-data-path "$TRAIN_DATA" \
    --output-dir "$GEN_DIR" \
    --seq-length "$SEQ_LENGTH" \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.85 \
    --turn-dropout \
    --batch-size 4 \
    --max-output-size-gb "${MAX_OUTPUT_SIZE_GB:-1024}" \
    --continuous \
    --difficulty-scores-path "$DIFFICULTY_SCORES" \
    > "$OUTPUT_PATH/datagen_a.log" 2>&1 &
PID_A=$!
echo "[datagen-A] PID=$PID_A"

###############################################################################
# Process B: GPU 4-7, writes to gen_b/
###############################################################################
echo "[datagen-B] Starting on GPU 4-7..."
CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/data_generation_offline.py \
    --target-model-path "$VERIFIER_NAME_OR_PATH" \
    --train-data-path "$TRAIN_DATA" \
    --output-dir "$GEN_DIR_B" \
    --seq-length "$SEQ_LENGTH" \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.85 \
    --turn-dropout \
    --batch-size 4 \
    --max-output-size-gb "${MAX_OUTPUT_SIZE_GB:-1024}" \
    --continuous \
    --difficulty-scores-path "$GEN_DIR_B/difficulty_scores.json" \
    > "$OUTPUT_PATH/datagen_b.log" 2>&1 &
PID_B=$!
echo "[datagen-B] PID=$PID_B"

###############################################################################
# Merger: Move gen_b/*.pt -> gen/ with index offset
###############################################################################
echo "[merger] Starting merger loop (offset=$INDEX_OFFSET, interval=10s)..."
(
    moved_total=0
    while true; do
        moved=0
        for f in "$GEN_DIR_B"/data_*.pt; do
            [ -f "$f" ] || continue
            basename=$(basename "$f")
            # Extract index: data_12345.pt -> 12345
            idx_str="${basename#data_}"
            idx_str="${idx_str%.pt}"
            idx=$((10#$idx_str))
            new_idx=$((idx + INDEX_OFFSET))
            new_name="data_${new_idx}.pt"
            if mv "$f" "$GEN_DIR/$new_name" 2>/dev/null; then
                moved=$((moved + 1))
            fi
        done
        if [ "$moved" -gt 0 ]; then
            moved_total=$((moved_total + moved))
            echo "[merger] Moved $moved files (total: $moved_total)"
        fi
        sleep 10
    done
) > "$OUTPUT_PATH/merger.log" 2>&1 &
MERGER_PID=$!
echo "[merger] PID=$MERGER_PID"

###############################################################################
# Monitor: log combined stats every 5 minutes
###############################################################################
(
    while true; do
        sleep 300
        count_a=$(find "$GEN_DIR" -name 'data_*.pt' -maxdepth 1 2>/dev/null | wc -l)
        count_b=$(find "$GEN_DIR_B" -name 'data_*.pt' -maxdepth 1 2>/dev/null | wc -l)
        echo "[$(date)] gen/ files: $count_a, gen_b/ pending: $count_b"
    done
) > "$OUTPUT_PATH/monitor.log" 2>&1 &

###############################################################################
# Wait for datagen processes (should never exit in continuous mode)
###############################################################################
echo "[main] Datagen PIDs: A=$PID_A, B=$PID_B, Merger=$MERGER_PID"
wait $PID_A $PID_B

echo "[WARN] Datagen process(es) exited unexpectedly"
kill $MERGER_PID 2>/dev/null || true
