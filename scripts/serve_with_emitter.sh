#!/bin/bash
set -euo pipefail

###############################################################################
# Launch vLLM server with hidden states emission for online training.
#
# Environment variables:
#   MODEL_PATH      - Model name or path (required as $1 or env var)
#   LAYER_IDS       - Space-separated layer indices (default: auto from model)
#   OUTPUT_DIR      - Directory for .pt output files (default: ./serving_data)
#   MAX_BUFFER_GB   - Max buffer size in GB (default: 100)
#   MIN_SEQ_LEN     - Min sequence length to capture (default: 64)
#   SAMPLE_RATE     - Fraction of requests to capture (default: 1.0)
#   TP_SIZE         - Tensor parallel size (default: 1)
#   PORT            - Server port (default: 8000)
#
# Usage:
#   ./scripts/serve_with_emitter.sh /data/models/MiniMax-M2.5
#   MODEL_PATH=/data/models/MiniMax-M2.5 TP_SIZE=4 ./scripts/serve_with_emitter.sh
###############################################################################

MODEL_PATH="${1:-${MODEL_PATH:-}}"
if [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 <model_path> [extra_args...]"
    echo "  or set MODEL_PATH environment variable"
    exit 1
fi

LAYER_IDS="${LAYER_IDS:-2 15 29 31}"
OUTPUT_DIR="${OUTPUT_DIR:-./serving_data}"
MAX_BUFFER_GB="${MAX_BUFFER_GB:-100}"
MIN_SEQ_LEN="${MIN_SEQ_LEN:-64}"
SAMPLE_RATE="${SAMPLE_RATE:-1.0}"
TP_SIZE="${TP_SIZE:-1}"
PORT="${PORT:-8000}"

echo "============================================="
echo " vLLM Server with Hidden States Emission"
echo " Model:       $MODEL_PATH"
echo " Layers:      $LAYER_IDS"
echo " Output:      $OUTPUT_DIR"
echo " Buffer:      ${MAX_BUFFER_GB}GB"
echo " Sample Rate: $SAMPLE_RATE"
echo " TP Size:     $TP_SIZE"
echo " Port:        $PORT"
echo "============================================="

python scripts/serve_with_emitter.py \
    --model "$MODEL_PATH" \
    --layer-ids $LAYER_IDS \
    --output-dir "$OUTPUT_DIR" \
    --max-buffer-gb "$MAX_BUFFER_GB" \
    --min-seq-len "$MIN_SEQ_LEN" \
    --sample-rate "$SAMPLE_RATE" \
    --tensor-parallel-size "$TP_SIZE" \
    --port "$PORT" \
    "${@:2}"
