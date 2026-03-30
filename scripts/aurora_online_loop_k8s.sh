#!/bin/bash
set -euo pipefail

###############################################################################
# Aurora Online Loop — K8s Orchestrator
#
# Runs on HOST (not in k8s). Launches datagen/train pods and coordinates
# via kubectl + rsync. Run from .17 host.
#
# Usage:
#   bash scripts/aurora_online_loop_k8s.sh [--start-round N] [--max-rounds N]
###############################################################################

# Config
OUTPUT_DIR="/data/output/minimax_m2.5_eagle3_aurora_online"
DATAGEN_NODE="10.83.115.18"
TRAIN_NODE="10.83.115.17"  # localhost
FILES_PER_ROUND=2000
EPOCHS_PER_ROUND=5
MAX_ROUNDS=20
START_ROUND=0
SEED_CKPT="/data/output/minimax_m2.5_eagle3_novita0320/checkpoints/67"
DATAGEN_YAML="k8s/k8s-aurora-online-datagen.yaml"
TRAIN_YAML="k8s/k8s-aurora-online-train.yaml"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --start-round) START_ROUND="$2"; shift 2 ;;
        --max-rounds) MAX_ROUNDS="$2"; shift 2 ;;
        --files-per-round) FILES_PER_ROUND="$2"; shift 2 ;;
        --epochs-per-round) EPOCHS_PER_ROUND="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

LOG_FILE="$OUTPUT_DIR/orchestrator.log"
mkdir -p "$OUTPUT_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Helper: patch env var in yaml and apply
apply_pod() {
    local yaml="$1"
    local pod_name="$2"
    local round="$3"

    # Delete old pod if exists
    kubectl delete pod "$pod_name" --ignore-not-found=true 2>/dev/null

    # Patch AURORA_ROUND in yaml and apply
    sed "s/AURORA_ROUND\"$/AURORA_ROUND\"/" "$yaml" | \
    sed "s/value: \"[0-9]*\"  # ROUND/value: \"$round\"  # ROUND/" | \
    kubectl apply -f - 2>/dev/null || true

    # If sed didn't match the comment pattern, use kubectl directly with env override
    # Simpler: just edit the yaml in-place temporarily
    local tmp_yaml="/tmp/aurora_pod_$pod_name.yaml"
    sed "s/name: AURORA_ROUND/name: AURORA_ROUND/;
         /name: AURORA_ROUND/{n;s/value: \"[0-9]*\"/value: \"$round\"/}" \
         "$yaml" > "$tmp_yaml"
    sed -i "/name: AURORA_FILES_PER_ROUND/{n;s/value: \"[0-9]*\"/value: \"$FILES_PER_ROUND\"/}" "$tmp_yaml"
    sed -i "/name: AURORA_EPOCHS_PER_ROUND/{n;s/value: \"[0-9]*\"/value: \"$EPOCHS_PER_ROUND\"/}" "$tmp_yaml"

    kubectl delete pod "$pod_name" --ignore-not-found=true 2>/dev/null
    sleep 2
    kubectl apply -f "$tmp_yaml"
}

# Helper: wait for pod to complete
wait_pod() {
    local pod_name="$1"
    local timeout="${2:-7200}"  # default 2h

    log "Waiting for pod $pod_name (timeout=${timeout}s)..."
    if ! kubectl wait --for=condition=Ready "pod/$pod_name" --timeout=120s 2>/dev/null; then
        # Pod might have already completed or errored before Ready
        local status
        status=$(kubectl get pod "$pod_name" -o jsonpath='{.status.phase}' 2>/dev/null)
        if [[ "$status" == "Succeeded" ]]; then
            log "Pod $pod_name already completed."
            return 0
        elif [[ "$status" == "Failed" ]]; then
            log "ERROR: Pod $pod_name failed!"
            kubectl logs "$pod_name" --tail=30 2>/dev/null | tee -a "$LOG_FILE"
            return 1
        fi
    fi

    # Wait for completion
    local elapsed=0
    while true; do
        local status
        status=$(kubectl get pod "$pod_name" -o jsonpath='{.status.phase}' 2>/dev/null)
        case "$status" in
            Succeeded)
                log "Pod $pod_name completed successfully."
                return 0
                ;;
            Failed)
                log "ERROR: Pod $pod_name failed!"
                kubectl logs "$pod_name" --tail=50 2>/dev/null | tee -a "$LOG_FILE"
                return 1
                ;;
            *)
                if (( elapsed >= timeout )); then
                    log "ERROR: Pod $pod_name timed out after ${timeout}s"
                    return 1
                fi
                sleep 30
                elapsed=$((elapsed + 30))
                # Print progress every 5 min
                if (( elapsed % 300 == 0 )); then
                    log "  Still running ($((elapsed/60))m)... $(kubectl logs "$pod_name" --tail=1 2>/dev/null)"
                fi
                ;;
        esac
    done
}

###############################################################################
# Pre-flight: ensure vocab mapping on both nodes
###############################################################################
VOCAB_DIR="$OUTPUT_DIR/vocab_mapping"
if [[ ! -f "$VOCAB_DIR/d2t.npy" ]]; then
    log "Copying vocab mapping from existing experiment..."
    mkdir -p "$VOCAB_DIR"
    cp /data/output/minimax_m2.5_eagle3_aurora_loss/vocab_mapping/d2t.npy "$VOCAB_DIR/"
    cp /data/output/minimax_m2.5_eagle3_aurora_loss/vocab_mapping/t2d.npy "$VOCAB_DIR/"
fi

log "Syncing vocab mapping to $DATAGEN_NODE..."
ssh "$DATAGEN_NODE" "mkdir -p $VOCAB_DIR"
rsync -az "$VOCAB_DIR/" "$DATAGEN_NODE:$VOCAB_DIR/"

log "============================================="
log " Aurora Online Loop — Starting"
log " Rounds: $START_ROUND → $((MAX_ROUNDS - 1))"
log " Files/Round: $FILES_PER_ROUND"
log " Epochs/Round: $EPOCHS_PER_ROUND"
log "============================================="

###############################################################################
# Main loop
###############################################################################
for (( R=START_ROUND; R<MAX_ROUNDS; R++ )); do
    ROUND_DIR="$OUTPUT_DIR/round_${R}"
    REMOTE_ROUND="$DATAGEN_NODE:$ROUND_DIR"

    log ""
    log ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    log "  Round $R / $((MAX_ROUNDS - 1))"
    log "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"

    #---------------------------------------------------------------------------
    # Stage A: Datagen + Mask on .18
    #---------------------------------------------------------------------------
    log "[round $R] Stage A: Launching datagen pod on .18"
    apply_pod "$DATAGEN_YAML" "aurora-online-datagen" "$R"
    wait_pod "aurora-online-datagen" 7200 || { log "FATAL: datagen failed at round $R"; exit 1; }

    # Capture datagen summary
    kubectl logs aurora-online-datagen --tail=5 2>/dev/null | tee -a "$LOG_FILE"

    #---------------------------------------------------------------------------
    # Stage B: Rsync data + masks from .18 to .17
    #---------------------------------------------------------------------------
    log "[round $R] Stage B: Syncing data from .18 to .17"
    mkdir -p "$ROUND_DIR/gen" "$ROUND_DIR/masks"
    rsync -az "$DATAGEN_NODE:$ROUND_DIR/gen/" "$ROUND_DIR/gen/"
    rsync -az "$DATAGEN_NODE:$ROUND_DIR/masks/" "$ROUND_DIR/masks/"

    PT_COUNT=$(find "$ROUND_DIR/gen" -name "data_*.pt" | wc -l)
    MASK_COUNT=$(find "$ROUND_DIR/masks" -name "mask_*.pt" | wc -l)
    log "[round $R] Synced: $PT_COUNT data files, $MASK_COUNT masks"

    #---------------------------------------------------------------------------
    # Stage C: Training on .17
    #---------------------------------------------------------------------------
    log "[round $R] Stage C: Launching training pod on .17"
    apply_pod "$TRAIN_YAML" "aurora-online-train" "$R"
    wait_pod "aurora-online-train" 7200 || { log "FATAL: training failed at round $R"; exit 1; }

    # Capture training summary
    kubectl logs aurora-online-train | grep -E "loss=|epoch=|complete" | tail -5 | tee -a "$LOG_FILE"

    #---------------------------------------------------------------------------
    # Stage D: Sync latest checkpoint back to .18
    #---------------------------------------------------------------------------
    CKPT_DIR="$OUTPUT_DIR/checkpoints"
    LATEST_CKPT=$(ls -1d "$CKPT_DIR"/* 2>/dev/null | sort -t/ -k$(echo "$CKPT_DIR" | tr '/' '\n' | wc -l) -n | tail -1)
    if [[ -z "$LATEST_CKPT" ]]; then
        log "ERROR: No checkpoint found at $CKPT_DIR"
        exit 1
    fi

    # Update latest_ckpt symlink/copy on .18
    log "[round $R] Stage D: Syncing checkpoint to .18"
    ssh "$DATAGEN_NODE" "mkdir -p $OUTPUT_DIR/latest_ckpt"
    rsync -az "$LATEST_CKPT/" "$DATAGEN_NODE:$OUTPUT_DIR/latest_ckpt/"
    log "[round $R] Synced checkpoint: $LATEST_CKPT → .18:$OUTPUT_DIR/latest_ckpt/"

    #---------------------------------------------------------------------------
    # Cleanup: delete completed pods
    #---------------------------------------------------------------------------
    kubectl delete pod aurora-online-datagen aurora-online-train --ignore-not-found=true 2>/dev/null

    log "[round $R] ✓ Round complete"
    log ""
done

log "============================================="
log " Aurora Online Loop complete!"
log " Rounds: $START_ROUND → $((MAX_ROUNDS - 1))"
log " Final ckpt: $OUTPUT_DIR/checkpoints/"
log "============================================="
