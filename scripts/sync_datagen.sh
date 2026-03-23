#!/usr/bin/env bash
# Sync .pt files from datagen node(s) to training node and update manifest.
#
# Usage:
#   ./scripts/sync_datagen.sh \
#       --datagen-nodes "host-10-83-115-21" \
#       --remote-dir /data/output/gen \
#       --local-dir /data/output/gen \
#       --manifest-path /data/output/gen/manifest.json \
#       --poll-interval 30 \
#       --target-train-count 10

set -euo pipefail

DATAGEN_NODES=""
REMOTE_DIR=""
LOCAL_DIR=""
MANIFEST_PATH=""
POLL_INTERVAL=30
MAX_SYNC_SIZE_GB=0
TARGET_TRAIN_COUNT=10

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datagen-nodes)      DATAGEN_NODES="$2"; shift 2 ;;
        --remote-dir)         REMOTE_DIR="$2"; shift 2 ;;
        --local-dir)          LOCAL_DIR="$2"; shift 2 ;;
        --manifest-path)      MANIFEST_PATH="$2"; shift 2 ;;
        --poll-interval)      POLL_INTERVAL="$2"; shift 2 ;;
        --max-sync-size-gb)   MAX_SYNC_SIZE_GB="$2"; shift 2 ;;
        --target-train-count) TARGET_TRAIN_COUNT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$DATAGEN_NODES" || -z "$REMOTE_DIR" || -z "$LOCAL_DIR" || -z "$MANIFEST_PATH" ]]; then
    echo "Error: --datagen-nodes, --remote-dir, --local-dir, --manifest-path are required"
    exit 1
fi

mkdir -p "$LOCAL_DIR"

IFS=',' read -ra NODES <<< "$DATAGEN_NODES"

datagen_marked_complete=false

update_manifest() {
    # Scan local dir for .pt files and build manifest
    python3 -c "
import os, json, tempfile
from datetime import datetime, timezone

local_dir = '$LOCAL_DIR'
manifest_path = '$MANIFEST_PATH'

files = []
for fname in sorted(os.listdir(local_dir)):
    if not fname.endswith('.pt'):
        continue
    idx_str = fname.replace('data_', '').replace('.pt', '')
    try:
        idx = int(idx_str)
    except ValueError:
        continue
    fpath = os.path.join(local_dir, fname)
    size_bytes = os.path.getsize(fpath)
    files.append({
        'idx': idx,
        'path': fname,
        'length': 0,
        'size_bytes': size_bytes,
        'train_count': 0,
    })

# Merge with existing manifest to preserve train_count, length, and extra fields
extra = {}
if os.path.exists(manifest_path):
    with open(manifest_path) as f:
        old = json.load(f)
    old_by_idx = {e['idx']: e for e in old.get('files', [])}
    extra = {k: v for k, v in old.items() if k not in ('status', 'files', 'updated_at')}

    # Read ledger for re-entered files
    ledger = {}
    ledger_path = os.path.join(local_dir, '.eviction_ledger')
    if os.path.exists(ledger_path):
        with open(ledger_path) as lf:
            ledger = json.load(lf)

    for f_entry in files:
        old_entry = old_by_idx.get(f_entry['idx'])
        if old_entry:
            f_entry['train_count'] = old_entry.get('train_count', 0)
            f_entry['length'] = old_entry.get('length', 0)
        elif f_entry['path'] in ledger:
            # Re-entered file: restore cumulative count from ledger
            f_entry['train_count'] = ledger[f_entry['path']]

# Check if all remote manifests are complete
all_complete = True
for node in '${DATAGEN_NODES}'.split(','):
    # Check if we can find a remote manifest marked complete
    # For simplicity, check if any remote manifest.json has status=complete
    pass

manifest = {
    'status': 'generating',
    'files': files,
    'updated_at': datetime.now(tz=timezone.utc).isoformat(),
}
manifest.update(extra)

dir_name = os.path.dirname(os.path.abspath(manifest_path))
os.makedirs(dir_name, exist_ok=True)
fd, tmp = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
with os.fdopen(fd, 'w') as fp:
    json.dump(manifest, fp, indent=2)
os.rename(tmp, manifest_path)
print(f'Updated manifest: {len(files)} files')
"
}

check_remote_complete() {
    local node="$1"
    # Check if remote manifest has status=complete
    ssh "$node" "cat ${REMOTE_DIR}/manifest*.json 2>/dev/null" | \
        python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('status')=='complete' else 1)" \
        2>/dev/null
}

generate_exclude_from_ledger() {
    # Generate a temporary exclude file from .eviction_ledger
    # Files that have reached the target train count are excluded from rsync
    python3 -c "
import json, os, sys

ledger_path = os.path.join('$LOCAL_DIR', '.eviction_ledger')
target = $TARGET_TRAIN_COUNT
exclude = os.path.join('$LOCAL_DIR', '.ledger_exclude')

if not os.path.exists(ledger_path):
    # No ledger yet — create empty exclude file
    open(exclude, 'w').close()
    sys.exit(0)

with open(ledger_path) as f:
    ledger = json.load(f)

with open(exclude, 'w') as f:
    for name, count in ledger.items():
        if count >= target:
            f.write(name + '\n')
"
}

mark_complete_with_remote_count() {
    local node="$1"
    local total_remote
    total_remote=$(ssh "$node" "find $REMOTE_DIR -name '*.pt' | wc -l" 2>/dev/null || echo "0")
    echo "[$(date)] Remote file count: $total_remote"

    # Write total_remote_files to manifest and mark complete
    python3 -c "
import json, os, tempfile
from datetime import datetime, timezone

manifest_path = '$MANIFEST_PATH'
total_remote = $total_remote

with open(manifest_path) as f:
    m = json.load(f)

m['status'] = 'complete'
m['total_remote_files'] = total_remote
m['updated_at'] = datetime.now(tz=timezone.utc).isoformat()

dir_name = os.path.dirname(os.path.abspath(manifest_path))
fd, tmp = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
with os.fdopen(fd, 'w') as fp:
    json.dump(m, fp, indent=2)
os.rename(tmp, manifest_path)
print(f'Marked manifest complete with total_remote_files={total_remote}')
"
}

echo "Starting sync loop: ${NODES[*]} -> $LOCAL_DIR (every ${POLL_INTERVAL}s, target_train_count=${TARGET_TRAIN_COUNT})"

# Bootstrap manifest from any existing local files so training can start
# while the first (potentially slow) rsync is still running.
update_manifest

while true; do
    all_complete=true

    for node in "${NODES[@]}"; do
        # Size gate: skip sync if local dir exceeds limit
        if [[ "$MAX_SYNC_SIZE_GB" -gt 0 ]]; then
            current_size_kb=$( (du -sk "$LOCAL_DIR" 2>/dev/null || true) | awk '{print $1}')
            current_size_kb=${current_size_kb:-0}
            max_size_kb=$((MAX_SYNC_SIZE_GB * 1024 * 1024))
            if [[ "$current_size_kb" -ge "$max_size_kb" ]]; then
                echo "[$(date)] Local dir at ${current_size_kb}KB >= ${max_size_kb}KB limit, skipping sync from $node"
                if ! check_remote_complete "$node" 2>/dev/null; then all_complete=false; fi
                continue
            fi
        fi

        # Generate dynamic exclude from eviction ledger (replaces .cleanup_exclude)
        generate_exclude_from_ledger
        EXCLUDE_FLAG="--exclude-from=$LOCAL_DIR/.ledger_exclude"

        echo "[$(date)] Syncing from $node:$REMOTE_DIR ..."
        rsync -az $EXCLUDE_FLAG \
            --include='*.pt' --include='sample_lengths.json' --exclude='*' \
            "$node:$REMOTE_DIR/" "$LOCAL_DIR/" || echo "rsync from $node failed, retrying next cycle"

        if ! check_remote_complete "$node" 2>/dev/null; then
            all_complete=false
        fi
    done

    update_manifest

    if $all_complete && ! $datagen_marked_complete; then
        echo "All datagen nodes complete. Marking manifest with remote file count."
        # Use first node to count remote files
        mark_complete_with_remote_count "${NODES[0]}"
        datagen_marked_complete=true
        echo "Datagen marked complete. Continuing sync for file re-entry..."
    fi

    sleep "$POLL_INTERVAL"
done
