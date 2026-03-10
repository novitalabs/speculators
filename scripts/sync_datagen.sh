#!/usr/bin/env bash
# Sync .pt files from datagen node(s) to training node and update manifest.
#
# Usage:
#   ./scripts/sync_datagen.sh \
#       --datagen-nodes "host-10-83-115-21" \
#       --remote-dir /data/output/gen \
#       --local-dir /data/output/gen \
#       --manifest-path /data/output/gen/manifest.json \
#       --poll-interval 30

set -euo pipefail

DATAGEN_NODES=""
REMOTE_DIR=""
LOCAL_DIR=""
MANIFEST_PATH=""
POLL_INTERVAL=30

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datagen-nodes) DATAGEN_NODES="$2"; shift 2 ;;
        --remote-dir)    REMOTE_DIR="$2"; shift 2 ;;
        --local-dir)     LOCAL_DIR="$2"; shift 2 ;;
        --manifest-path) MANIFEST_PATH="$2"; shift 2 ;;
        --poll-interval) POLL_INTERVAL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$DATAGEN_NODES" || -z "$REMOTE_DIR" || -z "$LOCAL_DIR" || -z "$MANIFEST_PATH" ]]; then
    echo "Error: --datagen-nodes, --remote-dir, --local-dir, --manifest-path are required"
    exit 1
fi

mkdir -p "$LOCAL_DIR"

IFS=',' read -ra NODES <<< "$DATAGEN_NODES"

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

# Merge with existing manifest to preserve train_count and length
if os.path.exists(manifest_path):
    with open(manifest_path) as f:
        old = json.load(f)
    old_by_idx = {e['idx']: e for e in old.get('files', [])}
    for f_entry in files:
        old_entry = old_by_idx.get(f_entry['idx'])
        if old_entry:
            f_entry['train_count'] = old_entry.get('train_count', 0)
            f_entry['length'] = old_entry.get('length', 0)

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

echo "Starting sync loop: ${NODES[*]} -> $LOCAL_DIR (every ${POLL_INTERVAL}s)"

while true; do
    all_complete=true

    for node in "${NODES[@]}"; do
        echo "[$(date)] Syncing from $node:$REMOTE_DIR ..."
        rsync -az --include='*.pt' --include='sample_lengths.json' --exclude='*' \
            "$node:$REMOTE_DIR/" "$LOCAL_DIR/" || echo "rsync from $node failed, retrying next cycle"

        if ! check_remote_complete "$node" 2>/dev/null; then
            all_complete=false
        fi
    done

    update_manifest

    if $all_complete; then
        echo "All datagen nodes complete. Marking manifest as complete."
        python3 -c "
import json, os, tempfile
from datetime import datetime, timezone
manifest_path = '$MANIFEST_PATH'
with open(manifest_path) as f:
    m = json.load(f)
m['status'] = 'complete'
m['updated_at'] = datetime.now(tz=timezone.utc).isoformat()
dir_name = os.path.dirname(os.path.abspath(manifest_path))
fd, tmp = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
with os.fdopen(fd, 'w') as fp:
    json.dump(m, fp, indent=2)
os.rename(tmp, manifest_path)
"
        echo "Sync complete. Exiting."
        exit 0
    fi

    sleep "$POLL_INTERVAL"
done
