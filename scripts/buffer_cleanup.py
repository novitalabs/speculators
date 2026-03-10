#!/usr/bin/env python3
"""Ring buffer cleanup for online Eagle3 training.

Deletes .pt files that have been trained on at least `--min-train-count` times,
keeping disk usage bounded.

Usage:
    python scripts/buffer_cleanup.py \
        --manifest-path /data/output/gen/manifest.json \
        --data-dir /data/output/gen \
        --min-train-count 2

    # Continuous mode (run alongside training):
    python scripts/buffer_cleanup.py \
        --manifest-path /data/output/gen/manifest.json \
        --data-dir /data/output/gen \
        --min-train-count 2 \
        --poll-interval 60
"""

import argparse
import logging
import os
import time

from speculators.train import manifest as manifest_mod

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [cleanup] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


def cleanup_once(manifest_path: str, data_dir: str, min_train_count: int) -> int:
    """Delete files with train_count >= min_train_count. Returns count deleted."""
    manifest = manifest_mod.read(manifest_path)
    to_keep = []
    deleted = 0

    for f in manifest["files"]:
        tc = f.get("train_count", 0)
        if tc >= min_train_count:
            fpath = os.path.join(data_dir, f["path"])
            if os.path.exists(fpath):
                os.remove(fpath)
                deleted += 1
                log.info(f"Deleted {f['path']} (train_count={tc})")
        else:
            to_keep.append(f)

    if deleted > 0:
        manifest_mod.write(manifest_path, to_keep, manifest["status"])
        log.info(f"Cleaned up {deleted} files, {len(to_keep)} remaining")

    return deleted


def main():
    parser = argparse.ArgumentParser(description="Ring buffer cleanup")
    parser.add_argument("--manifest-path", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--min-train-count", type=int, default=2)
    parser.add_argument(
        "--poll-interval", type=float, default=0,
        help="If >0, run continuously with this interval (seconds). 0 = run once.",
    )
    args = parser.parse_args()

    if args.poll_interval > 0:
        log.info(
            f"Continuous cleanup: min_train_count={args.min_train_count}, "
            f"poll every {args.poll_interval}s"
        )
        while True:
            cleanup_once(args.manifest_path, args.data_dir, args.min_train_count)
            time.sleep(args.poll_interval)
    else:
        deleted = cleanup_once(args.manifest_path, args.data_dir, args.min_train_count)
        log.info(f"Done. Deleted {deleted} files.")


if __name__ == "__main__":
    main()
