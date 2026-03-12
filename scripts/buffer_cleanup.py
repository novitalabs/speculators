#!/usr/bin/env python3
"""Ring buffer cleanup for online Eagle3 training.

Deletes .pt files that have been trained on at least `--min-train-count` times,
keeping disk usage bounded.  When `--max-size-gb` is set, oldest trained files
are also deleted to keep total gen directory size under the limit.

Usage:
    python scripts/buffer_cleanup.py \
        --manifest-path /data/output/gen/manifest.json \
        --data-dir /data/output/gen \
        --min-train-count 2 \
        --max-size-gb 1024

    # Continuous mode (run alongside training):
    python scripts/buffer_cleanup.py \
        --manifest-path /data/output/gen/manifest.json \
        --data-dir /data/output/gen \
        --min-train-count 2 \
        --max-size-gb 1024 \
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


def _dir_size_bytes(data_dir: str) -> int:
    """Return total size of .pt files in data_dir."""
    total = 0
    for fname in os.listdir(data_dir):
        if fname.endswith(".pt"):
            total += os.path.getsize(os.path.join(data_dir, fname))
    return total


def cleanup_once(
    manifest_path: str,
    data_dir: str,
    min_train_count: int,
    max_size_bytes: int | None = None,
) -> int:
    """Delete files with train_count >= min_train_count, then enforce size cap.
    Returns count deleted."""
    manifest = manifest_mod.read(manifest_path)
    to_keep = []
    deleted = 0

    # Phase 1: delete files trained enough times
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

    # Phase 2: enforce size cap by deleting oldest trained files first
    if max_size_bytes is not None:
        current_size = _dir_size_bytes(data_dir)
        if current_size > max_size_bytes:
            # Sort by train_count desc, then by idx asc (oldest first)
            trained = [f for f in to_keep if f.get("train_count", 0) > 0]
            trained.sort(key=lambda f: (-f.get("train_count", 0), f["idx"]))
            trained_paths = set()
            for f in trained:
                if current_size <= max_size_bytes:
                    break
                fpath = os.path.join(data_dir, f["path"])
                if os.path.exists(fpath):
                    fsize = os.path.getsize(fpath)
                    os.remove(fpath)
                    current_size -= fsize
                    deleted += 1
                    trained_paths.add(f["path"])
                    log.info(
                        f"Size cap: deleted {f['path']} "
                        f"(train_count={f.get('train_count', 0)}, "
                        f"remaining={current_size / 1e9:.1f}GB)"
                    )
            to_keep = [f for f in to_keep if f["path"] not in trained_paths]

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
        "--max-size-gb", type=float, default=0,
        help="Max total size of .pt files in GB. 0 = no limit.",
    )
    parser.add_argument(
        "--poll-interval", type=float, default=0,
        help="If >0, run continuously with this interval (seconds). 0 = run once.",
    )
    args = parser.parse_args()

    max_size_bytes = int(args.max_size_gb * 1e9) if args.max_size_gb > 0 else None

    if args.poll_interval > 0:
        log.info(
            f"Continuous cleanup: min_train_count={args.min_train_count}, "
            f"max_size={args.max_size_gb}GB, poll every {args.poll_interval}s"
        )
        while True:
            cleanup_once(
                args.manifest_path, args.data_dir,
                args.min_train_count, max_size_bytes,
            )
            time.sleep(args.poll_interval)
    else:
        deleted = cleanup_once(
            args.manifest_path, args.data_dir,
            args.min_train_count, max_size_bytes,
        )
        log.info(f"Done. Deleted {deleted} files.")


if __name__ == "__main__":
    main()
