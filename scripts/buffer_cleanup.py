#!/usr/bin/env python3
"""Ring buffer cleanup for online Eagle3 training.

Size-cap-only eviction: when `--max-size-gb` is exceeded, evict the most-trained
files first and record their cumulative train_count in `.eviction_ledger` so
sync can re-admit them later based on a target train count.

Usage:
    python scripts/buffer_cleanup.py \
        --manifest-path /data/output/gen/manifest.json \
        --data-dir /data/output/gen \
        --max-size-gb 1024

    # Continuous mode (run alongside training):
    python scripts/buffer_cleanup.py \
        --manifest-path /data/output/gen/manifest.json \
        --data-dir /data/output/gen \
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
    max_size_bytes: int | None = None,
    max_delete: int = 0,
    min_retain: int = 0,
) -> int:
    """Enforce size cap by evicting most-trained files first.
    Records evicted files in .eviction_ledger with cumulative train_count.
    Returns count deleted.

    Args:
        max_delete: Max files to delete per cycle (0 = unlimited).
        min_retain: Always keep at least this many files in manifest (0 = no minimum).
    """
    # Check epoch lock — skip cleanup while training is actively using files
    lock_file = os.path.join(data_dir, ".epoch_in_progress")
    if os.path.exists(lock_file):
        log.info("Epoch in progress, skipping cleanup")
        return 0

    # No size cap → nothing to do
    if max_size_bytes is None:
        return 0

    current_size = _dir_size_bytes(data_dir)
    if current_size <= max_size_bytes:
        return 0

    manifest = manifest_mod.read(manifest_path)
    all_files = list(manifest["files"])

    # Sort ALL files by train_count desc, idx asc (evict most-trained first)
    all_files.sort(key=lambda f: (-f.get("train_count", 0), f["idx"]))

    # Read existing ledger for cumulative counts
    ledger = manifest_mod.read_ledger(data_dir)

    deleted = 0
    evicted_counts: dict[str, int] = {}
    evicted_paths: set[str] = set()

    for f in all_files:
        if current_size <= max_size_bytes:
            break
        # Respect max_delete per cycle
        if max_delete > 0 and deleted >= max_delete:
            break
        # Respect min_retain — always keep enough files for training
        remaining = len(all_files) - len(evicted_paths)
        if min_retain > 0 and remaining <= min_retain:
            break

        fpath = os.path.join(data_dir, f["path"])
        if os.path.exists(fpath):
            fsize = os.path.getsize(fpath)
            os.remove(fpath)
            current_size -= fsize
            deleted += 1
            evicted_paths.add(f["path"])
            # Cumulative train count = ledger count (from previous evictions) + local count
            cumulative = ledger.get(f["path"], 0) + f.get("train_count", 0)
            evicted_counts[f["path"]] = cumulative
            log.info(
                f"Evicted {f['path']} "
                f"(local_tc={f.get('train_count', 0)}, "
                f"cumulative_tc={cumulative}, "
                f"remaining={current_size / 1e9:.1f}GB)"
            )

    if deleted > 0:
        # Write eviction ledger (replaces .cleanup_exclude)
        manifest_mod.update_ledger(data_dir, evicted_counts)
        log.info(f"Updated .eviction_ledger with {len(evicted_counts)} entries")

        # Rewrite manifest without deleted files, preserving extra fields
        kept = [f for f in manifest["files"] if f["path"] not in evicted_paths]
        extra = {k: v for k, v in manifest.items() if k not in ("status", "files", "updated_at")}
        manifest_mod.write(manifest_path, kept, manifest["status"], **extra)
        log.info(f"Cleaned up {deleted} files, {len(kept)} remaining")

    return deleted


def main():
    parser = argparse.ArgumentParser(description="Ring buffer cleanup")
    parser.add_argument("--manifest-path", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument(
        "--max-size-gb", type=float, default=0,
        help="Max total size of .pt files in GB. 0 = no limit.",
    )
    parser.add_argument(
        "--poll-interval", type=float, default=0,
        help="If >0, run continuously with this interval (seconds). 0 = run once.",
    )
    parser.add_argument(
        "--max-delete-per-cycle", type=int, default=5000,
        help="Max files to delete per cleanup cycle. 0 = unlimited. (default: 5000)",
    )
    parser.add_argument(
        "--min-retain-count", type=int, default=1000,
        help="Always keep at least this many files in manifest. 0 = no minimum. (default: 1000)",
    )
    args = parser.parse_args()

    max_size_bytes = int(args.max_size_gb * 1e9) if args.max_size_gb > 0 else None

    if args.poll_interval > 0:
        log.info(
            f"Continuous cleanup: "
            f"max_size={args.max_size_gb}GB, poll every {args.poll_interval}s, "
            f"max_delete={args.max_delete_per_cycle}, min_retain={args.min_retain_count}"
        )
        while True:
            cleanup_once(
                args.manifest_path, args.data_dir,
                max_size_bytes,
                max_delete=args.max_delete_per_cycle,
                min_retain=args.min_retain_count,
            )
            time.sleep(args.poll_interval)
    else:
        deleted = cleanup_once(
            args.manifest_path, args.data_dir,
            max_size_bytes,
            max_delete=args.max_delete_per_cycle,
            min_retain=args.min_retain_count,
        )
        log.info(f"Done. Deleted {deleted} files.")


if __name__ == "__main__":
    main()
