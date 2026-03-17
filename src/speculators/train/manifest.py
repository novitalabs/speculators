"""Manifest protocol for online Eagle3 training.

Provides atomic read/write of manifest.json that coordinates
datagen and training processes via the filesystem.
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Any


def read(manifest_path: str) -> dict[str, Any]:
    """Read manifest.json atomically. Returns empty manifest if file doesn't exist."""
    if not os.path.exists(manifest_path):
        return {"status": "generating", "files": [], "updated_at": None}
    with open(manifest_path) as f:
        return json.load(f)


def write(
    manifest_path: str,
    files: list[dict[str, Any]],
    status: str = "generating",
    **extra: Any,
) -> None:
    """Write manifest.json atomically (write to tmp, then os.rename).

    Extra keyword arguments are written as top-level fields in the manifest
    (e.g. total_remote_files, files_ever_seen_count, global_epoch).
    """
    manifest = {
        "status": status,
        "files": files,
        "updated_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    manifest.update(extra)
    dir_name = os.path.dirname(os.path.abspath(manifest_path))
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(manifest, f, indent=2)
        os.rename(tmp_path, manifest_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def add_file(
    manifest_path: str,
    idx: int,
    path: str,
    length: int,
    size_bytes: int | None = None,
) -> None:
    """Append a file entry to the manifest and write atomically."""
    manifest = read(manifest_path)
    entry: dict[str, Any] = {
        "idx": idx,
        "path": path,
        "length": length,
        "train_count": 0,
    }
    if size_bytes is not None:
        entry["size_bytes"] = size_bytes
    manifest["files"].append(entry)
    write(manifest_path, manifest["files"], manifest["status"])


def mark_complete(manifest_path: str) -> None:
    """Mark manifest status as complete."""
    manifest = read(manifest_path)
    write(manifest_path, manifest["files"], "complete")


def increment_train_count(manifest_path: str, trained_paths: set[str]) -> None:
    """Increment train_count for files that were trained on."""
    manifest = read(manifest_path)
    for f in manifest["files"]:
        if f["path"] in trained_paths:
            f["train_count"] = f.get("train_count", 0) + 1
    # Preserve extra top-level fields
    extra = {k: v for k, v in manifest.items() if k not in ("status", "files", "updated_at")}
    write(manifest_path, manifest["files"], manifest["status"], **extra)


# ---------------------------------------------------------------------------
# Eviction ledger — tracks cumulative train_count for evicted files
# ---------------------------------------------------------------------------

def read_ledger(data_dir: str) -> dict[str, int]:
    """Read .eviction_ledger — {filename: cumulative_train_count}."""
    path = os.path.join(data_dir, ".eviction_ledger")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def write_ledger(data_dir: str, ledger: dict[str, int]) -> None:
    """Write .eviction_ledger atomically (tmpfile + rename)."""
    path = os.path.join(data_dir, ".eviction_ledger")
    fd, tmp_path = tempfile.mkstemp(dir=data_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(ledger, f, indent=2)
        os.rename(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def update_ledger(data_dir: str, evicted: dict[str, int]) -> None:
    """Merge evicted files into ledger. evicted = {filename: cumulative_train_count}."""
    ledger = read_ledger(data_dir)
    ledger.update(evicted)
    write_ledger(data_dir, ledger)
