"""Async file writer for serving-captured hidden states.

Receives completed samples (CPU tensors) from the worker extension,
computes loss masks, and writes training-compatible .pt files to disk.
"""

import logging
import os
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoTokenizer

from speculators.data_generation.preprocessing import (
    _create_loss_mask_from_offsets,
    _detect_assistant_pattern,
    _supports_assistant_mask,
)
from speculators.train import manifest as manifest_mod

logger = logging.getLogger(__name__)


@dataclass
class EmitterConfig:
    """Configuration for the serving emitter."""

    output_dir: str
    tokenizer_name_or_path: str
    max_buffer_gb: float = 100.0
    min_seq_len: int = 64
    manifest_path: str | None = None

    def __post_init__(self):
        if self.manifest_path is None:
            self.manifest_path = os.path.join(self.output_dir, "manifest.json")


@dataclass
class PendingSample:
    """A completed sample ready for disk write."""

    request_id: str
    input_ids: torch.Tensor  # [seq_len], long, CPU
    hidden_states: list[torch.Tensor]  # N_layers x [seq_len, hidden], CPU


@dataclass
class EmitterStats:
    """Runtime statistics for monitoring."""

    files_written: int = 0
    samples_dropped: int = 0
    samples_filtered: int = 0
    buffer_bytes: int = 0
    queue_size: int = 0


def _get_dir_size_bytes(path: str) -> int:
    """Get total size of files in a directory (non-recursive, fast)."""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False) and entry.name.endswith(".pt"):
                total += entry.stat().st_size
    except OSError:
        pass
    return total


def _find_last_file_idx(output_dir: str) -> int:
    """Find the next available file index by scanning existing data_*.pt files."""
    max_idx = -1
    try:
        for entry in os.scandir(output_dir):
            name = entry.name
            if name.startswith("data_") and name.endswith(".pt"):
                try:
                    idx = int(name[5:-3])  # strip "data_" and ".pt"
                    max_idx = max(max_idx, idx)
                except ValueError:
                    continue
    except OSError:
        pass
    return max_idx + 1


class ServingEmitter:
    """Queue-based async writer that saves .pt files from serving-captured hidden states.

    Runs a background thread that:
    1. Dequeues completed samples (CPU tensors)
    2. Computes loss_mask from tokenizer chat template
    3. Applies quality filters (min seq len, empty mask)
    4. Saves as .pt files compatible with train_streaming.py
    5. Updates manifest.json atomically

    The submit() method is non-blocking and will drop samples
    if the queue is full or the buffer is at capacity.
    """

    def __init__(self, config: EmitterConfig):
        self._config = config
        self._output_dir = config.output_dir
        self._manifest_path = config.manifest_path
        self._max_buffer_bytes = int(config.max_buffer_gb * 1024**3)
        self._min_seq_len = config.min_seq_len

        # Create output directory
        os.makedirs(self._output_dir, exist_ok=True)

        # Load tokenizer and detect assistant pattern
        logger.info("Loading tokenizer: %s", config.tokenizer_name_or_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name_or_path, trust_remote_code=True
        )
        self._use_hf_mask = _supports_assistant_mask(self._tokenizer)
        if not self._use_hf_mask:
            pattern_str = _detect_assistant_pattern(self._tokenizer)
            self._assistant_pattern = re.compile(pattern_str, re.DOTALL)
            logger.info("Detected assistant pattern (regex fallback)")
        else:
            self._assistant_pattern = None
            logger.info("Using HF native assistant mask")

        # Resume from existing files
        self._file_idx = _find_last_file_idx(self._output_dir)
        self._stats = EmitterStats(
            buffer_bytes=_get_dir_size_bytes(self._output_dir)
        )
        self._lock = threading.Lock()

        # Bounded queue for backpressure
        self._queue: queue.Queue[PendingSample | None] = queue.Queue(maxsize=128)

        # Start writer thread
        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="serving-emitter"
        )
        self._writer_thread.start()
        logger.info(
            "ServingEmitter started: output_dir=%s, resume_idx=%d, "
            "buffer=%.1fGB/%.1fGB, min_seq_len=%d",
            self._output_dir,
            self._file_idx,
            self._stats.buffer_bytes / 1024**3,
            config.max_buffer_gb,
            self._min_seq_len,
        )

    def submit(self, sample: PendingSample) -> bool:
        """Submit a completed sample for async writing.

        Returns True if enqueued, False if dropped (buffer full or queue full).
        """
        # Check buffer capacity
        if self._stats.buffer_bytes >= self._max_buffer_bytes:
            with self._lock:
                self._stats.samples_dropped += 1
            return False

        try:
            self._queue.put_nowait(sample)
            return True
        except queue.Full:
            with self._lock:
                self._stats.samples_dropped += 1
            return False

    def shutdown(self, timeout: float = 30.0):
        """Graceful shutdown: drain queue and stop writer thread."""
        logger.info("Shutting down emitter (queue size: %d)...", self._queue.qsize())
        self._queue.put(None)  # sentinel
        self._writer_thread.join(timeout=timeout)
        if self._writer_thread.is_alive():
            logger.warning("Emitter writer thread did not stop within %.1fs", timeout)

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "files_written": self._stats.files_written,
                "samples_dropped": self._stats.samples_dropped,
                "samples_filtered": self._stats.samples_filtered,
                "buffer_gb": round(self._stats.buffer_bytes / 1024**3, 2),
                "queue_size": self._queue.qsize(),
            }

    def _writer_loop(self):
        """Background thread: dequeue samples and write to disk."""
        while True:
            sample = self._queue.get()
            if sample is None:
                break
            try:
                self._write_sample(sample)
            except Exception:
                logger.exception("Failed to write sample %s", sample.request_id)

    def _write_sample(self, sample: PendingSample):
        """Compute loss mask, apply quality filters, save .pt file."""
        input_ids = sample.input_ids
        seq_len = len(input_ids)

        # Quality filter: minimum sequence length
        if seq_len < self._min_seq_len:
            with self._lock:
                self._stats.samples_filtered += 1
            return

        # Compute loss mask
        loss_mask = self._compute_loss_mask(input_ids)

        # Quality filter: skip if no assistant content
        if loss_mask.sum() == 0:
            with self._lock:
                self._stats.samples_filtered += 1
            return

        # Wait for buffer space (blocks writer thread, not serving)
        while self._stats.buffer_bytes >= self._max_buffer_bytes:
            logger.debug(
                "Buffer full (%.1fGB), waiting...",
                self._stats.buffer_bytes / 1024**3,
            )
            time.sleep(10)
            # Recompute in case cleanup freed space
            with self._lock:
                self._stats.buffer_bytes = _get_dir_size_bytes(self._output_dir)

        # Build data dict (same format as offline datagen)
        data = {
            "input_ids": input_ids,
            "hidden_states": [h.contiguous() for h in sample.hidden_states],
            "loss_mask": loss_mask,
        }

        # Save .pt file
        with self._lock:
            file_idx = self._file_idx
            self._file_idx += 1

        filename = f"data_{file_idx}.pt"
        filepath = os.path.join(self._output_dir, filename)
        torch.save(data, filepath)

        # Get file size and update stats
        try:
            size_bytes = os.path.getsize(filepath)
        except OSError:
            size_bytes = 0

        with self._lock:
            self._stats.buffer_bytes += size_bytes
            self._stats.files_written += 1

        # Update manifest
        manifest_mod.add_file(
            self._manifest_path,
            idx=file_idx,
            path=filename,
            length=seq_len,
            size_bytes=size_bytes,
        )

    def _compute_loss_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute loss mask from input_ids using tokenizer chat template detection.

        Falls back to all-ones mask if detection fails.
        """
        try:
            text = self._tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)

            if self._use_hf_mask:
                # Use HF native assistant mask via re-tokenization
                # This path requires the original conversation, which we don't have.
                # Fall back to regex approach.
                pass

            if self._assistant_pattern is not None:
                # Re-tokenize with offset mapping
                encoding = self._tokenizer(
                    text,
                    return_offsets_mapping=True,
                    add_special_tokens=False,
                    return_tensors=None,
                )
                offsets = encoding["offset_mapping"]

                # Validate length matches (tokenizer round-trip may differ)
                if len(offsets) != len(input_ids):
                    logger.debug(
                        "Token length mismatch after round-trip: %d vs %d, "
                        "using all-ones mask",
                        len(offsets),
                        len(input_ids),
                    )
                    return torch.ones(len(input_ids), dtype=torch.long)

                return _create_loss_mask_from_offsets(
                    text, offsets, self._assistant_pattern
                )

            # No pattern available: train on everything
            return torch.ones(len(input_ids), dtype=torch.long)

        except Exception:
            logger.debug("Loss mask computation failed, using all-ones mask", exc_info=True)
            return torch.ones(len(input_ids), dtype=torch.long)
