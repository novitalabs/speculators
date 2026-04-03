"""Serving-mode worker extension for capturing hidden states during vLLM inference.

Extends HiddenStatesWorkerExtension with a "self-draining" mode: instead of
accumulating states for batch retrieval (offline), it processes completed
prefills inline and submits them to a background emitter for async disk write.

Usage:
    Specify in vLLM config as:
        worker_extension_cls="speculators.serving.worker_extension.ServingHiddenStatesExtension"

    Then call via collective_rpc:
        engine.collective_rpc("_setup_serving_capture", args=(layer_ids, output_dir, ...))
"""

import logging
import random
from dataclasses import dataclass, field

import torch
from vllm.distributed import get_tp_group

from speculators.data_generation.custom_worker import HiddenStatesWorkerExtension
from speculators.serving.emitter import EmitterConfig, PendingSample, ServingEmitter

logger = logging.getLogger(__name__)


@dataclass
class _PendingRequest:
    """Tracks a request whose prefill is still in progress."""

    prompt_token_ids: list[int]
    num_prompt_tokens: int
    num_prefill_captured: int = 0
    # hidden_chunks[layer_idx] = list of CPU tensors from each scheduler step
    hidden_chunks: list[list[torch.Tensor]] = field(default_factory=list)
    copy_events: list[torch.cuda.Event] = field(default_factory=list)


class ServingHiddenStatesExtension(HiddenStatesWorkerExtension):
    """Worker extension for capturing hidden states during production vLLM serving.

    Overrides _store_captured_states to process each scheduler step inline:
    1. Identify prefill tokens per request from input_batch metadata
    2. Slice corresponding hidden states and async-copy to CPU
    3. When a request's prefill completes, submit to ServingEmitter

    Only runs capture logic on TP rank 0 (inherited from base class).
    """

    def _setup_serving_capture(
        self,
        layer_ids: list[int],
        output_dir: str,
        tokenizer_name_or_path: str,
        max_buffer_gb: float = 100.0,
        min_seq_len: int = 64,
        sample_rate: float = 1.0,
    ) -> None:
        """Initialize serving capture mode. Called via collective_rpc."""
        self._is_capture_rank = get_tp_group().rank_in_group == 0
        self._serving_mode = True
        self._sample_rate = sample_rate
        self._pending_requests: dict[str, _PendingRequest] = {}
        # Set of request IDs we decided to skip (not sampled)
        self._skipped_requests: set[str] = set()

        # Call parent to patch the model forward
        self._setup_hidden_states_capture(layer_ids)

        # Async copy stream (only on capture rank)
        self._copy_stream = (
            torch.cuda.Stream() if self._is_capture_rank else None
        )

        # Only rank 0 creates the emitter
        if self._is_capture_rank:
            config = EmitterConfig(
                output_dir=output_dir,
                tokenizer_name_or_path=tokenizer_name_or_path,
                max_buffer_gb=max_buffer_gb,
                min_seq_len=min_seq_len,
            )
            self._emitter = ServingEmitter(config)
            logger.info(
                "Serving capture initialized: layers=%s, sample_rate=%.2f",
                list(layer_ids),
                sample_rate,
            )
        else:
            self._emitter = None

    def _store_captured_states(self, aux_hidden_states: list[torch.Tensor]):
        """Override: process captured states inline for serving mode.

        In serving mode, we:
        1. Identify which tokens are prefill for each request
        2. Async-copy prefill hidden states to CPU
        3. Submit completed samples to emitter

        Falls back to parent implementation if not in serving mode.
        """
        if not getattr(self, "_serving_mode", False):
            return super()._store_captured_states(aux_hidden_states)

        if not self._is_capture_rank:
            return

        input_batch = self.model_runner.input_batch
        num_reqs = input_batch.num_reqs

        # Get current batch metadata
        req_ids = input_batch.req_ids[:num_reqs]

        # Register new requests (probabilistic sampling)
        for i, req_id in enumerate(req_ids):
            if req_id in self._pending_requests or req_id in self._skipped_requests:
                continue

            # Probabilistic sampling decision
            if random.random() > self._sample_rate:
                self._skipped_requests.add(req_id)
                continue

            num_prompt = int(input_batch.num_prompt_tokens[i])
            prompt_ids = input_batch.token_ids_cpu[i, :num_prompt].tolist()

            self._pending_requests[req_id] = _PendingRequest(
                prompt_token_ids=prompt_ids,
                num_prompt_tokens=num_prompt,
                hidden_chunks=[[] for _ in range(len(aux_hidden_states))],
            )

        # Process each request's prefill tokens in this step
        offset = 0
        for i, req_id in enumerate(req_ids):
            num_computed_before = int(input_batch.num_computed_tokens_cpu[i])
            num_total_now = int(input_batch.num_tokens_no_spec[i])
            num_scheduled = num_total_now - num_computed_before

            if req_id not in self._pending_requests:
                offset += num_scheduled
                continue

            pending = self._pending_requests[req_id]

            # How many of these scheduled tokens are prefill?
            num_prefill_in_step = max(
                0,
                min(num_scheduled, pending.num_prompt_tokens - num_computed_before),
            )

            if num_prefill_in_step > 0:
                # Async copy prefill slice from each layer to CPU
                with torch.cuda.stream(self._copy_stream):
                    for layer_idx, layer_tensor in enumerate(aux_hidden_states):
                        prefill_slice = layer_tensor[
                            offset : offset + num_prefill_in_step
                        ]
                        cpu_tensor = prefill_slice.to("cpu", non_blocking=True)
                        pending.hidden_chunks[layer_idx].append(cpu_tensor)

                    event = torch.cuda.Event()
                    event.record(self._copy_stream)
                    pending.copy_events.append(event)

                pending.num_prefill_captured += num_prefill_in_step

            offset += num_scheduled

        # Emit completed prefills and clean up stale requests
        self._drain_completed(req_ids)

    def _drain_completed(self, active_req_ids: list[str]):
        """Submit completed prefills to emitter and clean up stale requests."""
        active_set = set(active_req_ids)
        to_remove = []

        for req_id, pending in self._pending_requests.items():
            prefill_done = pending.num_prefill_captured >= pending.num_prompt_tokens
            still_active = req_id in active_set

            if prefill_done:
                # Synchronize all async copies
                for event in pending.copy_events:
                    event.synchronize()

                # Concatenate chunks per layer
                hidden_states = []
                for layer_chunks in pending.hidden_chunks:
                    if len(layer_chunks) == 1:
                        hidden_states.append(layer_chunks[0])
                    else:
                        hidden_states.append(torch.cat(layer_chunks, dim=0))

                sample = PendingSample(
                    request_id=req_id,
                    input_ids=torch.tensor(
                        pending.prompt_token_ids, dtype=torch.long
                    ),
                    hidden_states=hidden_states,
                )

                if not self._emitter.submit(sample):
                    logger.debug("Sample dropped (buffer full): %s", req_id)

                to_remove.append(req_id)

            elif not still_active:
                # Request left the batch before prefill completed (aborted/preempted)
                logger.debug(
                    "Stale request cleaned up: %s (captured %d/%d)",
                    req_id,
                    pending.num_prefill_captured,
                    pending.num_prompt_tokens,
                )
                to_remove.append(req_id)

        for req_id in to_remove:
            del self._pending_requests[req_id]

        # Clean up skipped request IDs that are no longer active
        self._skipped_requests &= set(active_req_ids)

    def _get_emitter_stats(self) -> dict:
        """Return emitter statistics for monitoring. Called via collective_rpc."""
        if self._emitter is None:
            return {}
        return self._emitter.stats

    def _shutdown_serving_capture(self):
        """Graceful shutdown. Called via collective_rpc."""
        if self._emitter is not None:
            self._emitter.shutdown()
            logger.info("Serving emitter shut down")
