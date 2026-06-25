"""Custom worker extension for hidden states capture."""

import inspect
import logging
import types
from collections import defaultdict
from itertools import islice
from typing import Any

import torch
from vllm.distributed import get_pp_group, get_tp_group
from vllm.sequence import IntermediateTensors

__all__ = ["HiddenStatesWorkerExtension"]

logger = logging.getLogger(__name__)


def _rank_in_group(group) -> int:
    rank = getattr(group, "rank_in_group", 0)
    if callable(rank):
        return int(rank())
    return int(rank)


def _is_capture_rank() -> bool:
    return _rank_in_group(get_tp_group()) == 0


def _compute_llama_4_scaling(model, positions):
    llama_4_scaling_config = getattr(
        getattr(model, "config", None), "llama_4_scaling", None
    )
    if llama_4_scaling_config is None:
        return None

    try:
        from vllm.model_executor.models.deepseek_v2 import _get_llama_4_scaling
    except ImportError as exc:  # pragma: no cover - depends on vLLM build.
        raise RuntimeError(
            "Model config requires llama_4_scaling, but the vLLM DeepSeek helper "
            "could not be imported."
        ) from exc

    return _get_llama_4_scaling(
        original_max_position_embeddings=llama_4_scaling_config[
            "original_max_position_embeddings"
        ],
        scaling_beta=llama_4_scaling_config["beta"],
        positions=positions,
    )


def _call_decoder_layer(model, layer, positions, hidden_states, residual, llama_4_scaling):
    layer_call_style = getattr(  # noqa: SLF001
        model._extension, "_layer_call_style", "positional"
    )
    if layer_call_style == "deepseek":
        return layer(positions, hidden_states, residual, llama_4_scaling)
    if layer_call_style == "keyword":
        return layer(
            hidden_states=hidden_states,
            positions=positions,
            residual=residual,
        )
    return layer(positions, hidden_states, residual)


def _capture_tensor(hidden_states, residual):
    if residual is None:
        return hidden_states.clone()
    return (hidden_states + residual).clone()


def _patched_forward(
    self,
    input_ids,
    positions,
    intermediate_tensors=None,
    inputs_embeds=None,
    **_kwargs,
):
    """Patched forward pass that captures hidden states from specified layers.

    This function is bound to base_model instances via types.MethodType.
    It expects base_model to have an _extension attribute pointing to the
    HiddenStatesWorkerExtension instance.

    Args:
        deepstack_input_embeds: For multimodal models with deepstack (Qwen3VL)
    """
    if get_pp_group().is_first_rank:
        hidden_states = (
            inputs_embeds
            if inputs_embeds is not None
            else self.embed_input_ids(input_ids)
        )
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    aux_hidden_states = []
    extension = self._extension  # noqa: SLF001
    extension._capture_forward_calls += 1  # noqa: SLF001
    # Only capture on TP rank 0 to avoid duplicates
    should_capture = _is_capture_rank()
    target_layers = extension._layer_ids if should_capture else frozenset()  # noqa: SLF001
    llama_4_scaling = None
    if getattr(extension, "_layer_call_style", None) == "deepseek":
        llama_4_scaling = _compute_llama_4_scaling(self, positions)

    for idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):
        hidden_states, residual = _call_decoder_layer(
            self, layer, positions, hidden_states, residual, llama_4_scaling
        )
        absolute_layer_idx = self.start_layer + idx

        # Capture intermediate layers (not the last) before norm
        if absolute_layer_idx in target_layers:
            aux_hidden_states.append(_capture_tensor(hidden_states, residual))

    # Return early if not last PP rank
    if not get_pp_group().is_last_rank:
        intermediate = IntermediateTensors(
            {"hidden_states": hidden_states, "residual": residual}
        )
        if getattr(extension, "_use_aux_hidden_state_outputs", False):
            return intermediate, []
        return intermediate

    hidden_states, _ = self.norm(hidden_states, residual)
    use_aux_output = getattr(extension, "_use_aux_hidden_state_outputs", False)
    if should_capture and aux_hidden_states:
        # Replace the last captured layer with post-norm version if it was the final layer
        # vLLM runtime passes post-norm hidden_states to MTP, so training data must match
        last_layer_idx = self.end_layer - 1
        if last_layer_idx in target_layers:
            aux_hidden_states[-1] = hidden_states.clone()
        extension._capture_last_aux_shapes = [  # noqa: SLF001
            tuple(h.shape) for h in aux_hidden_states
        ]

    if use_aux_output:
        extension._capture_aux_output_forward_calls += 1  # noqa: SLF001
        if aux_hidden_states:
            return hidden_states, torch.stack(aux_hidden_states, dim=0)
        return hidden_states, hidden_states.new_empty((0, *hidden_states.shape))

    should_store = getattr(extension, "_current_request_metadata", None) is not None
    should_store = should_store or getattr(extension, "_serving_mode", False)
    if should_capture and aux_hidden_states and should_store:
        extension._capture_side_effect_store_calls += 1  # noqa: SLF001
        extension._store_captured_states(aux_hidden_states)  # noqa: SLF001

    return hidden_states


class HiddenStatesWorkerExtension:
    """Worker extension that adds hidden states capture functionality.

    This extension hooks into VLLM's Worker initialization by being specified
    in ParallelConfig.worker_extension_cls. It patches the model's forward pass
    to intercept and capture intermediate layer hidden states during inference.

    Key behaviors:
    - Only captures on tensor parallel (TP) rank 0 to avoid duplicate data when
      using tensor parallelism. All TP ranks compute the same hidden states, so
      capturing from rank 0 is sufficient.
    - Stores captured states in GPU memory during batch processing as lists of
      tensors, concatenating them only when retrieved via _get_captured_states().
    - Supports pipeline parallelism by handling IntermediateTensors correctly.

    Attributes:
        _layer_ids: Frozenset of layer indices for O(1) lookup during capture
        _captured_states: Accumulated hidden states per layer (GPU tensors)
        model_runner: Reference to the VLLM model runner
    """

    def _reset_debug_counters(self):
        self._capture_forward_calls = 0
        self._capture_aux_output_forward_calls = 0
        self._capture_side_effect_store_calls = 0
        self._capture_aux_state_store_calls = 0
        self._capture_last_aux_present = False
        self._capture_last_aux_shapes = []
        self._capture_last_error = None

    @staticmethod
    def _select_base_model(model):
        # Vision-language / wrapper models such as Kimi-K2.5/K2.6.
        if hasattr(model, "get_language_model"):
            return model.get_language_model().model
        # Text models.
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model
        attrs = [a for a in dir(model) if not a.startswith("_")]
        raise AttributeError(
            f"Could not find base model with 'layers' attribute. "
            f"Model type: {type(model).__name__}, "
            f"Available attributes: {attrs}"
        )

    @staticmethod
    def _detect_layer_call_style(base_model) -> str:
        first_layer = next(
            islice(base_model.layers, base_model.start_layer, base_model.end_layer),
            None,
        )
        if first_layer is None:
            return "positional"
        try:
            params = inspect.signature(first_layer.forward).parameters
        except (TypeError, ValueError):
            return "positional"
        if "llama_4_scaling" in params:
            return "deepseek"
        if "hidden_states" in params and "positions" in params and "residual" in params:
            return "keyword"
        return "positional"

    def _store_captured_states(self, aux_hidden_states):
        if self._captured_states is None:  # type: ignore[has-type]
            self._captured_states = [[h] for h in aux_hidden_states]
        else:
            for i, h in enumerate(aux_hidden_states):
                self._captured_states[i].append(h)

        metadata = getattr(self, "_current_request_metadata", None)
        if metadata is not None:
            # Sort by vLLM's actual batch position (vLLM reorders requests internally)
            input_batch = self.model_runner.input_batch  # type: ignore[attr-defined]
            sorted_metadata = sorted(
                metadata.items(),
                key=lambda item: input_batch.req_id_to_index.get(item[0], float("inf")),
            )
            self._request_metadata.append(sorted_metadata)  # type: ignore[has-type]

    def _setup_hidden_states_capture(self, layer_ids: list[int]):
        """Setup model to capture auxiliary hidden states from specific layers"""
        self._layer_ids = frozenset(layer_ids)  # Convert once for O(1) lookup
        self._captured_states = None  # type: ignore[assignment]
        self._request_metadata = []  # type: ignore[assignment]
        self._current_request_metadata = None  # type: ignore[assignment]
        self._reset_debug_counters()

        model = self.model_runner.model  # type: ignore[attr-defined]
        base_model = self._select_base_model(model)

        self._model_type = type(model).__name__
        self._base_model_type = type(base_model).__name__
        self._base_model_start_layer = getattr(base_model, "start_layer", None)
        self._base_model_end_layer = getattr(base_model, "end_layer", None)
        self._base_model_num_layers = len(getattr(base_model, "layers", []))
        self._layer_call_style = self._detect_layer_call_style(base_model)
        self._base_model_compile_disabled = False

        # Use vLLM's native aux-output path so CUDA graph replay returns hidden
        # state tensors captured during graph capture instead of relying on
        # Python-side forward side effects, which are skipped during replay.
        self._use_aux_hidden_state_outputs = True
        self.model_runner.use_aux_hidden_state_outputs = True  # type: ignore[attr-defined]

        base_model._extension = self  # noqa: SLF001
        base_model.forward = types.MethodType(_patched_forward, base_model)
        if hasattr(base_model, "do_not_compile"):
            # vLLM creates the torch.compile callable during model construction.
            # After replacing forward, that cached callable still points to the
            # original method and returns no aux states. Bypass it here; the
            # outer vLLM CUDA graph wrapper still captures/replays this forward.
            base_model.do_not_compile = True
            self._base_model_compile_disabled = True
        logger.info(
            "Hidden states capture setup complete: model=%s base_model=%s "
            "layers=%s start=%s end=%s num_layers=%s call_style=%s "
            "aux_output=%s compile_disabled=%s",
            self._model_type,
            self._base_model_type,
            layer_ids,
            self._base_model_start_layer,
            self._base_model_end_layer,
            self._base_model_num_layers,
            self._layer_call_style,
            self._use_aux_hidden_state_outputs,
            self._base_model_compile_disabled,
        )

    def _set_request_metadata(self, request_metadata: dict[str, int]):
        """Set request metadata for the next forward pass.

        Args:
            request_metadata: Dict mapping request_id -> num_prefill_tokens
        """
        self._current_request_metadata = request_metadata  # type: ignore[assignment]

    def _store_last_aux_hidden_states(self):
        """Store aux hidden states from the last vLLM execute_model() call.

        This is the CUDA-graph-safe capture path. The patched model forward
        returns aux states as explicit model outputs; vLLM saves them in
        execute_model_state until sample_tokens() is called.
        """
        if not getattr(self, "_use_aux_hidden_state_outputs", False):
            return False
        if not _is_capture_rank():
            return False

        if getattr(self, "_current_request_metadata", None) is None:
            self._capture_last_aux_present = False
            self._capture_last_error = "current_request_metadata is None"
            return False

        state = getattr(  # type: ignore[attr-defined]
            self.model_runner, "execute_model_state", None
        )
        if state is None:
            self._capture_last_aux_present = False
            self._capture_last_error = "execute_model_state is None"
            return False

        aux_hidden_states = getattr(state, "aux_hidden_states", None)
        if aux_hidden_states is None or len(aux_hidden_states) == 0:
            self._capture_last_aux_present = False
            self._capture_last_error = "execute_model_state.aux_hidden_states is empty"
            return False

        num_tokens = int(state.scheduler_output.total_num_scheduled_tokens)
        try:
            if isinstance(aux_hidden_states, torch.Tensor):
                captured = [h[:num_tokens].clone() for h in aux_hidden_states]
            else:
                captured = [h[:num_tokens].clone() for h in aux_hidden_states]
            self._capture_last_aux_shapes = [tuple(h.shape) for h in captured]
            self._capture_last_aux_present = True
            self._capture_aux_state_store_calls += 1
            self._store_captured_states(captured)
            self._current_request_metadata = None  # type: ignore[assignment]
            return True
        except Exception as exc:  # pragma: no cover - defensive diagnostics.
            self._capture_last_aux_present = False
            self._capture_last_error = repr(exc)
            raise

    def _reset_capture(self):
        """Reset captured states before starting a new batch"""
        if not hasattr(self, "_layer_ids"):
            raise RuntimeError(
                "Must call _setup_hidden_states_capture before capturing states"
            )
        self._captured_states = None  # type: ignore[assignment]
        self._request_metadata = []  # type: ignore[assignment]
        self._current_request_metadata = None  # type: ignore[assignment]
        self._reset_debug_counters()

    def _get_captured_states(self):
        """Get the captured hidden states organized by request ID.

        Returns:
            Dict mapping request_id to list of tensors (one per layer),
            or None if no states captured.

        Track which tokens belong to which request across chunked prefill iterations.
        """
        if self._captured_states is None:
            return None

        # Concatenate captured states from all scheduler iterations
        concatenated_layers = [
            torch.cat(layer_tensors, dim=0) for layer_tensors in self._captured_states
        ]

        # Slice and group by request
        request_chunks: defaultdict[str, list[list[torch.Tensor]]] = defaultdict(
            lambda: [[] for _ in range(len(concatenated_layers))]
        )
        current_idx = 0

        for metadata in self._request_metadata:  # type: ignore[has-type]
            for req_id, num_tok in metadata:
                for layer_idx, layer_tensor in enumerate(concatenated_layers):
                    chunk = layer_tensor[current_idx : current_idx + num_tok].clone()
                    request_chunks[req_id][layer_idx].append(chunk)
                current_idx += num_tok

        # Concatenate chunks for each request
        result: dict[str, list[torch.Tensor]] = {
            req_id: [torch.cat(chunks, dim=0) for chunks in layer_chunks]
            for req_id, layer_chunks in request_chunks.items()
        }

        # Clear intermediate storage
        self._captured_states = None  # type: ignore[assignment]
        self._request_metadata = []  # type: ignore[assignment]
        return result

    def _get_capture_debug_state(self) -> dict[str, Any]:
        captured_layers = getattr(self, "_captured_states", None)
        try:
            tp_rank = _rank_in_group(get_tp_group())
        except Exception:  # pragma: no cover - diagnostic only.
            tp_rank = None
        try:
            pp_group = get_pp_group()
            pp_rank = _rank_in_group(pp_group)
            is_first_pp = bool(pp_group.is_first_rank)
            is_last_pp = bool(pp_group.is_last_rank)
        except Exception:  # pragma: no cover - diagnostic only.
            pp_rank = None
            is_first_pp = None
            is_last_pp = None

        return {
            "model_type": getattr(self, "_model_type", None),
            "base_model_type": getattr(self, "_base_model_type", None),
            "layer_ids": sorted(getattr(self, "_layer_ids", [])),
            "layer_call_style": getattr(self, "_layer_call_style", None),
            "start_layer": getattr(self, "_base_model_start_layer", None),
            "end_layer": getattr(self, "_base_model_end_layer", None),
            "num_layers": getattr(self, "_base_model_num_layers", None),
            "tp_rank": tp_rank,
            "pp_rank": pp_rank,
            "is_first_pp": is_first_pp,
            "is_last_pp": is_last_pp,
            "is_capture_rank": tp_rank == 0,
            "use_aux_hidden_state_outputs": getattr(
                self, "_use_aux_hidden_state_outputs", None
            ),
            "base_model_compile_disabled": getattr(
                self, "_base_model_compile_disabled", None
            ),
            "forward_calls": getattr(self, "_capture_forward_calls", None),
            "aux_output_forward_calls": getattr(
                self, "_capture_aux_output_forward_calls", None
            ),
            "side_effect_store_calls": getattr(
                self, "_capture_side_effect_store_calls", None
            ),
            "aux_state_store_calls": getattr(
                self, "_capture_aux_state_store_calls", None
            ),
            "last_aux_present": getattr(self, "_capture_last_aux_present", None),
            "last_aux_shapes": getattr(self, "_capture_last_aux_shapes", None),
            "last_error": getattr(self, "_capture_last_error", None),
            "captured_layer_count": 0
            if captured_layers is None
            else len(captured_layers),
            "captured_chunk_counts": None
            if captured_layers is None
            else [len(layer) for layer in captured_layers],
            "request_metadata_batches": len(getattr(self, "_request_metadata", [])),
            "current_request_metadata": getattr(self, "_current_request_metadata", None),
        }
