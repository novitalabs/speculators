"""MLA variants of the DFlash decoder layer, for Kimi-K3 / DeepSeek-V3 drafts.

``model_definitions.py`` builds the draft out of Qwen3 parts (``Qwen3MLP``,
``Qwen3RMSNorm``, GQA ``q_proj``/``k_proj``/``v_proj``). That is the right shape
for a Qwen3-family draft and the wrong one for K3, in two ways that both bite:

1. **It cannot even be constructed.** ``Qwen3MLP.__init__`` does
   ``ACT2FN[config.hidden_act]``, and K3's ``hidden_act`` is ``"situ"`` -- a
   Moonshot activation absent from HF's table, so draft construction dies with
   ``KeyError: 'situ'`` before any forward pass.

2. **Even with that patched, the trained weights would be unservable.** vLLM's
   ``K3DSparkDecoderLayer`` builds ``MultiHeadLatentAttention`` + ``KimiMLP``,
   and the published ``Inferact/Kimi-K3-DSpark`` checkpoint carries
   ``q_a_proj`` / ``q_b_proj`` / ``kv_a_proj_with_mqa`` / ``kv_b_proj`` /
   ``kv_a_layernorm`` per layer. A Qwen3 draft emits ``q_proj``/``k_proj``/
   ``v_proj`` instead: different names AND different shapes, so nothing loads.

So this module mirrors the *serving* geometry instead: MLA LoRA projections and
a fused-gate SiTU MLP, keeping DFlash's one real trick -- injecting the
verifier's ``target_hidden`` as context KV so the draft attends over the full
context without recomputing it.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.qwen3.modeling_qwen3 import GradientCheckpointingLayer

# DFlash's mask builders emit [B, H, Q_LEN, KV_LEN]; only then is the trailing
# slice to (q_len, kv_len) meaningful.
_MASK_NDIM = 4


def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    """YaRN attention-temperature factor (DeepSeek-V3's formulation)."""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class SituMLP(nn.Module):
    """K3's SiTU-gated MLP, in the shape the DSpark checkpoint stores.

    ``beta * tanh(gate/beta) * sigmoid(gate) * up``, optionally squashing ``up``
    through ``linear_beta * tanh(up/linear_beta)``. Read off K3's own
    ``modeling_kimi_linear.py:64-85`` (``SituAndMul``) rather than reimplemented
    from a description.

    Why not simply register K3's ``SituAndMul`` into ``ACT2FN`` and keep
    ``Qwen3MLP``: the interfaces disagree in a way that would compute silently
    wrong values rather than raise. ``SituAndMul.forward`` takes ONE fused
    ``[gate, up]`` tensor, splits it at ``x.shape[-1] // 2``, and does the
    multiply itself; ``Qwen3MLP.forward`` does
    ``act_fn(gate_proj(x)) * up_proj(x)`` -- activation over the gate alone at
    full ``intermediate_size``, multiply afterwards. Feeding the latter's gate
    into the former would halve the width and treat the top half of the gate as
    ``up``. It also needs ``beta``/``linear_beta``, which an ``ACT2FN[key]``
    lookup has no way to pass.

    Weight names match the checkpoint (``mlp.gate_proj`` / ``mlp.up_proj`` /
    ``mlp.down_proj``), so a non-SiTU ``hidden_act`` still works and falls back
    to the standard gated form.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        hidden_act = getattr(config, "hidden_act", "silu")
        self.is_situ = hidden_act == "situ"
        if self.is_situ:
            # `or 1.0` mirrors K3's _get_situ_activation_params: the config may
            # carry the key with a null value.
            self.beta = float(getattr(config, "activation_situ_beta", None) or 1.0)
            linear_beta = getattr(config, "activation_situ_linear_beta", None)
            self.linear_beta = float(linear_beta) if linear_beta is not None else None
            self.act_fn = None
        else:
            self.beta = 1.0
            self.linear_beta = None
            self.act_fn = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if not self.is_situ:
            return self.down_proj(self.act_fn(gate) * up)  # type: ignore[misc]
        # fp32 for the tanh/sigmoid product, as K3 does, then back to input dtype.
        gate32 = gate.to(torch.float32)
        up32 = up.to(torch.float32)
        situ_a = self.beta * torch.tanh(gate32 / self.beta) * torch.sigmoid(gate32)
        if self.linear_beta is not None:
            up32 = self.linear_beta * torch.tanh(up32 / self.linear_beta)
        return self.down_proj((situ_a * up32).to(x.dtype))


class MLARMSNorm(nn.Module):
    """RMSNorm over the last dim, fp32 accumulation. Matches DeepseekV3RMSNorm."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q_pe: torch.Tensor, k_pe: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to the rope-carrying slices only.

    MLA splits each head into a NoPE part and a rope part; only the latter is
    rotated. ``cos``/``sin`` are ``[B, T, qk_rope_head_dim]`` and are unsqueezed
    on the head axis so one shared k_pe head broadcasts across query heads.
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (
        q_pe * cos + _rotate_half(q_pe) * sin,
        k_pe * cos + _rotate_half(k_pe) * sin,
    )


class MLADFlashAttention(nn.Module):
    """MLA attention with DFlash's context-KV injection.

    Projection names and shapes are exactly what vLLM's
    ``MultiHeadLatentAttention`` expects and what the published DSpark
    checkpoint stores::

        q_a_proj            [H -> q_lora_rank]
        q_a_layernorm       [q_lora_rank]
        q_b_proj            [q_lora_rank -> num_heads * (qk_nope + qk_rope)]
        kv_a_proj_with_mqa  [H -> kv_lora_rank + qk_rope]
        kv_a_layernorm      [kv_lora_rank]
        kv_b_proj           [kv_lora_rank -> num_heads * (qk_nope + v_head_dim)]
        o_proj              [num_heads * v_head_dim -> H]

    DFlash's trick is preserved: K and V for the *context* come from the
    verifier's ``target_hidden`` rather than from the draft's own hidden states,
    so keys/values span ``ctx_len + q_len`` while queries span only ``q_len``.
    Concatenation happens on the compressed latent (pre-``kv_b_proj``), which is
    both cheaper than concatenating expanded K/V and the only order that keeps
    ``kv_a_layernorm`` applied to whole latents.
    """

    def __init__(self, config: Any, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        # DFlash attends over injected context; it is never causal-only.
        self.is_causal = False
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        bias = bool(getattr(config, "attention_bias", False))

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.q_head_dim, bias=bias
            )
        else:
            self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=bias)
            self.q_a_layernorm = MLARMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=bias
        )
        self.kv_a_layernorm = MLARMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim, self.hidden_size, bias=bias
        )

        # YaRN raises attention temperature; dropping it would silently train at
        # the wrong scale for exactly the long contexts K3 is extended to.
        self.softmax_scale = self.q_head_dim ** (-0.5)
        rope_scaling = getattr(config, "rope_scaling", None) or getattr(
            config, "rope_parameters", None
        )
        rope_type = (
            rope_scaling.get("rope_type", rope_scaling.get("type"))
            if rope_scaling
            else None
        )
        if rope_type == "yarn":
            factor = rope_scaling.get("factor", 1.0)
            mscale_all_dim = rope_scaling.get("mscale_all_dim", 0)
            if mscale_all_dim:
                mscale = yarn_get_mscale(factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

    def _q(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.q_lora_rank is None:
            return self.q_proj(hidden_states)
        return self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        # Accepted for signature parity with Qwen3DFlashAttention and unused here:
        # DFlash training is a single forward over injected context, never
        # incremental decode, so there is no cache to read or advance. Taking them
        # and ignoring them beats the alternative of the decoder layer having to
        # know which attention class it holds.
        past_key_values: Any = None,  # noqa: ARG002
        cache_position: torch.Tensor | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> tuple[torch.Tensor, None]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        kv_len = ctx_len + q_len

        # -- queries: draft positions only
        q = self._q(hidden_states)
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # -- keys/values: verifier context THEN the draft's own positions. This
        #    is the DFlash mechanism; concatenating on the compressed latent
        #    keeps kv_a_layernorm applied to whole latents.
        compressed = torch.cat(
            [
                self.kv_a_proj_with_mqa(target_hidden),
                self.kv_a_proj_with_mqa(hidden_states),
            ],
            dim=1,
        )
        compressed_kv, k_pe = torch.split(
            compressed, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bsz, kv_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(bsz, kv_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )
        k_nope, value_states = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        # -- RoPE on the rope slices only, and q and k do NOT share a position
        #    range: keys span the whole [0, kv_len) (context then draft) while
        #    queries are only the draft tail. Rotating q with cos[:, :q_len]
        #    would place every draft token at absolute position 0 and destroy
        #    the context alignment, so q takes the LAST q_len entries.
        cos, sin = position_embeddings
        cos_k, sin_k = cos[:, :kv_len], sin[:, :kv_len]
        cos_q, sin_q = cos_k[:, -q_len:], sin_k[:, -q_len:]
        q_pe = (q_pe * cos_q.unsqueeze(1)) + (_rotate_half(q_pe) * sin_q.unsqueeze(1))
        k_pe = (k_pe * cos_k.unsqueeze(1)) + (_rotate_half(k_pe) * sin_k.unsqueeze(1))

        # -- reassemble full heads; k_pe is one shared head, expanded per query head
        query_states = torch.cat([q_nope, q_pe], dim=-1)
        key_states = torch.cat(
            [k_nope, k_pe.expand(-1, self.num_heads, -1, -1)], dim=-1
        )

        # SDPA, not flex_attention: MLA's query/key head dim (qk_nope+qk_rope,
        # 192 for K3) differs from its value head dim (128), and the flex kernel
        # assumes one head_dim throughout -- train.py's kimi_k2 branch already
        # forces "sdpa" for exactly this reason. SDPA takes q/k and v with
        # different last dims, and accepts either a bool or an additive mask, so
        # both of DFlash's mask builders (create_mask -> bool, create_float_mask
        # -> -inf floats) pass through unchanged.
        attn_mask = attention_mask
        if attn_mask is not None and attn_mask.ndim == _MASK_NDIM:
            attn_mask = attn_mask[..., :q_len, :kv_len]
        attn_output = nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attn_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.softmax_scale,
        )
        attn_output = (
            attn_output.transpose(1, 2)
            .reshape(bsz, q_len, self.num_heads * self.v_head_dim)
            .contiguous()
        )
        return self.o_proj(attn_output), None


class MLADFlashDecoderLayer(GradientCheckpointingLayer):
    """Drop-in replacement for ``Qwen3DFlashDecoderLayer`` with MLA + SiTU.

    Same ``forward`` signature and same submodule names (``self_attn`` / ``mlp``
    / ``input_layernorm`` / ``post_attention_layernorm``), so ``DFlashDraftModel``
    swaps this in without any change to its loop. The submodule *contents* are
    what differ, and they are chosen to match the serving geometry rather than
    the Qwen3 one -- see the module docstring.
    """

    def __init__(self, config: Any, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MLADFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = SituMLP(config)
        self.input_layernorm = MLARMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MLARMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        target_hidden: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        # position_ids / output_attentions / use_cache are part of the shared
        # decoder-layer signature that DFlashDraftModel's loop calls with, and are
        # unused on both backbones: positions arrive pre-computed as
        # position_embeddings, and training never decodes incrementally.
        position_ids: torch.Tensor | None = None,  # noqa: ARG002
        past_key_value: Any = None,
        output_attentions: bool | None = False,  # noqa: ARG002
        use_cache: bool | None = False,  # noqa: ARG002
        cache_position: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert hidden_states is not None  # noqa: S101
        assert target_hidden is not None  # noqa: S101
        assert position_embeddings is not None  # noqa: S101
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class MLARotaryEmbedding(nn.Module):
    """RoPE over ``qk_rope_head_dim`` only, with YaRN-style linear scaling.

    ``Qwen3RotaryEmbedding`` sizes its inverse frequencies from ``head_dim``
    (or ``hidden_size // num_heads``), which for K3's MLA would be the wrong
    width entirely: only the ``qk_rope_head_dim`` slice is rotated.
    """

    def __init__(self, config: Any, device: Any = None) -> None:
        super().__init__()
        dim = config.qk_rope_head_dim
        rope = getattr(config, "rope_scaling", None) or getattr(
            config, "rope_parameters", None
        ) or {}
        base = rope.get("rope_theta") or getattr(config, "rope_theta", 10000.0)
        self.scaling_factor = float(rope.get("factor", 1.0) or 1.0)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Linear position scaling for YaRN factor>1; interpolates positions into
        # the model's native basis, matching what the verifier was extended with.
        pos = position_ids.to(torch.float32)
        if self.scaling_factor > 1.0:
            pos = pos / self.scaling_factor
        freqs = pos[..., None] * self.inv_freq.to(x.device)[None, None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)
