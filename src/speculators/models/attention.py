"""Shared attention utilities for speculator models.

This module contains attention functions and utilities shared across different
speculator architectures (EAGLE3, DFlash, etc.) to avoid code duplication.

Note (camelot fork): ``flex_attention_forward`` and the ``ALL_ATTENTION_FUNCTIONS``
registry are re-exported from ``speculators.models.eagle3.attention`` instead of
being redefined here. The eagle3 variant is a superset (compiled flex_attention +
dense-mask SDPA fallback) and both upstream versions register the same
"simple_flex_attention" key into the transformers ``AttentionInterface`` singleton,
so a second local definition would silently override it depending on import order.
"""

from collections.abc import Callable

import torch
from torch.nn.attention.flex_attention import (
    BlockMask,
)
from torch.nn.attention.flex_attention import (
    create_mask as _create_mask,
)

from speculators.models.eagle3.attention import (  # noqa: F401
    ALL_ATTENTION_FUNCTIONS,
    flex_attention_forward,
)


def create_float_mask(
    mask_mod: Callable,
    B: int | None = None,  # noqa: N803
    H: int | None = None,  # noqa: N803
    Q_LEN: int = 0,  # noqa: N803
    KV_LEN: int = 0,  # noqa: N803
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Wrap ``create_mask`` and convert the boolean result to a float mask.

    Non-flex attention backends (eager, SDPA) add the mask numerically
    (``scores + mask``) and need 0 for attended and ``-inf`` for masked.
    """
    bool_mask = _create_mask(
        mask_mod, B=B, H=H, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device
    )
    float_mask = torch.zeros(bool_mask.shape, dtype=dtype, device=device)
    float_mask.masked_fill_(~bool_mask, float("-inf"))
    return float_mask


def block_mask_to_dense_attention_mask(
    block_mask: BlockMask, device: torch.device, dtype: torch.dtype
):
    attention_mask = torch.ones(block_mask.shape, device=device, dtype=dtype)

    for q_idx in range(attention_mask.shape[2]):
        attention_mask[0, 0, q_idx, :] = block_mask.mask_mod(
            torch.zeros(1, device=device, dtype=torch.long),
            torch.zeros(1, device=device, dtype=torch.long),
            torch.ones(1, device=device, dtype=torch.long) * q_idx,
            torch.arange(attention_mask.shape[3], device=device, dtype=torch.long),
        )
    return attention_mask
