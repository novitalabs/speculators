"""Loss and metrics for the DSpark draft model.

loss = compound_loss(logits, targets) + conf_alpha * BCE(confidence, accept_rate)

The confidence target ``accept_rate = sum_v min(q_v, p_v) = 1 - d_TV`` is the
analytical acceptance rate (the overlap ``tv_loss`` already computes).
"""

import os
from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from torch.nn.functional import binary_cross_entropy_with_logits, softmax

from speculators.models.metrics import (
    LossConfig,
    compound_loss,
    compute_accuracy_multi_step,
    dflash_loss_decay,
)

__all__ = [
    "COUNTER_PREFIX",
    "SLOT0_COUNTER_KEYS",
    "compute_metrics",
    "eligible_anchor_token_count",
]

_EPS = 1e-8

# Additive counters, summed over micro-batches and then over ranks. The prefix is
# the contract the trainer keys off: `_aggregate_metric_floats` MEANS every other
# metric key, which is correct for a rate and wrong for a count, so these must be
# recognisable without a per-key list on the trainer side.
COUNTER_PREFIX = "ctr_"

# Fixed, config-independent and ORDERED. The trainer reduces exactly this vector,
# so it must not depend on what a local pack happened to produce -- a rank whose
# key set differed would change the collective's shape and desync NCCL.
SLOT0_COUNTER_KEYS = (
    # Raw slot-0 agreement, pooled: the estimator the ACC log already reports.
    f"{COUNTER_PREFIX}slot0_hit",            # h_i summed
    f"{COUNTER_PREFIX}slot0_sampled",        # k_i summed
    # Eligible positions per pack, i.e. the population each pack SAMPLED FROM.
    f"{COUNTER_PREFIX}slot0_eligible",       # N_i summed
    # Token-weighted estimator, as numerator/denominator so the ratio can be taken
    # after both sums instead of averaging per-pack rates.
    f"{COUNTER_PREFIX}slot0_tw_num",         # sum_i N_i * h_i / k_i
    f"{COUNTER_PREFIX}slot0_tw_den",         # sum_i N_i  (only packs with k_i > 0)
    # Packs excluded from the weighted estimator, so a reader can tell a small
    # denominator from a silently dropped pack.
    f"{COUNTER_PREFIX}slot0_packs",          # packs contributing
    f"{COUNTER_PREFIX}slot0_packs_no_sample",  # k_i == 0
    f"{COUNTER_PREFIX}slot0_packs_no_eligible",  # N_i == 0
)


def eligible_anchor_token_count(
    loss_mask: torch.Tensor,  # [1, T]
    block_size: int,
) -> torch.Tensor:
    """N_i: supervised positions a pack could have drawn an anchor from.

    MUST match ``select_anchors`` (dflash/utils.py:43-44) exactly, which clears the
    final ``block_size`` positions of the WHOLE PACKED ROW before sampling -- a
    block anchored there would run past the end of the row. Counting plain
    ``loss_mask.sum()`` instead would overstate the population by up to
    ``block_size`` positions per pack and quietly bias the weighting.

    This is the sampling POPULATION, not the sample: ``select_anchors`` then draws
    ``min(max_anchors, N_i)`` of these without replacement.
    """
    if block_size <= 0:
        raise ValueError(f"Expected block size > 0, got {block_size}")
    valid = loss_mask.bool().clone()
    valid[:, -block_size:] = False
    return valid.sum().to(torch.float64)


def _masked_decayed_mean(
    elementwise: torch.Tensor,  # [1, T]
    loss_mask: torch.Tensor,  # [1, T]
    pos_idx: torch.Tensor,  # [1, T]
    decay_fn: Callable[[torch.Tensor], torch.Tensor] | None,
    loss_scale: torch.Tensor | None = None,  # [1, T]
) -> torch.Tensor:
    """Masked, optionally position-decayed mean of a precomputed per-position term.

    ``loss_scale`` multiplies the NUMERATOR only, after mask and decay, exactly
    as in ``speculators.models.metrics.loss_function``: the denominator stays
    the boolean supervised-slot count so a per-position weight cannot be
    normalised away by the per-pack mean.
    """
    loss_mask = loss_mask.to(elementwise.dtype)
    weighted = elementwise * loss_mask
    if decay_fn is not None:
        weighted = weighted * decay_fn(pos_idx.to(weighted.dtype))
    if loss_scale is not None:
        weighted = weighted * loss_scale.to(weighted.dtype)
    denominator = loss_mask.sum(dim=1) + _EPS
    return (weighted.sum(dim=1) / denominator).mean()


def compute_metrics(
    logits: torch.Tensor,  # [1, T, draft_vocab_size] (Markov-corrected)
    targets: torch.Tensor,  # [1, T, draft_vocab_size]
    confidence_logits: torch.Tensor | None,  # [1, T] or None
    loss_mask: torch.Tensor,  # [1, T]
    block_size: int,
    loss_config: LossConfig,
    gamma: float = 4.0,
    confidence_head_alpha: float = 1.0,
    sample_from_anchor: bool = True,
    eligible_token_count: torch.Tensor | None = None,
    loss_scale: torch.Tensor | None = None,  # [1, T]
) -> tuple[torch.Tensor, dict]:
    """Compute the DSpark loss and a metrics dict (``*_sum``/``*_total`` pairs).

    ``loss_scale`` is an optional per-position non-negative multiplier on the
    loss NUMERATORS. It reaches all three trained terms -- CE and TV through
    ``compound_loss``, the confidence BCE through ``_masked_decayed_mean`` --
    and NOTHING else: every metric below is computed from ``loss_mask`` and the
    argmax-derived counts under ``torch.no_grad()``, so slot-0 agreement, the
    accuracy metrics and the eligible-token counters are byte-identical whether
    or not a scale is passed. ``loss``/``*_loss`` DO carry it and are therefore
    not comparable to a run without it.
    """

    device = logits.device
    seq_len = logits.shape[1]
    pos_idx = (torch.arange(seq_len, device=device) % block_size).unsqueeze(0)
    # Slot 0 is a real prediction only when sampling from the anchor; when it
    # is the anchor it is excluded from decay, prefix products and accuracy.
    start_pos = 0 if sample_from_anchor else 1
    decay_fn = partial(
        dflash_loss_decay, gamma=gamma, sample_from_anchor=sample_from_anchor
    )

    loss, term_losses = compound_loss(
        logits,
        targets,
        loss_mask,
        pos_idx,
        loss_config=loss_config,
        decay_fn=decay_fn,
        loss_scale=loss_scale,
    )

    # Analytical per-position acceptance rate = distributional overlap.
    with torch.no_grad():
        # Chunked over the sequence. The unchunked spelling upcast BOTH logits and
        # targets to fp32 and held them at once: at Kimi-K3's identity vocab
        # (V=163840) with max_anchors 384 * block 8 = 3072 positions that is
        # 1.88 GiB each, 3.75 GiB live, purely to reduce to a [1, T] vector.
        # exp50n OOM'd on exactly this (allocator asked for 3072*163840*4 =
        # 2013265920 bytes). softmax and the min-overlap sum are both along the
        # LAST dim, so slicing the sequence is exact rather than an approximation
        # -- no cross-position term exists to lose.
        _chunk = max(1, int(os.environ.get("SPECULATORS_ACCEPT_CHUNK", "512")))
        accept_rate = torch.cat(
            [
                torch.minimum(
                    softmax(logits[:, i : i + _chunk].float(), dim=-1),
                    softmax(targets[:, i : i + _chunk].float(), dim=-1),
                ).sum(dim=-1)
                for i in range(0, seq_len, _chunk)
            ],
            dim=1,
        )  # [1, T]
        # Per-block cumulative acceptance product over the drafted slots,
        # shared by the accept-length and calibration metrics.
        num_blocks = seq_len // block_size
        accept_blocks = accept_rate.view(num_blocks, block_size)
        draft_mask = loss_mask.to(accept_rate.dtype).view(num_blocks, block_size)[
            :, start_pos:
        ]
        accept_prefix = (accept_blocks[:, start_pos:] * draft_mask).cumprod(dim=-1)

    metrics: dict[str, Any] = {}
    if confidence_logits is not None:
        c_star = accept_rate.detach().to(confidence_logits.dtype)
        bce = binary_cross_entropy_with_logits(
            confidence_logits, c_star, reduction="none"
        )  # [1, T]
        conf_loss = _masked_decayed_mean(
            bce, loss_mask, pos_idx, decay_fn, loss_scale=loss_scale
        )
        loss = loss + confidence_head_alpha * conf_loss

        with torch.no_grad():
            mask_f = loss_mask.to(accept_rate.dtype)
            mask_total = mask_f.sum().clamp_min(1.0)
            conf_prob = confidence_logits.float().sigmoid()
            metrics["confidence_loss_sum"] = conf_loss.detach().clone()
            metrics["confidence_loss_total"] = torch.ones((), device=device)
            metrics["confidence_abs_error_sum"] = (
                (conf_prob - accept_rate).abs() * mask_f
            ).sum()
            metrics["confidence_abs_error_total"] = mask_total
            # Mean predicted vs. observed acceptance — a calibration sanity check.
            metrics["confidence_pred_mean_sum"] = (conf_prob * mask_f).sum()
            metrics["confidence_pred_mean_total"] = mask_total
            # Calibration of the cumulative acceptance product, which is what
            # dynamic draft-length thresholding consumes (signed pred - target).
            conf_prefix = (
                conf_prob.view(num_blocks, block_size)[:, start_pos:] * draft_mask
            ).cumprod(dim=-1)
            metrics["confidence_cumprod_bias_sum"] = (
                (conf_prefix - accept_prefix) * draft_mask
            ).sum()
            metrics["confidence_cumprod_bias_total"] = draft_mask.sum().clamp_min(1.0)

    ones = torch.ones((), device=device)
    metrics["loss_sum"] = loss.detach().clone()
    metrics["loss_total"] = ones
    for term_name, term_val in term_losses.items():
        metrics[f"{term_name}_sum"] = term_val
        metrics[f"{term_name}_total"] = ones

    # Mean acceptance rate of the (Markov-corrected) drafter.
    with torch.no_grad():
        mask_f = loss_mask.to(accept_rate.dtype)
        metrics["accept_rate_sum"] = (accept_rate * mask_f).sum()
        metrics["accept_rate_total"] = mask_f.sum().clamp_min(1.0)

    # Expected accepted draft length per block (DSpark's tau): the cumulative
    # acceptance product summed over draft slots, plus the always-emitted anchor.
    with torch.no_grad():
        per_block_len = accept_prefix.sum(dim=-1) + 1.0
        block_valid = (draft_mask.sum(dim=-1) > 0).to(accept_rate.dtype)
        metrics["accept_len_sum"] = (per_block_len * block_valid).sum()
        metrics["accept_len_total"] = block_valid.sum().clamp_min(1.0)

    # Per-position greedy accuracy.
    pred_ids = torch.argmax(logits, dim=-1)
    target_ids = torch.argmax(targets, dim=-1)
    correct_per_pos, total_per_pos = compute_accuracy_multi_step(
        pred_ids, target_ids, loss_mask, pos_idx, block_size
    )
    metrics["full_acc_sum"] = correct_per_pos[start_pos:].sum()
    metrics["full_acc_total"] = total_per_pos[start_pos:].sum()
    for pos in range(start_pos, block_size):
        metrics[f"position_{pos}_acc_sum"] = correct_per_pos[pos]
        metrics[f"position_{pos}_acc_total"] = total_per_pos[pos]

    # ---- slot-0 teacher agreement, eligible-token weighted -------------------
    # A DIAGNOSTIC ESTIMATOR, not a new training signal: every value below is
    # detached, derived from tensors the loss already computed, and reported only.
    # The loss, the sampling and every pre-existing metric are untouched.
    #
    # WHAT IT ESTIMATES. Each pack draws k_i anchors uniformly without replacement
    # from its N_i eligible positions, so the pooled rate sum(h)/sum(k) weights a
    # pack by how many anchors it happened to contribute, not by how much eligible
    # text it holds. Re-weighting each pack's own rate h_i/k_i by N_i estimates the
    # rate over eligible TOKENS instead:
    #
    #     sum_i N_i * (h_i / k_i)  /  sum_i N_i
    #
    # Both differ from serving acceptance, which is measured on served requests with
    # its own boundaries and labels. This is NOT an acceptance rate and must not be
    # relabelled as one.
    #
    # Emitted as numerator and denominator, never as a ratio: the trainer sums these
    # over micro-batches and across ranks and divides ONCE at the end. Averaging
    # per-pack ratios would silently re-weight by pack count.
    if eligible_token_count is not None:
        with torch.no_grad():
            h_i = correct_per_pos[0].detach().to(torch.float64)
            k_i = total_per_pos[0].detach().to(torch.float64)
            n_i = eligible_token_count.detach().reshape(()).to(torch.float64)
            zero = torch.zeros((), dtype=torch.float64, device=device)
            one = torch.ones((), dtype=torch.float64, device=device)
            # A pack with no sampled slot-0 anchor has NO estimate of its own rate,
            # so it contributes to NEITHER side of the ratio. Adding its N_i to the
            # denominator alone would drag the estimate toward zero; substituting the
            # pooled rate would invent an observation. Both cases are counted so the
            # exclusion is visible rather than inferred from a short denominator.
            usable = (k_i > 0) & (n_i > 0)
            metrics[SLOT0_COUNTER_KEYS[0]] = h_i
            metrics[SLOT0_COUNTER_KEYS[1]] = k_i
            metrics[SLOT0_COUNTER_KEYS[2]] = n_i
            # clamp_min(1) only guards the DIVISION; the result is masked to 0 by
            # `usable`, so a k_i of 0 never reaches the numerator as h_i/1.
            metrics[SLOT0_COUNTER_KEYS[3]] = torch.where(
                usable, n_i * h_i / k_i.clamp_min(1.0), zero
            )
            metrics[SLOT0_COUNTER_KEYS[4]] = torch.where(usable, n_i, zero)
            metrics[SLOT0_COUNTER_KEYS[5]] = torch.where(usable, one, zero)
            metrics[SLOT0_COUNTER_KEYS[6]] = torch.where(k_i > 0, zero, one)
            metrics[SLOT0_COUNTER_KEYS[7]] = torch.where(n_i > 0, zero, one)

    return loss, metrics
