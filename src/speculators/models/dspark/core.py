from typing import ClassVar

import os

import torch
from transformers import PretrainedConfig

from speculators.model import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dspark.config import DSparkSpeculatorConfig
from speculators.models.dspark.metrics import (
    compute_metrics,
    eligible_anchor_token_count,
)
from speculators.models.dspark.model_definitions import ConfidenceHead, MarkovHead
from speculators.models.metrics import LossConfig, kl_div_loss, resolve_loss_config
from speculators.models.utils import conditional_torch_compile

_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (kl_div_loss, 1.0)}

__all__ = [
    "DSparkDraftModel",
]




# --- ANCHOR AUDIT (default OFF) -------------------------------------------
# Read ONCE at import, never inside forward. `forward` is @conditional_torch_
# compile'd, and the INTENT of a module-level Python bool is that Dynamo
# specializes on it so the OFF build carries no runtime branch.
#
# MEASURED, on real GPU, at the TRAINING shape. eq0433 compared OFF/OFF2/ON on the
# real compiled K3 forward in BOTH infer and train mode and found
# n_off_vs_on_diff_fields=0 with identical draft_tokens_sha256, loss_hex and
# rng_state_sha256 -- but only at seq_len=74, and its own verdict recorded
# gate_authorises_50_steps=false because a short input does not cover the training
# graph. mrep0926a then re-ran it at seq_len=32768 / max_anchors=96, train mode,
# forward AND backward, on two real saved micro-batches (s1m0, s1m1):
# verdict PASS, 0 differing fields, 0 same-arm instability,
# gate_authorises_50_steps=true. Evidence:
# diary-job/evidence/exp73-teacher/obs4/audit/{eq0433,mrep0926a}/.
#
# What that does NOT establish, per those verdicts' own `not_established` lists:
# equivalence against a PRISTINE (unpatched) tree -- both arms run this same file,
# OFF vs ON -- and generalisation to shapes other than the two measured. So the
# claim is "OFF is bit-identical to ON at the training shape", not "the OFF build
# is byte-identical to a build without this block".
#
# The static properties still hold and are still the reason the above is possible:
# the flag is read at import, no os.environ read occurs inside forward (which would
# force a graph break), and the audit block draws no RNG.
#
# Reading os.environ inside forward would force a graph break, which is why it
# is not done here.
#
# The audit consumes NO RNG. select_anchors() draws an UNSEEDED torch.randperm
# (dflash/utils.py:62), so any extra random draw here would shift every
# subsequent anchor set and silently invalidate the comparison this instrument
# exists to make. Everything below is argmax/gather/detach only.
_ANCHOR_AUDIT = os.environ.get("CAMELOT_ANCHOR_AUDIT", "") == "1"
# Reserved prefix: the trainer pops these BEFORE _metrics_to_float and they
# never reach metric_parts, the loss, the aggregator or a checkpoint.
AUDIT_PREFIX = "__audit__"

@SpeculatorModel.register("dspark")
class DSparkDraftModel(DFlashDraftModel):
    """DFlash backbone plus a Markov logit-bias head and a confidence head.

    After the base draft logits are produced, the Markov head biases position
    ``k`` using the previous block token and the confidence head predicts each
    position's acceptance probability. Everything else is inherited from DFlash.
    """

    config_class: ClassVar[type[DSparkSpeculatorConfig]] = DSparkSpeculatorConfig  # type: ignore[misc,assignment]

    def __init__(self, config: DSparkSpeculatorConfig) -> None:
        super().__init__(config=config)

        hidden_size = config.transformer_layer_config.hidden_size

        self.markov_head: MarkovHead | None = None
        if config.markov_rank > 0:
            self.markov_head = MarkovHead(
                verifier_vocab_size=self.verifier_vocab_size,
                draft_vocab_size=self.draft_vocab_size,
                markov_rank=config.markov_rank,
                hidden_size=hidden_size,
                head_type=config.markov_head_type,
            )

        self.confidence_head: ConfidenceHead | None = None
        if config.enable_confidence_head:
            if config.confidence_head_with_markov and self.markov_head is None:
                raise ValueError(
                    "confidence_head_with_markov=True requires markov_rank > 0."
                )
            input_dim = hidden_size + (
                config.markov_rank if config.confidence_head_with_markov else 0
            )
            self.confidence_head = ConfidenceHead(input_dim)

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DSparkDraftModel":
        """Create a DSpark model from training arguments (mirrors DFlash)."""
        config = DSparkSpeculatorConfig(
            **cls._build_base_config_kwargs("dspark", verifier_config, **kwargs),
            markov_rank=kwargs.get("markov_rank", 256),
            markov_head_type=kwargs.get("markov_head_type", "vanilla"),
            enable_confidence_head=kwargs.get("enable_confidence_head", True),
            confidence_head_with_markov=kwargs.get("confidence_head_with_markov", True),
        )

        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Resolve DSpark's compound loss from ``--loss-fn``."""
        loss_config = resolve_loss_config(kwargs["loss_fn"])
        gamma = kwargs.get("dflash_decay_gamma", 4.0)
        max_anchors = kwargs.get("max_anchors", 3072)
        confidence_head_alpha = kwargs.get("confidence_head_alpha", 1.0)
        shared = {
            "loss_config": loss_config,
            "gamma": gamma,
            "max_anchors": max_anchors,
            "confidence_head_alpha": confidence_head_alpha,
        }
        return dict(shared), dict(shared)

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,  # [1, total_seq_len, num_hidden*hidden_size]
        input_ids: torch.Tensor,  # [1, total_seq_len]
        loss_mask: torch.Tensor,  # [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor,  # [1, total_seq_len, hidden_size]
        document_ids: torch.Tensor,  # [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # [1, total_seq_len]
        loss_config: LossConfig | None = None,
        gamma: float = 4.0,
        max_anchors: int = 3072,
        confidence_head_alpha: float = 1.0,
        **kwargs,
    ):
        hidden, logits, targets, aligned_loss_mask, anchored_block_indices = (
            self._backbone_forward(
                hidden_states,
                input_ids,
                loss_mask,
                verifier_last_hidden_states,
                document_ids,
                position_ids,
                max_anchors=max_anchors,
                **kwargs,
            )
        )

        # DSpark: add the Markov logit bias and predict per-position confidence.
        num_blocks = max_anchors
        block = self.block_size
        mask_tokens_size = num_blocks * block
        # Ground-truth block tokens (verifier vocab); slot 0 holds the anchor.
        block_tokens = input_ids[0, anchored_block_indices].view(num_blocks, block)
        # prev_token_ids[:, k] is the token preceding draft position k.
        if self.config.sample_from_anchor:
            # Slot k predicts anchor+k+1, so the token at the previous position
            # is block_tokens[:, k] itself -- no shift.
            prev_token_ids = block_tokens
        else:
            # Slot k predicts anchor+k, so the previous token is
            # block_tokens[:, k-1] (slot 0 is the untrained anchor).
            prev_token_ids = torch.cat(
                [block_tokens[:, :1], block_tokens[:, :-1]], dim=1
            )  # [num_blocks, block]
        hidden_blocks = hidden.view(num_blocks, block, -1)

        confidence_logits = None
        prev_emb = None
        if self.markov_head is not None:
            prev_emb = self.markov_head.prev_embeddings(prev_token_ids)
            markov_bias = self.markov_head.block_bias(
                prev_token_ids=prev_token_ids,
                hidden_states=hidden_blocks,
                prev_emb=prev_emb,
            )
            logits = (logits.view(num_blocks, block, -1) + markov_bias).view(
                1, mask_tokens_size, -1
            )

        if self.confidence_head is not None:
            # confidence_head_with_markov requires markov_rank > 0 (enforced in
            # __init__), so prev_emb is always set when the flag is on.
            if self.config.confidence_head_with_markov and prev_emb is not None:
                conf_features = torch.cat(
                    [hidden_blocks, prev_emb.to(hidden_blocks.dtype)], dim=-1
                )
            else:
                conf_features = hidden_blocks
            confidence_logits = self.confidence_head(conf_features).reshape(
                1, mask_tokens_size
            )

        loss, metrics = compute_metrics(
            logits,
            targets,
            confidence_logits,
            aligned_loss_mask,
            self.block_size,
            loss_config=loss_config or _DEFAULT_LOSS_CONFIG,
            gamma=gamma,
            confidence_head_alpha=confidence_head_alpha,
            sample_from_anchor=self.config.sample_from_anchor,
            # N_i for the eligible-token-weighted slot-0 estimator. Computed from
            # the ORIGINAL loss_mask, which is the only place the sampling
            # population is still visible: aligned_loss_mask has already been
            # gathered down to the k_i anchors that were actually drawn, so the
            # population cannot be recovered from it.
            eligible_token_count=eligible_anchor_token_count(
                loss_mask, self.block_size
            ),
        )
        draft_tokens = torch.argmax(logits, dim=-1)

        if _ANCHOR_AUDIT:
            # The REAL anchor trace, not an upper bound. `anchored_block_indices`
            # is the exact [num_anchors*block_size] gather index the forward
            # itself used, so slot k of block b is sequence position
            # anchored_block_indices[b*block_size + k]. That is what makes this a
            # replayable record: PACK-MAP could only bound which positions were
            # ELIGIBLE (select_anchors additionally clears the last block_size
            # positions and then samples), while this is the set actually taken.
            #
            # Everything is detach()ed and carried under AUDIT_PREFIX, which the
            # trainer pops before metrics are floated/aggregated -- no audit
            # value reaches the loss, the reported metrics or a checkpoint.
            # draft_tokens/targets are already computed above; the argmax over
            # targets duplicates dspark/metrics.py:158 exactly, so the audit adds
            # no new numeric path, only a second read of existing tensors.
            metrics[AUDIT_PREFIX + "anchor_idx"] = (
                anchored_block_indices.detach().to(torch.int32)
            )
            metrics[AUDIT_PREFIX + "draft_argmax"] = draft_tokens.detach().view(-1)
            metrics[AUDIT_PREFIX + "teacher_argmax"] = (
                torch.argmax(targets, dim=-1).detach().view(-1)
            )
            metrics[AUDIT_PREFIX + "aligned_mask"] = (
                aligned_loss_mask.detach().view(-1).to(torch.uint8)
            )
            metrics[AUDIT_PREFIX + "block_size"] = torch.tensor(
                block, dtype=torch.int32, device=logits.device
            )

        return draft_tokens, loss, metrics
