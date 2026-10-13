"""Optional DSpark numerator weights must leave the mask denominator intact."""

import torch

from speculators.models.dspark.metrics import compute_metrics
from speculators.models.metrics import loss_function


def test_loss_scale_changes_only_numerator():
    logits = torch.zeros(1, 4, 2)
    targets = torch.ones_like(logits)
    mask = torch.tensor([[1, 1, 0, 0]])
    pos = torch.arange(4).unsqueeze(0)
    def term(x, y):
        return ((x - y) ** 2).mean(dim=-1)
    base = loss_function(logits, targets, mask, pos, loss_fn=term)
    same = loss_function(logits, targets, mask, pos, loss_fn=term,
                         loss_scale=torch.ones_like(mask))
    weighted = loss_function(logits, targets, mask, pos, loss_fn=term,
                             loss_scale=torch.tensor([[2., 1., 99., 99.]]))
    torch.testing.assert_close(base, same, rtol=0, atol=0)
    torch.testing.assert_close(weighted, base * 1.5, rtol=0, atol=0)


def test_dspark_confidence_loss_scales_with_same_mask():
    logits = torch.tensor([[[1., 0.], [0., 1.]]])
    targets = torch.tensor([[[0., 1.], [1., 0.]]])
    conf = torch.zeros(1, 2)
    mask = torch.ones(1, 2)
    config = {"ce": (lambda x, y: ((x - y) ** 2).mean(dim=-1), 1.0)}
    base, base_metrics = compute_metrics(logits, targets, conf, mask, 2,
                                         loss_config=config)
    same, _ = compute_metrics(logits, targets, conf, mask, 2,
                              loss_config=config, loss_scale=torch.ones_like(mask))
    weighted, weighted_metrics = compute_metrics(
        logits, targets, conf, mask, 2, loss_config=config,
        loss_scale=torch.full_like(mask, 2.0))
    torch.testing.assert_close(base, same, rtol=0, atol=0)
    torch.testing.assert_close(weighted, base * 2, rtol=1e-6, atol=1e-6)
    for key, value in base_metrics.items():
        if key.endswith("loss_sum") or key == "loss_sum":
            continue
        torch.testing.assert_close(weighted_metrics[key], value, rtol=0, atol=0)
