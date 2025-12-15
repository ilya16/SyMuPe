from __future__ import annotations

import torch
import torch.nn.functional as F

from symupe.utils import ExplicitEnum


class Reduction(ExplicitEnum):
    NONE = "none"
    MEAN = "mean"
    BATCH_MEAN = "batch_mean"


def reduce_metrics(
        values: torch.Tensor,
        mask: torch.Tensor | None,
        reduction: str | Reduction = Reduction.MEAN
) -> torch.Tensor | None:
    if reduction == Reduction.NONE:
        return values
    elif mask is None:
        return values.mean()
    elif reduction == Reduction.MEAN:
        return values[mask].mean()
    elif reduction == Reduction.BATCH_MEAN:
        return masked_batch_mean(values, mask)
    raise ValueError(f"Unknown reduction: {reduction}")


def masked_batch_mean(tensor: torch.Tensor, mask: torch.Tensor, ignore_empty: bool = True) -> torch.Tensor:
    tensor = tensor.masked_fill(~mask, 0.)

    if tensor.ndim == 3:
        num = tensor.sum(dim=-1).sum(dim=-1)
        den = mask.sum(dim=-1).sum(dim=-1)
    else:
        num = tensor.sum(dim=-1)
        den = mask.sum(dim=-1)

    if ignore_empty:
        tensor = num[den != 0.] / den[den != 0.]
    else:
        tensor = num / den.clamp(min=1e-5)

    return tensor.mean()


def accuracy(
        predictions: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor | None = None,
        reduction: str | Reduction = Reduction.NONE
) -> torch.Tensor:
    values = (predictions == labels).float()
    return reduce_metrics(values, mask, reduction)


def accuracy_topk(
        logits: torch.Tensor,
        labels: torch.Tensor,
        k: int,
        mask: torch.Tensor | None = None,
        reduction: str | Reduction = Reduction.NONE
) -> torch.Tensor:
    _, predictions_k = torch.topk(logits, k, dim=-1)
    values = (predictions_k == labels[..., None]).any(dim=1).float()
    return reduce_metrics(values, mask, reduction)


def distance(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
        reduction: str | Reduction = Reduction.NONE
) -> torch.Tensor:
    values = (predictions - targets).abs().float()
    return reduce_metrics(values, mask, reduction)


def weighted_distance(
        probs: torch.Tensor,
        targets: torch.Tensor,
        token_values: torch.Tensor,
        mask: torch.Tensor | None = None,
        reduction: str | Reduction = Reduction.NONE
) -> torch.Tensor:
    values = ((targets[..., None] - token_values.to(targets.device)[None, None, :]).abs() * probs).sum(dim=-1)
    return reduce_metrics(values, mask, reduction)


def emd_cdf_loss(
        probs: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor | None = None,
        reduction: str | Reduction = Reduction.NONE
) -> torch.Tensor:
    cdf_pred = torch.cumsum(probs, dim=-1)

    target_one_hot = F.one_hot(torch.clamp(labels, min=0), num_classes=cdf_pred.shape[-1]).float()
    cdf_target = torch.cumsum(target_one_hot, dim=-1)
    cdf_target[labels < 0] = 0.

    diff_cdfs = torch.abs(cdf_pred[..., :-1] - cdf_target[..., :-1])
    emd_losses = diff_cdfs.mean(dim=-1)

    return reduce_metrics(emd_losses, mask, reduction)


def mean_reciprocal_rank(
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor | None = None,
        reduction: str | Reduction = Reduction.NONE
) -> tuple[torch.Tensor, torch.Tensor]:
    _, sorted_indices = torch.sort(logits, dim=-1, descending=True)

    # create rank positions: 1-indexed ranks for each class
    ranks = torch.arange(
        1, sorted_indices.shape[1] + 1, device=sorted_indices.device
    ).unsqueeze(0).expand(sorted_indices.shape[0], -1)

    # replace non-matching positions with a large number so that min() finds the correct rank
    rr = torch.where(sorted_indices == labels[:, None], ranks, torch.tensor(1e5, device=ranks.device))

    # get the minimum rank per sample (i.e. first occurrence of the correct label)
    first_ranks = rr.min(dim=1)[0].float()

    # compute the reciprocal rank for each sample and take the mean
    mrr = 1.0 / first_ranks

    return reduce_metrics(first_ranks, mask, reduction), reduce_metrics(mrr, mask, reduction)
