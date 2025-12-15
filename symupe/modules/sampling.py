""" Sampling functions. """
from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F


# nucleus

def top_p_filtering(
        logits: torch.Tensor,
        threshold: float = 0.9,
        filter_value: float = -float("inf")
) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > threshold
    sorted_indices_to_remove = F.pad(sorted_indices_to_remove, (1, -1), value=False)

    sorted_logits[sorted_indices_to_remove] = filter_value
    return sorted_logits.scatter(-1, sorted_indices, sorted_logits)


# topk

def top_k_filtering(
        logits: torch.Tensor,
        threshold: float = 0.1,
        k: int | None = None,
        filter_value: float = -float("inf")
) -> torch.Tensor:
    k = k or math.ceil(threshold * logits.shape[-1])
    val, ind = torch.topk(logits, k, dim=-1)
    logits = torch.full_like(logits, filter_value)
    logits.scatter_(-1, ind, val)
    return logits


# top_a

def top_a_filtering(
        logits: torch.Tensor,
        min_p_pow: float = 2.0,
        min_p_ratio: float = 0.02,
        filter_value: float = -float("inf")
) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    limit = torch.pow(torch.max(probs), min_p_pow) * min_p_ratio
    return torch.where(probs < limit, filter_value, logits)


# top_k + top_p
def filter_logits(
        logits: torch.Tensor,
        top_k: int | float = 0,
        top_p: float = 1.,
        filter_value: float = -float("inf")
) -> torch.Tensor:
    if top_k > 0:
        if isinstance(top_k, float):
            logits = top_k_filtering(logits, threshold=top_k, filter_value=filter_value)
        else:
            logits = top_k_filtering(logits, k=top_k, filter_value=filter_value)

    if top_p < 1.:
        logits = top_p_filtering(logits, threshold=top_p, filter_value=filter_value)

    return logits


# sampling

def filter_and_sample(
        logits: torch.Tensor,
        filter_logits_fn: Callable[any, any] = filter_logits,
        filter_kwargs: dict[str, object] | None = None,
        temperature: float = 1.,
        top_k: float | int = 0,
        top_p: float = 1.,
        sample: bool = True,
) -> torch.Tensor:
    filter_kwargs = filter_kwargs or {}
    if filter_logits_fn == filter_logits:
        filtered_logits = filter_logits_fn(logits, top_k=top_k, top_p=top_p, **filter_kwargs)
    else:
        filtered_logits = filter_logits_fn(logits, **filter_kwargs)

    probs = F.softmax(filtered_logits / temperature, dim=-1)
    if not sample:
        return probs
    if probs.ndim == 3:
        return torch.stack([torch.multinomial(probs[i], 1) for i in range(probs.shape[0])])
    return torch.multinomial(probs, 1)


# flow matching


def cubic_scheduler(t: float | torch.Tensor, a: float = 2.0, b: float = 0.5) -> float | torch.Tensor:
    return -2 * (t ** 3) + 3 * (t ** 2) + a * (t ** 3 - 2 * t ** 2 + t) + b * (t ** 3 - t ** 2)


def cubic_scheduler_derivative(t: float | torch.Tensor, a: float = 2.0, b: float = 0.5) -> float | torch.Tensor:
    return -6 * (t ** 2) + 6 * t + a * (3 * t ** 2 - 4 * t + 1) + b * (3 * t ** 2 - 2 * t)


def x2prob(x: torch.Tensor, dict_size: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(x, num_classes=dict_size)


def sample_p(probs: torch.Tensor) -> torch.Tensor:
    return torch.multinomial(probs.flatten(0, -2), 1, replacement=True).view(
        *probs.shape[:-1]
    )


def sample_cond_pt(p0: torch.Tensor, p1: torch.Tensor, t: torch.Tensor | float) -> torch.Tensor:
    p_t = (1 - cubic_scheduler(t)) * p0 + cubic_scheduler(t) * p1
    return sample_p(p_t)
