""" A set of utility classes and functions used throughout the repository. """

import random
import sys
from collections.abc import Sequence
from dataclasses import is_dataclass

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm


def prob2bool(prob: float) -> bool:
    return random.choices([True, False], weights=[prob, 1 - prob])[0]


def prob_mask_like(shape: Sequence[int], prob: float, device: torch.device) -> torch.Tensor:
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def find_closest(array: np.ndarray, values, return_values: bool = False) -> np.ndarray:
    """Finds indices of the values closest to `values` in a given array."""
    ids = np.searchsorted(array, values, side="left")

    # find indexes where previous index is closer
    arr_values = array[np.minimum(ids, len(array) - 1)]
    prev_values = array[np.maximum(ids - 1, 0)]
    prev_idx_is_less = (ids == len(array)) | (np.fabs(values - prev_values) < np.fabs(values - arr_values))

    if isinstance(ids, np.ndarray):
        ids[prev_idx_is_less] -= 1
    elif prev_idx_is_less:
        ids -= 1

    ids = np.maximum(0, ids)

    if return_values:
        return array[ids]
    else:
        return ids


def tqdm_iterator(iterable, desc=None, position=0, leave=False, file=sys.stdout, **kwargs):
    return tqdm(iterable, desc=desc, position=position, leave=leave, file=file, **kwargs)


def apply(elements, func, tqdm_enabled=True, desc=None):
    """ Apply a given `func` over a list of elements."""
    iterator = tqdm_iterator(elements, desc=desc) if tqdm_enabled else elements
    return [func(e) for e in iterator]


def apply_dict(elements, names, func, tqdm_enabled=True, desc=None):
    """ Apply a given `func` over a list of named elements."""
    elements = zip(elements, names)
    iterator = tqdm_iterator(elements, desc=desc) if tqdm_enabled else elements
    return {name: func(e) for e, name in iterator}


def asdict(data):
    if is_dataclass(data):
        return {k: asdict(v) for k, v in vars(data).items()}
    return data


def count_parameters(model: nn.Module, requires_grad: bool = False):
    if requires_grad:
        return sum(map(lambda p: p.numel() if p.requires_grad else 0, model.parameters()))
    return sum(map(lambda p: p.numel(), model.parameters()))


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def forward_fill(arr: torch.Tensor, nan_value: float | torch.Tensor) -> torch.Tensor:
    if isinstance(arr, torch.Tensor):
        idx = torch.where(arr != nan_value, torch.arange(len(arr)), torch.tensor(0, device=arr.device))
        idx = torch.cummax(idx, dim=0).values
    else:
        idx = np.where(arr != nan_value, np.arange(len(arr)), 0)
        np.maximum.accumulate(idx, axis=0, out=idx)
    out = arr[idx]
    return out


def backward_fill(arr: torch.Tensor, nan_value: float | torch.Tensor) -> torch.Tensor:
    return forward_fill(arr[::-1], nan_value=nan_value)[::-1]


def trim_mean(data: np.ndarray, tails: float = 0.05) -> np.ndarray:
    k = int(tails * len(data))

    if k > 0:
        sorted_data = np.sort(data)
        data = sorted_data[k:-k]

    return np.mean(data)


def fill_by_mask_and_indices(
        tensor: torch.Tensor,
        new_tensor: torch.Tensor,
        mask: torch.Tensor,
        dims: torch.Tensor | None = None
) -> torch.Tensor:
    if dims is not None:
        tensor_slice = tensor[..., dims]
        tensor_slice[mask] = new_tensor[mask]
        tensor[..., dims] = tensor_slice
    else:
        tensor[mask] = new_tensor[mask]

    return tensor
