""" Transformer Normalization layers. """
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, dim: int, bias: bool = False, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_buffer("bias", torch.zeros(dim), persistent=False)

    def forward(self, x) -> torch.Tensor:
        return F.layer_norm(x, x.shape[-1:], self.weight, self.bias, eps=self.eps)

    def extra_repr(self) -> str:
        return f"({self.dim},), bias={self.bias.requires_grad}"


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, dim: int, condition_dim: int, bias: bool = False, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.weight = nn.Linear(condition_dim, dim)
        nn.init.zeros_(self.weight.weight)
        nn.init.ones_(self.weight.bias)

        self.bias = None
        if bias:
            self.bias = nn.Linear(condition_dim, dim)
            nn.init.zeros_(self.bias.weight)
            nn.init.zeros_(self.bias.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        condition = condition.unsqueeze(1) if condition.ndim == 2 else condition

        weight = self.weight(condition)
        bias = self.bias(condition) if self.bias is not None else x.new_zeros(1)

        return weight * F.layer_norm(x, x.shape[-1:], None, None, eps=self.eps) + bias

    def extra_repr(self) -> str:
        return f"bias={self.bias is not None}"


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim,
        unit_offset = True
    ):
        super().__init__()
        self.unit_offset = unit_offset
        self.scale = dim ** 0.5

        self.gamma = nn.Parameter(torch.zeros(dim))
        nn.init.constant_(self.gamma, 1. - float(unit_offset))

    def forward(self, x) -> torch.Tensor:
        gamma = self.gamma + float(self.unit_offset)
        return F.normalize(x, dim = -1) * self.scale * gamma


class AdaptiveRMSNorm(nn.Module):
    def __init__(self, dim: int, condition_dim: int):
        super().__init__()
        self.scale = dim ** 0.5

        self.gamma = nn.Linear(condition_dim, dim)
        nn.init.zeros_(self.gamma.weight)

    def forward(self, x, condition: torch.Tensor) -> torch.Tensor:
        condition = condition.unsqueeze(1) if condition.ndim == 2 else condition

        normed = F.normalize(x, dim = -1)
        gamma = self.gamma(condition)
        return normed * self.scale * (gamma + 1.)