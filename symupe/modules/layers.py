from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .constructor import Constructor, ModuleConfig
from .transformer import SinusoidalEmbedding, LearnedSinusoidalEmbedding


class ScaledTanh(nn.Module):
    def __init__(self, min_value: float, max_value: float):
        super().__init__()
        self.min = min_value
        self.max = max_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = (self.max - self.min) / 2.0
        return self.min + (torch.tanh(x) + 1) * scale

    def extra_repr(self) -> str:
        return f"min={self.min}, max={self.max}"


@dataclass
class TimePositionalEmbeddingConfig(ModuleConfig):
    freq_dim: int = 256
    emb_dim: int = 512
    freq_scale: int = 1000.0
    learned: bool = False
    with_steps: bool = False


class TimePositionalEmbedding(nn.Module, Constructor):
    def __init__(
        self,
        freq_dim: int = 256,
        emb_dim: int = 512,
        theta: int = 1000.0,
        freq_scale: int = 1000.0,
        learned: bool = False,
        with_steps: bool = False,
    ):
        super().__init__()

        emb_cls = LearnedSinusoidalEmbedding if learned else SinusoidalEmbedding
        self.freq_emb = emb_cls(
            freq_dim,
            theta=theta,
            freq_scale=freq_scale,
            scale=False,
            with_positions=with_steps,
        )

        self.mlp = nn.Sequential(
            nn.Linear(freq_dim + int(with_steps), emb_dim, bias=True),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freq_emb = self.freq_emb(x)
        return self.mlp(freq_emb)
