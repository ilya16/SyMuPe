""" Transformer FeedForward layer. """
from dataclasses import dataclass

import torch.nn as nn

from ..constructor import Constructor, ModuleConfig


class GLU(nn.Module):
    def __init__(self, activation: nn.Module, dim: int = -1):
        super().__init__()
        self.activation = activation
        self.dim = dim

    def forward(self, x):
        x, gate = x.chunk(2, dim=self.dim)
        return x * self.activation(gate)


@dataclass
class FeedForwardConfig(ModuleConfig):
    dim: int = 512
    mult: float = 4
    glu: bool = False
    swish: bool = False
    post_act_ln: bool = False
    dropout: float = 0.
    bias: bool = False


class FeedForward(nn.Module, Constructor):
    def __init__(
            self,
            dim: int = 512,
            mult: float = 4,
            glu: bool = False,
            swish: bool = False,
            post_act_ln: bool = False,
            dropout: float = 0.,
            bias: bool = False
    ):
        super().__init__()

        inner_dim = int(dim * mult)
        activation = nn.SiLU() if swish else nn.GELU()

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim * (1 + int(glu)), bias=bias),
            GLU(activation) if glu else activation,
            nn.LayerNorm(inner_dim) if post_act_ln else nn.Identity(),
            nn.Dropout(dropout) if dropout > 0. else nn.Identity(),
            nn.Linear(inner_dim, dim, bias=bias)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
