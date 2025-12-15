# Adapted from https://github.com/Tomiinek/Multilingual_Text_to_Speech/blob/master/modules/classifier.py
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import MISSING

from symupe.modules.constructor import ModuleConfig, Constructor


class GradientReversalFunction(torch.autograd.Function):
    """Revert gradient without any further input modification."""

    @staticmethod
    def forward(ctx, x, l, c):
        ctx.l = l
        ctx.c = c
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.clamp(-ctx.c, ctx.c)
        return ctx.l * grad_output.neg(), None, None


@dataclass
class ReversalClassifierOutput:
    logits: torch.Tensor = None
    loss: torch.Tensor | None = None


@dataclass
class ReversalClassifierConfig(ModuleConfig):
    input_dim: int = MISSING
    num_classes: int = MISSING
    hidden_dims: list[int] | None = None
    grad_clip_threshold: float = 0.25
    scale_factor: float = 1.


class ReversalClassifier(nn.Module, Constructor):
    """Adversarial classifier (with two FC layers) with a gradient reversal layer.

    Params:
        input_dim -- size of the input layer (probably should match the output size of encoder)
        hidden_dim -- size of the hidden layer
        output_dim -- number of channels of the output (probably should match the number of speakers/languages)
        grad_clip_threshold (float) -- maximal value of the gradient which flows from this module
        scale_factor (float, default: 1.0)-- scale multiplier of the reversed gradients
    """

    def __init__(
            self,
            input_dim: int,
            num_classes: int,
            hidden_dims: list[int] | None = None,
            grad_clip_threshold: float = 0.25,
            scale_factor: float = 1.
    ):
        super().__init__()
        self._lambda = scale_factor
        self._clipping = grad_clip_threshold

        self.num_classes = num_classes

        hidden_dims = hidden_dims or []
        hidden_dims = [hidden_dims] if isinstance(hidden_dims, int) else hidden_dims
        hidden_dims = list(hidden_dims)

        in_dims = [input_dim] + hidden_dims
        out_dims = hidden_dims + [num_classes]

        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(in_dims) - 1:
                layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x,  labels: torch.Tensor | None = None) -> ReversalClassifierOutput:
        x = GradientReversalFunction.apply(x, self._lambda, self._clipping)
        logits = self.layers(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ReversalClassifierOutput(logits=logits, loss=loss)


@dataclass
class MultiHeadReversalClassifierOutput:
    logits: dict[str, torch.Tensor] = None
    loss: torch.Tensor | None = None
    losses: dict[str, torch.Tensor] | None = None


@dataclass
class MultiHeadReversalClassifierConfig(ModuleConfig):
    input_dim: int = MISSING
    num_classes: dict[str, int] = MISSING
    classifier: ReversalClassifierConfig = MISSING
    loss_weight: float = 0.01


class MultiHeadReversalClassifier(nn.Module, Constructor):
    def __init__(
            self,
            input_dim: int,
            num_classes: dict[str, int],
            classifier: ReversalClassifierConfig,
            loss_weight: float = 0.01
    ):
        super().__init__()

        self.num_classes = num_classes

        self.heads = nn.ModuleDict({})
        for key, num in num_classes.items():
            self.heads[key] = ReversalClassifier.init(
                config=classifier,
                input_dim=input_dim,
                num_classes=num,
            )

        self.loss_weight = loss_weight

    def forward(
            self,
            embeddings: torch.Tensor,
            labels: torch.Tensor | None = None
    ) -> MultiHeadReversalClassifierOutput:
        logits = {}
        loss, losses = 0., {}
        for i, (key, head) in enumerate(self.heads.items()):
            out = head(embeddings, labels=labels[..., i] if labels is not None else None)
            logits[key] = out.logits

            if out.loss:
                key = "rev_clf/" + key
                loss += out.loss
                losses[key] = out.loss

        loss = self.loss_weight * loss / len(self.heads)
        losses["rev_clf"] = loss

        return MultiHeadReversalClassifierOutput(
            logits=logits,
            loss=loss if labels is not None else None,
            losses=losses if labels is not None else None
        )
