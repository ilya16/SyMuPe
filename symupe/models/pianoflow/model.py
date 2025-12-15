"""
PianoFlow model based on the CFMMusicTransformer.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from omegaconf import DictConfig

from symupe.data.tokenizers import SyMuPe
from symupe.modules.tuple_transformer import TupleTransformerConfig
from ..music_transformer.model import CFMMusicTransformerConfig, CFMMusicTransformerOutput, CFMMusicTransformer


@dataclass
class PianoFlowConfig(CFMMusicTransformerConfig):
    ...


@dataclass
class PianoFlowOutput(CFMMusicTransformerOutput):
    ...


class PianoFlow(CFMMusicTransformer):
    def __init__(
            self,
            num_tokens: dict[str, int],
            dim: int,
            transformer: DictConfig | TupleTransformerConfig,
            context_num_tokens: dict[str, int] | None = None,
            score_num_tokens: dict[str, int] | None = None,
            context_vectors: bool = False,
            value_mean: list[float] | dict[str, int] | None = None,
            value_std: list[float] | dict[str, int] | None = None,
            value_log: list[str] | None = None,
            value_keys: list[str] | None = None,
            tokenizer: SyMuPe | None = None,
            wrapper_kwargs: dict | None = None
    ):
        super().__init__(
            num_tokens=num_tokens,
            dim=dim,
            transformer=transformer,
            context_num_tokens=context_num_tokens,
            score_num_tokens=score_num_tokens,
            pedal_stream=False,
            context_vectors=context_vectors,
            value_mean=value_mean,
            value_std=value_std,
            value_log=value_log,
            value_keys=value_keys,
            tokenizer=tokenizer,
            wrapper_kwargs=wrapper_kwargs
        )

    def forward(
            self,
            tokens: torch.Tensor,
            values: torch.Tensor,
            vectors: torch.Tensor,
            mask: torch.Tensor | None = None,

            context: torch.Tensor | None = None,
            context_mask: torch.Tensor | None = None,
            context_tokens: torch.Tensor | None = None,
            context_values: torch.Tensor | None = None,
            score_tokens: torch.Tensor | None = None,
            score_values: torch.Tensor | None = None,

            labels: torch.Tensor | None = None,
            targets: torch.Tensor | None = None,

            type_ids: torch.Tensor | None = None,
            task_ids: torch.Tensor | None = None,
            task_tokens: torch.Tensor | None = None,
            return_cache: bool = False,
            output_layer: int | None = None,
            ema_model: nn.Module | None = None,
            **kwargs
    ):
        return super().forward(
            tokens=tokens, values=values, vectors=vectors, mask=mask,
            context=context, context_mask=context_mask,
            context_tokens=context_tokens, context_values=context_values,
            score_tokens=score_tokens, score_values=score_values,
            labels=labels, targets=targets,
            type_ids=type_ids, task_ids=task_ids, task_tokens=task_tokens,
            return_cache=return_cache, output_layer=output_layer, ema_model=ema_model
        )

    def generate(
            self,
            tokens: torch.Tensor,
            values: torch.Tensor,
            tokenizer: SyMuPe,

            steps: int = 4,
            step_factor: float = 1.,
            method: str | None = None,
            x_0: torch.Tensor | None = None,

            context: torch.Tensor | None = None,
            context_mask: torch.Tensor | None = None,
            context_tokens: torch.Tensor | None = None,
            context_values: torch.Tensor | None = None,
            context_scale: float = 1.,
            pedals: torch.Tensor | None = None,

            disable_tqdm: bool = False,
            return_intermediates: bool = False,
            **kwargs
    ):
        return super().generate(
            tokens=tokens, values=values, tokenizer=tokenizer,
            steps=steps, step_factor=step_factor, method=method, x_0=x_0,
            context=context, context_mask=context_mask,
            context_tokens=context_tokens, context_values=context_values,
            context_scale=context_scale, pedals=pedals,
            disable_tqdm=disable_tqdm, return_intermediates=return_intermediates
        )
