""" ScorePerformer metrics evaluator. """
from __future__ import annotations

import torch

from symupe.data.collators import LMScorePerformanceInputs
from symupe.data.tokenizers import SyMuPeBase
from symupe.modules.tuple_transformer import TupleTransformerOutput
from .model import ScorePerformerOutput
from ..music_transformer.evaluator import MusicTransformerEvaluator


class ScorePerformerEvaluator(MusicTransformerEvaluator):
    def __init__(
            self,
            model,
            tokenizer: SyMuPeBase,
            label_pad_token_id: int = -100,
            normalized_targets: bool = True,
            ignore_keys: list[str] | None = None
    ):
        super().__init__(model, tokenizer, label_pad_token_id, normalized_targets, ignore_keys)

    @torch.no_grad()
    def __call__(
            self,
            inputs: dict | LMScorePerformanceInputs,
            outputs: TupleTransformerOutput | ScorePerformerOutput,
            ignore_keys: list[str] | None = None
    ) -> dict[str, torch.Tensor]:
        return super().__call__(inputs=inputs, outputs=outputs.perf_decoder, ignore_keys=ignore_keys)
