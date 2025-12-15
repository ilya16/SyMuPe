""" Seq2SeqMusicTransformer metrics evaluator. """
from __future__ import annotations

import torch

from symupe.data.collators import LMScorePerformanceInputs
from symupe.data.tokenizers import SyMuPeBase
from symupe.modules.tuple_transformer import TupleTransformerOutput, TupleTransformerFMOutput
from .model import Seq2SeqMusicTransformerOutput, Seq2SeqFMMusicTransformerOutput
from ..music_transformer.evaluator import MusicTransformerEvaluator, FMMusicTransformerEvaluator


class Seq2SeqMusicTransformerEvaluator(MusicTransformerEvaluator):
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
            outputs: TupleTransformerOutput | Seq2SeqMusicTransformerOutput,
            ignore_keys: list[str] | None = None
    ) -> dict[str, torch.Tensor]:
        return super().__call__(inputs=inputs, outputs=outputs.decoder, ignore_keys=ignore_keys)


class Seq2SeqFMMusicTransformerEvaluator(FMMusicTransformerEvaluator):
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
            outputs: TupleTransformerFMOutput | Seq2SeqFMMusicTransformerOutput,
            ignore_keys: list[str] | None = None
    ) -> dict[str, torch.Tensor]:
        return super().__call__(inputs=inputs, outputs=outputs.decoder, ignore_keys=ignore_keys)
