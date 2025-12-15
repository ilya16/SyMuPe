"""
ScorePerformer model.

Combines PerformanceDecoder with ScoreEncoder and PerformanceEncoder TupleTransformers.
"""
from __future__ import annotations

from dataclasses import dataclass, MISSING

import torch
import torch.nn as nn
from omegaconf import DictConfig

from symupe.data.collators import ScorePerformanceInputs
from symupe.data.datasets import ScorePerformanceDataset
from symupe.modules.classes import LanguageModelingMode
from symupe.modules.constructor import ModuleConfig
from symupe.modules.tuple_transformer import (
    TupleTransformerConfig,
    TupleTransformerOutput,
    TupleTransformer
)
from symupe.modules.tuple_transformer import (
    TupleTransformerWrappers,
    TupleTransformerLMWrapper
)
from symupe.utils import asdict
from .mmd_transformer import MMDTupleTransformer, MMDTupleTransformerOutput
from .reversal_classifier import (
    MultiHeadReversalClassifierConfig,
    MultiHeadReversalClassifierOutput,
    MultiHeadReversalClassifier
)
from ..base import Model
from ..classifiers.model import (
    MultiHeadEmbeddingClassifierConfig,
    MultiHeadEmbeddingClassifier,
    MultiHeadEmbeddingClassifierOutput
)


# ScorePerformer model

@dataclass
class ScorePerformerConfig(ModuleConfig):
    num_tokens: dict[str, int] = MISSING
    dim: int = MISSING
    perf_decoder: TupleTransformerConfig = MISSING
    score_encoder: TupleTransformerConfig | None = None
    perf_encoder: TupleTransformerConfig | None = None
    classifiers: DictConfig | MultiHeadEmbeddingClassifierConfig | None = None
    reversal_classifiers: DictConfig | MultiHeadReversalClassifierConfig | None = None
    tie_token_emb: bool = False
    tie_token_emb_full: bool = False
    tie_encoders: bool = False
    mode: str | None = None
    num_score_tokens: dict[str, int] | None = None


@dataclass
class ScorePerformerEncoderOutput:
    score_embeddings: torch.Tensor | None = None
    score_mask: torch.Tensor | None = None
    perf_embeddings: torch.Tensor | None = None
    score_encoder: TupleTransformerOutput | None = None
    perf_encoder: MMDTupleTransformerOutput | None = None


@dataclass
class ScorePerformerOutput:
    perf_decoder: TupleTransformerOutput
    score_encoder: TupleTransformerOutput | None = None
    perf_encoder: MMDTupleTransformerOutput | None = None
    classifiers: MultiHeadEmbeddingClassifierOutput | None = None
    reversal_classifiers: MultiHeadReversalClassifierOutput | None = None
    loss: torch.Tensor | None = None
    losses: dict[str, torch.Tensor] | None = None


class ScorePerformer(Model):
    def __init__(
            self,
            num_tokens: dict[str, int],
            dim: int,
            perf_decoder: DictConfig | TupleTransformerConfig,
            score_encoder: DictConfig | TupleTransformerConfig | None = None,
            perf_encoder: DictConfig | TupleTransformerConfig | None = None,
            classifiers: DictConfig | MultiHeadEmbeddingClassifierConfig | None = None,
            reversal_classifiers: DictConfig | MultiHeadReversalClassifierConfig | None = None,
            tie_token_emb: bool = False,
            tie_token_emb_full: bool = False,
            tie_encoders: bool = False,
            mode: str | None = None,
            num_score_tokens: dict[str, int] | None = None
    ):
        super().__init__()

        self.score_encoder = None
        if score_encoder is not None:
            self.score_encoder = TupleTransformer.init(
                score_encoder,
                num_tokens=num_score_tokens or num_tokens,
                dim=score_encoder.get("dim", dim),
                lm_head=None
            )

        self.perf_encoder = None
        if perf_encoder is not None:
            self.perf_encoder = MMDTupleTransformer.init(
                perf_encoder,
                num_tokens=num_tokens,
                dim=score_encoder.get("dim", dim),
                lm_head=None
            )

        if tie_encoders:
            assert self.score_encoder is not None and self.perf_encoder is not None
            self.score_encoder.token_emb = self.perf_encoder.token_emb
            self.score_encoder.transformer = self.perf_encoder.transformer

        self.classifiers = None
        if classifiers is not None and classifiers.num_classes is not None:
            assert self.perf_encoder is not None
            self.classifiers = MultiHeadEmbeddingClassifier.init(
                classifiers,
                input_dim=self.perf_encoder.embedding_dim
            )

        self.reversal_classifiers = None
        if reversal_classifiers is not None and reversal_classifiers.num_classes is not None:
            assert self.perf_encoder is not None and self.score_encoder is not None
            self.reversal_classifiers = MultiHeadReversalClassifier.init(
                reversal_classifiers,
                input_dim=self.perf_encoder.embedding_dim
            )

        perf_decoder.transformer.cross_attend = self.score_encoder is not None
        context_emb_dim = None if self.score_encoder is None else self.score_encoder.dim
        style_emb_dim = None if self.perf_encoder is None else self.perf_encoder.embedding_dim

        self.perf_decoder = TupleTransformer.init(
            perf_decoder,
            num_tokens=num_tokens,
            dim=dim,
            context_embedding_dim=context_emb_dim,
            style_embedding_dim=style_emb_dim
        )

        if tie_token_emb_full:
            if self.score_encoder is not None:
                assert (len(self.score_encoder.token_emb.embs) == len(self.perf_decoder.token_emb.embs)
                        and self.score_encoder.token_emb.total_emb_dim == self.perf_decoder.token_emb.total_emb_dim)
                self.score_encoder.token_emb = self.perf_decoder.token_emb
            if self.perf_encoder is not None:
                assert (len(self.perf_encoder.token_emb.embs) == len(self.perf_decoder.token_emb.embs)
                        and self.perf_encoder.token_emb.total_emb_dim == self.perf_decoder.token_emb.total_emb_dim)
                self.perf_encoder.token_emb = self.perf_decoder.token_emb
        elif tie_token_emb:
            for key, emb in self.perf_decoder.token_emb.embs.items():
                if self.score_encoder is not None and key in self.score_encoder.token_emb.embs:
                    self.score_encoder.token_emb.embs[key] = self.perf_decoder.token_emb.embs[key]
                if self.perf_encoder is not None and key in self.perf_encoder.token_emb.embs:
                    self.perf_encoder.token_emb.embs[key] = self.perf_decoder.token_emb.embs[key]

        self.num_tokens = num_tokens
        self.num_score_tokens = num_score_tokens or num_tokens

        self.mode = mode
        if self.mode == LanguageModelingMode.CLM:
            self.prepare_for_clm()
        elif self.mode == LanguageModelingMode.MixedLM:
            self.prepare_for_mixlm()

    def _prepare_for_lm(self, mode: LanguageModelingMode | str | None = None) -> ScorePerformer:
        if mode is None:
            if isinstance(self.perf_decoder, TupleTransformerLMWrapper):
                self.perf_decoder = self.perf_decoder.model
        else:
            if isinstance(self.perf_decoder, TupleTransformer):
                self.perf_decoder = TupleTransformerWrappers[mode](self.perf_decoder)
            elif isinstance(self.perf_decoder, TupleTransformerLMWrapper):
                self.perf_decoder = TupleTransformerWrappers[mode](self.perf_decoder.model)
        self.mode = mode
        return self

    def prepare_for_clm(self) -> ScorePerformer:
        return self._prepare_for_lm(mode=LanguageModelingMode.CLM)

    def prepare_for_mixlm(self) -> ScorePerformer:
        return self._prepare_for_lm(mode=LanguageModelingMode.MixedLM)

    def unwrap_model(self) -> ScorePerformer:
        return self._prepare_for_lm(mode=None)

    @property
    def unwrapped_decoder(self):
        if isinstance(self.transformer, TupleTransformerLMWrapper):
            return self.perf_decoder.model
        return self.perf_decoder

    def forward_encoders(
            self,
            perf: torch.Tensor | None = None,
            perf_values: torch.Tensor | None = None,
            perf_mask: torch.Tensor | None = None,
            score: torch.Tensor | None = None,
            score_values: torch.Tensor | None = None,
            score_mask: torch.Tensor | None = None,
            bars: torch.Tensor | None = None,
            beats: torch.Tensor | None = None,
            onsets: torch.Tensor | None = None,
            deadpan_mask: torch.Tensor | None = None,
            compute_loss: bool = True
    ) -> ScorePerformerEncoderOutput:
        score_emb = perf_emb = None
        score_enc_out = perf_enc_out = None

        if self.score_encoder is not None:
            score_enc_out = self.score_encoder(
                tokens=score, values=score_values, mask=score_mask
            )
            score_emb = score_enc_out.hidden_state

        if self.perf_encoder is not None:
            perf_enc_out = self.perf_encoder(
                tokens=perf, values=perf_values, mask=perf_mask,
                bars=bars, beats=beats, onsets=onsets,
                deadpan_mask=deadpan_mask,
                compute_loss=compute_loss
            )
            perf_emb = perf_enc_out.embeddings

        return ScorePerformerEncoderOutput(
            score_embeddings=score_emb,
            score_mask=score_mask,
            perf_embeddings=perf_emb,
            score_encoder=score_enc_out,
            perf_encoder=perf_enc_out
        )

    def forward(
            self,
            perf: torch.Tensor,
            perf_values: torch.Tensor | None = None,
            perf_mask: torch.Tensor | None = None,
            score: torch.Tensor | None = None,
            score_values: torch.Tensor | None = None,
            score_mask: torch.Tensor | None = None,
            noisy_perf: torch.Tensor | None = None,
            noisy_perf_values: torch.Tensor | None = None,
            noisy_perf_mask: torch.Tensor | None = None,
            masked_perf: torch.Tensor | None = None,
            masked_perf_values: torch.Tensor | None = None,
            labels: torch.Tensor | None = None,
            targets: torch.Tensor | None = None,
            bars: torch.Tensor | None = None,
            beats: torch.Tensor | None = None,
            onsets: torch.Tensor | None = None,
            directions: torch.Tensor | None = None,
            perf_context_mask: torch.Tensor | None = None,
            deadpan_mask: torch.Tensor | None = None
    ) -> ScorePerformerOutput:
        enc_out = self.forward_encoders(
            perf=noisy_perf if noisy_perf is not None else perf,
            perf_values=noisy_perf_values if noisy_perf_values is not None else perf_values,
            perf_mask=noisy_perf_mask if noisy_perf_mask is not None else perf_mask,
            score=score,
            score_values=score_values,
            score_mask=score_mask,
            bars=bars,
            beats=beats,
            onsets=onsets,
            deadpan_mask=deadpan_mask
        )

        perf_dec_out = self.perf_decoder(
            tokens=perf,
            values=perf_values,
            mask=perf_mask,
            context=enc_out.score_embeddings,
            context_mask=enc_out.score_mask,
            style_embeddings=enc_out.perf_embeddings,
            labels=labels,
            targets=targets,
            masked_tokens=masked_perf,
            masked_values=masked_perf_values
        )
        loss, losses = perf_dec_out.loss, perf_dec_out.losses

        if enc_out.perf_encoder is not None:
            if enc_out.perf_encoder.loss is not None:
                loss += enc_out.perf_encoder.loss
                losses.update(**enc_out.perf_encoder.losses)

        clf_out = None
        if self.classifiers is not None:
            clf_mask = perf_mask if deadpan_mask is None else perf_mask & (~deadpan_mask[:, None])
            clf_out = self.classifiers(
                embeddings=enc_out.perf_encoder.full_embeddings[clf_mask],
                labels=directions[clf_mask]
            )
            if clf_out.loss is not None:
                loss += clf_out.loss
                losses.update(**clf_out.losses)

        rev_clf_out = None
        if self.reversal_classifiers is not None:
            rev_labels = []
            for i, key in enumerate(self.perf_encoder.token_emb.embs.keys()):
                if key in self.reversal_classifiers.heads:
                    rev_labels.append(score[..., i])
            rev_clf_out = self.reversal_classifiers(
                embeddings=enc_out.perf_encoder.full_embeddings[perf_mask],
                labels=torch.stack(rev_labels, dim=-1)[perf_mask]
            )
            if rev_clf_out.loss is not None:
                loss += rev_clf_out.loss
                losses.update(**rev_clf_out.losses)

        return ScorePerformerOutput(
            perf_decoder=perf_dec_out,
            score_encoder=enc_out.score_encoder,
            perf_encoder=enc_out.perf_encoder,
            classifiers=clf_out,
            reversal_classifiers=rev_clf_out,
            loss=loss,
            losses=losses
        )

    def prepare_inputs(
            self,
            inputs: dict | ScorePerformanceInputs,
            ema_model: nn.Module | None = None
    ) -> dict[str, torch.Tensor]:
        if isinstance(inputs, ScorePerformanceInputs):
            inputs = asdict(inputs)

        inputs_dict = {
            "perf": inputs["performances"]["tokens"],
            "perf_values": inputs["performances"]["values"],
            "perf_mask": inputs["performances"]["mask"],
            "score": inputs["scores"]["tokens"],
            "score_values": inputs["performances"]["values"],
            "score_mask": inputs["scores"]["mask"],

            "perf_context_mask": inputs["performances"]["context_mask"]
        }

        if inputs.get("labels", None):
            inputs_dict["labels"] = inputs["labels"]["tokens"]
            inputs_dict["targets"] = inputs["labels"]["values"]

        if inputs.get("noisy_performances", None):
            inputs_dict["noisy_perf"] = inputs["noisy_performances"]["tokens"]
            inputs_dict["noisy_perf_values"] = inputs["noisy_performances"]["values"]
            inputs_dict["noisy_perf_mask"] = inputs["noisy_performances"]["mask"]

        if inputs.get("masked_performances", None):
            inputs_dict["masked_perf"] = inputs["masked_performances"]["tokens"]
            inputs_dict["masked_perf_values"] = inputs["masked_performances"]["values"]

        if inputs.get("segments", None):
            inputs_dict["bars"] = inputs["segments"]["bar"]
            inputs_dict["beats"] = inputs["segments"]["beat"]
            inputs_dict["onsets"] = inputs["segments"]["onset"]

        inputs_dict["directions"] = inputs.get("directions", None)
        inputs_dict["deadpan_mask"] = inputs.get("deadpan_mask", None)

        return inputs_dict

    @staticmethod
    def inject_data_config(
            config: DictConfig | ScorePerformerConfig | None,
            dataset: ScorePerformanceDataset | None
    ) -> DictConfig | ModuleConfig | None:
        assert isinstance(dataset, ScorePerformanceDataset)

        config["num_tokens"] = dataset.performance_token_sizes
        config["num_score_tokens"] = dataset.score_token_sizes

        for module in ["score_encoder", "perf_encoder", "perf_decoder"]:
            if config.get(module) is not None:
                token_emb_cfg = config[module]["token_embeddings"]
                token_emb_cfg["token_values"] = {
                    name: value.tolist() for name, value in dataset.tokenizer.token_values(
                        normalize=dataset.normalize_values
                    ).items()
                }
                token_emb_cfg["special_tokens"] = dataset.tokenizer.special_tokens_dict

        clf_cfg = config.get("classifiers")
        if clf_cfg is not None and dataset.performance_directions is not None:
            clf_cfg["num_classes"] = dict(dataset.performance_direction_sizes)
            clf_cfg["class_samples"] = dict(dataset.get_direction_class_weights()[1])

        rev_clf_cfg = config.get("reversal_classifiers")
        if rev_clf_cfg is not None and rev_clf_cfg.get("num_classes") is None:
            assert "_token_types_" in rev_clf_cfg
            rev_clf_cfg["num_classes"] = {
                key: num for key, num in dataset.performance_token_sizes.items()
                if key in rev_clf_cfg["_token_types_"]
            }
            del rev_clf_cfg["_token_types_"]

        return config

    @staticmethod
    def cleanup_config(
            config: DictConfig | ScorePerformerConfig | None,
    ) -> DictConfig | ModuleConfig | None:
        for key in ["score_encoder", "perf_encoder", "perf_decoder"]:
            if config.get(key) is not None:
                config[key]["token_embeddings"].pop("token_values", None)

        if config.get("classifiers") is not None:
            config["classifiers"].pop("class_samples", None)

        return config
