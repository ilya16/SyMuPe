"""
A family of Sequence-to-Sequence Music Transformer models.

Each model combines oEncoder and Decoder TupleTransformers
with a causal language modeling or flow matching wrapper for the Decoder.

Available models:
    - Seq2SeqMusicTransformer
    - Seq2SeqDFMMusicTransformer
    - Seq2SeqFMMusicTransformer
"""
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, MISSING

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from symupe.data.collators import Seq2SeqInputs
from symupe.data.datasets import SequenceDataset
from symupe.data.tokenizers import SyMuPe
from symupe.data.tokenizers.constants import SPECIAL_TOKENS_VALUE
from symupe.modules.classes import LanguageModelingMode, ModelWrapper
from symupe.modules.constructor import ModuleConfig
from symupe.modules.tuple_transformer import (
    TupleTransformerConfig, TupleTransformerOutput, TupleTransformer, TupleTransformerCache,
    TupleTransformerARWrapper,
    TupleTransformerDFMWrapper, DFMIntermediates,
    TupleTransformerFMWrapper, TupleTransformerFMOutput, FMIntermediates
)
from symupe.utils import asdict
from ..base import Model


# Seq2SeqMusicTransformer model

@dataclass
class Seq2SeqMusicTransformerConfig(ModuleConfig):
    num_tokens: dict[str, int] = MISSING
    dim: int = MISSING
    encoder: TupleTransformerConfig = MISSING
    decoder: TupleTransformerConfig = MISSING
    context_num_tokens: dict[str, int] | None = None
    score_num_tokens: dict[str, int] | None = None
    tie_token_emb: bool = False
    context_with_memory: bool = False
    token_keys: list[str] | None = None
    value_keys: list[str] | None = None
    wrapper_kwargs: dict | None = None


@dataclass
class Seq2SeqMusicTransformerOutput:
    encoder: TupleTransformerOutput
    decoder: TupleTransformerOutput
    loss: torch.Tensor | None = None
    losses: dict[str, torch.Tensor] | None = None


class _Seq2SeqMusicTransformer(Model):
    def __init__(
            self,
            num_tokens: dict[str, int],
            dim: int,
            encoder: DictConfig | TupleTransformerConfig,
            decoder: DictConfig | TupleTransformerConfig,
            context_num_tokens: dict[str, int] | None = None,
            score_num_tokens: dict[str, int] | None = None,
            tie_token_emb: bool = False,
            context_with_memory: bool = False,
            tokenizer: SyMuPe | None = None,
            token_keys: list[str] | None = None,
            value_keys: list[str] | None = None,
            wrapper_kwargs: dict | None = None
    ):
        super().__init__()

        self.encoder = TupleTransformer.init(
            encoder,
            dim=encoder.get("dim", dim),
            num_tokens=num_tokens,
            lm_head=None
        )

        self.decoder = TupleTransformer.init(
            decoder,
            dim=dim,
            num_tokens=num_tokens,
            context_embedding_dim=self.encoder.dim,
            context_num_tokens=context_num_tokens,
            score_num_tokens=score_num_tokens,
            token_keys=token_keys,
            value_keys=value_keys
        )

        if tie_token_emb:
            self.encoder.token_emb = self.decoder.token_emb
            if self.encoder.task_emb is not None and self.decoder.task_emb is not None:
                self.encoder.task_emb = self.decoder.task_emb

        self.context_with_memory = context_with_memory

        self.num_tokens = num_tokens
        self.tokenizer = tokenizer
        self.token_keys = self.decoder.token_keys
        self.value_keys = self.decoder.value_keys

        self.wrapper_kwargs = wrapper_kwargs or {}

        self.mode = None

    def unwrap_model(self):
        if isinstance(self.decoder, ModelWrapper):
            self.decoder = self.decoder.model
        self.mode = None
        return self

    @property
    def unwrapped_decoder(self):
        if isinstance(self.decoder, ModelWrapper):
            return self.decoder.model
        return self.decoder

    @property
    def unwrapped_transformer(self):
        return self.unwrapped_decoder

    def _build_context(
            self,
            enc_out: TupleTransformerOutput,
            enc_mask: torch.Tensor | None = None,
            dec_context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context, context_mask = enc_out.hidden_state, enc_mask

        if dec_context is not None:
            assert not self.context_with_memory
            context = torch.cat([context, dec_context], dim=-1)

        if self.context_with_memory:
            if context_mask is None:
                context_mask = torch.ones_like(context[..., 0]).bool()

            if enc_out.task_state is not None:
                context = torch.cat((enc_out.task_state, context), dim=1)
                context_mask = F.pad(context_mask, (1, 0), value=True)

            if enc_out.memory_state is not None:
                context = torch.cat((enc_out.memory_state, context), dim=1)
                context_mask = F.pad(context_mask, (enc_out.memory_state.shape[1], 0), value=True)
        return context, context_mask

    @abstractmethod
    def forward(self, *args, **kwargs):
        ...

    @abstractmethod
    def generate(self, *args, **kwargs):
        ...

    def prepare_inputs(
            self,
            inputs: dict | Seq2SeqInputs,
            ema_model: nn.Module | None = None
    ) -> dict[str, torch.Tensor]:
        if isinstance(inputs, Seq2SeqInputs):
            inputs = asdict(inputs)

        inputs_dict = {
            "enc_tokens": inputs["input_sequences"]["tokens"],
            "enc_values": inputs["input_sequences"]["values"],
            "enc_mask": inputs["input_sequences"]["mask"],
            "dec_tokens": inputs["output_sequences"]["tokens"],
            "dec_values": inputs["output_sequences"]["values"],
            "dec_mask": inputs["output_sequences"]["mask"]
        }

        if inputs["context_sequences"] is not None:
            inputs_dict["dec_context_tokens"] = inputs["context_sequences"]["tokens"]
            inputs_dict["dec_context_values"] = inputs["context_sequences"]["values"]

        if inputs["score_sequences"] is not None:
            inputs_dict["dec_score_tokens"] = inputs["score_sequences"]["tokens"]
            inputs_dict["dec_score_values"] = inputs["score_sequences"]["values"]

        if inputs["emotion_embeddings"] is not None:
            inputs_dict["dec_context"] = inputs["emotion_embeddings"]

        if inputs["output_sequences"]["type_ids"] is not None:
            inputs_dict["dec_type_ids"] = inputs["output_sequences"]["type_ids"]

        if inputs.get("labels", None):
            inputs_dict["labels"] = inputs["labels"]["tokens"]
            inputs_dict["targets"] = inputs["labels"]["values"]

        if inputs.get("full_labels", None) is not None:
            inputs_dict["full_labels"] = inputs["full_labels"]["tokens"]

        if inputs.get("task_ids", None) is not None:
            inputs_dict["task_ids"] = inputs["encoding_ids"][:, 1]  # target encoding ids

        if inputs.get("task_sequences", None) is not None:
            inputs_dict["task_tokens"] = inputs["task_sequences"]["tokens"]

        return inputs_dict

    @staticmethod
    def inject_data_config(
            config: DictConfig | Seq2SeqMusicTransformerConfig | None,
            dataset: SequenceDataset | None
    ) -> DictConfig | ModuleConfig | None:
        assert isinstance(dataset, SequenceDataset)

        if isinstance(dataset, SequenceDataset):
            config["num_tokens"] = dataset.token_sizes
            config["context_num_tokens"] = dataset.context_token_sizes
            config["score_num_tokens"] = dataset.score_token_sizes
        else:
            config["num_tokens"] = dataset.performance_token_sizes

        for module in ["encoder", "decoder"]:
            if config.get(module) is not None:
                token_emb_cfg = config[module]["token_embeddings"]
                token_emb_cfg["token_values"] = {
                    name: value.tolist() for name, value in dataset.tokenizer.token_values(
                        normalize=dataset.normalize_values
                    ).items()
                }
                token_emb_cfg["special_tokens"] = dataset.tokenizer.special_tokens_dict

                token_pos_emb_cfg = config[module].get("token_pos_embeddings", None)
                if token_pos_emb_cfg is not None and token_pos_emb_cfg.get("token_dims") is None:
                    assert "_token_types_" in token_pos_emb_cfg

                    token_pos_emb_cfg["token_dims"] = {
                        key: idx for idx, key in enumerate(dataset.token_sizes.keys())
                        if key in token_pos_emb_cfg["_token_types_"]
                    }
                    del token_pos_emb_cfg["_token_types_"]

                if "_task_embedding_" in config[module]:
                    config[module]["num_tasks"] = len(dataset.sequence_encoding_types)
                    del config[module]["_task_embedding_"]

        return config

    @staticmethod
    def cleanup_config(
            config: DictConfig | Seq2SeqMusicTransformerConfig | None,
    ) -> DictConfig | ModuleConfig | None:
        for key in ["encoder", "decoder"]:
            if config.get(key) is not None:
                config[key]["token_embeddings"].pop("token_values", None)

        return config


class Seq2SeqMusicTransformer(_Seq2SeqMusicTransformer):
    def __init__(
            self,
            num_tokens: dict[str, int],
            dim: int,
            encoder: DictConfig | TupleTransformerConfig,
            decoder: DictConfig | TupleTransformerConfig,
            context_num_tokens: dict[str, int] | None = None,
            score_num_tokens: dict[str, int] | None = None,
            tie_token_emb: bool = False,
            context_with_memory: bool = False,
            tokenizer: SyMuPe | None = None,
            token_keys: list[str] | None = None,
            value_keys: list[str] | None = None,
            wrapper_kwargs: dict | None = None
    ):
        super().__init__(
            num_tokens=num_tokens,
            dim=dim,
            encoder=encoder,
            decoder=decoder,
            context_num_tokens=context_num_tokens,
            score_num_tokens=score_num_tokens,
            tie_token_emb=tie_token_emb,
            context_with_memory=context_with_memory,
            tokenizer=tokenizer,
            token_keys=token_keys,
            value_keys=value_keys,
            wrapper_kwargs=wrapper_kwargs
        )

        self.prepare_for_clm()

    def prepare_for_clm(self) -> Seq2SeqMusicTransformer:
        if isinstance(self.decoder, TupleTransformer):
            self.decoder = TupleTransformerARWrapper(self.decoder, token_keys=self.token_keys, **self.wrapper_kwargs)
        self.mode = LanguageModelingMode.CLM
        return self

    def forward(
            self,
            enc_tokens: torch.Tensor,
            dec_tokens: torch.Tensor | None,
            enc_values: torch.Tensor | None = None,
            dec_values: torch.Tensor | None = None,
            enc_mask: torch.Tensor | None = None,
            dec_mask: torch.Tensor | None = None,

            labels: torch.Tensor | None = None,
            targets: torch.Tensor | None = None,
            full_labels: torch.Tensor | None = None,

            dec_context: torch.Tensor | None = None,
            dec_context_tokens: torch.Tensor | None = None,
            dec_context_values: torch.Tensor | None = None,
            dec_score_tokens: torch.Tensor | None = None,
            dec_score_values: torch.Tensor | None = None,

            dec_type_ids: torch.Tensor | None = None,
            task_ids: torch.Tensor | None = None,
            task_tokens: torch.Tensor | None = None
    ) -> Seq2SeqMusicTransformerOutput:
        enc_out = self.encoder(
            tokens=enc_tokens,
            values=enc_values,
            mask=enc_mask,
            task_ids=task_ids,
            task_tokens=task_tokens
        )
        context, context_mask = self._build_context(enc_out, enc_mask, dec_context=dec_context)

        dec_out = self.decoder(
            tokens=dec_tokens,
            values=dec_values,
            mask=dec_mask,
            context=context,
            context_mask=context_mask,
            context_tokens=dec_context_tokens,
            context_values=dec_context_values,
            score_tokens=dec_score_tokens,
            score_values=dec_score_values,
            labels=labels,
            targets=targets,
            full_labels=full_labels,
            type_ids=dec_type_ids,
            task_ids=task_ids,
            task_tokens=task_tokens
        )
        loss, losses = dec_out.loss, dec_out.losses

        return Seq2SeqMusicTransformerOutput(
            encoder=enc_out,
            decoder=dec_out,
            loss=loss,
            losses=losses
        )

    @torch.inference_mode()
    def generate(
            self,
            enc_tokens: torch.Tensor,
            dec_tokens: torch.Tensor | None,
            enc_values: torch.Tensor | None,
            dec_values: torch.Tensor | None,
            tokenizer: SyMuPe,

            dec_context: torch.Tensor | None = None,
            dec_context_tokens: torch.Tensor | None = None,
            dec_context_values: torch.Tensor | None = None,
            dec_score_tokens: torch.Tensor | None = None,
            dec_score_values: torch.Tensor | None = None,

            task_ids: torch.Tensor | None = None,
            task_tokens: torch.Tensor | None = None,
            seq_len: int | None = None,
            temperature: float = 1.,
            top_k: float | int = -1,
            top_p: float = 0.8,
            disable_tqdm: bool = False,
            return_cache: bool = False,
            force_known_tokens: bool = False,
            **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, TupleTransformerCache]:
        self.prepare_for_clm()

        num_dims = len(enc_tokens.shape)
        if num_dims == 2:
            enc_tokens = enc_tokens[None]
            enc_values = enc_values[None] if enc_values is not None else None

        enc_out = self.encoder(
            tokens=enc_tokens,
            values=enc_values
        )
        context, _ = self._build_context(enc_out, enc_mask=None, dec_context=dec_context)

        return self.decoder.generate(
            dec_tokens, values=dec_values,
            known_tokens=enc_tokens if force_known_tokens else None,
            known_values=enc_values if force_known_tokens else None,
            context=context,
            context_tokens=dec_context_tokens, context_values=dec_context_values,
            score_tokens=dec_score_tokens, score_values=dec_score_values,
            task_ids=task_ids, task_tokens=task_tokens,
            seq_len=seq_len or enc_tokens.shape[1] - 1, tokenizer=tokenizer,
            temperature=temperature, top_k=top_k, top_p=top_p,
            disable_tqdm=disable_tqdm, return_cache=return_cache,
            **kwargs
        )


class Seq2SeqDFMMusicTransformer(_Seq2SeqMusicTransformer):
    def __init__(
            self,
            num_tokens: dict[str, int],
            dim: int,
            encoder: DictConfig | TupleTransformerConfig,
            decoder: DictConfig | TupleTransformerConfig,
            context_num_tokens: dict[str, int] | None = None,
            score_num_tokens: dict[str, int] | None = None,
            tie_token_emb: bool = False,
            context_with_memory: bool = False,
            tokenizer: SyMuPe | None = None,
            wrapper_kwargs: dict | None = None
    ):
        super().__init__(
            num_tokens=num_tokens,
            dim=dim,
            encoder=encoder,
            decoder=decoder,
            context_num_tokens=context_num_tokens,
            score_num_tokens=score_num_tokens,
            tie_token_emb=tie_token_emb,
            context_with_memory=context_with_memory,
            tokenizer=tokenizer,
            wrapper_kwargs=wrapper_kwargs
        )

        self.prepare_for_dfm()

    def prepare_for_dfm(self) -> Seq2SeqDFMMusicTransformer:
        if isinstance(self.decoder, TupleTransformer):
            self.decoder = TupleTransformerDFMWrapper(
                self.decoder, tokenizer=self.tokenizer, **self.wrapper_kwargs
            )
        self.mode = LanguageModelingMode.DFM
        return self

    def forward(
            self,
            enc_tokens: torch.Tensor,
            dec_tokens: torch.Tensor | None,
            enc_values: torch.Tensor | None = None,
            dec_values: torch.Tensor | None = None,
            enc_mask: torch.Tensor | None = None,
            dec_mask: torch.Tensor | None = None,

            dec_context: torch.Tensor | None = None,
            dec_context_tokens: torch.Tensor | None = None,
            dec_context_values: torch.Tensor | None = None,
            dec_score_tokens: torch.Tensor | None = None,
            dec_score_values: torch.Tensor | None = None,

            labels: torch.Tensor | None = None,
            targets: torch.Tensor | None = None,
            full_labels: torch.Tensor | None = None,
            task_ids: torch.Tensor | None = None,
            task_tokens: torch.Tensor | None = None
    ) -> Seq2SeqMusicTransformerOutput:
        enc_out = self.encoder(
            tokens=enc_tokens,
            values=enc_values,
            mask=enc_mask,
            task_ids=task_ids,
            task_tokens=task_tokens
        )
        context, context_mask = self._build_context(enc_out, enc_mask, dec_context=dec_context)

        if self.decoder.tokenizer is None:
            self.decoder.set_tokenizer(self.tokenizer)

        dec_out = self.decoder(
            tokens=dec_tokens,
            values=dec_values,
            mask=dec_mask,
            context=context,
            context_mask=context_mask,
            context_tokens=dec_context_tokens,
            context_values=dec_context_values,
            score_tokens=dec_score_tokens,
            score_values=dec_score_values,
            labels=labels,
            targets=targets,
            full_labels=full_labels,
            task_ids=task_ids,
            task_tokens=task_tokens
        )
        loss, losses = dec_out.loss, dec_out.losses

        return Seq2SeqMusicTransformerOutput(
            encoder=enc_out,
            decoder=dec_out,
            loss=loss,
            losses=losses
        )

    @torch.inference_mode()
    def generate(
            self,
            enc_tokens: torch.Tensor,
            dec_tokens: torch.Tensor | None,
            enc_values: torch.Tensor | None,
            dec_values: torch.Tensor | None,
            tokenizer: SyMuPe,
            dec_context: torch.Tensor | None = None,
            dec_context_tokens: torch.Tensor | None = None,
            dec_context_values: torch.Tensor | None = None,
            dec_score_tokens: torch.Tensor | None = None,
            dec_score_values: torch.Tensor | None = None,
            task_ids: torch.Tensor | None = None,
            task_tokens: torch.Tensor | None = None,
            seq_len: int | None = None,
            steps: int = 4,
            step_factor: float = 1.,
            method: str | None = None,
            disable_tqdm: bool = False,
            return_cache: bool = False,
            force_known_tokens: bool = False,
            **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, DFMIntermediates]:
        self.prepare_for_dfm()

        num_dims = len(enc_tokens.shape)
        if num_dims == 2:
            enc_tokens = enc_tokens[None]
            enc_values = enc_values[None] if enc_values is not None else None

        enc_out = self.encoder(
            tokens=enc_tokens,
            values=enc_values
        )
        context, _ = self._build_context(enc_out, enc_mask=None, dec_context=dec_context)

        return self.decoder.generate(
            dec_tokens, values=dec_values,
            known_tokens=enc_tokens if force_known_tokens else None,
            known_values=enc_values if force_known_tokens else None,
            context=context,
            context_tokens=dec_context_tokens, context_values=dec_context_values,
            score_tokens=dec_score_tokens, score_values=dec_score_values,
            task_ids=task_ids, task_tokens=task_tokens,
            seq_len=seq_len or enc_tokens.shape[1] - 1, tokenizer=tokenizer,
            steps=steps, step_factor=step_factor, method=method,
            disable_tqdm=disable_tqdm, return_cache=return_cache,
            **kwargs
        )


@dataclass
class Seq2SeqFMMusicTransformerConfig(Seq2SeqMusicTransformerConfig):
    token_keys: list[str] = ...
    value_keys: list[str] = ...
    value_mean: list[float] | dict[str, int] | None = None
    value_std: list[float] | dict[str, int] | None = None


@dataclass
class Seq2SeqFMMusicTransformerOutput:
    encoder: TupleTransformerOutput
    decoder: TupleTransformerFMOutput
    pred_values: torch.Tensor | None = None
    pred_tokens: torch.Tensor | None = None
    loss: torch.Tensor | None = None
    losses: dict[str, torch.Tensor] | None = None


class Seq2SeqFMMusicTransformer(_Seq2SeqMusicTransformer):
    def __init__(
            self,
            num_tokens: dict[str, int],
            dim: int,
            encoder: DictConfig | TupleTransformerConfig,
            decoder: DictConfig | TupleTransformerConfig,
            token_keys: list[str],
            value_keys: list[str],
            context_num_tokens: dict[str, int] | None = None,
            score_num_tokens: dict[str, int] | None = None,
            tie_token_emb: bool = False,
            context_with_memory: bool = False,
            value_mean: list[float] | dict[str, int] | None = None,
            value_std: list[float] | dict[str, int] | None = None,
            tokenizer: SyMuPe | None = None,
            wrapper_kwargs: dict | None = None
    ):
        self.token_keys = list(token_keys)
        self.value_keys = list(value_keys)

        for key in self.token_keys:
            assert key in num_tokens
        for key in self.value_keys:
            assert key in num_tokens

        assert decoder.lm_head is not None
        decoder.lm_head["num_tokens"] = {key: num for key, num in num_tokens.items() if key in self.token_keys}

        assert decoder.value_head is not None
        decoder.value_head["num_features"] = len(self.value_keys)
        decoder["input_vectors_dim"] = len(self.value_keys)

        super().__init__(
            num_tokens=num_tokens,
            dim=dim,
            encoder=encoder,
            decoder=decoder,
            context_num_tokens=context_num_tokens,
            score_num_tokens=score_num_tokens,
            tie_token_emb=tie_token_emb,
            context_with_memory=context_with_memory,
            tokenizer=tokenizer,
            wrapper_kwargs=wrapper_kwargs
        )

        value_mean = value_mean or [0.]
        if isinstance(value_mean, (dict, DictConfig)):
            value_mean = [value_mean.get(key, 0.) for key in self.value_keys]

        value_std = value_std or [1.]
        if isinstance(value_std, (dict, DictConfig)):
            value_std = [value_std.get(key, 1.) for key in self.value_keys]

        self.register_buffer("value_mean", torch.tensor(value_mean))
        self.register_buffer("value_std", torch.tensor(value_std))

        self.prepare_for_fm()

    def prepare_for_fm(self) -> Seq2SeqFMMusicTransformer:
        if isinstance(self.decoder, TupleTransformer):
            self.decoder = TupleTransformerFMWrapper(
                self.decoder, tokenizer=self.tokenizer,
                token_keys=self.token_keys, value_keys=self.value_keys,
                value_mean=self.value_mean, value_std=self.value_std,
                **self.wrapper_kwargs
            )
        self.mode = LanguageModelingMode.FM
        return self

    @property
    def num_token_features(self) -> int:
        return len(self.token_keys)

    @property
    def num_value_features(self) -> int:
        return len(self.value_keys)

    def forward(
            self,
            enc_tokens: torch.Tensor,
            dec_tokens: torch.Tensor | None,
            dec_vectors: torch.Tensor,
            enc_values: torch.Tensor | None = None,
            dec_values: torch.Tensor | None = None,
            enc_mask: torch.Tensor | None = None,
            dec_mask: torch.Tensor | None = None,

            dec_context: torch.Tensor | None = None,
            dec_context_tokens: torch.Tensor | None = None,
            dec_context_values: torch.Tensor | None = None,
            dec_score_tokens: torch.Tensor | None = None,
            dec_score_values: torch.Tensor | None = None,

            labels: torch.Tensor | None = None,
            targets: torch.Tensor | None = None,
            full_labels: torch.Tensor | None = None,
            task_ids: torch.Tensor | None = None,
            task_tokens: torch.Tensor | None = None
    ) -> Seq2SeqFMMusicTransformerOutput:
        enc_out = self.encoder(
            tokens=enc_tokens,
            values=enc_values,
            mask=enc_mask,
            task_ids=task_ids,
            task_tokens=task_tokens
        )
        context, context_mask = self._build_context(enc_out, enc_mask, dec_context=dec_context)

        if self.decoder.tokenizer is None:
            self.decoder.set_tokenizer(self.tokenizer)

        xv_1 = dec_vectors.clone()
        xv_1 = self.normalize_values(xv_1)
        xv_1[dec_vectors <= SPECIAL_TOKENS_VALUE] = 0.

        dec_out = self.decoder(
            tokens=dec_tokens,
            values=dec_values,
            vectors=xv_1,
            mask=dec_mask,
            context=context,
            context_mask=context_mask,
            context_tokens=dec_context_tokens,
            context_values=dec_context_values,
            score_tokens=dec_score_tokens,
            score_values=dec_score_values,
            labels=labels,
            targets=targets,
            full_labels=full_labels,
            task_ids=task_ids,
            task_tokens=task_tokens
        )
        loss, losses = dec_out.loss, dec_out.losses

        return Seq2SeqFMMusicTransformerOutput(
            encoder=enc_out,
            decoder=dec_out,
            pred_tokens=dec_out.pred_tokens,
            pred_values=dec_out.pred_values,
            loss=loss,
            losses=losses
        )

    @torch.inference_mode()
    def generate(
            self,
            enc_tokens: torch.Tensor,
            dec_tokens: torch.Tensor | None,
            enc_values: torch.Tensor | None,
            dec_values: torch.Tensor | None,
            tokenizer: SyMuPe,
            dec_context: torch.Tensor | None = None,
            dec_context_tokens: torch.Tensor | None = None,
            dec_context_values: torch.Tensor | None = None,
            dec_score_tokens: torch.Tensor | None = None,
            dec_score_values: torch.Tensor | None = None,
            task_ids: torch.Tensor | None = None,
            task_tokens: torch.Tensor | None = None,
            seq_len: int | None = None,
            steps: int = 4,
            step_factor: float = 1.,
            method: str | None = None,
            disable_tqdm: bool = False,
            return_cache: bool = False,
            force_known_tokens: bool = False,
            **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, FMIntermediates]:
        self.prepare_for_fm()

        num_dims = len(enc_tokens.shape)
        if num_dims == 2:
            enc_tokens = enc_tokens[None]
            enc_values = enc_values[None] if enc_values is not None else None

        enc_out = self.encoder(
            tokens=enc_tokens,
            values=enc_values
        )
        context, _ = self._build_context(enc_out, enc_mask=None, dec_context=dec_context)

        return self.decoder.generate(
            dec_tokens, values=dec_values,
            known_tokens=enc_tokens if force_known_tokens else None,
            known_values=enc_values if force_known_tokens else None,
            context=context,
            context_tokens=dec_context_tokens, context_values=dec_context_values,
            score_tokens=dec_score_tokens, score_values=dec_score_values,
            task_ids=task_ids, task_tokens=task_tokens,
            seq_len=seq_len or enc_tokens.shape[1] - 1, tokenizer=tokenizer,
            steps=steps, step_factor=step_factor, method=method,
            disable_tqdm=disable_tqdm, return_cache=return_cache,
            **kwargs
        )

    def normalize_values(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.value_mean) / self.value_std

    def denormalize_values(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.value_std + self.value_mean

    def prepare_inputs(
            self,
            inputs: dict | Seq2SeqInputs,
            ema_model: nn.Module | None = None
    ) -> dict[str, torch.Tensor]:
        inputs_dict = super().prepare_inputs(inputs=inputs, ema_model=ema_model)

        inputs_dict["dec_vectors"] = inputs["full_labels"]["values"][..., self.num_token_features:]

        return inputs_dict

    @staticmethod
    def inject_data_config(
            config: DictConfig | Seq2SeqMusicTransformerConfig | None,
            dataset: SequenceDataset | None
    ) -> DictConfig | ModuleConfig | None:
        config = _Seq2SeqMusicTransformer.inject_data_config(config=config, dataset=dataset)

        # if isinstance(dataset, SequenceDataset):
        #     config["context_num_tokens"] = dataset.context_token_sizes

        config["decoder"]["input_vectors"] = "cat"

        return config
