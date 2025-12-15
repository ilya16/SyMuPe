"""
A family of MusicTransformer models.

Each model consists of an encoder/decoder TupleTransformer
with an optional language modeling or flow matching head.

Available models:
    - MusicTransformer
    - CFMMusicTransformer
    - DFMMusicTransformer
    - FMMusicTransformer
"""
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, MISSING

import torch
import torch.nn as nn
from omegaconf import DictConfig

from modules.tuple_transformer.language_modeling import TupleTransformerLMOutput
from symupe.data.collators import SequenceInputs, PerformanceInputs, Seq2SeqInputs
from symupe.data.datasets import SequenceDataset, PerformanceDataset
from symupe.data.tokenizers import OctupleM, SyMuPe
from symupe.modules.classes import LanguageModelingMode, ModelWrapper
from symupe.modules.constructor import ModuleConfig
from symupe.modules.tuple_transformer import (
    TupleTransformerConfig, TupleTransformer, TupleTransformerOutput,
    TupleTransformerWrappers,
    TupleTransformerLMWrapper,
    TupleTransformerCFMOutput, TupleTransformerCFMWrapper, CFMIntermediates,
    TupleTransformerDFMWrapper, TupleTransformerDFMOutput, DFMIntermediates,
    TupleTransformerFMWrapper, TupleTransformerFMOutput, FMIntermediates
)
from symupe.modules.tuple_transformer.flow_matching import resample
from symupe.utils import asdict
from ..base import Model


@dataclass
class _MusicTransformerConfig(ModuleConfig):
    num_tokens: dict[str, int] = MISSING
    dim: int = MISSING
    transformer: TupleTransformerConfig = MISSING
    context_num_tokens: dict[str, int] | None = None
    score_num_tokens: dict[str, int] | None = None
    token_keys: list[str] | None = None
    value_keys: list[str] | None = None


@dataclass
class _MusicTransformerOutput(TupleTransformerOutput):
    loss: torch.Tensor | None = None
    losses: dict[str, torch.Tensor] | None = None


class _MusicTransformer(Model):
    def __init__(
            self,
            num_tokens: dict[str, int],
            dim: int,
            transformer: DictConfig | TupleTransformerConfig,
            context_num_tokens: dict[str, int] | None = None,
            score_num_tokens: dict[str, int] | None = None,
            token_keys: list[str] | None = None,
            value_keys: list[str] | None = None,
            wrapper_kwargs: dict | None = None
    ):
        super().__init__()

        self.transformer = TupleTransformer.init(
            transformer,
            dim=dim,
            num_tokens=num_tokens,
            context_num_tokens=context_num_tokens,
            score_num_tokens=score_num_tokens,
            token_keys=token_keys,
            value_keys=value_keys
        )
        self.num_tokens = num_tokens

        self.token_keys = self.transformer.token_keys
        self.value_keys = self.transformer.value_keys
        self.token_indices = [idx for idx, key in enumerate(self.num_tokens) if key in token_keys] if token_keys else None
        self.value_indices = [idx for idx, key in enumerate(self.num_tokens) if key in value_keys] if value_keys else None

        self.wrapper_kwargs = wrapper_kwargs or {}

        self.mode = None

    def unwrap_model(self):
        if isinstance(self.transformer, ModelWrapper):
            self.transformer = self.transformer.model
        self.mode = None
        return self

    @property
    def unwrapped_transformer(self):
        if isinstance(self.transformer, ModelWrapper):
            return self.transformer.model
        return self.transformer

    @abstractmethod
    def forward(self, *args, **kwargs):
        ...

    @abstractmethod
    def generate(self, *args, **kwargs):
        ...

    def prepare_inputs(
            self,
            inputs: dict | SequenceInputs | PerformanceInputs | Seq2SeqInputs,
            ema_model: nn.Module | None = None
    ) -> dict[str, torch.Tensor]:
        if isinstance(inputs, (SequenceInputs, PerformanceInputs, Seq2SeqInputs)):
            inputs = asdict(inputs)

        seq_key = "sequences"
        if "performances" in inputs:
            seq_key = "performances"
        elif "output_sequences" in inputs:
            seq_key = "output_sequences"

        inputs_dict = {
            "tokens": inputs[seq_key]["tokens"],
            "values": inputs[seq_key]["values"],
            "mask": inputs[seq_key]["mask"]
        }

        if inputs["context_sequences"] is not None:
            inputs_dict["context_tokens"] = inputs["context_sequences"]["tokens"]
            inputs_dict["context_values"] = inputs["context_sequences"]["values"]

        if inputs["score_sequences"] is not None:
            inputs_dict["score_tokens"] = inputs["score_sequences"]["tokens"]
            inputs_dict["score_values"] = inputs["score_sequences"]["values"]

        if inputs["emotion_embeddings"] is not None:
            inputs_dict["context"] = inputs["emotion_embeddings"]

        if inputs.get("labels", None) is not None:
            inputs_dict["labels"] = inputs["labels"]["tokens"]
            inputs_dict["targets"] = inputs["labels"]["values"]

        if inputs.get("full_labels", None) is not None:
            inputs_dict["full_labels"] = inputs["full_labels"]["tokens"]

        if inputs.get(f"masked_{seq_key}", None) is not None:
            inputs_dict["masked_tokens"] = inputs[f"masked_{seq_key}"]["tokens"]
            inputs_dict["masked_values"] = inputs[f"masked_{seq_key}"]["values"]
        elif inputs.get("input_sequences", None) is not None:
            inputs_dict["masked_tokens"] = inputs["input_sequences"]["tokens"]
            inputs_dict["masked_values"] = inputs["input_sequences"]["values"]

        if inputs.get("type_ids", None) is not None:
            inputs_dict["type_ids"] = inputs["type_ids"]

        if inputs.get("task_ids", None) is not None:
            inputs_dict["task_ids"] = inputs["task_ids"]

        if inputs.get("task_sequences", None) is not None:
            inputs_dict["task_tokens"] = inputs["task_sequences"]["tokens"]

        # if inputs.get("segments", None) is not None:
        #     inputs_dict["bars"] = inputs["segments"]["bar"]
        #     inputs_dict["beats"] = inputs["segments"]["beat"]
        #     inputs_dict["onsets"] = inputs["segments"]["onset"]

        return inputs_dict

    @staticmethod
    def inject_data_config(
            config: DictConfig | MusicTransformerConfig | None,
            dataset: SequenceDataset | PerformanceDataset | None
    ) -> DictConfig | ModuleConfig | None:
        assert isinstance(dataset, (SequenceDataset, PerformanceDataset))

        if isinstance(dataset, SequenceDataset):
            config["num_tokens"] = dataset.token_sizes
            config["context_num_tokens"] = dataset.context_token_sizes
            config["score_num_tokens"] = dataset.score_token_sizes
        else:
            config["num_tokens"] = dataset.performance_token_sizes

        token_emb_cfg = config["transformer"]["token_embeddings"]
        if token_emb_cfg is not None:
            token_emb_cfg["token_values"] = {
                key: value.tolist() for key, value in dataset.tokenizer.token_values(normalize=True).items()
            }
            token_emb_cfg["special_tokens"] = dataset.tokenizer.special_tokens_dict

        token_pos_emb_cfg = config["transformer"].get("token_pos_embeddings", None)
        if token_pos_emb_cfg is not None and token_pos_emb_cfg.get("token_dims") is None:
            assert "_token_types_" in token_pos_emb_cfg

            token_pos_emb_cfg["token_dims"] = {
                key: idx for idx, key in enumerate(dataset.token_sizes.keys())
                if key in token_pos_emb_cfg["_token_types_"]
            }
            del token_pos_emb_cfg["_token_types_"]

        if config["transformer"].get("_task_embedding_", False):
            config["transformer"]["num_tasks"] = len(dataset.sequence_encoding_types)
            del config["transformer"]["_task_embedding_"]

        return config

    @staticmethod
    def cleanup_config(
            config: DictConfig | MusicTransformerConfig | None,
    ) -> DictConfig | ModuleConfig | None:
        if config["transformer"].get("token_embeddings", None) is not None:
            config["transformer"]["token_embeddings"].pop("token_values", None)
        return config


@dataclass
class MusicTransformerConfig(_MusicTransformerConfig):
    mode: str | None = None


@dataclass
class MusicTransformerOutput(_MusicTransformerConfig):
    ...


class MusicTransformer(_MusicTransformer):
    def __init__(
            self,
            num_tokens: dict[str, int],
            dim: int,
            transformer: DictConfig | TupleTransformerConfig,
            mode: str | None = None,
            context_num_tokens: dict[str, int] | None = None,
            score_num_tokens: dict[str, int] | None = None,
            token_keys: list[str] | None = None,
            value_keys: list[str] | None = None,
            wrapper_kwargs: dict | None = None
    ):
        super().__init__(
            num_tokens=num_tokens,
            dim=dim,
            transformer=transformer,
            context_num_tokens=context_num_tokens,
            score_num_tokens=score_num_tokens,
            token_keys=token_keys,
            value_keys=value_keys,
            wrapper_kwargs=wrapper_kwargs
        )

        self.mode = mode

        if self.mode == LanguageModelingMode.MLM:
            self.prepare_for_mlm()
        elif self.mode == LanguageModelingMode.CLM:
            self.prepare_for_clm()
        elif self.mode == LanguageModelingMode.MixedLM:
            self.prepare_for_mixlm()

    def _prepare_for_lm(self, mode: LanguageModelingMode | str | None = None) -> MusicTransformer:
        if mode is None:
            if isinstance(self.transformer, TupleTransformerLMWrapper):
                self.transformer = self.transformer.model
        else:
            if isinstance(self.transformer, TupleTransformer):
                self.transformer = TupleTransformerWrappers[mode](
                    self.transformer, token_keys=self.token_keys, **self.wrapper_kwargs
                )
            elif isinstance(self.transformer, TupleTransformerLMWrapper):
                self.transformer = TupleTransformerWrappers[mode](
                    self.transformer.model, token_keys=self.token_keys, **self.wrapper_kwargs
                )
        self.mode = mode
        return self

    def prepare_for_mlm(self) -> MusicTransformer:
        return self._prepare_for_lm(mode=LanguageModelingMode.MLM)

    def prepare_for_clm(self) -> MusicTransformer:
        return self._prepare_for_lm(mode=LanguageModelingMode.CLM)

    def prepare_for_mixlm(self) -> MusicTransformer:
        return self._prepare_for_lm(mode=LanguageModelingMode.MixedLM)

    def forward(
            self,
            tokens: torch.Tensor,
            values: torch.Tensor | None = None,
            mask: torch.Tensor | None = None,

            context: torch.Tensor | None = None,
            context_mask: torch.Tensor | None = None,
            context_tokens: torch.Tensor | None = None,
            context_values: torch.Tensor | None = None,
            score_tokens: torch.Tensor | None = None,
            score_values: torch.Tensor | None = None,

            labels: torch.Tensor | None = None,
            targets: torch.Tensor | None = None,
            full_labels: torch.Tensor | None = None,

            masked_tokens: torch.Tensor | None = None,
            masked_values: torch.Tensor | None = None,

            type_ids: torch.Tensor | None = None,
            task_ids: torch.Tensor | None = None,
            task_tokens: torch.Tensor | None = None,
            return_cache: bool = False,
            output_layer: int | None = None
    ) -> TupleTransformerOutput | TupleTransformerLMOutput:
        out: TupleTransformerOutput | TupleTransformerLMOutput = self.transformer(
            tokens, values=values, mask=mask,
            context=context, context_mask=context_mask,
            context_tokens=context_tokens, context_values=context_values,
            score_tokens=score_tokens, score_values=score_values,
            type_ids=type_ids, task_ids=task_ids, task_tokens=task_tokens,
            labels=labels, targets=targets, full_labels=full_labels,
            masked_tokens=masked_tokens, masked_values=masked_values,
            return_cache=return_cache, output_layer=output_layer
        )

        return out

    @torch.inference_mode()
    def generate(
            self,
            tokens: torch.Tensor,
            values: torch.Tensor,
            tokenizer: OctupleM,
            temperature: float = 1.,
            top_k: float | int = -1,
            top_p: float = 0.8,
            disable_tqdm: bool = False,
            **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        assert self.mode is not None

        if self.mode in (LanguageModelingMode.MLM, LanguageModelingMode.MixedLM):
            inference_fn = self.transformer.unmask
        else:
            inference_fn = self.transformer.generate

        out_tokens, out_values = inference_fn(
            tokens, values=values, tokenizer=tokenizer,
            temperature=temperature, top_k=top_k, top_p=top_p,
            disable_tqdm=disable_tqdm, **kwargs
        )

        return out_tokens, out_values, None


@dataclass
class CFMMusicTransformerConfig(_MusicTransformerConfig):
    pedal_stream: bool = False
    value_mean: list[float] | dict[str, int] | None = None
    value_std: list[float] | dict[str, int] | None = None
    tokenizer: SyMuPe | None = None
    wrapper_kwargs: dict | None = None


@dataclass
class CFMMusicTransformerOutput(_MusicTransformerOutput):
    pred_values: torch.Tensor | None = None


class CFMMusicTransformer(_MusicTransformer):
    def __init__(
            self,
            num_tokens: dict[str, int],
            dim: int,
            transformer: DictConfig | TupleTransformerConfig,
            context_num_tokens: dict[str, int] | None = None,
            score_num_tokens: dict[str, int] | None = None,
            pedal_stream: bool = False,
            context_vectors: bool = False,
            value_mean: list[float] | dict[str, int] | None = None,
            value_std: list[float] | dict[str, int] | None = None,
            value_log: list[str] | None = None,
            value_keys: list[str] | None = None,
            tokenizer: SyMuPe | None = None,
            wrapper_kwargs: dict | None = None
    ):
        value_keys = value_keys or list(num_tokens.keys())

        transformer["input_vectors"] = "cat"
        transformer["input_vectors_dim"] = (int(context_vectors) + 1) * (len(value_keys) + 2 * int(pedal_stream))

        super().__init__(
            num_tokens=num_tokens,
            dim=dim,
            transformer=transformer,
            context_num_tokens=context_num_tokens,
            score_num_tokens=score_num_tokens,
            value_keys=value_keys,
            wrapper_kwargs=wrapper_kwargs
        )

        value_mean = value_mean or {}
        if isinstance(value_mean, (dict, DictConfig)):
            value_mean = [value_mean.get(key, 0.) for key in self.value_keys]

        value_std = value_std or {}
        if isinstance(value_std, (dict, DictConfig)):
            value_std = [value_std.get(key, 1.) for key in self.value_keys]

        value_log = value_log or []
        value_log_ids = []
        for i, key in enumerate(self.num_tokens):
            if key in value_log:
                value_log_ids.append(i)
                value_mean[i], value_std[i] = 0., 1.

        self.register_buffer("value_mean", torch.tensor(value_mean))
        self.register_buffer("value_std", torch.tensor(value_std))
        self.register_buffer("value_log_ids", torch.tensor(value_log_ids))

        self.pedal_stream = pedal_stream
        self.context_vectors = context_vectors
        self.tokenizer = tokenizer

        self.prepare_for_cfm()

    def prepare_for_cfm(self) -> CFMMusicTransformer:
        if isinstance(self.transformer, TupleTransformer):
            self.transformer = TupleTransformerCFMWrapper(
                self.transformer, tokenizer=self.tokenizer,
                value_mean=self.value_mean, value_std=self.value_std,
                context_vectors=self.context_vectors,
                value_log_ids=self.value_log_ids, value_keys=self.value_keys,
                **self.wrapper_kwargs
            )
        return self

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

            pedals: torch.Tensor | None = None,
            type_ids: torch.Tensor | None = None,
            task_ids: torch.Tensor | None = None,
            task_tokens: torch.Tensor | None = None,
            return_cache: bool = False,
            output_layer: int | None = None,
            ema_model: nn.Module | None = None
    ) -> TupleTransformerCFMOutput:
        out: TupleTransformerCFMOutput = self.transformer(
            tokens, values=values, vectors=vectors, mask=mask,
            context=context, context_mask=context_mask,
            context_tokens=context_tokens, context_values=context_values,
            score_tokens=score_tokens, score_values=score_values,
            pedals=pedals if self.pedal_stream else None,
            type_ids=type_ids, task_ids=task_ids, task_tokens=task_tokens,
            labels=labels, targets=targets,
            return_cache=return_cache, output_layer=output_layer, ema_model=ema_model
        )

        return out

    def generate(
            self,
            tokens: torch.Tensor,
            values: torch.Tensor,
            tokenizer: OctupleM,

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

            loss_fn: nn.Module | None = None,
            context_len: int = 0,
            gamma: float = 1,
            norm_fn: Callable | None = None,
            schedule_fn: Callable | None = None,
            num_resample: int = 1,
            resample_period: int = 5,
            resample_fn: Callable = resample,

            disable_tqdm: bool = False,
            return_intermediates: bool = False,
            **kwargs
    ) -> (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]
          | tuple[torch.Tensor, torch.Tensor, torch.Tensor, CFMIntermediates]
    ):
        self.prepare_for_cfm()

        if self.pedal_stream:
            if pedals is None:
                pedals = torch.full(values.shape[:-1] + (2,), fill_value=-1., device=tokens.device)
        else:
            pedals = None

        vectors = values[..., self.value_indices] if self.value_indices is not None else values

        out_tokens, out_values, out_pedals, intermediates = self.transformer.generate(
            tokens, values=values, vectors=vectors,
            steps=steps, step_factor=step_factor, method=method, x_0=x_0,
            tokenizer=tokenizer,
            context=context, context_mask=context_mask,
            context_tokens=context_tokens, context_values=context_values,
            context_scale=context_scale,
            pedals=pedals,
            loss_fn=loss_fn, context_len=context_len,
            gamma=gamma, norm_fn=norm_fn, schedule_fn=schedule_fn,
            num_resample=num_resample, resample_period=resample_period, resample_fn=resample_fn,
            disable_tqdm=disable_tqdm, **kwargs
        )

        if return_intermediates:
            return out_tokens, out_values, out_pedals, intermediates
        return out_tokens, out_values, out_pedals

    def prepare_inputs(
            self,
            inputs: dict | SequenceInputs | PerformanceInputs | Seq2SeqInputs,
            ema_model: nn.Module | None = None
    ) -> dict[str, torch.Tensor]:
        inputs_dict = super().prepare_inputs(inputs=inputs, ema_model=ema_model)

        inputs_dict["vectors"] = inputs["full_labels"]["values"]

        if inputs["pedals"] is not None:
            inputs_dict["pedals"] = inputs["pedals"]

        for key in ("bars", "beats", "onsets", "full_labels", "masked_tokens", "masked_values"):
            inputs_dict.pop(key, None)

        if ema_model is not None:
            inputs_dict["ema_model"] = ema_model.transformer

        return inputs_dict


@dataclass
class DFMMusicTransformerConfig(_MusicTransformerConfig):
    ...


@dataclass
class DFMMusicTransformerOutput(_MusicTransformerOutput):
    pred_tokens: torch.Tensor | None = None


class DFMMusicTransformer(_MusicTransformer):
    def __init__(
            self,
            num_tokens: dict[str, int],
            dim: int,
            transformer: DictConfig | TupleTransformerConfig,
            context_num_tokens: dict[str, int] | None = None,
            score_num_tokens: dict[str, int] | None = None,
            token_keys: list[str] | None = None,
            tokenizer: SyMuPe | None = None,
            wrapper_kwargs: dict | None = None
    ):
        super().__init__(
            num_tokens=num_tokens,
            dim=dim,
            transformer=transformer,
            context_num_tokens=context_num_tokens,
            score_num_tokens=score_num_tokens,
            token_keys=token_keys,
            wrapper_kwargs=wrapper_kwargs
        )

        self.tokenizer = tokenizer

        self.prepare_for_dfm()

    def prepare_for_dfm(self) -> DFMMusicTransformer:
        if isinstance(self.transformer, TupleTransformer):
            self.transformer = TupleTransformerDFMWrapper(
                self.transformer, tokenizer=self.tokenizer,
                token_keys=self.token_keys, **self.wrapper_kwargs
            )
        self.mode = LanguageModelingMode.DFM
        return self

    def forward(
            self,
            tokens: torch.Tensor,
            values: torch.Tensor,
            mask: torch.Tensor | None = None,

            context: torch.Tensor | None = None,
            context_mask: torch.Tensor | None = None,
            context_tokens: torch.Tensor | None = None,
            context_values: torch.Tensor | None = None,
            score_tokens: torch.Tensor | None = None,
            score_values: torch.Tensor | None = None,

            labels: torch.Tensor | None = None,
            targets: torch.Tensor | None = None,
            full_labels: torch.Tensor | None = None,

            masked_tokens: torch.Tensor | None = None,
            masked_values: torch.Tensor | None = None,

            type_ids: torch.Tensor | None = None,
            task_ids: torch.Tensor | None = None,
            task_tokens: torch.Tensor | None = None,
            return_cache: bool = False,
            output_layer: int | None = None,
            ema_model: nn.Module | None = None
    ) -> TupleTransformerDFMOutput:
        self.transformer.tokenizer = self.tokenizer

        out: TupleTransformerDFMOutput = self.transformer(
            tokens, values=values,  mask=mask,
            context=context, context_mask=context_mask,
            context_tokens=context_tokens, context_values=context_values,
            score_tokens=score_tokens, score_values=score_values,
            task_ids=task_ids, task_tokens=task_tokens,
            labels=labels, targets=targets, full_labels=full_labels,
            masked_tokens=masked_tokens, masked_values=masked_values,
            return_cache=return_cache, output_layer=output_layer, ema_model=ema_model
        )

        return out

    def generate(
            self,
            tokens: torch.Tensor,
            values: torch.Tensor,
            tokenizer: OctupleM | None,

            steps: int = 4,
            step_factor: float = 1.,
            method: str | None = None,

            context: torch.Tensor | None = None,
            context_mask: torch.Tensor | None = None,
            context_tokens: torch.Tensor | None = None,
            context_values: torch.Tensor | None = None,
            context_scale: float = 1.,

            disable_tqdm: bool = False,
            return_intermediates: bool = False,
            **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, DFMIntermediates]:
        self.prepare_for_dfm()
        tokenizer = tokenizer or self.tokenizer

        out_tokens, out_values, intermediates = self.transformer.generate(
            tokens, values=values,
            steps=steps, step_factor=step_factor, method=method, tokenizer=tokenizer,
            context=context, context_mask=context_mask,
            context_tokens=context_tokens, context_values=context_values,
            context_scale=context_scale,
            disable_tqdm=disable_tqdm, **kwargs
        )

        if return_intermediates:
            return out_tokens, out_values, intermediates
        return out_tokens, out_values

    def prepare_inputs(
            self,
            inputs: dict | SequenceInputs | PerformanceInputs | Seq2SeqInputs,
            ema_model: nn.Module | None = None
    ) -> dict[str, torch.Tensor]:
        inputs_dict = super().prepare_inputs(inputs=inputs, ema_model=ema_model)

        for key in ("bars", "beats", "onsets"):
            inputs_dict.pop(key, None)

        if ema_model is not None:
            inputs_dict["ema_model"] = ema_model.transformer

        return inputs_dict


@dataclass
class FMMusicTransformerConfig(_MusicTransformerConfig):
    context_vectors: bool = False
    value_mean: list[float] | dict[str, int] | None = None
    value_std: list[float] | dict[str, int] | None = None


@dataclass
class FMMusicTransformerOutput(_MusicTransformerOutput):
    pred_tokens: torch.Tensor | None = None
    pred_values: torch.Tensor | None = None


class FMMusicTransformer(_MusicTransformer):
    def __init__(
            self,
            num_tokens: dict[str, int],
            dim: int,
            transformer: DictConfig | TupleTransformerConfig,
            token_keys: list[str],
            value_keys: list[str],
            context_vectors: bool = False,
            value_mean: list[float] | dict[str, int] | None = None,
            value_std: list[float] | dict[str, int] | None = None,
            context_num_tokens: dict[str, int] | None = None,
            score_num_tokens: dict[str, int] | None = None,
            tokenizer: SyMuPe | None = None,
            wrapper_kwargs: dict | None = None
    ):
        self.token_keys = list(token_keys)
        self.value_keys = list(value_keys)

        for key in self.token_keys:
            assert key in num_tokens
        for key in self.value_keys:
            assert key in num_tokens

        assert transformer.lm_head is not None
        transformer.lm_head["num_tokens"] = {key: num for key, num in num_tokens.items() if key in self.token_keys}

        assert transformer.value_head is not None
        transformer.value_head["num_features"] = len(self.value_keys)
        transformer["input_vectors"] = "cat"
        transformer["input_vectors_dim"] = (int(context_vectors) + 1) * len(self.value_keys)  # ctx + noisy

        super().__init__(
            num_tokens=num_tokens,
            dim=dim,
            transformer=transformer,
            context_num_tokens=context_num_tokens,
            score_num_tokens=score_num_tokens,
            token_keys=token_keys,
            value_keys=value_keys,
            wrapper_kwargs=wrapper_kwargs
        )

        self.tokenizer = tokenizer
        self.context_vectors = context_vectors
        self.wrapper_kwargs = wrapper_kwargs or {}

        value_mean = value_mean or [0.]
        if isinstance(value_mean, (dict, DictConfig)):
            value_mean = [value_mean.get(key, 0.) for key in self.value_keys]

        value_std = value_std or [1.]
        if isinstance(value_std, (dict, DictConfig)):
            value_std = [value_std.get(key, 1.) for key in self.value_keys]

        self.register_buffer("value_mean", torch.tensor(value_mean))
        self.register_buffer("value_std", torch.tensor(value_std))

        self.prepare_for_fm()

    def prepare_for_fm(self) -> FMMusicTransformer:
        if isinstance(self.transformer, TupleTransformer):
            self.transformer = TupleTransformerFMWrapper(
                self.transformer, tokenizer=self.tokenizer,
                token_keys=self.token_keys, value_keys=self.value_keys,
                context_vectors=self.context_vectors,
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
            full_labels: torch.Tensor | None = None,

            masked_tokens: torch.Tensor | None = None,
            masked_values: torch.Tensor | None = None,

            type_ids: torch.Tensor | None = None,
            task_ids: torch.Tensor | None = None,
            task_tokens: torch.Tensor | None = None,
            return_cache: bool = False,
            output_layer: int | None = None,
            ema_model: nn.Module | None = None
    ) -> TupleTransformerFMOutput:
        self.transformer.tokenizer = self.tokenizer

        out: TupleTransformerFMOutput = self.transformer(
            tokens, values=values, vectors=vectors, mask=mask,
            context=context, context_mask=context_mask,
            context_tokens=context_tokens, context_values=context_values,
            score_tokens=score_tokens, score_values=score_values,
            type_ids=type_ids, task_ids=task_ids, task_tokens=task_tokens,
            labels=labels, targets=targets, full_labels=full_labels,
            masked_tokens=masked_tokens, masked_values=masked_values,
            return_cache=return_cache, output_layer=output_layer, ema_model=ema_model
        )

        return out

    def generate(
            self,
            tokens: torch.Tensor,
            values: torch.Tensor,
            tokenizer: OctupleM | None,

            steps: int = 4,
            step_factor: float = 1.,
            method: str | None = None,

            masked_tokens: torch.Tensor | None = None,
            masked_values: torch.Tensor | None = None,

            context: torch.Tensor | None = None,
            context_mask: torch.Tensor | None = None,
            context_tokens: torch.Tensor | None = None,
            context_values: torch.Tensor | None = None,
            context_scale: float = 1.,

            disable_tqdm: bool = False,
            return_intermediates: bool = False,
            **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, FMIntermediates]:
        self.prepare_for_fm()
        tokenizer = tokenizer or self.tokenizer

        out_tokens, out_values, intermediates = self.transformer.generate(
            tokens, values=values,
            steps=steps, step_factor=step_factor, method=method, tokenizer=tokenizer,
            masked_tokens=masked_tokens, masked_values=masked_values,
            context=context, context_mask=context_mask,
            context_tokens=context_tokens, context_values=context_values,
            context_scale=context_scale,
            disable_tqdm=disable_tqdm, **kwargs
        )

        if return_intermediates:
            return out_tokens, out_values, intermediates
        return out_tokens, out_values

    def prepare_inputs(
            self,
            inputs: dict | SequenceInputs | PerformanceInputs | Seq2SeqInputs,
            ema_model: nn.Module | None = None
    ) -> dict[str, torch.Tensor]:
        inputs_dict = super().prepare_inputs(inputs=inputs, ema_model=ema_model)

        inputs_dict["vectors"] = inputs["full_labels"]["values"][..., self.num_token_features:]

        for key in ("bars", "beats", "onsets"):
            inputs_dict.pop(key, None)

        if ema_model is not None:
            inputs_dict["ema_model"] = ema_model.transformer

        return inputs_dict
