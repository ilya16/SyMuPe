""" TupleTransformer: Transformer with support for tuple token and value sequences. """
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, MISSING, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from omegaconf import DictConfig

from symupe.data.tokenizers.constants import SPECIAL_TOKENS_VALUE
from symupe.modules.constructor import Constructor, ModuleConfig
from symupe.modules.transformer import (
    TransformerRegistry, TransformerConfig,
    TransformerIntermediates, TransformerOutput,
    AbsolutePositionalEmbedding, LayerNorm
)
from symupe.utils import ExplicitEnum, prob_mask_like
from .embeddings import (
    TupleTransformerEmbeddingsRegistry,
    TupleTransformerEmbeddingsConfig, TupleTransformerEmbeddings,
    PositionTupleTransformerEmbeddingsConfig, PositionTupleTransformerEmbeddings
)
from .heads import (
    TupleTransformerHeadsRegistry, TupleTransformerHeadsConfig,
    TupleTransformerCausalLMHead, TupleTransformerDFMHead,
    TupleTransformerSplitValueHead
)
from ..layers import TimePositionalEmbedding, LegacyTimePositionalEmbedding


class EmbeddingMode(ExplicitEnum):
    SUM = "sum"
    CONCAT = "cat"
    PREFIX = "prefix"
    ATTENTION = "attention"
    ATTENTION_PREFIX = "attention_prefix"
    ADANORM = "adanorm"
    LAYER_CONCAT = "layer_cat"
    LAYER_SUM = "layer_sum"
    TOKEN_PREFIX = "token_prefix"


@dataclass
class TupleTransformerCache:
    token_emb: torch.Tensor | None = None
    transformer: TransformerIntermediates | None = None

    def add_past_cache(self, past_cache: TupleTransformerCache | None) -> TupleTransformerCache:
        if past_cache is None:
            return self

        if past_cache.transformer is not None and self.transformer is not None:
            self.transformer = self.transformer.add_past_cache(past_cache.transformer)

        return self


@dataclass
class TupleTransformerOutput:
    hidden_state: torch.Tensor
    memory_state: torch.Tensor | None = None
    task_state: torch.Tensor | None = None
    mode_state: torch.Tensor | None = None
    logits: dict[str, torch.Tensor] | None = None
    attentions: list[torch.Tensor] | None = None
    cache: TupleTransformerCache | None = None
    values: torch.Tensor | dict[str, torch.Tensor] | None = None
    task_logits: torch.Tensor | None = None
    type_logits: torch.Tensor | None = None


@dataclass
class TupleTransformerConfig(ModuleConfig):
    num_tokens: dict[str, int] = MISSING
    dim: int = 512
    max_seq_len: int = 1024
    transformer: DictConfig | TransformerConfig = field(
        default_factory=lambda: TransformerConfig(_target_="default")
    )

    token_embeddings: DictConfig | TupleTransformerEmbeddingsConfig | None = field(
        default_factory=lambda: TupleTransformerEmbeddingsConfig()
    )
    token_pos_embeddings: DictConfig | PositionTupleTransformerEmbeddingsConfig | None = None
    use_abs_pos_emb: bool = True
    emb_norm: bool = False
    emb_dropout: float = 0.0
    project_bias: bool = True

    context_embedding: str = EmbeddingMode.ATTENTION
    context_embedding_dim: int | None = None
    context_project_dim: int | None = None
    context_norm: bool = False
    context_dropout: float = 0.
    null_context: bool = False
    context_layer_ids: Sequence[int] | None = None

    context_token_embedding: str | None = None
    context_num_tokens: dict[str, int] | None = None
    context_token_embeddings: DictConfig | TupleTransformerEmbeddingsConfig | None = None
    context_tokens_dropout: float = 0.

    score_num_tokens: dict[str, int] | None = None
    score_token_embeddings: DictConfig | TupleTransformerEmbeddingsConfig | None = None
    score_tokens_dropout: float = 0.

    style_embedding: str = EmbeddingMode.CONCAT
    style_embedding_dim: int | None = None
    style_embedding_dropout: float = 0.

    time_embedding: str | None = None  # EmbeddingMode.ADANORM
    time_embedding_num: int = 1
    time_embedding_dim: int | None = None
    time_embedding_freq_dim: int | None = 256
    time_embedding_legacy: bool = False

    num_tasks: int | None = None
    task_embedding: str = EmbeddingMode.PREFIX
    task_embedding_dim: int | None = None
    task_head: bool = False

    num_types: int | None = None
    type_embedding: str | None = None
    type_embedding_dim: int | None = None
    type_head: bool = False

    mode_embedding: str = EmbeddingMode.PREFIX
    mode_embedding_dim: int | None = None

    lm_head: DictConfig | TupleTransformerHeadsConfig | None = None
    value_head: DictConfig | TupleTransformerHeadsConfig | None = None
    token_keys: list[str] | None = None
    value_keys: list[str] | None = None

    transformer_output_layer: int | None = None


class TupleTransformer(nn.Module, Constructor):
    def __init__(
            self,
            num_tokens: dict[str, int],
            dim: int = 512,
            max_seq_len: int = 1024,
            transformer: DictConfig | TransformerConfig = TransformerConfig(_target_="default"),

            token_embeddings: DictConfig | TupleTransformerEmbeddingsConfig | None = TupleTransformerEmbeddingsConfig(),
            token_pos_embeddings: DictConfig | PositionTupleTransformerEmbeddingsConfig | None = None,
            use_abs_pos_emb: bool = True,
            emb_norm: bool = False,
            emb_dropout: float = 0.0,
            project_bias: bool = True,

            input_vectors: str = EmbeddingMode.CONCAT,
            input_vectors_dim: int | None = None,

            context_embedding: str = EmbeddingMode.ATTENTION,
            context_embedding_dim: int | None = None,
            context_project_dim: int | None = None,
            context_norm: bool = False,
            context_dropout: float = 0.,
            null_context: bool = False,
            context_layer_ids: Sequence[int] | None = None,

            context_num_tokens: dict[str, int] | None = None,
            context_token_embeddings: DictConfig | TupleTransformerEmbeddingsConfig | None = None,
            context_tokens_dropout: float = 0.,

            score_num_tokens: dict[str, int] | None = None,
            score_token_embeddings: DictConfig | TupleTransformerEmbeddingsConfig | None = None,
            score_tokens_dropout: float = 0.,

            style_embedding: str = EmbeddingMode.CONCAT,
            style_embedding_dim: int | None = None,
            style_embedding_dropout: float = 0.,

            time_embedding: str | None = None,
            time_embedding_num: int = 1,
            time_embedding_dim: int | None = None,
            time_embedding_freq_dim: int | None = 256,
            time_embedding_legacy: bool = False,

            num_tasks: int | None = None,
            task_embedding: str = EmbeddingMode.PREFIX,
            task_embedding_dim: int | None = None,
            task_head: bool = False,

            num_types: int | None = None,
            type_embedding: str | None = None,
            type_embedding_dim: int | None = None,
            type_head: bool = False,

            mode_embedding: str | None = None,
            mode_embedding_dim: int | None = None,

            lm_head: DictConfig | TupleTransformerHeadsConfig | None = None,
            value_head: DictConfig | TupleTransformerHeadsConfig | None = None,
            token_keys: list[str] | None = None,
            value_keys: list[str] | None = None,
            transformer_output_layer: int | None = None
    ):
        super().__init__()

        self.dim = dim
        self.max_seq_len = max_seq_len
        emb_dim = dim  # default(emb_dim, dim)

        # token/value embeddings

        assert token_embeddings is not None or input_vectors_dim > 0

        self.num_tokens = num_tokens
        self.token_emb = None
        if token_embeddings is not None:
            self.token_emb = TupleTransformerEmbeddingsRegistry.instantiate(
                config=token_embeddings,
                num_tokens=num_tokens,
                emb_dims=token_embeddings.get("emb_dims", emb_dim),
                project_emb_dim=emb_dim
            )
        else:
            emb_dim = len(num_tokens)  # use only values

        self.is_multiseq = self.token_emb is not None and hasattr(self.token_emb, "multi_mode")

        # token-based position embeddings

        self.token_pos_emb = None
        if token_pos_embeddings:
            self.token_pos_emb = PositionTupleTransformerEmbeddings.init(
                config=token_pos_embeddings,
                emb_dims=token_pos_embeddings.get("emb_dims", emb_dim),
                project_emb_dim=emb_dim,
                special_tokens=self.token_emb.special_tokens
            )

        # absolute positional embeddings

        self.pos_emb = None
        if use_abs_pos_emb:
            self.pos_emb = AbsolutePositionalEmbedding(emb_dim, self.max_seq_len)
            nn.init.kaiming_normal_(self.pos_emb.emb.weight)

        self.emb_norm = LayerNorm(emb_dim) if emb_norm else nn.Identity()
        self.emb_dropout = nn.Dropout(emb_dropout) if emb_dropout > 0. else nn.Identity()

        # optional extra input embeddings

        assert input_vectors == EmbeddingMode.CONCAT
        self.input_vectors = input_vectors
        self.input_vectors_dim = input_vectors_dim or 0

        # context embeddings

        self.context_embedding = context_embedding
        self.context_embedding_dim = context_embedding_dim or 0
        self.context_project_dim = self.context_total_embedding_dim = context_project_dim or self.context_embedding_dim
        self.project_context = nn.Linear(
            self.context_embedding_dim, self.context_project_dim
        ) if self.context_project_dim != self.context_embedding_dim else None
        self.context_dropout = context_dropout
        self.context_norm = LayerNorm(self.context_project_dim) if context_norm and self.context_project_dim > 0 else None
        self.null_context = nn.Parameter(torch.randn(self.context_project_dim)) if null_context else None

        if self.context_embedding != EmbeddingMode.ATTENTION:
            transformer.cross_attend = False
            transformer.cross_attention = None

        # context token embeddings

        self.context_token_embeddings = None
        if context_token_embeddings is not None:
            self.context_token_embeddings = TupleTransformerEmbeddings.init(
                config=context_token_embeddings,
                num_tokens=context_num_tokens,
                emb_dims=context_token_embeddings.get("emb_dims", emb_dim)
            )
            self.context_total_embedding_dim += self.context_token_embeddings.project_emb_dim
        self.context_tokens_dropout = context_tokens_dropout

        # score token embeddings

        self.score_token_embeddings = None
        if score_token_embeddings is not None:
            self.score_token_embeddings = TupleTransformerEmbeddings.init(
                config=score_token_embeddings,
                num_tokens=score_num_tokens,
                emb_dims=score_token_embeddings.get("emb_dims", emb_dim)
            )
            self.context_total_embedding_dim += self.score_token_embeddings.project_emb_dim
        self.score_tokens_dropout = score_tokens_dropout

        # style embeddings

        self.style_embedding = style_embedding
        self.style_embedding_dim = style_embedding_dim or 0
        self.style_embedding_dropout = style_embedding_dropout

        # time embedding

        self.time_embedding = time_embedding
        self.time_embedding_num = time_embedding_num
        self.time_emb = None
        if time_embedding is not None:
            time_embedding_dim = time_embedding_dim or dim // time_embedding_num
            if time_embedding_legacy:
                self.time_emb = LegacyTimePositionalEmbedding(
                    dim=dim, out_features=time_embedding_dim, activation=True,
                )
            else:
                time_embedding_freq_dim = time_embedding_freq_dim or time_embedding_dim
                self.time_emb = TimePositionalEmbedding(
                    freq_dim=time_embedding_freq_dim, emb_dim=time_embedding_dim, with_steps=True
                )
            time_embedding_dim *= time_embedding_num
        self.time_embedding_dim = time_embedding_dim or 0

        # input embeddings projection

        self.project_emb = nn.Identity()
        total_emb_dim = (
                emb_dim
                + int(self.input_vectors == EmbeddingMode.CONCAT) * self.input_vectors_dim
                + int(self.context_embedding == EmbeddingMode.CONCAT) * self.context_total_embedding_dim
                + int(self.style_embedding == EmbeddingMode.CONCAT) * self.style_embedding_dim
                + int(self.time_embedding == EmbeddingMode.CONCAT) * self.time_embedding_dim
        )
        if total_emb_dim != dim:
            self.project_emb = nn.Linear(total_emb_dim, dim, bias=project_bias)

        # task embedding

        assert task_embedding in (
            EmbeddingMode.PREFIX, EmbeddingMode.SUM, EmbeddingMode.TOKEN_PREFIX, EmbeddingMode.ADANORM
        )
        self.task_embedding = task_embedding
        self.task_emb = None
        if num_tasks is not None and task_embedding != EmbeddingMode.TOKEN_PREFIX:
            task_embedding_dim = task_embedding_dim or dim
            self.task_emb = nn.Embedding(num_tasks, task_embedding_dim)
            nn.init.kaiming_normal_(self.task_emb.weight)
        self.task_embedding_dim = task_embedding_dim or 0

        self.type_embedding = type_embedding
        self.type_emb = None
        if type_embedding is not None:
            assert type_embedding in (EmbeddingMode.SUM, EmbeddingMode.ADANORM)
            type_embedding_dim = type_embedding_dim or dim
            _is_sum = type_embedding == EmbeddingMode.SUM
            self.type_emb = nn.Embedding(
                2 if num_types is None else num_types,
                type_embedding_dim, padding_idx=0 if _is_sum else None
            )
            nn.init.kaiming_normal_(self.type_emb.weight[int(_is_sum):])
        self.type_embedding_dim = type_embedding_dim or 0

        self.mode_embedding = mode_embedding
        self.mode_emb = None
        if mode_embedding is not None:
            assert mode_embedding in (EmbeddingMode.PREFIX, EmbeddingMode.SUM, EmbeddingMode.ADANORM)
            mode_embedding_dim = mode_embedding_dim or dim
            self.mode_emb = nn.Embedding(2, mode_embedding_dim)
            nn.init.kaiming_normal_(self.mode_emb.weight)
        self.mode_embedding_dim = mode_embedding_dim or 0

        # transformer

        self.adaptive_norm = any(
            emb_type == EmbeddingMode.ADANORM
            for emb_type in (
                self.style_embedding, self.time_embedding, self.task_embedding,
                self.type_embedding, self.mode_embedding
            )
        )
        self.adaptive_condition_dim = (
                int(self.style_embedding == EmbeddingMode.ADANORM) * self.style_embedding_dim
                + int(self.time_embedding == EmbeddingMode.ADANORM) * self.time_embedding_dim
                + int(self.task_embedding == EmbeddingMode.ADANORM) * self.task_embedding_dim
                + int(self.type_embedding == EmbeddingMode.ADANORM) * self.type_embedding_dim
                + int(self.mode_embedding == EmbeddingMode.ADANORM) * self.mode_embedding_dim
        ) or None

        self.transformer = TransformerRegistry.instantiate(
            transformer,
            dim=dim,
            adaptive_norm=self.adaptive_norm,
            condition_dim=self.adaptive_condition_dim,
            context_as_input_prefix=context_embedding == EmbeddingMode.PREFIX,
            context_as_attention_prefix=context_embedding == EmbeddingMode.ATTENTION_PREFIX,
            context_as_layer_input=context_embedding in (EmbeddingMode.LAYER_CONCAT, EmbeddingMode.LAYER_SUM),
            context_as_layer_input_sum=context_embedding == EmbeddingMode.LAYER_SUM,
            context_dim=self.context_total_embedding_dim,
            context_layer_ids=context_layer_ids
        )
        self.transformer_output_layer = transformer_output_layer

        # language modeling and value heads

        self.lm_head = None
        self.token_keys = None
        self.token_indices = None
        if lm_head is not None:
            self.lm_head = TupleTransformerHeadsRegistry.instantiate(
                config=lm_head, dim=dim, embeddings=self.token_emb, keys=token_keys
            )
            self.token_keys = list(token_keys) if token_keys is not None else list(self.num_tokens.keys())
            self.token_indices = [idx for idx, key in enumerate(self.num_tokens) if key in self.token_keys]

        self.value_head = None
        self.value_keys = None
        if value_head is not None:
            self.value_head = TupleTransformerHeadsRegistry.instantiate(
                config=value_head, dim=dim, keys=value_keys
            )
            if value_keys is not None:
                self.value_keys = list(value_keys)
            else:
                self.value_keys = list(self.num_tokens.keys())[:self.value_head.num_features]

        self.task_head = None
        if num_tasks is not None and task_head:
            self.task_head = nn.Linear(dim, num_tasks, bias=False)

        self.type_head = None
        if num_types is not None and type_head:
            self.type_head = nn.Linear(dim, num_types, bias=False)

    def forward(
            self,
            tokens: torch.Tensor | list[torch.Tensor],
            values: torch.Tensor | list[torch.Tensor] | None = None,
            vectors: torch.Tensor | None = None,
            mask: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            causal: torch.Tensor | None = None,

            context: torch.Tensor | None = None,
            context_mask: torch.Tensor | None = None,
            context_dropout: float | None = None,

            context_tokens: torch.Tensor | None = None,
            context_values: torch.Tensor | None = None,
            context_tokens_dropout: float | None = None,

            score_tokens: torch.Tensor | None = None,
            score_values: torch.Tensor | None = None,
            score_tokens_dropout: float | None = None,

            style_embeddings: torch.Tensor | None = None,
            time_steps: torch.Tensor | None = None,
            full_labels: torch.Tensor | None = None,
            type_ids: torch.Tensor | None = None,
            interpolated: torch.Tensor | None = None,
            task_ids: torch.Tensor | None = None,
            task_tokens: torch.Tensor | None = None,

            cache: TupleTransformerCache | None = None,
            return_cache: bool = False,
            output_keys: list | None = None,
            return_embeddings: bool = False,
            return_attn: bool = False,
            output_layer: int | None = None,
            **kwargs
    ) -> TupleTransformerOutput:
        token_emb = None
        if self.token_emb is not None:
            token_emb = self.token_emb(
                tokens[0] if not self.is_multiseq and isinstance(tokens, list) else tokens,
                values=values[0] if not self.is_multiseq and isinstance(values, list) else values,
                cache=cache.token_emb if cache is not None else None
            )
            x = token_emb

            if self.token_pos_emb is not None:
                x = x + self.token_pos_emb(tokens, values)

            if self.pos_emb is not None:
                x = x + self.pos_emb(x)
            x = self.emb_dropout(self.emb_norm(x))

            if vectors is not None:
                x = torch.cat([x, vectors], dim=-1)
        else:
            assert values is not None and vectors is not None
            values = values.clone()
            values[values <= SPECIAL_TOKENS_VALUE] = 0.
            x = torch.cat([values, vectors], dim=-1)

        batch, seq_len = x.shape[:2]
        device = x.device

        contexts = []
        if context is not None:
            if self.context_embedding == EmbeddingMode.CONCAT:
                context = context[:, :x.shape[1]]

            if self.project_context is not None:
                context = self.project_context(context)

            if context is not None:
                if self.context_norm is not None:
                    context = self.context_norm(context)

            context_dropout = context_dropout or (self.context_dropout if self.training else 0.)
            if context_dropout > 0.:
                keep_mask = prob_mask_like((batch,), 1 - context_dropout, device=device)
                if self.null_context is not None:
                    context = torch.where(
                        keep_mask[:, None, None],
                        context,
                        self.null_context[None, None]
                    )
                else:
                    context = context * keep_mask[:, None, None]

            contexts.append(context)

        if self.context_token_embeddings is not None:
            if context_tokens is not None or context_values is not None:
                context_token_emb = self.context_token_embeddings(tokens=context_tokens, values=context_values)

                context_tokens_dropout = context_tokens_dropout or (
                    self.context_tokens_dropout if self.training else 0.
                )
                if context_tokens_dropout > 0.:
                    keep_mask = prob_mask_like((batch,), 1 - context_tokens_dropout, device=device)
                    context_token_emb = context_token_emb * keep_mask[:, None, None]
            else:
                context_token_emb = x.new_zeros(batch, seq_len, self.score_token_embeddings.total_emb_dim)

            contexts.append(context_token_emb)

        if self.score_token_embeddings is not None:
            if score_tokens is not None or score_values is not None:
                score_token_emb = self.score_token_embeddings(tokens=score_tokens, values=score_values)

                score_tokens_dropout = score_tokens_dropout or (self.score_tokens_dropout if self.training else 0.)
                if score_tokens_dropout > 0.:
                    keep_mask = prob_mask_like((batch,), 1 - score_tokens_dropout, device=device)
                    score_token_emb = score_token_emb * keep_mask[:, None, None]
            else:
                score_token_emb = x.new_zeros(batch, seq_len, self.score_token_embeddings.total_emb_dim)

            contexts.append(score_token_emb)

        context = torch.cat(contexts, dim=-1) if len(contexts) > 0 else None

        if self.context_embedding == EmbeddingMode.CONCAT:
            context = context[:, :x.shape[1]]
            x = torch.cat([x, context], dim=-1)
            context = context_mask = None

        if style_embeddings is not None:
            if self.training and self.style_embedding_dropout > 0.:
                keep_mask = prob_mask_like((batch,), 1 - self.style_embedding_dropout, device=device)
                style_embeddings = style_embeddings * keep_mask[:, None, None]

            style_embeddings = style_embeddings[:, :x.shape[1]]
            if self.style_embedding == EmbeddingMode.CONCAT:
                x = torch.cat([x, style_embeddings], dim=-1)

        time_embeddings = None
        if self.time_emb is not None:
            assert time_steps is not None

            if time_steps.ndim == 0:
                time_steps = repeat(time_steps, "-> b", b=batch)

            if time_steps.ndim == 1 and time_steps.shape[0] == 1:
                time_steps = repeat(time_steps, "1 -> b", b=batch)

            time_embeddings = self.time_emb(time_steps.view(-1)).view(time_steps.shape + (-1,))[:, None]
            if self.time_embedding_num > 1:
                assert time_embeddings.ndim == 4
                time_embeddings = time_embeddings.view(batch, 1, -1)

            if self.time_embedding == EmbeddingMode.CONCAT:
                x = torch.cat([x, time_embeddings.expand(-1, x.shape[1], -1)], dim=-1)

        task_embeddings = None
        if task_ids is not None and self.task_emb is not None:
            task_embeddings = self.task_emb(task_ids)[:, None]
        elif task_tokens is not None and self.task_embedding == EmbeddingMode.TOKEN_PREFIX:
            task_embeddings = self.token_emb(task_tokens)

        type_embeddings = None
        if type_ids is not None and self.type_emb is not None:
            type_embeddings = self.type_emb(type_ids.long()[:, :x.shape[1]])

        mode_embeddings = None
        if causal is not None and self.mode_emb is not None:
            mode_embeddings = self.mode_emb(causal.long())[:, None]

        adaptive_condition = []

        if self.style_embedding == EmbeddingMode.ADANORM and style_embeddings is not None:
            adaptive_condition.append(style_embeddings)

        if self.time_embedding == EmbeddingMode.ADANORM and time_embeddings is not None:
            adaptive_condition.append(time_embeddings)

        if self.task_embedding == EmbeddingMode.ADANORM and task_embeddings is not None:
            adaptive_condition.append(task_embeddings)

        if self.type_embedding == EmbeddingMode.ADANORM and type_embeddings is not None:
            adaptive_condition.append(type_embeddings)

        if self.mode_embedding == EmbeddingMode.ADANORM and mode_embeddings is not None:
            adaptive_condition.append(mode_embeddings)

        adaptive_condition = [
            adaptive_cond.expand(-1, x.shape[1], -1) if adaptive_cond.shape[1] == 1 else adaptive_cond
            for adaptive_cond in adaptive_condition
        ]

        adaptive_condition = torch.cat(adaptive_condition, dim=-1) if len(adaptive_condition) else None

        x = self.project_emb(x)

        mask_pad_len = 0
        if task_embeddings is not None:
            if self.task_embedding == EmbeddingMode.SUM:
                x = x + task_embeddings
            elif self.task_embedding in (EmbeddingMode.PREFIX, EmbeddingMode.TOKEN_PREFIX):
                x = torch.cat([task_embeddings, x], dim=1)
                mask_pad_len += task_embeddings.shape[1]

        if type_embeddings is not None:
            if self.type_embedding == EmbeddingMode.SUM:
                x[:, mask_pad_len:] = x[:, mask_pad_len:] + type_embeddings

        if mode_embeddings is not None:
            if self.mode_embedding == EmbeddingMode.SUM:
                x[:, mask_pad_len:] = x[:, mask_pad_len:] + mode_embeddings
            elif self.mode_embedding == EmbeddingMode.PREFIX:
                x = torch.cat([mode_embeddings, x], dim=1)
                mask_pad_len += 1

        if mask_pad_len > 0:
            if mask is not None:
                mask = F.pad(mask, (mask_pad_len, 0), value=1.)
            if attention_mask is not None:
                attention_mask = F.pad(attention_mask, (mask_pad_len, 0, mask_pad_len, 0), value=True)

        output_layer = output_layer or self.transformer_output_layer
        return_embeddings = return_embeddings or (output_layer and output_layer < len(self.transformer.layers) - 1)

        output: TransformerOutput = self.transformer(
            x,
            mask=mask,
            attention_mask=attention_mask,
            causal=causal,
            context=context,
            context_mask=context_mask,
            adaptive_condition=adaptive_condition,
            cache=cache.transformer if cache is not None else None,
            return_cache=True,
            output_layer=output_layer
        )
        out, memory_tokens, intermediates = output.out, output.memory_tokens, output.intermediates
        if (self.transformer_output_layer is not None
                and self.transformer_output_layer < len(self.transformer.layers) - 1):
            out = self.transformer.layers[self.transformer_output_layer + 1].attention_norm(out)

        mode_out = None
        if mode_embeddings is not None and self.mode_embedding == EmbeddingMode.PREFIX:
            mode_out, out = out[:, :1], out[:, 1:]

        task_out = None
        if task_embeddings is not None and self.task_embedding in (EmbeddingMode.PREFIX, EmbeddingMode.TOKEN_PREFIX):
            task_out, out = out[:, :task_embeddings.shape[1]], out[:, task_embeddings.shape[1]:]

        logits = None
        if not return_embeddings and self.lm_head is not None:
            if isinstance(self.lm_head, (TupleTransformerCausalLMHead, TupleTransformerDFMHead)):
                if self.token_indices is not None:
                    full_labels = full_labels[..., self.token_indices]
                logits = self.lm_head(out, labels=full_labels, keys=output_keys)
            else:
                logits = self.lm_head(out, keys=output_keys)

        pred_values = None
        if not return_embeddings and self.value_head is not None:
            if isinstance(self.value_head, TupleTransformerSplitValueHead):
                pred_values = self.value_head(out, keys=output_keys)
            else:
                pred_values = self.value_head(out)

        task_logits = None
        if not return_embeddings and self.task_head is not None:
            task_logits = self.task_head(out)

        type_logits = None
        if not return_embeddings and self.type_head is not None:
            type_logits = self.type_head(out)

        attn_maps = None
        if return_attn:
            attn_maps = list(map(lambda t: t.attention.scores, intermediates.layers))

        cache = None
        if return_cache:
            cache = TupleTransformerCache(
                token_emb=token_emb,
                transformer=intermediates
            )

        return TupleTransformerOutput(
            hidden_state=out,
            memory_state=memory_tokens,
            task_state=task_out,
            mode_state=mode_out,
            logits=logits,
            attentions=attn_maps,
            cache=cache,
            values=pred_values,
            task_logits=task_logits,
            type_logits=type_logits
        )
