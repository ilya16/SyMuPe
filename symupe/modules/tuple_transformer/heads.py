""" TupleTransformer's language and vector modeling heads. """
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import MISSING, DictConfig

from symupe.data.tokenizers import OctupleM
from symupe.modules.constructor import Registry, VariableModuleConfig, Constructor
from symupe.modules.layers import ScaledTanh, TimePositionalEmbedding
from symupe.modules.sampling import (
    filter_and_sample,
    x2prob, cubic_scheduler, cubic_scheduler_derivative, sample_p
)
from symupe.modules.transformer import (
    TransformerConfig, TransformerOutput,
    DecoderTransformerConfig, DecoderTransformer,
    EncoderTransformerConfig, EncoderTransformer
)
from .embeddings import TupleTransformerEmbeddings

TupleTransformerHeadsRegistry = type("_TupleTransformerHeadsRegistry", (Registry,), {})()


@dataclass
class TupleTransformerHeadsConfig(VariableModuleConfig):
    dim: int = MISSING


# Language Modeling Heads


@dataclass
class _TupleTransformerLMHeadConfig(TupleTransformerHeadsConfig):
    num_tokens: dict[str, int] | None = None
    embeddings: TupleTransformerEmbeddings | None = None
    keys: list[str] | None = None


class _TupleTransformerLMHead(nn.Module, Constructor):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
            self,
            x: torch.Tensor,
            labels: torch.Tensor | None = None,
            keys: list[str | int] | None = None
    ):
        ...

    def infer(
            self,
            x: torch.Tensor,
            keys: list[str | int] | None = None,
            temperature: float = 1.,
            top_k: float | int = -1,
            top_p: float = 0.8,
            num_first_mask: int = 1,
            filter_key_ids: dict[str, list] | None = None,
            ignore_non_special: list[str] | None = None,
            tokenizer: OctupleM = None
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        mask_value = -float("Inf")
        filter_key_ids = filter_key_ids or {}
        ignore_non_special = ignore_non_special or []

        logits = self(x, keys=keys)

        pred_tokens = {}
        for key, key_logits in logits.items():
            if num_first_mask > 0:
                key_logits[..., :num_first_mask] = mask_value

            filter_ids = filter_key_ids.get(key, None)
            if filter_ids is not None:
                key_logits[:, filter_ids] = mask_value

            if key in ignore_non_special:
                key_logits[:, tokenizer.zero_token:] = mask_value

            pred_tokens[key] = filter_and_sample(
                key_logits, temperature=temperature, top_k=top_k, top_p=top_p
            )

        pred_tokens = torch.cat(list(pred_tokens.values()), dim=-1)[None]
        pred_values = tokenizer.decode_values(pred_tokens, token_type=list(logits.keys()), normalize=True).float()

        return logits, pred_tokens, pred_values


@dataclass
class TupleTransformerLMHeadConfig(_TupleTransformerLMHeadConfig):
    _target_: str = "lm"
    bias: bool = False


@TupleTransformerHeadsRegistry.register("lm")
class TupleTransformerLMHead(_TupleTransformerLMHead):
    def __init__(
            self,
            dim: int,
            num_tokens: dict[str, int] | None = None,
            embeddings: TupleTransformerEmbeddings | None = None,
            keys: list[str] | None = None,
            bias: bool = False
    ):
        assert num_tokens is not None or embeddings is not None
        super().__init__()

        self.num_tokens = num_tokens or embeddings.num_tokens
        self.heads = nn.ModuleDict({
            key: nn.Linear(dim, num, bias=bias)
            for key, num in self.num_tokens.items()
            if not keys or key in keys
        })
        self.keys = list(self.heads.keys())

    def forward(self, x: torch.Tensor, keys: list[str | int] | None = None, **kwargs) -> dict[str, torch.Tensor]:
        logits = {
            key: head(x)
            for i, (key, head) in enumerate(self.heads.items())
            if keys is None or i in keys or key in keys
        }
        return logits


@dataclass
class TupleTransformerTiedLMHeadConfig(_TupleTransformerLMHeadConfig):
    _target_: str = "lm-tied"
    reuse_projection: bool = True
    emb_norm: bool = True


@TupleTransformerHeadsRegistry.register("lm-tied")
class TupleTransformerTiedLMHead(_TupleTransformerLMHead):
    def __init__(
            self,
            dim: int,
            embeddings: TupleTransformerEmbeddings,
            reuse_projection: bool = True,
            emb_norm: bool = True,
            keys: list[str] | None = None
    ):
        super().__init__()

        self.embs = [embeddings.embs]
        self.total_emb_dim = embeddings.total_emb_dim
        self.split_dims = [token_emb.embedding_dim for token_emb in embeddings.embs.values()]

        self.project_emb = nn.Linear(dim, self.total_emb_dim, bias=False)
        if reuse_projection:
            assert dim == embeddings.project_emb.out_features, \
                f"Projection layer could be reused only if last input tensor dimension " \
                f"is equal to projection layer's `out_features = {embeddings.project_emb.out_features}`"
            self.project_emb.weight = embeddings.project_emb.weight

        self.norm = nn.LayerNorm(self.total_emb_dim) if emb_norm else nn.Identity()

    def forward(self, x: torch.Tensor, keys: list[str | int] | None = None, **kwargs) -> dict[str, torch.Tensor]:
        token_embs = self.norm(x @ self.project_emb.weight).split(self.split_dims, dim=-1)

        logits = {
            key: token_embs[i] @ self.embs[0][key].weight.t()
            for i, key in enumerate(self.embs[0].keys())
            if keys is None or i in keys or key in keys
        }
        return logits


@dataclass
class TupleTransformerTiedProjLMHeadConfig(_TupleTransformerLMHeadConfig):
    _target_: str = "lm-tied-proj"
    emb_norm: bool = True


@TupleTransformerHeadsRegistry.register("lm-tied-proj")
class TupleTransformerTiedProjLMHead(_TupleTransformerLMHead):
    def __init__(
            self,
            dim: int,
            embeddings: TupleTransformerEmbeddings,
            num_tokens: dict[str, int] | None = None,
            keys: list[str] | None = None,
            split: bool = False,
            emb_norm: bool = True
    ):
        assert num_tokens is not None or embeddings is not None
        super().__init__()

        self.total_emb_dim = embeddings.total_emb_dim

        self.split = split
        self.split_dims = [token_emb.embedding_dim for token_emb in embeddings.embs.values()]

        self.project_emb = nn.Linear(dim, self.total_emb_dim, bias=False)
        assert dim == embeddings.project_emb.out_features, \
            f"Projection layer could be reused only if last input tensor dimension " \
            f"is equal to projection layer's `out_features = {embeddings.project_emb.out_features}`"
        self.project_emb.weight = embeddings.project_emb.weight

        self.norm = nn.LayerNorm(self.total_emb_dim) if emb_norm else nn.Identity()

        num_tokens = num_tokens or embeddings.num_tokens
        self.heads = nn.ModuleDict({
            key: nn.Linear(embeddings.embs[key].embedding_dim if split else self.total_emb_dim, num, bias=False)
            for key, num in num_tokens.items()
            if not keys or key in keys
        })

    def forward(self, x: torch.Tensor, keys: list[str | int] | None = None, **kwargs) -> dict[str, torch.Tensor]:
        token_emb = self.norm(x @ self.project_emb.weight)

        if self.split:
            token_embs = token_emb.split(self.split_dims, dim=-1)
            logits = {
                key: head(token_embs[i])
                for i, (key, head) in enumerate(self.heads.items())
                if keys is None or i in keys or key in keys
            }
        else:
            logits = {
                key: head(token_emb)
                for i, (key, head) in enumerate(self.heads.items())
                if keys is None or i in keys or key in keys
            }
        return logits


@dataclass
class TupleTransformerTiedSplitLMHeadConfig(_TupleTransformerLMHeadConfig):
    _target_: str = "lm-tied-split"


@TupleTransformerHeadsRegistry.register("lm-tied-split")
class TupleTransformerTiedSplitLMHead(_TupleTransformerLMHead):
    def __init__(
            self,
            dim: int,
            embeddings: TupleTransformerEmbeddings,
            keys: list[str] | None = None
    ):
        super().__init__()

        to_embs = {}
        for key, token_emb in embeddings.embs.items():
            if not keys or key in keys:
                to_embs[key] = nn.Sequential(
                    nn.Linear(dim, token_emb.embedding_dim),
                    nn.LayerNorm(token_emb.embedding_dim),
                )

        self.to_embs = nn.ModuleDict(to_embs)
        self.embs = [embeddings.embs]

    def forward(self, x: torch.Tensor, keys: list[str | int] | None = None, **kwargs) -> dict[str, torch.Tensor]:
        logits = {
            key: self.to_embs[key](x) @ self.embs[0][key].weight.t()
            for i, key in enumerate(self.embs[0].keys())
            if keys is None or i in keys or key in keys
        }
        return logits


@dataclass
class TupleTransformerCausalLMHeadConfig(_TupleTransformerLMHeadConfig):
    _target_: str = "lm-causal"
    transformer: DictConfig | TransformerConfig = field(default_factory=lambda: DecoderTransformerConfig())
    type_embeddings: bool = False
    prior_input: bool = True
    bias: bool = False


@TupleTransformerHeadsRegistry.register("lm-causal")
class TupleTransformerCausalLMHead(_TupleTransformerLMHead):
    def __init__(
            self,
            dim: int,
            num_tokens: dict[str, int] | None = None,
            embeddings: TupleTransformerEmbeddings | None = None,
            transformer: DictConfig | TransformerConfig = DecoderTransformerConfig(),
            type_embeddings: bool = False,
            prior_input: bool = True,
            keys: list[str] | None = None,
            bias: bool = False
    ):
        assert embeddings is not None
        super().__init__()

        self.embedding_dim = list(embeddings.embs.values())[0].embedding_dim
        assert all([token_emb.embedding_dim == self.embedding_dim for token_emb in embeddings.embs.values()])

        num_embeddings = len(keys) if keys is not None else len(embeddings.embs)

        self.token_emb = [embeddings]
        self.project_emb = nn.Linear(dim, num_embeddings * self.embedding_dim, bias=False)

        self.type_embeddings = None
        if type_embeddings:
            self.type_embeddings = nn.Parameter(0.01 * torch.randn((num_embeddings, self.embedding_dim)))

        self.prior_input = prior_input

        input_dim = self.embedding_dim * 2 if prior_input else self.embedding_dim
        self.project_in = nn.Linear(
            input_dim, transformer.dim, bias=False
        ) if input_dim != transformer.dim else nn.Identity()

        self.transformer = DecoderTransformer.init(transformer)

        self.num_tokens = num_tokens or embeddings.num_tokens
        self.heads = nn.ModuleDict({
            key: nn.Linear(transformer.dim, num, bias=bias)
            for key, num in self.num_tokens.items()
            if not keys or key in keys
        })
        self.keys = list(self.heads.keys())

    def forward(
            self,
            x: torch.Tensor,
            labels: torch.Tensor,
            keys: list[str | int] | None = None,
    ) -> dict[str, torch.Tensor]:
        b, t = x.shape[:2]

        labels_embs = torch.stack(list(
            self.token_emb[0]._forward_embeddings(labels, keys=self.keys).values()
        ), dim=-2)
        labels_embs = labels_embs.view(b * t, -1, self.embedding_dim)
        labels_embs = F.pad(labels_embs, (0, 0, 1, 0))[:, :-1]

        if self.type_embeddings is not None:
            labels_embs = labels_embs + self.type_embeddings[None]

        prior_embs = self.project_emb(x)  # initial embedding for each token
        prior_embs = prior_embs.view(b * t, -1, self.embedding_dim)

        x = labels_embs
        if self.prior_input:
            x = torch.cat([prior_embs, labels_embs], dim=-1)  # combine with embeddings of known past labels

        x = self.project_in(x)
        x = self.transformer(
            x,
            context=prior_embs if self.transformer.expects_context else None
        ).out  # pass through transformer
        x = x.view(b, t, prior_embs.shape[-2], -1)

        logits = {
            key: head(x[..., i, :])
            for i, (key, head) in enumerate(self.heads.items())
            if keys is None or i in keys or key in keys
        }

        return logits

    def infer(
            self,
            x: torch.Tensor,
            keys: list[str | int] | None = None,
            temperature: float = 1.,
            top_k: float | int = -1,
            top_p: float = 0.8,
            num_first_mask: int = 1,
            filter_key_ids: dict[str, list] | None = None,
            ignore_non_special: list[str] | None = None,
            tokenizer: OctupleM = None
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        x = x[:, None] if x.ndim == 2 else x
        b, t = x.shape[:2]
        mask_value = -float("Inf")
        filter_key_ids = filter_key_ids or {}
        ignore_non_special = ignore_non_special or []

        prior_embs = self.project_emb(x)  # initial embedding for each token
        prior_embs = prior_embs.view(b * t, -1, self.embedding_dim)

        labels_embs = torch.zeros_like(prior_embs)
        if self.type_embeddings is not None:
            labels_embs[:, 0] = self.type_embeddings[0]

        x = labels_embs
        if self.prior_input:
            x = torch.cat([prior_embs, labels_embs], dim=-1)  # combine with embeddings of known past labels

        logits, pred_tokens = {}, []
        cache = None
        for i, (key, head) in enumerate(self.heads.items()):
            x_i = self.project_in(x[:, :i + 1])
            output: TransformerOutput = self.transformer(
                x_i, context=prior_embs if self.transformer.expects_context else None,
                cache=cache, return_cache=True
            )
            out_i, cache = output.out, output.intermediates
            key_logits = head(out_i[:, -1])

            key_logits[..., :num_first_mask] = mask_value

            filter_ids = filter_key_ids.get(key, None)
            if filter_ids is not None:
                key_logits[..., filter_ids] = mask_value

            if key in ignore_non_special:
                key_logits[..., tokenizer.zero_token:] = mask_value

            pred_tokens_i = filter_and_sample(key_logits, temperature=temperature, top_k=top_k, top_p=top_p)
            if keys is None or key in keys:
                logits[key] = key_logits
                pred_tokens.append(pred_tokens_i.view(b, t, 1))

            if i + 1 < x.shape[-2]:
                next_label_emb = self.token_emb[0].embs[key](pred_tokens_i.squeeze(1))
                if self.type_embeddings is not None:
                    next_label_emb = next_label_emb + self.type_embeddings[i + 1]
                x[:, i + 1, -self.embedding_dim:] = next_label_emb

        pred_tokens = torch.cat(pred_tokens, dim=-1)
        pred_values = tokenizer.decode_values(pred_tokens, token_type=list(logits.keys()), normalize=True).float()

        return logits, pred_tokens, pred_values


@dataclass
class TupleTransformerDFMHeadConfig(_TupleTransformerLMHeadConfig):
    _target_: str = "dfm"
    transformer: DictConfig | TransformerConfig = field(default_factory=lambda: EncoderTransformerConfig())
    time_embedding_dim: int = 64,
    type_embeddings: bool = False
    distribution: str = "uniform"
    bias: bool = False


@TupleTransformerHeadsRegistry.register("dfm")
class TupleTransformerDFMHead(_TupleTransformerLMHead):
    def __init__(
            self,
            dim: int,
            num_tokens: dict[str, int] | None = None,
            embeddings: TupleTransformerEmbeddings | None = None,
            transformer: DictConfig | TransformerConfig = EncoderTransformerConfig(),
            time_embedding_dim: int = 64,
            type_embeddings: bool = False,
            distribution: str = "uniform",
            keys: list[str] | None = None,
            bias: bool = False
    ):
        assert embeddings is not None
        super().__init__()

        self.embedding_dim = list(embeddings.embs.values())[0].embedding_dim
        assert all([token_emb.embedding_dim == self.embedding_dim for token_emb in embeddings.embs.values()])

        num_embeddings = len(keys) if keys is not None else len(embeddings.embs)

        self.token_emb = [embeddings]
        self.project_emb = nn.Linear(dim, num_embeddings * self.embedding_dim, bias=False)

        self.type_embeddings = None
        if type_embeddings:
            self.type_embeddings = nn.Parameter(0.01 * torch.randn((num_embeddings, self.embedding_dim)))

        input_dim = self.embedding_dim * 2
        self.project_in = nn.Linear(
            input_dim, transformer.dim, bias=False
        ) if input_dim != transformer.dim else nn.Identity()

        self.time_embedding_dim = time_embedding_dim
        self.time_emb = TimePositionalEmbedding(
            freq_dim=time_embedding_dim, emb_dim=time_embedding_dim, with_steps=True
        )

        self.transformer = EncoderTransformer.init(
            transformer,
            adaptive_norm=True,
            condition_dim=self.time_embedding_dim
        )

        self.num_tokens = num_tokens or embeddings.num_tokens
        self.heads = nn.ModuleDict({
            key: nn.Linear(transformer.dim, num, bias=bias)
            for key, num in self.num_tokens.items()
            if not keys or key in keys
        })
        self.keys = list(self.heads.keys())

        self.register_buffer("token_nums", torch.tensor(list(self.num_tokens.values())), persistent=False)

        self.mask_token_id = 1
        assert distribution in ("mask", "uniform")
        self.distribution = distribution

    def forward(
            self,
            x: torch.Tensor,
            labels: torch.Tensor,
            keys: list[str | int] | None = None
    ) -> dict[str, torch.Tensor]:
        b, t = x.shape[:2]

        # main conditional flow logic
        x_1 = labels.clone().view(b * t, -1)

        # x_0 is noisy/masked input
        if self.distribution == "uniform":
            x_0 = (torch.rand(b * t, len(self.token_nums), device=x.device) * self.token_nums).long()
        else:
            x_0 = torch.full_like(x_1, fill_value=self.mask_token_id)

        # random times
        time_steps = torch.rand((b * t, 1), dtype=x.dtype, device=x.device)

        # sample x_t
        x_t = x_0.clone()
        mask = torch.rand_like(x_1.float()) < cubic_scheduler(time_steps)
        x_t[mask] = x_1[mask]

        x_t_embs = torch.stack(list(
            self.token_emb[0]._forward_embeddings(x_t, keys=self.keys).values()
        ), dim=-2)

        if self.type_embeddings is not None:
            x_t_embs = x_t_embs + self.type_embeddings[None]

        prior_embs = self.project_emb(x)  # initial embedding for each token
        prior_embs = prior_embs.view(b * t, -1, self.embedding_dim)

        x = torch.cat([prior_embs, x_t_embs], dim=-1)  # combine with embeddings of known past labels
        x = self.project_in(x)

        time_embeddings = self.time_emb(time_steps)

        out = self.transformer(
            x,
            adaptive_condition=time_embeddings
        ).out  # pass through transformer
        out = out.view(b, t, prior_embs.shape[-2], -1)

        logits = {
            key: head(out[..., i, :])
            for i, (key, head) in enumerate(self.heads.items())
            if keys is None or i in keys or key in keys
        }

        return logits

    def infer(
            self,
            x: torch.Tensor,
            keys: list[str | int] | None = None,
            temperature: float = 1.,
            top_k: float | int = -1,
            top_p: float = 0.8,
            steps: int = 4,
            num_first_mask: int = 1,
            filter_key_ids: dict[str, list] | None = None,
            ignore_non_special: list[str] | None = None,
            tokenizer: OctupleM = None
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        x = x[:, None] if x.ndim == 2 else x
        b, t = x.shape[:2]
        mask_value = -float("Inf")
        filter_key_ids = filter_key_ids or {}
        ignore_non_special = ignore_non_special or []

        prior_embs = self.project_emb(x)  # initial embedding for each token
        prior_embs = prior_embs.view(b * t, -1, self.embedding_dim)

        if self.distribution == "uniform":
            x_0 = (torch.rand(b * t, len(self.token_nums), device=x.device) * self.token_nums).long()
        else:
            x_0 = torch.full((b * t, len(self.keys)), fill_value=self.mask_token_id, dtype=torch.long, device=x.device)

        time_steps = torch.linspace(0, 1, steps + 1)
        time_embeddings = self.time_emb(time_steps)

        x_t = x_0.clone()
        p_1t = {}

        for i, time_step in enumerate(time_steps[:-1]):
            dt = time_steps[i + 1] - time_steps[i]

            delta_t = {key: x2prob(x_t[..., i], num) for i, (key, num) in enumerate(self.num_tokens.items())}

            x_t_embs = torch.stack(list(
                self.token_emb[0]._forward_embeddings(x_t, keys=self.keys).values()
            ), dim=-2)

            if self.type_embeddings is not None:
                x_t_embs = x_t_embs + self.type_embeddings[None]

            x_i = torch.cat([prior_embs, x_t_embs], dim=-1)  # combine with embeddings of known past labels
            x_i = self.project_in(x_i)

            out = self.transformer(
                x_i,
                adaptive_condition=time_embeddings[i][None]
            ).out

            p_1t = {
                key: head(out[..., i, :])
                for i, (key, head) in enumerate(self.heads.items())
                if keys is None or i in keys or key in keys
            }

            for key, key_logits in p_1t.items():
                filter_ids = filter_key_ids.get(key, None)
                if filter_ids is not None:
                    key_logits[..., filter_ids] = mask_value

                if key in ignore_non_special:
                    key_logits[..., tokenizer.zero_token:] = mask_value

            if time_step < time_steps[-2]:
                kappa_coeff = cubic_scheduler_derivative(time_step) / (1 - cubic_scheduler(time_step))

                p_t, x_t = {}, {}
                for i, (key, num) in enumerate(self.num_tokens.items()):
                    p_t[key] = delta_t[key] + dt * kappa_coeff * (p_1t[key].softmax(-1) - delta_t[key])
                    x_t[key] = sample_p(p_t[key])

                x_t = torch.stack(list(x_t.values()), dim=-1)

        logits = {key: p1t_i.view(b, t, -1) for key, p1t_i in p_1t.items()}

        x_1 = {}
        for key, p1t_i in p_1t.items():
            x_1[key] = sample_p(p1t_i.softmax(-1))
        pred_tokens = torch.stack(list(x_1.values()), dim=-1).view(b, t, -1)
        pred_values = tokenizer.decode_values(pred_tokens, token_type=list(logits.keys()), normalize=True).float()

        return logits, pred_tokens, pred_values


# Value Prediction Heads


@dataclass
class _TupleTransformerValueHeadConfig(TupleTransformerHeadsConfig):
    ...


class _TupleTransformerValueHead(nn.Module, Constructor):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor):
        ...

    def infer(
            self,
            x: torch.Tensor,
            keys: list[str | int] | None = None,
            tokenizer: OctupleM = None
    ) -> tuple[list[str] | None, torch.Tensor, torch.Tensor]:
        pred_values = self(x, keys=keys)

        token_types = None
        if isinstance(pred_values, dict):
            token_types = list(pred_values.keys())
            pred_values = torch.cat(list(pred_values.values()), dim=-1)[None].float()

        pred_tokens = tokenizer.encode_tokens(pred_values, token_type=token_types, denormalize=True)

        return token_types, pred_tokens, pred_values


@dataclass
class TupleTransformerValueHeadConfig(_TupleTransformerValueHeadConfig):
    _target_: str = "value"
    num_features: int | None = MISSING
    keys: list[str] | None = None
    ranges: list[tuple[float, float]] | None = None


@TupleTransformerHeadsRegistry.register("value")
class TupleTransformerValueHead(_TupleTransformerValueHead):
    def __init__(
            self,
            dim: int,
            num_features: int | None = None,
            keys: list[str] | None = None,
            ranges: list[tuple[float, float] | None] | None = None
    ):
        super().__init__()
        assert num_features is not None or keys is not None
        self.num_features = num_features or len(keys)
        self.head = nn.Linear(dim, self.num_features)

        self.limiters = None
        if ranges is not None:
            assert len(ranges) == self.num_features
            self.limiters = nn.ModuleList([
                ScaledTanh(*value_range) if value_range is not None else nn.Identity()
                for value_range in ranges
            ])

    def forward(self, x: torch.Tensor, keys: list[str | int] | None = None) -> torch.Tensor:
        values = self.head(x)

        if self.limiters is not None:
            values = torch.stack([
                self.limiters[i](values[..., i])
                for i in range(len(self.limiters))
            ], dim=-1)

        return values


@dataclass
class TupleTransformerSplitValueHeadConfig(_TupleTransformerValueHeadConfig):
    _target_: str = "value-split"
    keys: list[str] = MISSING
    ranges: dict[str, tuple[float, float] | None] | None = None


@TupleTransformerHeadsRegistry.register("value-split")
class TupleTransformerSplitValueHead(_TupleTransformerValueHead):
    def __init__(
            self,
            dim: int,
            keys: list[str],
            ranges: dict[str, tuple[float, float] | None] | None = None
    ):
        super().__init__()

        self.num_features = len(keys)

        ranges = ranges or {}
        self.heads = nn.ModuleDict({
            key: nn.Sequential(
                nn.Linear(dim, 1),
                ScaledTanh(*ranges[key]) if key in ranges and ranges[key] is not None else nn.Identity()
            )
            for key in keys
        })

    def forward(self, x: torch.Tensor, keys: list[str | int] | None = None) -> dict[str, torch.Tensor]:
        values = {
            key: layer(x).squeeze(-1)
            for i, (key, layer) in enumerate(self.heads.items())
            if keys is None or i in keys or key in keys
        }

        return values


@dataclass
class TupleTransformerEmbeddingHeadConfig(TupleTransformerHeadsConfig):
    _target_: str = "embedding"
    emb_dim: int = MISSING
    hidden_dim: int | None = None
    depth: int = 2
    detach_inputs: bool | float = False


@TupleTransformerHeadsRegistry.register("embedding")
class TupleTransformerEmbeddingHead(nn.Module, Constructor):
    def __init__(
            self,
            dim: int,
            emb_dim: int,
            hidden_dim: int | None = None,
            depth: int = 2,
            detach_inputs: bool | float = False
    ):
        super().__init__()

        hidden_dim = hidden_dim or emb_dim

        input_dims = [dim] + [hidden_dim] * (depth - 1)
        output_dims = [hidden_dim] * (depth - 1) + [emb_dim]

        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(input_dims, output_dims)):
            layers.append(nn.Linear(in_dim, out_dim))
            if i < depth - 1:
                layers.append(nn.Mish())

        self.layers = nn.Sequential(*layers)

        self.detach_inputs = detach_inputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.detach_inputs * x.detach() + (1 - self.detach_inputs) * x
        embeddings = self.layers(x)
        return embeddings
