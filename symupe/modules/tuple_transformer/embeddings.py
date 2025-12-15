""" TupleTransformer's token and value embeddings. """
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from omegaconf import MISSING

from symupe.data.tokenizers.constants import MASK_TOKEN, SOS_TOKEN, EOS_TOKEN
from symupe.modules.constructor import Constructor, Registry, VariableModuleConfig
from symupe.modules.transformer import (
    DiscreteContinuousEmbedding, DiscreteSinusoidalEmbedding,
    AttentionConfig, Attention
)
from symupe.utils import ExplicitEnum


class EmbeddingMode(ExplicitEnum):
    CONCAT = "cat"
    SUM = "sum"


TupleTransformerEmbeddingsRegistry = type("_TupleTransformerEmbeddingsRegistry", (Registry,), {})()


@dataclass
class TupleTransformerEmbeddingsConfig(VariableModuleConfig):
    _target_: str = "simple"
    num_tokens: dict[str, int] = MISSING
    emb_dims: dict[str, int] | int = MISSING
    mode: str | EmbeddingMode = EmbeddingMode.CONCAT
    project_emb_dim: int | None = None
    project_bias: bool = True
    emb_norm: bool = False
    discrete: bool = True
    continuous: bool | list[str] = False
    sinusoidal: bool | dict[str, list] = False
    sinusoidal_learned: bool = False
    sinusoidal_dense: bool = False
    token_values: dict[str, list] | None = None
    num_first_discrete: int = 0
    special_tokens: dict[str, int] | None = None
    embedding_kwargs: dict | None = None
    tie_keys: dict[str, str] | None = None
    attention: AttentionConfig | None = None
    type_embeddings: bool = False


@TupleTransformerEmbeddingsRegistry.register("simple")
class TupleTransformerEmbeddings(nn.Module, Constructor):
    def __init__(
            self,
            num_tokens: dict[str, int],
            emb_dims: dict[str, int] | int,
            mode: str = EmbeddingMode.CONCAT,
            project_emb_dim: int | None = None,
            project_bias: bool = True,
            emb_norm: bool = False,
            discrete: bool | list[str] = True,
            continuous: bool | list[str] = False,
            sinusoidal: bool | list[str] = False,
            sinusoidal_learned: bool = False,
            sinusoidal_dense: bool = False,
            token_values: dict[str, list] | None = None,
            num_first_discrete: int = 0,
            special_tokens: dict[str, int] | None = None,
            embedding_kwargs: dict | None = None,
            tie_keys: dict[str, str] | None = None,
            attention: AttentionConfig | None = None,
            type_embeddings: bool = False
    ):
        super().__init__()

        self.mode = mode

        if self.mode == EmbeddingMode.SUM or attention is not None:
            assert (
                    isinstance(emb_dims, int)
                    or all([emb_dim == list(emb_dims.values())[0] for emb_dim in emb_dims.values()])
            ), "`emb_dims` in TupleTokenEmbeddings' `sum`/`attention` mode should be the same for all keys."

        if isinstance(discrete, bool):
            discrete_keys = [key for key in num_tokens] if discrete else []
        else:
            discrete_keys = discrete

        if isinstance(continuous, bool):
            continuous_keys = [key for key in num_tokens] if continuous else []
        else:
            continuous_keys = continuous

        if isinstance(sinusoidal, bool):
            sinusoidal_keys = [key for key in num_tokens] if sinusoidal else []
        else:
            sinusoidal_keys = sinusoidal

        for key in num_tokens:
            assert not (key in continuous_keys and key in sinusoidal_keys), \
                f"Embedding for token type `{key}` should be either Continuous or Sinusoidal."
            if key not in continuous_keys and key not in sinusoidal_keys and key not in discrete_keys:
                discrete_keys.append(key)

        token_values = token_values or {}
        embedding_kwargs = embedding_kwargs or {}
        special_tokens = special_tokens or {}
        num_first_discrete = max(num_first_discrete, len(special_tokens))

        embeddings, total_emb_dim = {}, 0
        for key, num in num_tokens.items():
            emb_dim = emb_dims if isinstance(emb_dims, int) else emb_dims[key]
            if tie_keys and key in tie_keys:
                embeddings[key] = embeddings[tie_keys[key]]
                emb_dim = emb_dims if isinstance(emb_dims, int) else emb_dims[tie_keys[key]]
            elif key in continuous_keys:
                embeddings[key] = DiscreteContinuousEmbedding(
                    num_embeddings=num,
                    embedding_dim=emb_dim,
                    discrete=key in discrete_keys,
                    continuous=True,
                    num_first_discrete=num_first_discrete,
                    token_values=token_values.get(key, None),
                    padding_idx=0,
                    **embedding_kwargs
                )
            elif key in sinusoidal_keys:
                sinusoidal_learned = sinusoidal_learned or sinusoidal_dense
                embeddings[key] = DiscreteSinusoidalEmbedding(
                    num_embeddings=num,
                    embedding_dim=emb_dim,
                    learned=sinusoidal_learned,
                    discrete=key in discrete_keys,
                    num_first_discrete=num_first_discrete,
                    token_values=token_values.get(key, None),
                    padding_idx=0,
                    **embedding_kwargs
                )
            else:
                embeddings[key] = nn.Embedding(num, emb_dim, padding_idx=0)
            total_emb_dim += emb_dim if self.mode == EmbeddingMode.CONCAT else emb_dim - total_emb_dim

        self.embs = nn.ModuleDict(embeddings)
        self.norm = nn.LayerNorm(total_emb_dim) if emb_norm else nn.Identity()

        project_emb_dim = project_emb_dim or total_emb_dim
        self.project_emb = nn.Identity()
        if self.mode == EmbeddingMode.CONCAT:
            self.project_emb = nn.Linear(total_emb_dim, project_emb_dim, bias=project_bias)

        self.attention = None
        if attention is not None:
            emb_dim = emb_dims if isinstance(emb_dims, int) else list(emb_dims.values())[0]
            self.attention = Attention.init(
                attention,
                dim=emb_dim,
                causal=False
            )

        self.type_embeddings = None
        if type_embeddings:
            emb_dim = emb_dims if isinstance(emb_dims, int) else list(emb_dims.values())[0]
            self.type_embeddings = nn.Parameter(torch.zeros((len(self.embs), emb_dim)))

        self.num_tokens = num_tokens
        self.emb_dims = emb_dims
        self.total_emb_dim = total_emb_dim
        self.project_emb_dim = project_emb_dim

        self.discrete = len(discrete_keys)
        self.discrete_keys = discrete_keys

        self.continuous = len(continuous_keys)
        self.continuous_keys = continuous_keys

        self.sinusoidal = len(sinusoidal_keys)
        self.sinusoidal_keys = sinusoidal_keys

        self.num_first_discrete = num_first_discrete
        self.special_tokens = special_tokens
        self.token_values = token_values

        self.init_()

    def init_(self):
        for key, emb in self.embs.items():
            if isinstance(emb, nn.Embedding):
                nn.init.xavier_uniform_(emb.weight)
        if self.mode == EmbeddingMode.CONCAT:
            nn.init.xavier_uniform_(self.project_emb.weight)

    def _forward_embeddings(
            self,
            x: torch.Tensor,
            values: torch.Tensor | None = None,
            keys: list[str] | None = None
    ) -> dict[str, torch.Tensor]:
        keys = keys or self.embs.keys()
        return {
            key: self.embs[key](x[..., i], values=values[..., i] if values is not None else values)
            if key in self.continuous_keys or key in self.sinusoidal_keys else self.embs[key](x[..., i])
            for i, key in enumerate(keys)
        }

    def _forward_project(
            self,
            token_embs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        token_embs = list(token_embs.values())

        if self.type_embeddings is not None:
            token_embs = [
                token_emb + self.type_embeddings[i]
                for i, token_emb in enumerate(token_embs)
            ]

        if self.attention is not None:
            x = torch.stack(token_embs, dim=-2)
            b, t, c, d = x.shape
            x = x.view(-1, c, d)
            x = self.attention(x)[0]
            token_embs = x.view(b, t, c, d)

            if self.mode == EmbeddingMode.CONCAT:
                total_token_emb = token_embs.view(b, t, c * d)
            else:
                total_token_emb = token_embs.sum(dim=-2)
        else:
            if self.mode == EmbeddingMode.CONCAT:
                total_token_emb = torch.cat(token_embs, dim=-1)
            else:
                total_token_emb = sum(token_embs)

        return self.project_emb(self.norm(total_token_emb))

    def forward(
            self,
            tokens: torch.Tensor,
            values: torch.Tensor | None = None,
            cache: torch.Tensor | None = None,
            return_embeddings: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if cache is not None:
            tokens = tokens[:, cache.shape[1]:]
            values = values[:, cache.shape[1]:] if values is not None else values

        token_embs = self._forward_embeddings(tokens, values)
        token_emb = self._forward_project(token_embs)

        if cache is not None:
            token_emb = torch.cat([cache, token_emb], dim=1)

        if return_embeddings:
            return token_emb, token_embs
        else:
            return token_emb


class MultiSeqEmbeddingMode(ExplicitEnum):
    PRE_SUM = "pre-sum"
    POST_SUM = "post-sum"
    POST_CAT = "post-cat"


@dataclass
class MultiSeqTupleTransformerEmbeddingsConfig(TupleTransformerEmbeddingsConfig):
    _target_: str = "multi-seq"
    multi_mode: str = MultiSeqEmbeddingMode.PRE_SUM
    num_sequences: int = 2


@TupleTransformerEmbeddingsRegistry.register("multi-seq")
class MultiSeqTupleTransformerEmbeddings(TupleTransformerEmbeddings):
    def __init__(
            self,
            num_tokens: dict[str, int],
            emb_dims: dict[str, int] | int,
            mode: str = EmbeddingMode.CONCAT,
            project_emb_dim: int | None = None,
            project_bias: bool = True,
            emb_norm: bool = False,
            discrete: bool | list[str] = True,
            continuous: bool | list[str] = False,
            sinusoidal: bool | list[str] = False,
            sinusoidal_learned: bool = False,
            sinusoidal_dense: bool = False,
            token_values: dict[str, list] | None = None,
            num_first_discrete: int = 0,
            special_tokens: dict[str, int] | None = None,
            embedding_kwargs: dict | None = None,
            tie_keys: dict[str, str] | None = None,
            multi_mode: str = MultiSeqEmbeddingMode.PRE_SUM,
            num_sequences: int = 2
    ):
        super().__init__(
            num_tokens=num_tokens,
            emb_dims=emb_dims,
            mode=mode,
            project_emb_dim=project_emb_dim,
            project_bias=project_bias,
            emb_norm=emb_norm,
            discrete=discrete,
            continuous=continuous,
            sinusoidal=sinusoidal,
            sinusoidal_learned=sinusoidal_learned,
            sinusoidal_dense=sinusoidal_dense,
            token_values=token_values,
            num_first_discrete=num_first_discrete,
            special_tokens=special_tokens,
            embedding_kwargs=embedding_kwargs,
            tie_keys=tie_keys
        )

        self.multi_mode = multi_mode
        self.num_sequences = num_sequences

        if self.multi_mode == MultiSeqEmbeddingMode.POST_CAT:
            self.project_multiemb = nn.Linear(num_sequences * project_emb_dim, project_emb_dim)

    def forward(
            self,
            tokens: torch.Tensor | list[torch.Tensor],
            values: torch.Tensor | list[torch.Tensor] | None = None,
            cache: torch.Tensor | None = None,
            return_embeddings: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        if isinstance(tokens, list) and len(tokens) == 1:
            tokens = tokens[0]
            values = values[0] if isinstance(values, list) else values

        if isinstance(tokens, torch.Tensor):
            return super().forward(tokens, values=values, cache=cache, return_embeddings=return_embeddings)

        if cache is not None:
            tokens = [t[:, cache.shape[1]:] for t in tokens]
            values = [v[:, cache.shape[1]:] for v in values] if values is not None else values

        tokens = torch.stack(tokens, dim=0)
        values = torch.stack(values, dim=0) if values is not None else None
        token_embs = self._forward_embeddings(tokens, values=values)

        if self.multi_mode == MultiSeqEmbeddingMode.PRE_SUM:
            token_embs = {
                key: [e.squeeze(dim=0) for e in token_emb.split(1, dim=0)]
                for key, token_emb in token_embs.items()
            }

            total_token_embs = {
                key: sum(token_embs[key])
                for key in token_embs
            }
            token_emb = self._forward_project(total_token_embs)
        elif self.multi_mode.startswith("post"):
            token_embs = [
                {key: token_embs[key][i] for key, token_emb in token_embs.items()}
                for i in range(len(tokens))
            ]
            token_embs_proj = [self._forward_project(te) for te in token_embs]

            if self.multi_mode == MultiSeqEmbeddingMode.POST_CAT:
                assert len(token_embs_proj) == self.num_sequences
                token_emb = self.project_multiemb(torch.cat(token_embs_proj, dim=-1))
            else:
                token_emb = sum(token_embs_proj)
        else:
            return None

        if cache is not None:
            token_emb = torch.cat([cache, token_emb], dim=1)

        if return_embeddings:
            return token_emb, token_embs
        else:
            return token_emb


def complex_log(float_input: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    eps = float_input.new_tensor(eps)
    real = float_input.abs().maximum(eps).log()
    imag = (float_input < 0).to(float_input.dtype) * torch.pi
    return torch.complex(real, imag)


def associative_scan(values: torch.Tensor, coeffs: torch.Tensor, dim: int = -1, decimals: int = 4) -> torch.Tensor:
    # https://github.com/pytorch/pytorch/issues/53095
    log_values = complex_log(values.float())
    log_coeffs = complex_log(coeffs.float())
    a_star = torch.cumsum(log_coeffs, dim=dim)
    log_x0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=dim)
    log_x = a_star + log_x0_plus_b_star
    return torch.exp(log_x).real.round(decimals=decimals)


@dataclass
class PositionTupleTransformerEmbeddingsConfig(VariableModuleConfig):
    _target_: str = "position"
    token_dims: dict[str, int] = MISSING
    emb_dims: dict[str, int] | int = MISSING
    prefix: bool = True
    interval: bool = True
    project_emb_dim: int = 512
    emb_norm: bool = False
    sinusoidal_learned: bool = True
    special_tokens: dict[str, int] | None = None
    embedding_kwargs: dict | None = None


@TupleTransformerEmbeddingsRegistry.register("position")
class PositionTupleTransformerEmbeddings(nn.Module, Constructor):
    def __init__(
            self,
            token_dims: dict[str, int],
            emb_dims: dict[str, int] | int,
            prefix: bool = True,
            interval: bool = True,
            project_emb_dim: int = 512,
            emb_norm: bool = False,
            sinusoidal_learned: bool = True,
            special_tokens: dict[str, int] | None = None,
            embedding_kwargs: dict | None = None
    ):
        super().__init__()

        assert prefix or interval
        self.prefix = prefix
        self.interval = interval
        num_sequences = int(prefix) + int(interval)

        embedding_kwargs = embedding_kwargs or {}
        embedding_kwargs["depth"] = embedding_kwargs.get("depth", 1)

        special_tokens = special_tokens or {}
        num_first_discrete = len(special_tokens)

        embeddings, total_emb_dim = {}, 0
        for key in token_dims:
            emb_dim = emb_dims if isinstance(emb_dims, int) else emb_dims[key]

            embeddings[key] = DiscreteSinusoidalEmbedding(
                num_embeddings=num_first_discrete + 1,
                embedding_dim=emb_dim,
                learned=sinusoidal_learned,
                num_first_discrete=num_first_discrete,
                padding_idx=0,
                **embedding_kwargs
            )

            total_emb_dim += emb_dim * num_sequences

        self.embs = nn.ModuleDict(embeddings)
        self.norm = nn.LayerNorm(total_emb_dim) if emb_norm else nn.Identity()

        self.project_emb = nn.Linear(total_emb_dim, project_emb_dim, bias=False)

        self.register_buffer("token_dims", torch.tensor(list(token_dims.values())))
        self.emb_dims = emb_dims
        self.total_emb_dim = total_emb_dim

        self.special_tokens = special_tokens
        self.num_first_discrete = num_first_discrete

        self.mask_token_id = self.special_tokens.get(MASK_TOKEN, -1)
        self.sos_token_id = self.special_tokens.get(SOS_TOKEN, -1)
        self.eos_token_id = self.special_tokens.get(EOS_TOKEN, -1)

        self.init_()

    def init_(self):
        for key, emb in self.embs.items():
            if isinstance(emb, nn.Embedding):
                nn.init.xavier_uniform_(emb.weight)
        nn.init.xavier_uniform_(self.project_emb.weight)
        nn.init.normal_(self.project_emb.weight, std=1e-2)

    def _forward_embeddings(
            self,
            x: torch.Tensor,
            values: torch.Tensor,
            label: str
    ) -> dict[str, torch.Tensor]:
        return {
            f"{key}/{label}": emb(x[..., i], values=values[..., i])
            for i, (key, emb) in enumerate(self.embs.items())
        }

    def _forward_project(
            self,
            token_embs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        token_embs = list(token_embs.values())
        total_token_emb = torch.cat(token_embs, dim=-1)
        return self.project_emb(self.norm(total_token_emb))

    def forward(
            self,
            tokens: torch.Tensor,
            values: torch.Tensor,
            cache: torch.Tensor | None = None,
            return_embeddings: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        assert values is not None, "`values` should be used with PositionTupleTokenEmbeddings"

        if cache is not None:
            tokens = tokens[:, cache.shape[1]:]
            values = values[:, cache.shape[1]:]

        tokens = tokens[..., self.token_dims]
        values = values[..., self.token_dims].clone()

        special_mask = tokens <= self.num_first_discrete
        tokens = tokens.masked_fill(~special_mask, self.num_first_discrete)
        values[special_mask] = 0.
        special_mask &= (tokens != self.sos_token_id) * (tokens != self.eos_token_id)

        pos_embs = {}
        if self.prefix:
            unknown_mask = torch.cumsum(special_mask, dim=1).bool()

            pos_known = torch.cumsum(values, dim=1)
            pos_known[unknown_mask] = 0.

            tokens_known = tokens.clone()
            tokens_known[unknown_mask * (tokens_known == self.num_first_discrete)] = self.mask_token_id

            prefix_pos_embs = self._forward_embeddings(tokens_known, pos_known, "prefix")
            pos_embs.update(**prefix_pos_embs)

        if self.interval:
            pos_interval = associative_scan(values, (~special_mask).float(), dim=1)

            interval_pos_embs = self._forward_embeddings(tokens, pos_interval, "interval")
            pos_embs.update(**interval_pos_embs)

        pos_emb = self._forward_project(pos_embs)

        if cache is not None:
            pos_emb = torch.cat([cache, pos_emb], dim=1)

        if return_embeddings:
            return pos_emb, pos_embs
        else:
            return pos_emb
