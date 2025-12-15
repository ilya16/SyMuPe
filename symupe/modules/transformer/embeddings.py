""" General purpose transformer embeddings. """
from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum


class DiscreteContinuousEmbedding(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            discrete: bool = True,
            continuous: bool = True,
            num_first_discrete: int = 0,
            depth: int = 1,
            token_values: list[float] | torch.Tensor | None = None,
            padding_idx: int | None = None,
            activation=None,
            _weight: torch.Tensor | None = None,
            device=None,
            dtype=None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_first_discrete = num_first_discrete
        self.num_discrete_embeddings = num_embeddings if discrete else self.num_first_discrete

        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, "Padding_idx must be within num_embeddings"
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, "Padding_idx must be within num_embeddings"
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx

        assert discrete or continuous, "`DiscreteContinuousEmbedding` should be at least discrete or continuous"
        self.discrete = discrete
        self.continuous = continuous

        self.index_weight = None
        if self.has_discrete:
            num_discrete_embeddings = num_embeddings if self.discrete else self.num_first_discrete
            if _weight is None:
                self.index_weight = nn.Parameter(
                    torch.empty((num_discrete_embeddings, embedding_dim), **factory_kwargs)
                )
            else:
                assert list(_weight.shape) == [num_discrete_embeddings, embedding_dim], \
                    "Shape of weight does not match num_embeddings and embedding_dim"
                self.index_weight = nn.Parameter(_weight)

        self.value_layer = None
        self.activation = None
        if self.continuous:
            if token_values is not None:
                if not isinstance(token_values, torch.Tensor):
                    token_values = torch.tensor(token_values)
            else:
                token_values = torch.cat([
                    100 * torch.arange(-self.num_first_discrete, 0, 1),
                    torch.linspace(0., 1., self.num_embeddings - self.num_first_discrete)
                ])
            token_values = token_values.to(**factory_kwargs)

            layers = [
                nn.Linear(1, embedding_dim, **factory_kwargs),
                nn.SiLU() if depth > 1 else nn.Identity()
            ]

            for i in range(depth - 1):
                layers.extend([
                    nn.Linear(embedding_dim, embedding_dim, **factory_kwargs),
                    nn.SiLU() if i < depth - 2 else nn.Identity()
                ])

            layers = layers[:-1] if isinstance(layers[-1], nn.Identity) else layers

            self.value_layer = nn.Sequential(*layers)
            self.activation = activation

        self.register_buffer("token_values", token_values)
        self._ignore_values = False

        self.register_buffer("_token_weight", None, persistent=False)
        self.register_buffer("_value_weight", None, persistent=False)
        if _weight is None:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.has_discrete:
            nn.init.xavier_uniform_(self.index_weight)
            nn.init.normal_(self.index_weight[self.num_first_discrete:], std=1e-2)
        if self.continuous:
            for module in self.value_layer.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                if self.has_discrete:
                    self.index_weight[self.padding_idx].fill_(0)

    def forward(self, tokens: torch.Tensor | None = None, values: torch.Tensor | None = None) -> torch.Tensor:
        assert not self.has_discrete or tokens is not None
        if values is None:  # use fixed `token_values`
            assert not self._ignore_values or self.token_values is not None, \
                f"`{self.__class__.__name__}.token_values` cannot be empty when no `values` are provided"
            token_weight = self.token_weight if self.has_discrete else 0
            value_weight = self.value_weight if self.continuous else 0
            weight = token_weight + value_weight
            return F.embedding(tokens, weight, self.padding_idx)
        else:
            token_emb = F.embedding(tokens, self.token_weight, self.padding_idx) if self.has_discrete else 0
            if not self.training and (
                    not self.continuous or (self.has_discrete and torch.all(tokens < self.num_discrete_embeddings))
            ):
                return token_emb
            else:
                value_emb = self.forward_values(tokens, values=values)
                return token_emb + value_emb

    def forward_indices(self, tokens: torch.Tensor) -> torch.Tensor:
        assert self.has_discrete
        return F.embedding(tokens, self.token_weight, self.padding_idx)

    def forward_values(self, tokens: torch.Tensor, values: torch.Tensor | None = None) -> torch.Tensor:
        values = None if self._ignore_values else values
        assert self.continuous
        if values is None:  # use fixed `token_values`
            assert not self._ignore_values or self.token_values is not None, \
                f"`{self.__class__.__name__}.token_values` cannot be empty when no `values` are provided"
            return F.embedding(tokens, self.value_weight, self.padding_idx)
        else:
            value_emb = self._compute_value_embeddings(values)
            if self.num_first_discrete > 0:
                value_emb[tokens < self.num_first_discrete] = 0.
            return value_emb

    def _compute_value_embeddings(self, values: torch.Tensor) -> torch.Tensor:
        assert self.continuous
        shape = values.shape
        values_emb = self.value_layer(values.reshape(-1, 1)).view(*shape, -1)
        values_emb = values_emb if self.activation is None else self.activation(values_emb)
        return values_emb

    @property
    def token_weight(self) -> torch.Tensor | None:
        if self._token_weight is None:
            if self.discrete:
                return self.index_weight
            elif self.num_first_discrete > 0:
                return torch.cat([
                    self.index_weight,
                    self.index_weight.new_zeros((self.num_embeddings - self.num_first_discrete, self.embedding_dim))
                ])
        return self._token_weight

    @property
    def value_weight(self) -> torch.Tensor | None:
        if self.continuous:
            if self.token_values is None:
                value_weight = self._compute_value_embeddings(
                    torch.arange(self.num_embeddings - self.num_first_discrete, device=self.index_weight.device)
                )
                return torch.cat([torch.zeros_like(self.index_weight), value_weight], dim=0)
            elif self._value_weight is None:
                value_weight = self._compute_value_embeddings(self.token_values.view(-1))
                if self.num_first_discrete > 0:
                    value_weight[:self.num_first_discrete] = 0.
                return value_weight
            return self._value_weight

    @property
    def weight(self) -> torch.Tensor:
        if self.has_discrete:
            if self.token_values is None:
                return self.token_weight
            return self.token_weight + self.value_weight
        else:
            return self.value_weight

    @property
    def has_discrete(self) -> bool:  # fully discrete or has discrete some token indices
        return self.discrete or self.num_first_discrete > 0

    def train(self, mode=True):
        if mode or self.token_values is None:
            self._token_weight, self._value_weight = None, None
        else:
            if self.has_discrete:
                self._token_weight = self.token_weight.detach().to(device=self.index_weight.device)
            if self.continuous:
                self._value_weight = self.value_weight.detach().to(device=self.token_values.device)
        return super().train(mode)

    def extra_repr(self) -> str:
        s = "{num_embeddings}, {embedding_dim}"
        if self.padding_idx is not None:
            s += ", discrete={discrete}, num_first_discrete={num_first_discrete}, padding_idx={padding_idx}"
        return s.format(**self.__dict__)


class DiscreteSinusoidalEmbedding(DiscreteContinuousEmbedding):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            learned: bool = False,
            depth: int = 0,
            theta: int | None = 1000.,
            freq_scale: float | None = 1.,
            with_positions: bool = True,
            log_inv_freq: bool = False,
            ignore_values: bool = False,
            discrete: bool = False,
            num_first_discrete: int = 0,
            token_values: list[float] | torch.Tensor | None = None,
            padding_idx: int | None = None,
            activation=None,
            _weight: torch.Tensor | None = None,
            device=None,
            dtype=None
    ) -> None:
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            discrete=discrete,
            continuous=True,
            num_first_discrete=num_first_discrete,
            token_values=token_values,
            padding_idx=padding_idx,
            device=device,
            dtype=dtype
        )

        factory_kwargs = {"device": device, "dtype": dtype}

        if token_values is not None:
            min_diff = torch.diff(torch.sort(self.token_values[self.num_first_discrete:]).values.unique()).min().item()
            max_value = max(1., self.token_values[self.num_first_discrete:].max().item())

            theta = math.ceil(round(max_value / min_diff) / 100) * 100.
            freq_scale = math.pi / 2 / min_diff

        cls = SinusoidalEmbedding
        if learned:
            cls = partial(LearnedSinusoidalEmbedding, log_inv_freq=log_inv_freq)

        layers = [
            cls(
                dim=embedding_dim,
                theta=theta,
                freq_scale=freq_scale,
                scale=False,
                with_positions=with_positions
            )
        ]

        if depth > 0:
            layers.extend([
                nn.Linear(embedding_dim + int(with_positions), embedding_dim, **factory_kwargs),
                nn.SiLU() if depth > 1 or activation else nn.Identity()
            ])

            for i in range(1, depth):
                layers.extend([
                    nn.Linear(embedding_dim, embedding_dim, **factory_kwargs),
                    nn.SiLU() if i < depth - 1 or activation else nn.Identity()
                ])

        layers = layers[:-1] if isinstance(layers[-1], nn.Identity) else layers

        self.value_layer = nn.Sequential(*layers) if len(layers) > 0 else layers[0]
        self.activation = nn.SiLU if activation else None

        if ignore_values:
            self.register_buffer(
                "token_values",
                torch.cat([
                    100 * torch.arange(-self.num_first_discrete, 0, 1),
                    torch.arange(self.num_embeddings - self.num_first_discrete)
                ])
            )
        self._ignore_values = ignore_values

    def reset_parameters(self) -> None:
        if self.has_discrete:
            nn.init.xavier_uniform_(self.index_weight)
        self._fill_padding_idx_with_zero()

    @property
    def value_weight(self) -> torch.Tensor:
        if self._value_weight is not None:
            return self._value_weight
        if self.token_values is None:
            value_weight = self._compute_value_embeddings(
                torch.arange(self.num_embeddings - self.num_first_discrete, device=self.index_weight.device)
            )
            value_weight = torch.cat([torch.zeros_like(self.index_weight), value_weight], dim=0)
        else:
            value_weight = self._compute_value_embeddings(self.token_values.view(-1))
            if self.num_first_discrete > 0:
                value_weight[:self.num_first_discrete] = 0.
        return value_weight


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        self.dim = dim
        self.scale = dim ** -0.5
        self.max_seq_len = max_seq_len
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[1]
        assert seq_len <= self.max_seq_len

        if pos is None:
            pos = torch.arange(seq_len, device=x.device)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return pos_emb

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class SinusoidalEmbedding(nn.Module):
    def __init__(
            self,
            dim: int,
            theta: float = 10000,
            freq_scale: float = 1.,
            scale: bool | float | None = 1.,
            with_positions: bool = False
    ):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

        self.theta = theta
        self.register_buffer("freq_scale", torch.ones(1) * freq_scale, persistent=True)

        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = theta ** -freq_seq
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        if isinstance(scale, bool) and scale:
            scale = dim ** -0.5
        if scale:
            self.scale = nn.Parameter(torch.ones(1) * scale)
        else:
            self.register_buffer("scale", torch.ones(1))

        self.with_positions = with_positions

    def forward(self, x: torch.Tensor, is_pos: bool = True, seq_dim: int = 1, offset: float = 0) -> torch.Tensor:
        pos = x if is_pos else torch.arange(x.shape[seq_dim], device=x.device)

        inv_freq = self.get_inv_freq()
        pos = pos.type_as(inv_freq) + offset
        emb = pos.unsqueeze(-1) * self.freq_scale * inv_freq
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        emb = self.scale * emb

        if self.with_positions:
            return torch.cat((pos[:, None], emb), dim=-1)
        return emb

    def get_inv_freq(self) -> torch.Tensor:
        return self.inv_freq

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, "
            f"theta={float(self.theta):.3f}, "
            f"freq_scale={float(self.freq_scale):.3f}, "
            f"with_positions={self.with_positions}"
        )


class LearnedSinusoidalEmbedding(SinusoidalEmbedding):
    def __init__(
            self,
            dim: int,
            theta: float = 10000,
            freq_scale: float = 1.,
            scale: bool | float | None = 1.,
            with_positions: bool = False,
            log_inv_freq: bool = False
    ):
        super().__init__(dim=dim, theta=theta, freq_scale=freq_scale, scale=scale, with_positions=with_positions)

        if log_inv_freq:
            self.log_inv_freq = nn.Parameter(torch.log(self.inv_freq))
            self.inv_freq = None
        else:
            self.inv_freq = nn.Parameter(self.inv_freq)
            self.log_inv_freq = None

    def get_inv_freq(self) -> torch.Tensor:
        return self.inv_freq if self.log_inv_freq is None else self.log_inv_freq.exp()


class ALiBiPositionalBias(nn.Module):
    def __init__(
            self,
            heads: int | tuple[int, int, int],
            total_heads: int,
            contextual_heads: int = 0,
            symmetric: bool = True,
            prefix: bool = False,
            ignore_positive: bool = False
    ):
        super().__init__()

        assert isinstance(heads, int) or len(heads) == 3
        heads = (heads, 0, 0) if isinstance(heads, int) else heads

        contextual_heads = contextual_heads or 0
        assert contextual_heads <= sum(heads) <= total_heads

        self.heads = heads
        self.contextual_heads = contextual_heads
        self.total_heads = total_heads
        self.symmetric = symmetric
        self.ignore_positive = ignore_positive

        slopes = torch.Tensor(self._compute_slopes(sum(heads))).view(-1, 1, 1)
        if not symmetric:
            slopes = torch.stack([slopes, torch.roll(slopes, -1)])
        self.register_buffer("slopes", slopes, persistent=False)

        self.cross_attn_bias = nn.Parameter(torch.zeros(self.total_heads, 1, 1)) if prefix else None
        self.ignore_attn_bias = nn.Parameter(torch.zeros(self.total_heads, 1, 1)) if ignore_positive else None

    @staticmethod
    def _compute_slopes(heads) -> list[float]:
        def slopes_power_of_2(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(heads).is_integer():
            return slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return slopes_power_of_2(closest_power_of_2) \
            + slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads - closest_power_of_2]

    def get_bias(self, i: int, j: int, offset: int = 0, pos: torch.Tensor | None = None) -> torch.Tensor:
        if pos is None:
            i_arange = torch.arange(offset, i + offset, dtype=torch.int, device=self.slopes.device)
            j_arange = torch.arange(j, dtype=torch.int, device=self.slopes.device)
            return -torch.abs(j_arange[None, None, :] - i_arange[None, :, None])
        else:
            bias = -torch.abs(pos[:, None] - pos[:, :, None])[:, offset:i, :j]
            if self.ignore_positive:
                is_neg = pos < 0
                is_neg = is_neg[:, None] + is_neg[:, :, None]
                bias[is_neg] = 1
            return bias[:, None]

    def get_slopes(self) -> torch.Tensor:
        return self.slopes

    def forward(
            self,
            i: int, j: int,
            offset: int = 0, prefix: int = 0,
            bias: torch.Tensor | None = None,
            q: torch.Tensor | None = None,
            k: torch.Tensor | None = None
    ) -> torch.Tensor:
        if bias is not None and bias.shape[-2] >= i and bias.shape[-1] >= j:
            bias = bias[..., :i, :j] if bias.shape[-2] > i or bias.shape[-1] > j else bias
        else:
            bias = self.get_bias(i, j, offset)

        if self.contextual_heads > 0:
            assert q is not None and k is not None
            q = q[:, :self.contextual_heads]
            k = k[:, None] if k.ndim == 3 else k[:, :self.contextual_heads]
            dots = einsum(f"b h i d, b h j d -> b h i j", q, k) * (q.shape[-1] ** -0.5)
            c_bias = -dots.sigmoid()
            c_pos = c_bias.triu(diagonal=offset + 1).cumsum(dim=-1) \
                + c_bias.tril(diagonal=offset - 1).flip(-1).cumsum(dim=-1).flip(-1)

            bias = torch.cat([
                c_pos,
                bias[None].expand(q.shape[0], self.total_heads - self.contextual_heads, -1, -1),
            ], dim=1)

        slopes = self.get_slopes()
        if self.total_heads - slopes.shape[-3] > 0:
            slopes = F.pad(slopes, (0, 0, 0, 0, 0, self.total_heads - slopes.shape[-3]))

        if self.symmetric:
            attn_bias = slopes * bias
        else:
            attn_bias = slopes[0] * torch.tril(bias) + slopes[1] * torch.triu(bias)

        if self.heads[0] != sum(self.heads):
            mask_value = -torch.finfo(attn_bias.dtype).max
            mask_bias = attn_bias.new_full((attn_bias.shape[-2], attn_bias.shape[-1]), fill_value=mask_value // 2)

            if self.heads[1] > 0:
                attn_bias[self.heads[0]:sum(self.heads[:2])] += torch.triu(mask_bias, diagonal=offset + 1)
            if self.heads[2] > 0:
                attn_bias[sum(self.heads[:2]):sum(self.heads)] += torch.tril(mask_bias, diagonal=offset - 1)

        if prefix > 0:
            is_prefix = torch.arange(j, device=attn_bias.device) < prefix
            is_cross_attn = is_prefix[:, None] ^ is_prefix[None, :]
            is_cross_attn = is_cross_attn[offset:]

            attn_bias = torch.where(
                is_cross_attn,
                self.cross_attn_bias if self.cross_attn_bias is not None
                else attn_bias.new_zeros((self.total_heads, 1, 1)),
                attn_bias
            )

        if self.ignore_positive and self.ignore_attn_bias is not None:
            attn_bias = torch.where(bias > 0, self.ignore_attn_bias, attn_bias)

        return attn_bias

    def extra_repr(self) -> str:
        s = "heads={heads}, total_heads={total_heads}, symmetric={symmetric}"
        return s.format(**self.__dict__)


class LearnedALiBiPositionalBias(ALiBiPositionalBias):
    def __init__(
            self,
            heads: int | tuple[int, int, int],
            total_heads: int,
            contextual_heads: int = 0,
            symmetric: bool = True,
            prefix: bool = False,
            ignore_positive: bool = False
    ):
        super().__init__(
            heads, total_heads, contextual_heads=contextual_heads,
            symmetric=symmetric, prefix=prefix, ignore_positive=ignore_positive
        )
        log_slopes = torch.log(self.slopes)
        self.learned_logslopes = nn.Parameter(log_slopes)

    def get_slopes(self) -> torch.Tensor:
        return self.learned_logslopes.exp()


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        use_xpos: bool = False,
        scale_base: float = 512,
        interpolation_factor: float = 1.,
        base: float = 10000,
        base_rescale_factor: float = 1.
    ):
        super().__init__()
        self.dim = dim

        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self.register_buffer("freqs", torch.empty(1, 0, dim), persistent=False)

        assert interpolation_factor >= 1.
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.register_buffer("scale", None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.register_buffer("scale", scale)

    @torch.autocast("cuda", enabled=False)
    def get_pos_emb(self, x: torch.Tensor, seq_len: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[1] if seq_len is None else seq_len

        if seq_len > self.freqs.shape[-2]:
            t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=x.device)[None]
            max_pos = t.max()

            freqs = torch.einsum("b i , j -> b i j", t.type_as(self.inv_freq), self.inv_freq) / self.interpolation_factor
            freqs = torch.stack((freqs, freqs), dim = -1)
            shape = freqs.shape
            freqs = freqs.view(shape[:-2] + (-1,))
            self.freqs = freqs
        else:
            freqs = self.freqs

        if self.scale is None:
            return freqs, torch.tensor(1., device=x.device)

        power = (t - (max_pos // 2)) / self.scale_base
        scale = self.scale ** power[:, None]
        scale = torch.stack((scale, scale), dim = -1)
        scale = scale.view(shape)

        return freqs, scale

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.view(shape[:-1] + (-1, 2))
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return x.view(shape)

    def apply_rotary_pos_emb(self, t: torch.Tensor, freqs: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        rot_dim, seq_len, orig_dtype = freqs.shape[-1], t.shape[-2], t.dtype

        freqs = freqs[:, -seq_len:, :]
        scale = scale[:, -seq_len:, :] if scale.ndim > 1 else scale

        freqs = freqs[:, None] if t.ndim == 4 and freqs.ndim == 3 else freqs

        # partial rotary embeddings, Wang et al. GPT-J
        t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
        t = (t * freqs.cos() * scale) + (self.rotate_half(t) * freqs.sin() * scale)
        out = torch.cat((t, t_unrotated), dim=-1)

        return out.type(orig_dtype)

    def forward(
            self,
            x: torch.Tensor,
            seq_len: int | None = None,
            pos_emb: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> torch.Tensor:
        pos_emb, scale = self.get_pos_emb(x, seq_len=seq_len) if pos_emb is None else pos_emb
        l = pos_emb.shape[-1]
        xl, xr = x[..., :l], x[..., l:]
        with torch.autocast("cuda", enabled=False):
            xl = self.apply_rotary_pos_emb(xl, pos_emb, scale=scale)
        return torch.cat((xl, xr), dim=-1)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"
