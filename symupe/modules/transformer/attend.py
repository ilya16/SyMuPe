"""
Attention with efficient memory attention support.

Adapted from: https://github.com/lucidrains/x-transformers
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange
from packaging import version
from torch import nn, einsum


@dataclass
class AttentionIntermediates:
    queries: torch.Tensor | None = None
    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None

    scores: torch.Tensor | None = None

    def add_past_cache(self, past_cache: AttentionIntermediates | None) -> AttentionIntermediates:
        if past_cache is None:
            return self

        if past_cache.queries is not None and self.queries is not None:
            self.queries = torch.cat([past_cache.queries, self.queries], dim=-2)

        if past_cache.scores is not None and self.scores is not None:
            past_scores = past_cache.scores
            if past_cache.scores.shape[-1] != self.scores.shape[-1]:
                past_scores = F.pad(past_cache.scores, (0, 1), value=-torch.finfo(past_cache.scores.dtype).max)

            self.scores = torch.cat([past_scores, self.scores], dim=-2)

        return self

    def to_tuple(self):
        return self.queries, self.keys, self.values, self.scores


# main class

class Attend(nn.Module):
    def __init__(
            self,
            *,
            dropout: float = 0.,
            causal: bool = False,
            scale: float | None = None,
            enable_flash: bool = False,
            enable_math: bool = True,
            enable_mem_efficient: bool = True
    ):
        super().__init__()

        self.causal = causal
        self.scale = scale

        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        # efficient attention
        torch_version = version.parse(torch.__version__)
        self.efficient = torch_version >= version.parse("2.0.0")

        sdp_kwargs = {
            "enable_flash": enable_flash,
            "enable_math": enable_math,
            "enable_mem_efficient": enable_mem_efficient
        }

        if torch_version >= version.parse("2.3.0"):
            from torch.nn.attention import SDPBackend

            str_to_backend = dict(
                enable_flash=SDPBackend.FLASH_ATTENTION,
                enable_mem_efficient=SDPBackend.EFFICIENT_ATTENTION,
                enable_math=SDPBackend.MATH,
                enable_cudnn=SDPBackend.CUDNN_ATTENTION
            )

            sdpa_backends = [str_to_backend[enable_str] for enable_str, enable in sdp_kwargs.items() if enable]

            self.sdp_context_manager = partial(torch.nn.attention.sdpa_kernel, sdpa_backends)
        else:
            self.sdp_context_manager = partial(torch.backends.cuda.sdp_kernel, **sdp_kwargs)

    def efficient_attn(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask: torch.Tensor | None = None,
            attn_bias: torch.Tensor | None = None,
            offset: int = 0
    ) -> tuple[torch.Tensor, AttentionIntermediates]:
        batch, heads, q_len, _ = q.shape
        k_len, device = k.shape[-2], q.device

        # Recommended for multi-query single-key-value attention by Tri Dao
        # kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])

        intermediates = AttentionIntermediates(queries=q.detach(), keys=k.detach(), values=v.detach())

        if k.ndim == 3:
            k = rearrange(k, "b ... -> b 1 ...").expand(-1, heads, -1, -1)

        if v.ndim == 3:
            v = rearrange(v, "b ... -> b 1 ...").expand(-1, heads, -1, -1)

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        causal = self.causal

        if mask is not None:
            assert mask.ndim == 4
            mask = mask.expand(batch, heads, q_len, k_len)

            # manually handle causal mask, if another mask was given

            if causal:
                causal_mask = torch.ones(
                    (q_len, k_len), dtype=torch.bool, device=device
                ).triu(k_len - q_len + 1 - offset)
                mask = mask & ~causal_mask
                causal = False

        # handle alibi positional bias
        # convert from bool to float

        if attn_bias is not None:
            if attn_bias.ndim == 3:
                attn_bias = rearrange(attn_bias, "h i j -> 1 h i j").expand(batch, -1, -1, -1)

            # if mask given, the mask would already contain the causal mask from above logic
            # otherwise, if no mask given but still causal, mask out alibi positional bias to a large negative number

            mask_value = -torch.finfo(q.dtype).max

            if mask is not None:
                attn_bias = attn_bias.masked_fill(~mask, mask_value // 2)
            elif causal:
                causal_mask = torch.ones(
                    (q_len, k_len), dtype=torch.bool, device=device
                ).triu(k_len - q_len + 1 - offset)
                attn_bias = attn_bias.masked_fill(causal_mask, mask_value // 2)
                causal = False

            # scaled_dot_product_attention handles attn_mask either as bool or additive bias
            # make it an additive bias here

            mask = attn_bias

        # pytorch 2.0 attention: q, k, v, mask, dropout, causal, softmax_scale

        with self.sdp_context_manager():
            out = F.scaled_dot_product_attention(
                q.contiguous(), k.contiguous(), v.contiguous(),
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.,
                is_causal=causal and q_len > 1
            )

        return out, intermediates

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask: torch.Tensor | None = None,
            attn_bias: torch.Tensor | None = None,
            offset: int = 0
    ) -> tuple[torch.Tensor, AttentionIntermediates]:
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        if self.efficient and q.shape[0] <= 65535:
            return self.efficient_attn(
                q, k, v, mask=mask, attn_bias=attn_bias, offset=offset
            )

        n, device = q.shape[-2], q.device
        scale = self.scale if self.scale is not None else q.shape[-1] ** -0.5

        kv_einsum_eq = "b j d" if k.ndim == 3 else "b h j d"

        dots = einsum(f"b h i d, {kv_einsum_eq} -> b h i j", q, k) * scale

        if attn_bias is not None:
            dots = dots + attn_bias

        dtype = dots.dtype
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            dots = dots.masked_fill(~mask, mask_value)

        if self.causal:
            i, j = dots.shape[-2:]
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=device).triu(j - i + 1 - offset)
            dots = dots.masked_fill(causal_mask, mask_value)

        attn = dots.softmax(dim=-1)
        attn = attn.type(dtype)

        attn = self.attn_dropout(attn)

        out = einsum(f"b h i j, {kv_einsum_eq} -> b h i d", attn, v)

        intermediates = AttentionIntermediates(
            queries=q.detach(), keys=k.detach(), values=v.detach(),
            scores=dots.detach()
        )

        return out, intermediates
