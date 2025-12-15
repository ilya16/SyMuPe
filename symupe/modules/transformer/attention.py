""" Transformer Attention with kv caching support for inference. """
from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .attend import AttentionIntermediates, Attend
from .embeddings import ALiBiPositionalBias, LearnedALiBiPositionalBias, RotaryEmbedding
from .normalization import LayerNorm
from ..constructor import Constructor, ModuleConfig, Registry

AttentionRegistry = type("_AttentionRegistry", (Registry,), {})()


@dataclass
class AttentionSharedIntermediates:
    rel_pos_bias: torch.Tensor | None = None
    rotary_pos_emb: tuple[torch.Tensor, float] | None = None


@dataclass
class AttentionConfig(ModuleConfig):
    _target_: str = "global"
    dim: int = 512
    heads: int = 8
    head_dim: int = 64
    causal: bool = False
    dropout: float = 0.
    one_kv_head: bool = False

    context_dim: int | None = None
    context_norm: bool = False
    context_as_input_prefix: bool = False
    context_as_attention_prefix: bool = False

    num_mem_kv: int = 0
    max_attend: int | None = None

    alibi_pos_bias: bool = False
    alibi_heads: int | None = None
    alibi_contextual_heads: int | None = None
    alibi_symmetric: bool = True
    alibi_learned: bool = False

    rotary_pos_emb: bool = False
    rotary_emb_dim: int | None = None
    rotary_emb_base: float = 10000.


@AttentionRegistry.register("global")
class Attention(nn.Module, Constructor):
    def __init__(
            self,
            dim: int,
            heads: int = 8,
            head_dim: int = 64,
            causal: bool = False,
            dropout: float = 0.,
            one_kv_head: bool = False,
            context_dim: int | None = None,
            context_norm: bool = False,
            context_as_input_prefix: bool = False,
            context_as_attention_prefix: bool = False,
            num_mem_kv: int = 0,
            max_attend: int | None = None,
            alibi_pos_bias: bool = False,
            alibi_heads: int | tuple[int, int, int] | None = None,
            alibi_contextual_heads: int | None = None,
            alibi_symmetric: bool = True,
            alibi_learned: bool = False,
            rotary_pos_emb: bool = False,
            rotary_emb_dim: int | None = None,
            rotary_emb_base: float = 10000.
    ):
        super().__init__()

        self.heads = heads
        self.causal = causal
        self.max_attend = max_attend

        self.head_dim = head_dim
        self.one_kv_head = one_kv_head
        out_dim = q_dim = head_dim * heads
        kv_dim = head_dim if one_kv_head else head_dim * heads
        context_dim = context_dim or dim

        self.to_q = nn.Linear(dim, q_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, 2 * kv_dim, bias=False)

        self.scale = head_dim ** -0.5

        # relative positional bias

        self.rel_pos = None
        if alibi_pos_bias:
            alibi_heads = alibi_heads or heads
            _alibi_heads = alibi_heads if isinstance(alibi_heads, int) else sum(alibi_heads)
            assert _alibi_heads <= heads, "number of ALiBi heads must be less than the total number of heads"
            alibi_pos_cls = LearnedALiBiPositionalBias if alibi_learned else ALiBiPositionalBias
            self.rel_pos = alibi_pos_cls(
                heads=alibi_heads,
                total_heads=heads,
                contextual_heads=alibi_contextual_heads,
                symmetric=alibi_symmetric or causal,
                prefix=context_as_input_prefix or context_as_attention_prefix
            )

        self.rotary_pos_emb = None
        if rotary_pos_emb:
            rotary_emb_dim = min(max(rotary_emb_dim or self.head_dim // 2, 32), self.head_dim)
            self.rotary_pos_emb = RotaryEmbedding(dim=rotary_emb_dim, base=rotary_emb_base)
            # self.rotary_pos_emb = torch.jit.script(self.rotary_pos_emb)

        # attend class - includes core attention algorithm

        self.attend = Attend(
            causal=causal,
            dropout=dropout,
            scale=self.scale
        )

        # context processing

        self.context_norm = LayerNorm(context_dim, bias=False) if context_norm else None
        self.context_as_input_prefix = context_as_input_prefix  # prepended before the 1st transformer layer
        self.context_as_attention_prefix = context_as_attention_prefix

        # add memory key / values

        self.num_mem_kv = num_mem_kv
        self.mem_kv = nn.Parameter(torch.randn(2, num_mem_kv, head_dim)) if num_mem_kv > 0 else None

        # output layer

        self.to_out = nn.Linear(out_dim, dim, bias=False)

    def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor | None = None,
            context: torch.Tensor | None = None,
            context_mask: torch.Tensor | None = None,
            input_prefix_len: int = 0,
            attn_mask: torch.Tensor | None = None,
            causal: torch.Tensor | None = None,
            memory: torch.Tensor | None = None,
            cache: AttentionIntermediates | None = None,
            shared_cache: AttentionSharedIntermediates | None = None
    ) -> tuple[torch.Tensor, AttentionIntermediates, AttentionSharedIntermediates]:
        b, n = x.shape[:2]
        h, device = self.heads, x.device
        has_context, has_memory, has_cache = context is not None, memory is not None, cache is not None
        cond_as_prefix = self.context_as_input_prefix or self.context_as_attention_prefix
        cross_attention = has_context and not cond_as_prefix
        assert not (has_memory and cross_attention), "memory keys are incompatible with cross attention"

        if has_context:
            if self.context_norm is not None:
                context = self.context_norm(context)

            if context_mask is None:
                context_mask = torch.ones(context.shape[:2], device=device, dtype=torch.bool)

        kv_input = x if cond_as_prefix or context is None else context

        # process memories

        if has_memory:
            assert not self.context_as_attention_prefix
            kv_input = torch.cat((memory, kv_input), dim=-2)

        # take care of prefix-based self attention conditioning
        # make sure to either concat the to the self attention mask or lengthen it accordingly

        if self.context_as_attention_prefix:
            kv_input = torch.cat((context, kv_input), dim=-2)
            prefix_len = context.shape[-2]

            if mask is None:
                mask = torch.ones((b, n), device=device, dtype=torch.bool)

            if context_mask is not None:
                mask = torch.cat((context_mask, mask), dim=-1)
            else:
                mask = F.pad(mask, (prefix_len, 0), value=True)
        else:
            prefix_len = input_prefix_len

        # project for queries, keys, values

        q = self.to_q(x)
        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        if has_cache and cross_attention:
            k, v = cache.keys, cache.values
        else:
            k, v = self.to_kv(kv_input).chunk(2, dim=-1)

            if not self.one_kv_head:
                k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (k, v))

        input_mask = mask if not has_context or cond_as_prefix else context_mask

        # kv cache

        cached_q, cached_k, cached_v = q, k, v
        if has_cache and not cross_attention:
            prefix_memory_len = prefix_len if cond_as_prefix else 0
            prefix_memory_len += memory.shape[1] if has_memory else 0

            if prefix_memory_len > 0:
                mk, k = k[:, :prefix_memory_len], k[:, prefix_memory_len:]
                mv, v = v[:, :prefix_memory_len], v[:, prefix_memory_len:]

            k = cached_k = torch.cat([cache.keys, k], dim=-2)
            v = cached_v = torch.cat([cache.values, v], dim=-2)

            if prefix_memory_len > 0:
                k = torch.cat([mk, k], dim=-2)
                v = torch.cat([mv, v], dim=-2)

        # rotary positional embedding

        rotary_pos_emb = None
        if not has_context and self.rotary_pos_emb is not None:
            if shared_cache is not None and shared_cache.rotary_pos_emb is not None:
                rotary_pos_emb = shared_cache.rotary_pos_emb
            else:
                rotary_pos_emb = self.rotary_pos_emb.get_pos_emb(x, seq_len=k.shape[-2])
            q = self.rotary_pos_emb(q, pos_emb=rotary_pos_emb)
            k = self.rotary_pos_emb(k, pos_emb=rotary_pos_emb)

        # null key / values

        if self.num_mem_kv > 0:
            if self.one_kv_head:
                mem_k, mem_v = repeat(self.mem_kv, "kv n d -> kv b n d", b=b).unbind(dim=0)
            else:
                mem_k, mem_v = repeat(self.mem_kv, "kv n d -> kv b h n d", b=b, h=h).unbind(dim=0)
            k = torch.cat((mem_k, k), dim=-2)
            v = torch.cat((mem_v, v), dim=-2)

            if input_mask is not None:
                input_mask = F.pad(input_mask, (self.num_mem_kv, 0), value=True)

        # handle all masks

        i, j = map(lambda t: t.shape[-2], (q, k))

        final_attn_mask = None

        if input_mask is not None:
            input_mask = rearrange(input_mask, "b j -> b 1 1 j")
            final_attn_mask = ~input_mask

        if attn_mask is not None:
            assert 2 <= attn_mask.ndim <= 4, \
                "attention mask must have greater than 2 dimensions but less than or equal to 4"
            if attn_mask.ndim == 2:
                attn_mask = rearrange(attn_mask, "i j -> 1 1 i j")
            elif attn_mask.ndim == 3:
                attn_mask = rearrange(attn_mask, "h i j -> 1 h i j")
            attn_mask = attn_mask[:, :, -1:] if has_cache else attn_mask
            final_attn_mask = final_attn_mask | (~attn_mask) if final_attn_mask is not None else ~attn_mask

        if causal is not None:
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=device).triu(j - i + 1)
            causal_mask = causal_mask[None, None].repeat(b, 1, 1, 1)
            causal_mask[~causal, :] = False
            final_attn_mask = final_attn_mask | causal_mask if final_attn_mask is not None else causal_mask

        if self.max_attend is not None:
            range_q = torch.arange(j - i, j, device=device)
            range_k = torch.arange(j, device=device)
            dist = rearrange(range_q, "i -> 1 1 i 1") - rearrange(range_k, "j -> 1 1 1 j")
            max_attend_mask = torch.logical_or(dist < -self.max_attend, dist > self.max_attend)
            final_attn_mask = final_attn_mask | max_attend_mask if final_attn_mask is not None else max_attend_mask

        final_attn_mask = ~final_attn_mask if final_attn_mask is not None else None  # True for attended positions

        # prepare relative positional bias, if needed

        rel_pos_bias, attn_bias = None, None
        if self.rel_pos is not None:
            if shared_cache is not None and shared_cache.rel_pos_bias is not None:
                rel_pos_bias = shared_cache.rel_pos_bias
            else:
                rel_pos_bias = self.rel_pos.get_bias(i, j, offset=j - i).to(dtype=q.dtype)

            attn_bias = self.rel_pos(i, j, offset=j - i, prefix=prefix_len, bias=rel_pos_bias, q=q, k=k)

        # attention is all we need

        out, intermediates = self.attend(
            q, k, v,
            mask=final_attn_mask,
            attn_bias=attn_bias
        )

        # update cache with tensors without any memory and prefix embeddings

        intermediates.queries = cached_q
        intermediates.keys = cached_k
        intermediates.values = cached_v

        # merge heads

        out = rearrange(out, "b h n d -> b n (h d)")

        # combine the heads

        out = self.to_out(out)

        if mask is not None:
            mask = mask[:, -1:] if has_cache else mask
            out = out * mask[..., None]

        shared_intermediates = AttentionSharedIntermediates(
            rel_pos_bias=rel_pos_bias,
            rotary_pos_emb=rotary_pos_emb
        )

        return out, intermediates, shared_intermediates


def pad_to_multiple(tensor: torch.Tensor, multiple: int, dim: int = -1, value: float = 0.) -> tuple[bool, torch.Tensor]:
    assert -2 <= dim <= -1
    seqlen = tensor.shape[dim]
    if seqlen % multiple == 0:
        return False, tensor
    m = seqlen / multiple
    remainder = ceil(m) * multiple - seqlen
    if dim == -2:
        return True, F.pad(tensor, (0, 0, 0, remainder), value=value)
    else:
        return True, F.pad(tensor, (0, remainder), value=value)


def look_around(x, backward: int = 1, forward: int = 0, pad_value: float = -1) -> torch.Tensor:
    assert 3 <= len(x.shape) <= 4
    t = x.shape[1]
    if len(x.shape) == 4:
        padded_x = F.pad(x, (0, 0, 0, 0, backward, forward), value=pad_value)
    else:
        padded_x = F.pad(x, (0, 0, backward, forward), value=pad_value)
    tensors = [padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)]
    return torch.cat(tensors, dim=2)


@dataclass
class LocalAttentionConfig(AttentionConfig):
    _target_: str = "local"
    window_size: int = 256
    local_windows: int | None = 1
    exact_window: bool = True


@AttentionRegistry.register("local")
class LocalAttention(Attention):
    def __init__(
            self,
            dim: int = 256,
            heads: int = 4,
            head_dim: int | None = 64,
            causal: bool = False,
            window_size: int = 256,
            local_windows: int | None = 1,
            exact_window: bool = True,
            dropout: float = 0.0,
            one_kv_head: bool = False,
            context_as_input_prefix: bool = False,
            context_as_attention_prefix: bool = False,
            num_mem_kv: int = 0,
            max_attend: int | None = None,
            alibi_pos_bias: bool = False,
            alibi_heads: int | None = None,
            alibi_symmetric: bool = True,
            alibi_learned: bool = False
    ):
        super().__init__(
            dim=dim,
            heads=heads,
            head_dim=head_dim,
            causal=causal,
            dropout=dropout,
            one_kv_head=one_kv_head,
            context_as_input_prefix=context_as_input_prefix,
            context_as_attention_prefix=context_as_attention_prefix,
            num_mem_kv=num_mem_kv,
            max_attend=max_attend,
            alibi_pos_bias=alibi_pos_bias,
            alibi_heads=alibi_heads,
            alibi_symmetric=alibi_symmetric,
            alibi_learned=alibi_learned
        )

        self.window_size = window_size
        self.exact_window = exact_window
        self.local_windows = 0 if local_windows is None else local_windows

        assert not self.context_as_input_prefix or not self.context_as_attention_prefix, \
            "LocalAttention does not support conditioning as prefix yet"

    def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor | None = None,
            context: torch.Tensor | None = None,
            context_mask: torch.Tensor | None = None,
            input_prefix_len: int = 0,
            attn_mask: torch.Tensor | None = None,
            causal: torch.Tensor | None = None,
            memory: torch.Tensor | None = None,
            cache: AttentionIntermediates | None = None,
            shared_cache: AttentionSharedIntermediates | None = None
    ) -> tuple[torch.Tensor, AttentionIntermediates, AttentionSharedIntermediates]:
        assert context is None, "LocalAttention does not support contextual attention"
        assert cache is None, "LocalAttention does not fully support caching"
        assert memory is None, "LocalAttention does not support memory yet"

        b, n = x.shape[:2]
        h, device = self.heads, x.device
        window_size, local_windows = self.window_size, self.local_windows
        pad_value = -1.

        local_windows = 0 if n <= window_size else local_windows
        offset = window_size * local_windows
        max_attend = offset if self.max_attend is None else self.max_attend

        kv_input = x if context is None else context

        # project for queries, keys, values

        q = self.to_q(x)
        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        if cache is not None and context is not None:
            k, v = cache.keys, cache.values
        else:
            k, v = self.to_kv(kv_input).chunk(2, dim=-1)

            kvh = 1 if self.one_kv_head else h
            k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=kvh), (k, v))

        # kv cache

        if cache is not None and context is None:
            k = torch.cat([cache.keys, k], dim=-2)
            v = torch.cat([cache.values, v], dim=-2)

        if self.one_kv_head:
            k, v = k.expand_as(q), q.expand_as(q)

        orig_seq_len = q.shape[2]

        # pack and pad

        q, packed_shape = rearrange(q, "b h n d -> (b h) n d"), q.shape[:2]
        needed_pad, q = pad_to_multiple(q, window_size, dim=-2)
        _, k = pad_to_multiple(rearrange(k, "b h n d -> (b h) n d"), window_size, dim=-2)
        _, v = pad_to_multiple(rearrange(v, "b h n d -> (b h) n d"), window_size, dim=-2)

        bh, n, head_dim = q.shape
        device, dtype = q.device, q.dtype
        windows = n // window_size

        seq = torch.arange(n, device=device)
        b_t = rearrange(seq, "(w n) -> 1 w n", w=windows, n=window_size)

        # bucketing

        bq, bk, bv = map(lambda t: rearrange(t, "b (w n) d -> b w n d", w=windows), (q, k, v))

        bk = look_around(bk, backward=local_windows, forward=local_windows, pad_value=pad_value)
        bv = look_around(bv, backward=local_windows, forward=local_windows, pad_value=pad_value)

        # calculate positions for masking

        bq_t = b_t
        bq_k = look_around(b_t, backward=local_windows, forward=local_windows, pad_value=pad_value)

        bq_t = bq_t[..., :, None]
        bq_k = bq_k[..., None, :]

        pad_mask = bq_k == pad_value

        # handle all masks, starting with the mask for padding value

        final_attn_mask = pad_mask  # True for attended positions

        # input padding mask

        if mask is not None:
            batch = mask.shape[0]
            assert (b % batch) == 0

            _, input_mask = pad_to_multiple(mask, window_size, dim=-1, value=0.)

            input_mask = rearrange(input_mask, "b (w n) -> b w n", w=windows)
            input_mask = look_around(input_mask, backward=local_windows, forward=local_windows, pad_value=0.)
            input_mask = rearrange(input_mask, "... j -> ... 1 j")
            input_mask = repeat(input_mask, "b ... -> (b h) ...", h=h)

            final_attn_mask = final_attn_mask | (~input_mask)

        # causal mask

        if self.causal:
            causal_mask = bq_t < bq_k

            if max_attend > 0:
                causal_mask = causal_mask | (bq_t > (bq_k + max_attend))

            final_attn_mask = final_attn_mask | causal_mask

        # mask out for exact window size for non-causal

        if not self.causal and max_attend > 0:
            window_mask = ((bq_k - max_attend) > bq_t) | (bq_t > (bq_k + max_attend))  # forward + backward
            final_attn_mask = final_attn_mask | window_mask

        final_attn_mask = ~final_attn_mask

        # prepare relative positional bias, if needed

        rel_pos_bias: torch.Tensor | None = None
        attn_bias: torch.Tensor | None = None
        if self.rel_pos is not None:
            i, j = bq.shape[-2], bk.shape[-2]
            if shared_cache is not None and shared_cache.rel_pos_bias is not None:
                rel_pos_bias = shared_cache.rel_pos_bias
            else:
                rel_pos_bias = self.rel_pos.get_bias(i, j, offset=offset).to(dtype=q.dtype)
            attn_bias = repeat(self.rel_pos(i, j, offset=offset, bias=rel_pos_bias), "h ... -> (b h) w ...", b=b, w=windows)

        # attend

        out, intermediates = self.attend(
            bq, bk, bv,
            mask=final_attn_mask,
            attn_bias=attn_bias,
            offset=offset
        )

        # merge heads

        out = rearrange(out, "b w n d -> b (w n) d")
        out = out[:, :orig_seq_len, :]
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)

        # combine the heads

        out = self.to_out(out)

        if mask is not None:
            mask = mask[:, -1:] if cache is not None else mask
            out = out * mask[..., None]

        shared_intermediates = AttentionSharedIntermediates(rel_pos_bias=rel_pos_bias)

        return out, intermediates, shared_intermediates
