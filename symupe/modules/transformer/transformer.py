""" Transformer Attention Layers with data caching support for inference. """
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from .attend import AttentionIntermediates
from .attention import AttentionRegistry, Attention, AttentionConfig, AttentionSharedIntermediates
from .feedforward import FeedForward, FeedForwardConfig
from .normalization import LayerNorm, AdaptiveLayerNorm, RMSNorm, AdaptiveRMSNorm
from ..constructor import Constructor, Registry, ModuleConfig, VariableModuleConfig


@dataclass
class TransformerLayerIntermediates:
    attention: AttentionIntermediates | None = None
    cross_attention: AttentionIntermediates | None = None
    output: torch.Tensor | None = None

    def add_past_cache(self, past_cache: TransformerLayerIntermediates | None) -> TransformerLayerIntermediates:
        if past_cache is None:
            return self

        if self.attention is not None:
            self.attention.add_past_cache(past_cache.attention)

        if self.cross_attention is not None:
            self.cross_attention.add_past_cache(past_cache.cross_attention)

        return self


@dataclass
class TransformerLayerOutput:
    out: torch.Tensor
    intermediates: TransformerLayerIntermediates | None = None
    shared_intermediates: AttentionSharedIntermediates | None = None


@dataclass
class TransformerLayerConfig(ModuleConfig):
    dim: int = 384
    causal: bool = False
    attention: AttentionConfig | DictConfig | None = field(default_factory=lambda: AttentionConfig())
    cross_attention: AttentionConfig | DictConfig | None = None
    feed_forward: FeedForwardConfig | DictConfig = field(default_factory=lambda: FeedForwardConfig())
    pre_norm: bool = True
    rms_norm: bool = False
    adaptive_norm: bool = False
    condition_dim: int | None = None
    context_as_input_prefix: bool = False
    context_as_attention_prefix: bool = False


class TransformerLayer(nn.Module, Constructor):
    def __init__(
            self,
            dim: int = 384,
            causal: bool = False,
            attention: AttentionConfig | DictConfig | None = AttentionConfig(),
            cross_attention: AttentionConfig | DictConfig | None = None,
            feed_forward: FeedForwardConfig | DictConfig = FeedForwardConfig(),
            pre_norm: bool = True,
            rms_norm: bool = False,
            adaptive_norm: bool = False,
            condition_dim: int | None = None,
            context_as_input_prefix: bool = False,
            context_as_attention_prefix: bool = False
    ):
        super().__init__()

        self.pre_norm = pre_norm
        self.adaptive_norm = adaptive_norm

        assert not adaptive_norm or condition_dim is not None
        if rms_norm:
            norm_fn = AdaptiveRMSNorm if adaptive_norm else RMSNorm
        else:
            norm_fn = AdaptiveLayerNorm if adaptive_norm else LayerNorm
        norm_kwargs = dict(condition_dim=condition_dim) if adaptive_norm else dict()

        self.context_as_input_prefix = context_as_input_prefix
        self.context_as_attention_prefix = context_as_attention_prefix

        self.attention, self.attention_norm = None, None
        if attention:
            self.attention_norm = norm_fn(dim, **norm_kwargs)
            self.attention = AttentionRegistry.instantiate(
                attention,
                dim=dim,
                causal=causal,
                context_as_input_prefix=context_as_input_prefix,
                context_as_attention_prefix=context_as_attention_prefix
            )

        self.cross_attention, self.cross_attention_norm = None, None
        if cross_attention:
            self.cross_attention_norm = norm_fn(dim, **norm_kwargs)
            self.cross_attention = Attention.init(cross_attention, dim=dim)

        self.feed_forward_norm = norm_fn(dim, **norm_kwargs)
        self.feed_forward = FeedForward.init(feed_forward, dim=dim)

    def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor | None = None,
            context: torch.Tensor | None = None,
            context_mask: torch.Tensor | None = None,
            input_prefix_len: int = 0,
            attention_mask: torch.Tensor | None = None,
            causal: torch.Tensor | None = None,
            adaptive_condition: torch.Tensor | None = None,
            memory: list[torch.Tensor] | None = None,
            cache: TransformerLayerIntermediates | None = None,
            shared_cache: AttentionSharedIntermediates | None = None
    ) -> TransformerLayerOutput:
        assert not self.adaptive_norm or adaptive_condition is not None, \
            "`adaptive_condition` should be provided for AdaptiveLayerNorm"

        norm_kwargs = dict(condition=adaptive_condition) if self.adaptive_norm else dict()

        # attention layer + normalization

        attn_cache: AttentionIntermediates | None = None
        attn_shared_cache: AttentionSharedIntermediates | None = None
        if self.attention is not None:
            residual = x
            if self.pre_norm:  # (pre-layer) normalization
                x = self.attention_norm(x, **norm_kwargs)

            x, attn_cache, attn_shared_cache = self.attention(
                x,
                mask=mask,
                context=context if self.context_as_attention_prefix else None,
                context_mask=context_mask,
                input_prefix_len=input_prefix_len,
                attn_mask=attention_mask,
                causal=causal,
                memory=memory,
                cache=cache.attention if cache is not None else None,
                shared_cache=shared_cache
            )
            x = x + residual  # residual connection

            if not self.pre_norm:  # (post-layer) normalization
                x = self.attention_norm(x, **norm_kwargs)

        # cross attention + normalization

        cross_attn_cache: AttentionIntermediates | None = None
        if self.cross_attention is not None:
            residual = x
            if self.pre_norm:  # (pre-layer) normalization
                x = self.cross_attention_norm(x, **norm_kwargs)

            x, cross_attn_cache, _ = self.cross_attention(
                x,
                mask=mask,
                context=context,
                context_mask=context_mask,
                cache=cache.cross_attention if cache is not None else None
            )
            x = x + residual  # residual connection

            if not self.pre_norm:  # (post-layer) normalization
                x = self.cross_attention_norm(x, **norm_kwargs)

        # position-wise feed-forward + normalization

        residual = x
        if self.pre_norm:  # (pre-layer) normalization
            x = self.feed_forward_norm(x, **norm_kwargs)

        x = self.feed_forward(x)
        x = x + residual  # residual connection

        if not self.pre_norm:  # (post-layer) normalization
            x = self.feed_forward_norm(x, **norm_kwargs)

        # x = x * mask[..., None] if mask is not None and cache is None else x

        intermediates = TransformerLayerIntermediates(
            attention=attn_cache,
            cross_attention=cross_attn_cache,
            output=torch.cat([cache.output, x], dim=1) if cache is not None else x
        )

        return TransformerLayerOutput(
            out=x,
            intermediates=intermediates,
            shared_intermediates=attn_shared_cache
        )


TransformerRegistry = type("_TransformerRegistry", (Registry,), {})()


@dataclass
class TransformerIntermediates:
    output: torch.Tensor
    layers: list[TransformerLayerIntermediates] | None = None
    memories: list[torch.Tensor] | None = None

    def add_past_cache(self, past_cache: TransformerIntermediates | None) -> TransformerIntermediates:
        if past_cache is None:
            return self

        if past_cache.layers is not None and self.layers is not None:
            for past_layer_cache, layer_cache in zip(past_cache.layers, self.layers):
                layer_cache.add_past_cache(past_layer_cache)

        return self


@dataclass
class TransformerOutput:
    out: torch.Tensor
    memory_tokens: torch.Tensor | None = None
    intermediates: TransformerIntermediates | None = None


@dataclass
class TransformerConfig(VariableModuleConfig):
    _target_: str = "default"

    dim: int = 384
    depth: int = 6
    causal: bool = False

    attention: AttentionConfig | DictConfig | None = AttentionConfig()
    feed_forward: FeedForwardConfig | DictConfig = FeedForwardConfig()
    cross_attend: bool = False
    cross_attention: AttentionConfig | DictConfig | None = None

    pre_norm: bool = True
    rms_norm: bool = False
    adaptive_norm: bool = False
    condition_dim: int | None = None
    final_norm_bias: bool = True

    context_as_input_prefix: bool = False
    context_as_attention_prefix: bool = False
    context_as_layer_input: bool = False
    context_as_layer_input_sum: bool = False
    context_layer_ids: Sequence[int] | None = None

    skip_connections: bool = False
    skip_connection_scale: float | None = None

    share_rel_pos_bias: bool = False
    memory_tokens: int | None = None
    max_memory_len: int | None = None


@TransformerRegistry.register("default")
class Transformer(nn.Module, Constructor):
    def __init__(
            self,
            dim: int = 384,
            depth: int = 6,
            causal: bool = False,
            attention: AttentionConfig | DictConfig | None = AttentionConfig(),
            feed_forward: FeedForwardConfig | DictConfig = FeedForwardConfig(),
            cross_attend: bool = False,
            cross_attention: AttentionConfig | DictConfig | None = None,
            pre_norm: bool = True,
            rms_norm: bool = False,
            adaptive_norm: bool = False,
            condition_dim: int | None = None,
            final_norm_bias: bool = True,
            context_dim: int | None = None,
            context_as_input_prefix: bool = False,
            context_as_attention_prefix: bool = False,
            context_as_layer_input: bool = False,
            context_as_layer_input_sum: bool = False,
            context_layer_ids: Sequence[int] | None = None,
            skip_connections: bool = False,
            skip_connection_scale: float | None = None,
            share_rel_pos_bias: bool = False,
            memory_tokens: int | None = None,
            max_memory_len: int | None = None
    ):
        super().__init__()

        self.dim = dim
        self.causal = causal
        self.pre_norm = pre_norm
        self.adaptive_norm = adaptive_norm

        cross_attention = cross_attention if cross_attend and cross_attention is not None else cross_attention
        self.cross_attend = cross_attend or cross_attention is not None

        assert (
                int(self.cross_attend) + int(context_as_input_prefix)
                + int(context_as_attention_prefix) + int(context_as_layer_input) <= 1
        ), ("Only one of `cross_attend`/`cross_attention`, `context_as_input_prefix` "
            "and `context_as_attention_prefix` can be used at the same time")

        self.context_as_input_prefix = context_as_input_prefix
        self.context_as_attention_prefix = context_as_attention_prefix
        self.context_as_layer_input = context_as_layer_input
        self.context_as_layer_input_sum = context_as_layer_input_sum

        self.expects_context = (
                self.cross_attend or self.context_as_input_prefix
                or self.context_as_attention_prefix or self.context_as_layer_input
        )
        self.context_layer_ids = context_layer_ids or set(range(depth)) if self.expects_context else set()

        self.layers = nn.ModuleList([
            TransformerLayer.init(
                dim=dim,
                causal=causal,
                attention=attention,
                cross_attention=cross_attention if i in self.context_layer_ids else None,
                feed_forward=feed_forward,
                pre_norm=pre_norm,
                rms_norm=rms_norm,
                adaptive_norm=adaptive_norm,
                condition_dim=condition_dim,
                context_as_input_prefix=context_as_input_prefix and i in self.context_layer_ids,
                context_as_attention_prefix=context_as_attention_prefix and i in self.context_layer_ids
            )
            for i in range(depth)
        ])

        self.skip_connection_scale = skip_connection_scale or 2 ** -0.5
        self.skip_projections = None
        if skip_connections:
            self.skip_projections = nn.ModuleList([
                nn.Linear(dim * 2, dim, bias=False) if (i + 1) > ceil(depth / 2) else None
                for i in range(depth)
            ])

        self.context_projections = None
        if context_as_layer_input:
            context_dim = context_dim or dim
            self.context_projections = nn.ModuleList([
                nn.Linear(int(1 - context_as_layer_input_sum) * dim + context_dim, dim, bias=False)
                if i in self.context_layer_ids else None
                for i in range(depth)
            ])

        if share_rel_pos_bias:
            for layer in self.layers[1:]:
                layer.attention.rel_pos = self.layers[0].attention.rel_pos

        memory_tokens = memory_tokens or 0
        self.memory_tokens = None
        if memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(memory_tokens, dim))

        self.max_memory_len = max_memory_len or 0

        self.norm = LayerNorm(dim, bias=final_norm_bias) if self.pre_norm else nn.Identity()

    def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor | None = None,
            context: torch.Tensor | None = None,
            context_mask: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            causal: torch.Tensor | None = None,
            adaptive_condition: torch.Tensor | None = None,
            cache: TransformerIntermediates | None = None,
            return_cache: bool = False,
            memories: list[torch.Tensor] | None = None,
            return_memories: bool = False,
            output_layer: int | None = None
    ) -> TransformerOutput:
        assert not (self.expects_context ^ (context is not None)), \
            "`context` must be passed for `cross_attention`/`context_as_input_prefix`/`context_as_attention_prefix`"
        assert not self.adaptive_norm or adaptive_condition is not None, \
            "`adaptive_condition` must be passed for `AdaptiveLayerNorm`"

        if context is not None and context_mask is None:
            context_mask = torch.ones(context.shape[:2], device=x.device, dtype=torch.bool)

        # self attention kwargs

        prefix_len = 0
        if self.context_as_input_prefix:
            b, n, _, device = *x.shape, x.device

            prefix_len = context.shape[1]
            x = torch.cat((context, x), dim=1)

            if mask is None:
                mask = torch.ones((b, n), device=device, dtype=torch.bool)

            mask = torch.cat((context_mask, mask), dim=-1)

        memories = memories if memories is not None else [None] * len(self.layers)

        if cache is not None:
            x = x[:, -1:]
            adaptive_condition = adaptive_condition[:, -1:] if adaptive_condition is not None else None

        num_mem = 0
        if self.memory_tokens is not None:
            num_mem = self.memory_tokens.shape[0]

            if cache is None:
                mem = self.memory_tokens[None].expand(x.shape[0], -1, -1)
                x = torch.cat((mem, x), dim=1)

            if mask is not None:
                mask = F.pad(mask, (num_mem, 0), value=1.)
            if attention_mask is not None:
                attention_mask = F.pad(attention_mask, (num_mem, 0, num_mem, 0), value=True)
            if adaptive_condition is not None and adaptive_condition.shape[1] > 1:
                adaptive_condition = F.pad(adaptive_condition, (0, 0, num_mem, 0), value=0.)
            if context is not None and self.context_projections is not None:
                context = F.pad(context, (0, 0, num_mem, 0), value=0.)

        output_layer = output_layer or -1
        output_layer = len(self.layers) + output_layer if output_layer < 0 else output_layer
        assert 0 <= output_layer < len(self.layers), \
            f"Transformer has only {len(self.layers)}, while the passed `output_layer` asks for layer #{output_layer}"
        full_path = output_layer == len(self.layers) - 1

        layer_cache, shared_cache = None, None
        layer_intermediates: list[TransformerLayerIntermediates] = []
        new_memories: list[torch.Tensor] = []
        skip_connections: list[torch.Tensor] = []
        for layer_idx, layer in enumerate(self.layers):
            if cache is not None:
                layer_cache = cache.layers[layer_idx]
            memory = memories[layer_idx]

            skip_projection = self.skip_projections[layer_idx] if self.skip_projections is not None else None
            if skip_projection is not None:
                x = torch.cat((x, self.skip_connection_scale * skip_connections.pop()), dim=-1)
                x = skip_projection(x)

            context_projection = self.context_projections[layer_idx] if self.context_projections is not None else None
            if context is not None and context_projection is not None:
                if self.context_as_layer_input_sum:
                    x = x + context_projection(context)
                else:
                    x = torch.cat((x, context), dim=-1)
                    x = context_projection(x)

            layer_output: TransformerLayerOutput = layer(
                x,
                mask=mask,
                context=context,
                context_mask=context_mask,
                input_prefix_len=prefix_len,
                attention_mask=attention_mask,
                causal=causal,
                adaptive_condition=adaptive_condition,
                memory=memory,
                cache=layer_cache,
                shared_cache=shared_cache
            )
            x = layer_output.out
            shared_cache = layer_output.shared_intermediates

            if return_cache:
                layer_intermediates.append(layer_output.intermediates)

            if return_memories and layer_output.intermediates:
                if memory is None:
                    new_memory = layer_output.intermediates.output[:, num_mem:]
                else:
                    new_memory = torch.cat([memory, layer_output.intermediates.output[:, num_mem:]], dim=1)
                new_memories.append(new_memory[:, -self.max_mem_len:])

            if self.skip_projections is not None and layer_idx < len(self.layers) // 2:
                skip_connections.append(x)

            if output_layer == layer_idx:
                break

        if full_path:
            out = self.norm(x)
            out = out * mask[..., None] if mask is not None else out
        else:
            out = x

        mem = None
        if self.memory_tokens is not None:
            mem, out = out[:, :num_mem], out[:, num_mem:]

        if self.context_as_input_prefix:
            out = out[:, prefix_len:]

        if full_path and cache is not None:
            out = torch.cat([cache.output, out[:, -1:]], dim=1)

        intermediates = TransformerIntermediates(
            output=out,
            layers=layer_intermediates,
            memories=new_memories
        )

        return TransformerOutput(
            out=out,
            memory_tokens=mem,
            intermediates=intermediates
        )


@dataclass
class EncoderTransformerConfig(TransformerConfig):
    _target_: str = "encoder"
    causal: bool = False


@TransformerRegistry.register("encoder")
class EncoderTransformer(Transformer):
    def __init__(self, **kwargs):
        kwargs.pop("causal", None)
        super().__init__(causal=False, **kwargs)


@dataclass
class DecoderTransformerConfig(TransformerConfig):
    _target_: str = "decoder"
    causal: bool = True


@TransformerRegistry.register("decoder")
class DecoderTransformer(Transformer):
    def __init__(self, **kwargs):
        kwargs.pop("causal", None)
        super().__init__(causal=True, **kwargs)
