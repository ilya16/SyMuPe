""" TupleTransformer with MMD-VAE output embedding heads. """
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from modules.tuple_transformer.transformer import EmbeddingMode
from symupe.modules.transformer import TransformerConfig
from symupe.modules.tuple_transformer import TupleTransformerEmbeddingsConfig
from symupe.modules.tuple_transformer import (
    TupleTransformerOutput, TupleTransformerConfig, TupleTransformer,
    TupleTransformerHeadsConfig, TupleTransformerSplitValueHeadConfig
)
from symupe.utils import ExplicitEnum


class EmbeddingAggregateMode(ExplicitEnum):
    GLOBAL = "global"
    BAR = "bar"
    BEAT = "beat"
    ONSET = "onset"
    NOTE = "note"


@dataclass
class MMDTupleTransformerOutput(TupleTransformerOutput):
    latents: list[torch.Tensor] | None = None
    embeddings: torch.Tensor | None = None
    full_embeddings: torch.Tensor | None = None
    dropout_mask: torch.Tensor | None = None
    loss: torch.Tensor | None = None
    losses: dict[str, torch.Tensor] | None = None


@dataclass
class LatentConfig:
    dim: int = 64
    dropout: float = 0.
    reg_attributes: list[str | None] | None = None


@dataclass
class MMDTupleTransformerConfig(TupleTransformerConfig):
    latent_hierarchy: str | list[str] = None
    latents: dict[str | EmbeddingAggregateMode, LatentConfig | dict] | DictConfig = field(
        default_factory=lambda: {EmbeddingAggregateMode.GLOBAL.value: LatentConfig(dim=64)}
    )
    residual: bool = False
    hierarchical: bool = False
    hierarchical_cumulative: bool = True
    inclusive_latent_dropout: bool = True
    regularization_regression: bool = False
    regularization_residual: bool = False
    mmd_gamma: float | None = None
    loss_weight: float = 1.0
    reg_loss_weight: float = 1.0


class MMDVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.linear = nn.Linear(input_dim, latent_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


class MMDTupleTransformer(TupleTransformer):
    def __init__(
            self,
            num_tokens: dict[str, int],
            dim: int = 512,
            max_seq_len: int = 1024,
            transformer: DictConfig | TransformerConfig = TransformerConfig(_target_="default"),
            token_embeddings: DictConfig | TupleTransformerEmbeddingsConfig = TupleTransformerEmbeddingsConfig(),
            use_abs_pos_emb: bool = True,
            emb_norm: bool = False,
            emb_dropout: float = 0.0,
            context_embedding: str = EmbeddingMode.ATTENTION,
            context_embedding_dim: int | None = None,
            style_embedding: str = EmbeddingMode.CONCAT,
            style_embedding_dim: int | None = None,
            lm_head: DictConfig | TupleTransformerHeadsConfig | None = None,
            value_head: DictConfig | TupleTransformerSplitValueHeadConfig | None = None,
            transformer_output_layer: int | None = None,
            latent_hierarchy: str | list[str] = None,
            latents: dict[str | EmbeddingAggregateMode, LatentConfig | dict] | DictConfig = field(
                default_factory=lambda: {EmbeddingAggregateMode.GLOBAL.value: LatentConfig(dim=64)}
            ),
            residual: bool = False,
            hierarchical: bool = False,
            hierarchical_cumulative: bool = True,  # condition of note embeddings and all preceding latents
            inclusive_latent_dropout: bool = True,  # dropout all lower-level latents for dropped out segment
            deadpan_zero_latent: bool = False,  # optimize latents to zero for deadpan performances
            regularization_regression: bool = False,  # regularize absolute attribute differences for projected latents
            regularization_residual: bool = False,  # regularize residual attributes
            mmd_gamma: float | None = 0.5,
            loss_weight: float = 1.0,
            reg_loss_weight: float = 1.0
    ):
        super().__init__(
            num_tokens=num_tokens,
            dim=dim,
            max_seq_len=max_seq_len,
            transformer=transformer,
            token_embeddings=token_embeddings,
            use_abs_pos_emb=use_abs_pos_emb,
            emb_norm=emb_norm,
            emb_dropout=emb_dropout,
            context_embedding=context_embedding,
            context_embedding_dim=context_embedding_dim,
            style_embedding=style_embedding,
            style_embedding_dim=style_embedding_dim,
            lm_head=lm_head,
            value_head=value_head,
            transformer_output_layer=transformer_output_layer
        )

        latent_hierarchy = [] if isinstance(latent_hierarchy, str) else list(latent_hierarchy)
        self.latents = {
            mode: LatentConfig(**latents[mode])
            for mode in latent_hierarchy
        }

        for mode in latent_hierarchy:
            assert EmbeddingAggregateMode.has_value(mode), \
                f"`{mode}` is not a valid `aggregate_mode`, available modes: {EmbeddingAggregateMode.list()}"

        self.residual = residual
        self.hierarchical = hierarchical
        self.hierarchical_cumulative = hierarchical_cumulative

        self.inclusive_latent_dropout = inclusive_latent_dropout
        self.deadpan_zero_latent = deadpan_zero_latent

        self.vae_heads = nn.ModuleDict()
        input_dim, latent_dims = dim, []
        has_reg_attributes = False
        for mode, latent in self.latents.items():
            latent_dim_i = latent.dim
            latent_dims.append(latent_dim_i)

            assert (latent.reg_attributes is None or len(latent.reg_attributes) == 0
                    or latent_dim_i >= len(latent.reg_attributes))
            has_reg_attributes = has_reg_attributes or (
                    latent.reg_attributes is not None and len(latent.reg_attributes) > 0
            )

            self.vae_heads[mode] = MMDVAE(
                input_dim=input_dim,
                latent_dim=latent_dim_i
            )
            if self.hierarchical:
                if self.hierarchical_cumulative:
                    input_dim += latent_dim_i
                else:
                    input_dim = latent_dim_i

        self.latent_dims = latent_dims
        self.embedding_dim = sum(latent_dims)

        self.criterion = MMDLoss(gamma=mmd_gamma)
        self.loss_weight = loss_weight

        self.reg_heads = None
        if regularization_regression:
            self.reg_heads = nn.ModuleDict()
            for mode, latent in self.latents.items():
                num_reg_attributes = len(latent.reg_attributes)
                if latent.reg_attributes is None or num_reg_attributes == 0:
                    continue
                self.reg_heads[mode] = nn.Conv1d(
                    in_channels=num_reg_attributes,
                    out_channels=num_reg_attributes,
                    kernel_size=1,
                    groups=num_reg_attributes
                )

        self.regularization_regression = regularization_regression
        self.regularization_residual = regularization_residual
        self._type_to_idx = {key: i for i, key in enumerate(self.token_emb.embs.keys())}

        self.reg_criterion = AttributeRegularizationLoss(
            regression=regularization_regression
        ) if has_reg_attributes else None
        self.reg_loss_weight = reg_loss_weight

    def forward(
            self,
            tokens: torch.Tensor | list[torch.Tensor],
            values: torch.Tensor | list[torch.Tensor] | None = None,
            mask: torch.Tensor | None = None,
            latents: torch.Tensor | list[torch.Tensor] | None = None,
            bars: torch.Tensor | None = None,
            beats: torch.Tensor | None = None,
            onsets: torch.Tensor | None = None,
            deadpan_mask: torch.Tensor | None = None,
            return_embeddings: bool = False,
            return_attn: bool = False,
            compute_loss: bool = True,
            **kwargs
    ) -> MMDTupleTransformerOutput:
        x = tokens[0] if isinstance(tokens, list) else tokens

        transformer_outputs = super().forward(
            tokens=tokens, values=values, mask=mask,
            return_embeddings=return_embeddings, return_attn=return_attn,
            **kwargs
        )

        note_embeddings = transformer_outputs.hidden_state

        if mask is None:
            mask = torch.ones_like(note_embeddings[..., :1], dtype=torch.bool, device=x.device)
        else:
            mask = mask.unsqueeze(-1)
            note_embeddings = note_embeddings * mask

        assert not self.deadpan_zero_latent or deadpan_mask is not None

        _latents = latents
        prior_drop_mask = None
        latents, embeddings, drop_masks = [], [], []
        loss, losses, reg_losses = None, {}, {}

        # segm_values
        for i, (aggregate_mode, latent_config) in enumerate(self.latents.items()):
            segments = self._get_segments(aggregate_mode, bars=bars, beats=beats, onsets=onsets)
            latents_i, latents_mask_i, embeddings_i, note_embeddings_i, segm_values_i, values_i, drop_mask_i = (
                self._forward_latents(
                    note_embeddings, mask, aggregate_mode,
                    values=values, segments=segments, latents=None if _latents is None else _latents[i]
                )
            )

            if self.training and self.inclusive_latent_dropout:
                if prior_drop_mask is None:
                    prior_drop_mask = drop_mask_i
                elif drop_mask_i is not None:
                    prior_drop_mask = drop_mask_i = prior_drop_mask + drop_mask_i

            latents.append(latents_i)
            embeddings.append(embeddings_i)
            drop_masks.append(drop_mask_i.expand_as(embeddings_i))

            if self.residual:
                note_embeddings = note_embeddings - note_embeddings_i  # .detach()
            if self.regularization_residual:
                values = values - values_i

            if self.hierarchical:
                if self.hierarchical_cumulative:
                    note_embeddings = torch.concatenate([note_embeddings, embeddings_i], dim=-1)
                else:
                    note_embeddings = embeddings_i

            if compute_loss:
                losses[f"MMD/{aggregate_mode}"] = self.criterion(latents_i, mask=latents_mask_i)

                if self.deadpan_zero_latent:
                    deadpan_latents_i = latents_i[deadpan_mask[:, None] * latents_mask_i]
                    if torch.any(deadpan_latents_i):
                        losses[f"MMD/{aggregate_mode}/deadpan"] = F.mse_loss(
                            deadpan_latents_i, torch.zeros_like(deadpan_latents_i)
                        )

                reg_attributes = self.latents[aggregate_mode].reg_attributes or []
                if values is not None and len(reg_attributes) > 0 and self.reg_criterion is not None:
                    reg_latents = latents_i[..., :len(reg_attributes)]
                    if self.regularization_regression:
                        reg_latents = self.reg_heads[aggregate_mode](
                            latents_i[..., :len(reg_attributes)].transpose(1, 2)
                        ).transpose(1, 2)
                    for z_idx, key in enumerate(reg_attributes):
                        if key is None or key == "None":
                            continue
                        reg_losses[f"reg/{key}/{aggregate_mode}"] = self.reg_criterion(
                            reg_latents[..., z_idx], segm_values_i[..., self._type_to_idx[key]], mask=latents_mask_i
                        )

        embeddings = torch.cat(embeddings, dim=-1)
        drop_mask = torch.cat(drop_masks, dim=-1)

        embeddings = embeddings * mask

        if self.training:
            full_embeddings = embeddings.clone()
            drop_mask = drop_mask * mask
            if deadpan_mask is not None:
                drop_mask = drop_mask * (~deadpan_mask[:, None, None])
            embeddings = embeddings * (~drop_mask)
        else:
            full_embeddings = embeddings
            drop_mask = None

        if compute_loss:
            reg_losses = reg_losses or {}
            loss = losses["MMD"] = self.loss_weight * sum(losses.values())
            if len(reg_losses):
                losses["reg"] = self.reg_loss_weight * sum(reg_losses.values()) / len(reg_losses)
                loss = losses["MMD"] + losses["reg"]
                losses.update(**reg_losses)

        return MMDTupleTransformerOutput(
            hidden_state=transformer_outputs.hidden_state,
            logits=transformer_outputs.logits,
            attentions=transformer_outputs.attentions,
            latents=latents,
            embeddings=embeddings,
            full_embeddings=full_embeddings,
            dropout_mask=drop_mask,
            loss=loss,
            losses=losses
        )

    def _forward_latents(
            self,
            note_embeddings: torch.Tensor,
            mask: torch.Tensor,
            aggregate_mode: str,
            values: torch.Tensor | None = None,
            segments: torch.Tensor | None = None,
            latents: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t = note_embeddings.shape[:2]

        segment_mode = aggregate_mode in (
            EmbeddingAggregateMode.BAR,
            EmbeddingAggregateMode.BEAT,
            EmbeddingAggregateMode.ONSET
        )

        values_mask = False
        if values is not None:
            values_mask = mask.clone().detach()[..., 0]
            values_mask[(values < -100.).any(dim=-1)] = False
            values[~values_mask] = 0.

        segm_note_embeddings = note_embeddings
        segm_values = values
        latents_mask = None
        if latents is None:
            if aggregate_mode == EmbeddingAggregateMode.GLOBAL:
                segm_note_embeddings = note_embeddings.sum(dim=1) / mask.sum(dim=1)
                segm_note_embeddings = segm_note_embeddings.unsqueeze(1)
                latents_mask = torch.ones_like(segm_note_embeddings[..., 0], dtype=torch.bool)
                if values is not None:
                    segm_values = values.sum(dim=1) / values_mask.sum(dim=1)
            elif segment_mode:
                # build segment alignment
                alignment = torch.zeros(b, t, segments.max() + 1, device=note_embeddings.device)
                indices = (
                    torch.arange(b).repeat_interleave(t),
                    torch.arange(t).repeat(b),
                    segments.view(-1)
                )
                alignment[indices] = 1.

                # aggregate note embeddings by segments
                counts = torch.maximum(torch.tensor(1), alignment.sum(dim=1))[..., None]
                segm_note_embeddings = (note_embeddings.transpose(1, 2) @ alignment).transpose(1, 2) / counts

                alignment[~values_mask] = 0.
                counts = torch.maximum(torch.tensor(1), alignment.sum(dim=1))[..., None]
                segm_values = (values.transpose(1, 2) @ alignment).transpose(1, 2) / counts

                latents_mask = torch.all(segm_note_embeddings != 0., dim=-1)
            else:
                latents_mask = mask[..., 0]

            latents = self.vae_heads[aggregate_mode](segm_note_embeddings)
            latents = latents * latents_mask[..., None]

        embeddings = latents

        latent_dropout = self.latents[aggregate_mode].dropout
        if aggregate_mode != EmbeddingAggregateMode.GLOBAL and self.training and latent_dropout > 0.:
            drop_mask = dropout_latent_mask(latents_mask, latent_dropout)
        else:
            drop_mask = torch.zeros_like(latents_mask[..., None], dtype=torch.bool)

        note_embeddings = segm_note_embeddings
        values = segm_values
        if aggregate_mode == EmbeddingAggregateMode.GLOBAL:
            embeddings = embeddings.expand(-1, t, -1)
            note_embeddings = segm_note_embeddings.expand(-1, t, -1)
            if segm_values is not None:
                values = segm_values.expand(-1, t, -1)
            if drop_mask is not None:
                drop_mask = drop_mask.expand(-1, t, -1)
        elif segment_mode:
            # distribute embeddings
            indices = torch.arange(b).repeat_interleave(t)
            embeddings = embeddings[(indices, segments.view(-1))].view(b, t, -1)
            note_embeddings = segm_note_embeddings[(indices, segments.view(-1))].view(b, t, -1)
            if segm_values is not None:
                values = segm_values[(indices, segments.view(-1))].view(b, t, -1)
            if drop_mask is not None:
                drop_mask = drop_mask[(indices, segments.view(-1))].view(b, t, -1)

        embeddings = embeddings * mask

        return latents, latents_mask, embeddings, note_embeddings, segm_values, values, drop_mask

    @staticmethod
    def _get_segments(
            aggregate_mode: str,
            bars: torch.Tensor | None = None,
            beats: torch.Tensor | None = None,
            onsets: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        if aggregate_mode == EmbeddingAggregateMode.BAR:
            assert bars is not None, f"`bars` should be provided as inputs for aggregate_mode `{aggregate_mode}`"
            return bars
        elif aggregate_mode == EmbeddingAggregateMode.BEAT:
            assert beats is not None, f"`beats` should be provided as inputs for aggregate_mode `{aggregate_mode}`"
            return beats
        elif aggregate_mode == EmbeddingAggregateMode.ONSET:
            assert onsets is not None, f"`onsets` should be provided as inputs for aggregate_mode `{aggregate_mode}`"
            return onsets
        return None

    def embeddings_to_latents(
            self,
            embeddings: torch.Tensor,
            mask: torch.Tensor | None = None,
            bars: torch.Tensor | None = None,
            beats: torch.Tensor | None = None,
            onsets: torch.Tensor | None = None
    ) -> torch.Tensor:
        embeddings = embeddings.split(self.latent_dims, dim=-1)
        latents = []
        for i, aggregate_mode in enumerate(self.latents.keys()):
            segments = self._get_segments(aggregate_mode, bars=bars, beats=beats, onsets=onsets)
            latents_i = self._embeddings_to_latents(
                embeddings[i], aggregate_mode, segments=segments, mask=mask
            )
            latents.append(latents_i)

        return latents

    @staticmethod
    def _embeddings_to_latents(
            embeddings: torch.Tensor,
            aggregate_mode: str,
            mask: torch.Tensor | None = None,
            segments: torch.Tensor | None = None
    ):
        b, t = embeddings.shape[:2]

        segment_mode = aggregate_mode in (
            EmbeddingAggregateMode.BAR,
            EmbeddingAggregateMode.BEAT,
            EmbeddingAggregateMode.ONSET
        )

        if aggregate_mode == EmbeddingAggregateMode.GLOBAL:
            if mask is None:
                latents = embeddings.mean(dim=1)
            else:
                latents = embeddings.sum(dim=1) / mask.sum(dim=1)
            latents = latents.unsqueeze(1)
        elif segment_mode:
            # build segment alignment
            alignment = torch.zeros(b, t, segments.max() + 1, device=embeddings.device)
            indices = (
                torch.arange(b).repeat_interleave(t),
                torch.arange(t).repeat(b),
                segments.view(-1)
            )
            alignment[indices] = 1.

            # aggregate output embeddings by segments
            counts = torch.maximum(torch.tensor(1), alignment.sum(dim=1))[..., None]
            latents = (embeddings.transpose(1, 2) @ alignment).transpose(1, 2) / counts
        else:
            latents = embeddings

        return latents

    def latents_to_embeddings(
            self,
            latents: torch.Tensor,
            seq_len: torch.Tensor,
            bars: torch.Tensor | None = None,
            beats: torch.Tensor | None = None,
            onsets: torch.Tensor | None = None
    ) -> torch.Tensor:
        embeddings = []
        for i, aggregate_mode in enumerate(self.latents.keys()):
            segments = self._get_segments(aggregate_mode, bars=bars, beats=beats, onsets=onsets)
            embeddings_i = self._latents_to_embeddings(
                latents[i], seq_len, aggregate_mode, segments=segments
            )
            embeddings.append(embeddings_i)

        embeddings = torch.cat(embeddings, dim=-1)

        return embeddings

    @staticmethod
    def _latents_to_embeddings(
            latents: torch.Tensor,
            seq_len: torch.Tensor,
            aggregate_mode: str,
            segments: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, t = latents.shape[0], seq_len

        segment_mode = aggregate_mode in (
            EmbeddingAggregateMode.BAR,
            EmbeddingAggregateMode.BEAT,
            EmbeddingAggregateMode.ONSET
        )

        embeddings = latents
        if aggregate_mode == EmbeddingAggregateMode.GLOBAL:
            embeddings = embeddings.expand(-1, t, -1)
        elif segment_mode:
            # distribute embeddings
            embeddings = embeddings[(torch.arange(b).repeat_interleave(t), segments.view(-1))].view(b, t, -1)

        return embeddings


class MMDLoss(nn.Module):
    def __init__(self, num_samples: int = 256, max_num_latents: int = 1024, gamma: float | None = None):
        super().__init__()
        self.num_samples = num_samples
        self.max_num_latents = max_num_latents
        self.gamma = gamma

    def forward(self, latents: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            latents = latents[mask]

        if latents.shape[0] > self.max_num_latents:
            # avoid memory overflow: sample `max_num_latents` and compute loss only for them
            latents = latents[torch.randperm(latents.shape[0])[:self.max_num_latents]]

        z = torch.randn(self.num_samples, latents.shape[-1], device=latents.device, dtype=latents.dtype)
        return self.compute_mmd(z, latents)

    def gaussian_kernel(self, x, y) -> torch.Tensor:
        x_core = x.unsqueeze(1).expand(-1, y.size(0), -1)  # (x_dim, y_dim, dim)
        y_core = y.unsqueeze(0).expand(x.size(0), -1, -1)  # (x_dim, y_dim, dim)
        gamma = 1 / x.size(-1) if self.gamma is None else self.gamma
        numerator = (x_core - y_core).pow(2).mean(2) * gamma  # (x_dim, y_dim)
        return torch.exp(-numerator)

    def compute_mmd(self, x, y) -> torch.Tensor:
        x_kernel = self.gaussian_kernel(x, x)
        y_kernel = self.gaussian_kernel(y, y)
        xy_kernel = self.gaussian_kernel(x, y)
        return x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()


class AttributeRegularizationLoss(nn.Module):
    def __init__(
            self,
            delta: float = 1.,
            regression: bool = False,
            max_num_latents: int = 1024
    ):
        super().__init__()
        self.delta = delta
        self.regression = regression
        self.max_num_latents = max_num_latents

    def forward(
            self,
            latents: torch.Tensor,
            attributes: torch.Tensor,
            mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if mask is not None:
            latents, attributes = latents[mask], attributes[mask]

        if latents.shape[0] > self.max_num_latents:
            # avoid memory overflow: sample `max_num_latents` and compute loss only for them
            ids = torch.randperm(latents.shape[0])[:self.max_num_latents]
            latents, attributes = latents[ids], attributes[ids]

        dist_z = latents[None] - latents[:, None]
        dist_a = attributes[None] - attributes[:, None]

        if self.regression:
            return F.l1_loss(dist_z, dist_a)
        return F.l1_loss(torch.tanh(self.delta * dist_z), torch.sign(dist_a))


def dropout_latent_mask(mask: torch.Tensor, dropout: float) -> torch.Tensor:
    bs, ts = torch.where(mask)
    drop_ids = torch.rand(bs.shape[0]) < dropout
    drop_mask = torch.zeros_like(mask, dtype=torch.bool)
    drop_mask[bs[drop_ids], ts[drop_ids]] = True
    return drop_mask[..., None]
