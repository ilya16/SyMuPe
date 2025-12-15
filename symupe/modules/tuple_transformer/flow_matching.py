""" TupleTransformer wrappers for Flow Matching modeling tasks. """
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm.asyncio import tqdm

from symupe.data.tokenizers import SyMuPe
from symupe.data.tokenizers.constants import SPECIAL_TOKENS_VALUE, EOS_TOKEN, EOD_TOKEN
from symupe.modules.classes import ModelWrapper
from symupe.modules.metrics import masked_batch_mean
from symupe.modules.sampling import (
    top_p_filtering,
    cubic_scheduler, cubic_scheduler_derivative, x2prob, sample_p
)
from symupe.modules.tuple_transformer import TupleTransformerOutput, TupleTransformer
from symupe.utils import fill_by_mask_and_indices


@dataclass
class TupleTransformerCFMOutput(TupleTransformerOutput):
    x_1_pred: torch.Tensor | None = None
    pred_values: torch.Tensor | None = None
    pred_pedals: torch.Tensor | None = None
    loss: torch.Tensor | None = None
    losses: dict[str, torch.Tensor] | None = None


@dataclass
class CFMIntermediates:
    time_steps: list[torch.Tensor] | None = None
    points: list[torch.Tensor] | None = None
    vectors: list[torch.Tensor] | None = None
    losses: list[torch.Tensor] | None = None
    grads: list[torch.Tensor] | None = None


def resample(x_t: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, x_0: torch.Tensor | None = None) -> torch.Tensor:
    # x_t = (t * x_t + dt * torch.randn_like(x_t)) / (t + dt)
    x_t = (t / (t + dt)) * x_t
    noise_factor = (2 * torch.clamp(1 - t - dt, min=0.) * dt / (t + dt)) ** 0.5
    x_t = x_t + noise_factor * torch.randn_like(x_t)
    return x_t


class TupleTransformerCFMWrapper(ModelWrapper):
    def __init__(
            self,
            model: TupleTransformer,
            tokenizer: SyMuPe,
            sigma: float = 1e-5,
            cosine_loss: bool = False,
            consistency_loss: bool = False,
            delta: float = 1e-3,
            global_tempo_loss: bool = False,
            method: str = "euler",
            context_vectors: bool = False,
            value_mean: torch.Tensor | None = None,
            value_std: torch.Tensor | None = None,
            value_log_ids: torch.Tensor | None = None,
            value_keys: list[str] | None = None,
            mask_token_id: int = 1,
            ignore_index: int = -100
    ):
        super().__init__(model=model)

        self.tokenizer = tokenizer

        self.sigma = sigma
        self.cosine_loss = cosine_loss
        self.consistency_loss = consistency_loss
        self.delta = delta

        self.context_vectors = context_vectors
        self.global_tempo_loss = global_tempo_loss

        assert method in ("euler", "midpoint")
        self.method = method

        self.mask_token_id = mask_token_id
        self.ignore_index = ignore_index

        self.token_types = list(self.model.num_tokens.keys())
        self.value_keys = value_keys or self.token_types
        self.value_indices = None
        if self.value_keys:
            self.value_indices = [idx for idx, key in enumerate(self.token_types) if key in value_keys]

        self.pos_shift_index = self.token_types.index("PositionShift") if "PositionShift" in self.token_types else None
        self.time_shift_index = self.value_keys.index("TimeShift") if "TimeShift" in self.value_keys else None

        self.register_buffer("value_mean", value_mean, persistent=False)
        self.register_buffer("value_std", value_std, persistent=False)
        self.register_buffer("value_log_ids", value_log_ids, persistent=False)

    def normalize_values(self, values: torch.Tensor) -> torch.Tensor:
        values = (values - self.value_mean) / self.value_std
        if len(self.value_log_ids) > 0:
            v = values[..., self.value_log_ids]
            values[..., self.value_log_ids] = torch.sign(v) * torch.log1p(torch.abs(v))
        return values

    def denormalize_values(self, values: torch.Tensor) -> torch.Tensor:
        values = values * self.value_std + self.value_mean
        if len(self.value_log_ids) > 0:
            v = values[..., self.value_log_ids]
            values[..., self.value_log_ids] = torch.sign(v) * torch.expm1(torch.abs(v))
        return values

    def normalize_pedals(self, pedals: torch.Tensor) -> torch.Tensor:
        if self.time_shift_index is not None:
            pedals[..., 1] = (pedals[..., 1] - self.value_mean[self.time_shift_index]) / self.value_std[self.time_shift_index]
        else:
            pedals[..., 1] = torch.where(
                pedals[..., 1] >= 0., torch.log1p(torch.abs(pedals[..., 1])), torch.tensor(-1., device=pedals.device)
            )
        return pedals

    def denormalize_pedals(self, pedals: torch.Tensor) -> torch.Tensor:
        if self.time_shift_index is not None:
            pedals[..., 1] = pedals[..., 1] * self.value_std[self.time_shift_index] + self.value_mean[self.time_shift_index]
        else:
            pedals[..., 1] = torch.where(
                pedals[..., 1] >= 0., torch.expm1(pedals[..., 1]), torch.tensor(-1., device=pedals.device)
            )
        return pedals

    @staticmethod
    def process_pedal_predictions(pedals: torch.Tensor) -> torch.Tensor:
        pedals[..., 1][pedals[..., 1] <= 0.] = -1.

        pedals[..., 0][pedals[..., 0] > 0.5] = 1.
        pedals[..., 0][pedals[..., 0] < -0.5] = -1.
        pedals[..., 0][(pedals[..., 0] > -0.5) & (pedals[..., 0] < 0.5)] = 0.

        return pedals

    def forward(
            self,
            tokens: torch.Tensor | None = None,
            values: torch.Tensor | None = None,
            vectors: torch.Tensor | None = None,
            labels: torch.Tensor | None = None,
            targets: torch.Tensor | None = None,
            pedals: torch.Tensor | None = None,
            compute_loss: bool = True,
            ema_model: nn.Module | None = None,
            **kwargs
    ) -> TupleTransformerCFMOutput:
        vectors_all = vectors.clone()
        vectors = vectors[..., self.value_indices] if self.value_indices is not None else vectors

        # main conditional flow logic
        x_1 = self.normalize_values(vectors.clone())
        x_1[vectors <= SPECIAL_TOKENS_VALUE] = 0.

        if pedals is not None:
            pedals = self.normalize_pedals(pedals.clone())
            x_1 = torch.cat([x_1, pedals], dim=-1)

        # x_0 is gaussian noise
        x_0 = torch.randn_like(x_1)

        # random times
        time_steps = torch.rand((x_1.shape[0],), dtype=x_1.dtype, device=x_1.device)
        t = rearrange(time_steps, "b -> b 1 1")

        # sample x_t (w in the paper)
        x_t = (1 - (1 - self.sigma) * t) * x_0 + t * x_1
        flow = x_1 - (1 - self.sigma) * x_0

        xv_t = x_t
        if self.context_vectors:
            values_ctx = values.clone()
            values_ctx = values_ctx[..., self.value_indices] if self.value_indices is not None else values_ctx
            x_ctx = self.normalize_values(values_ctx)
            x_ctx[values_ctx <= SPECIAL_TOKENS_VALUE] = 0.
            xv_t = torch.cat([x_ctx, x_t], dim=-1)

        # predict

        out: TupleTransformerOutput = self.model(
            tokens=tokens,
            values=values,
            vectors=xv_t,
            time_steps=time_steps,
            **kwargs
        )
        u_t = out.values.float()

        with torch.no_grad():
            x_1_pred = x_t + u_t * (1 - t)
            pred_values = self.denormalize_values(x_1_pred[..., :vectors.shape[-1]])

            is_known = tokens != self.mask_token_id if labels is None else labels == self.ignore_index
            values_1 = values.clone()
            if self.value_indices is not None:
                values_1, is_known = values_1[..., self.value_indices], is_known[..., self.value_indices]
            pred_values[is_known] = values_1[is_known]

            pred_pedals = None
            if pedals is not None:
                pred_pedals = x_1_pred[..., vectors.shape[-1]:]

                pred_pedals = self.denormalize_pedals(pred_pedals)
                pred_pedals = self.process_pedal_predictions(pred_pedals)

        loss, losses = None, None
        if compute_loss:
            if self.value_indices is not None:
                labels = labels[..., self.value_indices]
            loss_mask = labels != self.ignore_index

            if pedals is not None:
                loss_mask = torch.cat([
                    loss_mask,
                    torch.ones(loss_mask.shape[:2] + (pedals.shape[-1],), device=pedals.device, dtype=torch.bool),
                ], dim=-1)

            mask = kwargs.get("mask", None)
            if mask is not None:
                loss_mask = loss_mask * mask[..., None]

            token_types = self.value_keys + (["Pedal", "PedalShift"] if pedals is not None else [])

            cfm_losses = {
                f"{key}/cfm_mse": masked_batch_mean(
                    F.mse_loss(
                        u_t[..., i],
                        flow[..., i],
                        reduction="none"
                    ),
                    mask=loss_mask[..., i]
                )
                for i, key in enumerate(token_types)
                if torch.any(loss_mask[..., i])
            }
            loss = loss_cfm = masked_batch_mean(F.mse_loss(u_t, flow, reduction="none"), loss_mask)

            losses = {"cfm_mse": loss_cfm}
            losses.update(**cfm_losses)

            if self.cosine_loss:
                loss_sim = 1 - masked_batch_mean(
                    torch.nn.functional.cosine_similarity(u_t, flow, dim=-1),
                    torch.any(loss_mask, dim=-1)
                )
                loss = loss + loss_sim
                losses.update(**{"cfm_sim": loss_sim})

            if self.consistency_loss:
                with torch.no_grad():
                    r = torch.clamp(t + self.delta, max=1.)
                    x_r = (1 - (1 - self.sigma) * r) * x_0 + r * x_1
                    ema_model = self.model if ema_model is None else ema_model.model
                    v_r = ema_model(
                        tokens=tokens,
                        values=values,
                        vectors=x_r,
                        time_steps=torch.clamp(time_steps + self.delta, max=1.),
                        **kwargs
                    ).values.float()

                f_t = x_t + u_t * (1 - t)
                f_r = x_r + v_r * (1 - r)

                if loss_mask is None:
                    loss_f = F.mse_loss(f_t, f_r)
                    loss_v = F.mse_loss(u_t, v_r)
                else:
                    loss_f = masked_batch_mean(F.mse_loss(f_t, f_r, reduction="none"), loss_mask)
                    loss_v = masked_batch_mean(F.mse_loss(u_t, v_r, reduction="none"), loss_mask)

                loss = loss + loss_f + loss_v
                losses.update(**{"cfm_consistency/f": loss_f, "cfm_consistency/v": loss_v})

            if self.global_tempo_loss:
                assert self.pos_shift_index is not None and self.time_shift_index is not None

                pos_shifts = vectors_all[..., self.pos_shift_index]
                pos_shifts[pos_shifts <= SPECIAL_TOKENS_VALUE] = 0.
                vectors_1 = vectors.clone()

                # x_1_est = x_0 + u_t
                x_1_est = x_t + u_t * (1 - t)
                est_values = self.denormalize_values(x_1_est[..., :vectors.shape[-1]])

                est_values[vectors_1 <= SPECIAL_TOKENS_VALUE] = 0.
                vectors_1[vectors_1 <= SPECIAL_TOKENS_VALUE] = 0.

                est_time_shifts = est_values[..., self.time_shift_index]
                time_shifts = vectors_1[..., self.time_shift_index]

                total_pos_shift = torch.sum(pos_shifts, dim=-1)
                spw = torch.sum(time_shifts, dim=-1) / (total_pos_shift + 1e-5)
                est_spw = torch.sum(est_time_shifts, dim=-1) / (total_pos_shift + 1e-5)

                loss_mask = (mask.sum(dim=-1) > 16) & (total_pos_shift > 0.5)

                if torch.any(loss_mask):
                    loss_tempo = 0.1 * F.mse_loss(spw[loss_mask], est_spw[loss_mask])
                    loss = loss + loss_tempo
                    losses.update(**{"cfm_endpoint/tempo": loss_tempo})

        return TupleTransformerCFMOutput(
            x_1_pred=x_1_pred,
            pred_values=pred_values,
            pred_pedals=pred_pedals,
            loss=loss,
            losses=losses,
            **out.__dict__
        )

    def generate(
            self,
            tokens: torch.Tensor,
            values: torch.Tensor,
            vectors: torch.Tensor,
            tokenizer: SyMuPe,
            mask: torch.Tensor | None = None,

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

            value_denorm_fn: Callable | None = None,
            disable_tqdm: bool = False,
            **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, CFMIntermediates]:
        tokenizer = tokenizer or self.tokenizer

        was_training = self.model.training
        if was_training:
            self.model.eval()

        num_dims = len(tokens.shape)
        if num_dims == 2:
            tokens = tokens[None]
            values = values[None]
            vectors = vectors[None]
            pedals = pedals[None] if pedals is not None and pedals.ndim != values.ndim else None

        x_1 = self.normalize_values(vectors.clone())
        x_1[vectors <= SPECIAL_TOKENS_VALUE] = 0.

        pedals_1 = None
        if pedals is not None:
            pedals_1 = pedals.clone()
            pedals = self.normalize_pedals(pedals.clone())
            x_1 = torch.cat([x_1, pedals], dim=-1)

        out_tokens = tokens.clone().detach()
        out_values = values.clone().detach()
        out_pedals = pedals.clone().detach() if pedals is not None else None

        if mask is None:
            mask = torch.full_like(out_tokens[..., 0], True, dtype=torch.bool, device=out_tokens.device)

        num_values = vectors.shape[-1]

        unmask = out_tokens == self.mask_token_id
        if self.value_indices is not None:
            unmask = unmask[..., self.value_indices]

        if pedals_1 is not None:
            unmask = torch.cat([unmask, pedals_1 == -1], dim=-1)

        x_0 = torch.randn_like(x_1) if x_0 is None else x_0.view_as(vectors)

        method = method or self.method
        assert method in ("euler", "midpoint")

        if step_factor == 1.:
            time_steps = torch.linspace(0, 1, steps + 1)
        else:
            assert step_factor < 1.
            time_steps = -torch.diff(torch.logspace(0, steps, steps + 1, base=step_factor))
            time_steps = torch.cat([torch.tensor([0.]), time_steps])
            time_steps = torch.cumsum(time_steps / time_steps.sum(), dim=0)

        num_resample = 1 if loss_fn is None else num_resample

        x_t = x_0.clone()

        if context_scale != 1.:
            tokens = torch.cat([tokens, tokens], dim=0)
            values = torch.cat([values, values], dim=0)
            if context is not None:
                context = torch.cat([context, torch.zeros_like(context)], dim=0)
            if context_tokens is not None:
                context_tokens = torch.cat([
                    context_tokens,
                    torch.full_like(context_tokens, fill_value=self.mask_token_id)
                ], dim=0)
            if context_values is not None:
                context_values = torch.cat([
                    context_values,
                    torch.full_like(context_values, fill_value=SPECIAL_TOKENS_VALUE - self.mask_token_id)
                ], dim=0)

        points, vectors, losses, grads = [], [], [], []
        pbar = time_steps[:-1] if disable_tqdm else tqdm(time_steps[:-1], leave=False)
        for i, t in enumerate(pbar):
            dt = time_steps[i + 1] - time_steps[i]

            loss, u_t = None, None
            for k in range(num_resample):
                x_t_known = (1 - (1 - self.sigma) * t) * x_0 + t * x_1
                x_t[~unmask] = x_t_known[~unmask]
                if k == 0:
                    points.append(x_t.clone().detach())

                if loss_fn is not None:
                    x_t = x_t.detach().requires_grad_()

                _x_t = xv_t = torch.cat([x_t, x_t], dim=0) if context_scale != 1. else x_t

                if self.context_vectors:
                    values_ctx = values.clone()
                    values_ctx = values_ctx[..., self.value_indices] if self.value_indices is not None else values_ctx
                    x_ctx = self.normalize_values(values_ctx)
                    x_ctx[values_ctx <= SPECIAL_TOKENS_VALUE] = 0.
                    xv_t = torch.cat([x_ctx, x_t], dim=-1)

                with torch.inference_mode(mode=loss_fn is None):
                    outputs: TupleTransformerOutput = self.model(
                        tokens=tokens,
                        values=values,
                        vectors=xv_t,
                        time_steps=t,
                        mask=mask,
                        context=context,
                        context_mask=context_mask,
                        context_tokens=context_tokens,
                        context_values=context_values,
                        **kwargs
                    )
                    if context_scale == 1.:
                        u_t = outputs.values
                    else:
                        u_t, null_u_t = torch.chunk(outputs.values, 2, dim=0)
                        u_t = null_u_t + (u_t - null_u_t) * context_scale

                    if method == "midpoint":
                        outputs: TupleTransformerOutput = self.model(
                            tokens=tokens,
                            values=values,
                            vectors=_x_t + (dt / 2) * u_t,
                            time_steps=t + dt / 2,
                            mask=mask,
                            context=context,
                            context_mask=context_mask,
                            context_tokens=context_tokens,
                            context_values=context_values,
                            **kwargs
                        )
                        if context_scale == 1.:
                            u_t = outputs.values
                        else:
                            u_t, null_u_t = torch.chunk(outputs.values, 2, dim=0)
                            u_t = null_u_t + (u_t - null_u_t) * context_scale

                if loss_fn is not None:
                    # code by @realfolkcode
                    x_clean = x_t + (1 - (1 - self.sigma) * t) * u_t
                    x_clean = value_denorm_fn(x_clean)

                    loss = loss_fn(x_clean[:, context_len:])
                    grad = torch.autograd.grad(loss, x_t)[0]
                    if norm_fn is not None:
                        grad = norm_fn(
                            grad=grad,
                            vector_field=u_t,
                            t=t,
                            x_t=x_t
                        )

                    if schedule_fn is not None:
                        schedule = schedule_fn(t)
                    else:
                        schedule = 1

                    if k == 0:
                        grads.append(grad.clone().detach())
                    u_t = u_t - gamma * schedule * grad

                x_t = x_t + u_t * dt

                if loss_fn is None:
                    continue

                if i % resample_period != 0:
                    break

                if k < num_resample - 1:
                    x_t = resample_fn(x_t, t, dt, x_0)
                elif num_resample > 1:
                    with torch.no_grad():
                        outputs = self.model(
                            tokens=tokens[:1],
                            values=values[:1],
                            vectors=x_t,
                            time_steps=t,
                            mask=mask,
                            context=context[:1] if context is not None else None,
                            context_mask=context_mask,
                            context_tokens=context_tokens[:1] if context_tokens is not None else None,
                            context_values=context_values[:1] if context_values is not None else None,
                            **kwargs
                        )
                    x_0 = x_t - (t + dt) * outputs.values

            vectors.append(u_t.detach())

            if loss is not None:
                losses.append(loss.item())

        if loss_fn is not None:
            loss = loss_fn(value_denorm_fn(x_t[:, context_len:]))
            losses.append(loss.item())

        x_t = x_t.detach()
        pred_values = self.denormalize_values(x_t[..., :num_values])
        out_values = fill_by_mask_and_indices(
            out_values, pred_values, mask=unmask[..., :num_values], dims=self.value_indices
        )
        x_t[~unmask] = x_1[~unmask]
        points.append(x_t.clone().detach())

        pred_tokens = tokenizer.encode_tokens(pred_values, token_type=self.value_keys, denormalize=True)
        out_tokens = fill_by_mask_and_indices(
            out_tokens, pred_tokens, mask=unmask[..., :num_values], dims=self.value_indices
        )

        if pedals is not None:
            pred_pedals = x_t[..., num_values:]

            pred_pedals = self.denormalize_pedals(pred_pedals)
            pred_pedals = self.process_pedal_predictions(pred_pedals)

            out_pedals[unmask[..., num_values:]] = pred_pedals[unmask[..., num_values:]]

        if num_dims == 2:
            out_tokens = out_tokens.squeeze(0)
            out_values = out_values.squeeze(0)

        intermediates = CFMIntermediates(
            time_steps=time_steps[:-1],
            points=points,
            vectors=vectors,
            losses=losses,
            grads=grads
        )

        if was_training:
            self.model.train(was_training)

        return out_tokens, out_values, out_pedals, intermediates


@dataclass
class TupleTransformerDFMOutput(TupleTransformerOutput):
    pred_tokens: torch.Tensor | None = None
    loss: torch.Tensor | None = None
    losses: dict[str, torch.Tensor] | None = None


@dataclass
class DFMIntermediates:
    time_steps: list[torch.Tensor] | None = None
    states: list[torch.Tensor] | None = None
    probs: list[dict[str, torch.Tensor]] | None = None


class TupleTransformerDFMWrapper(ModelWrapper):
    def __init__(
            self,
            model: TupleTransformer,
            tokenizer: SyMuPe,
            token_keys: list[str] | None = None,
            distribution: str = "uniform",
            method: str = "euler",
            mask_token_id: int = 1,
            ignore_index: int = -100
    ):
        super().__init__(model=model)

        self.tokenizer = tokenizer

        assert distribution in ("mask", "uniform")
        self.distribution = distribution

        assert method in ("euler",)
        self.method = method

        self.mask_token_id = mask_token_id
        self.eos_token_id = None
        self.eod_token_id = None
        self.ignore_index = ignore_index

        self.token_types = list(self.model.num_tokens.keys())
        self.token_keys = token_keys or self.token_types
        self.token_indices = None
        if self.token_keys:
            self.token_indices = [idx for idx, key in enumerate(self.token_types) if key in token_keys]
        self.num_tokens = self.model.token_emb.num_tokens
        self.register_buffer("token_nums", torch.tensor(list(self.num_tokens.values())), persistent=False)

    def set_tokenizer(self, tokenizer: SyMuPe):
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer[0, EOS_TOKEN] if EOS_TOKEN in tokenizer.special_tokens else None
        self.eod_token_id = tokenizer[0, EOD_TOKEN] if EOD_TOKEN in tokenizer.special_tokens else None

    def forward(
            self,
            tokens: torch.Tensor,
            values: torch.Tensor | None = None,
            labels: torch.Tensor | None = None,
            targets: torch.Tensor | None = None,
            full_labels: torch.Tensor | None = None,
            compute_loss: bool = True,
            ema_model: nn.Module | None = None,
            **kwargs
    ) -> TupleTransformerDFMOutput:
        # main conditional flow logic
        x_1 = full_labels.clone()

        unmask = tokens == self.mask_token_id
        if self.token_indices is not None:
            unmask = unmask[..., self.token_indices]

        # x_0 is noisy/masked input
        if self.distribution == "uniform":
            x_0 = (torch.rand_like(tokens) * self.token_nums).long()
            x_0[tokens == 0] = 0
            x_0[~unmask] = tokens[~unmask]
        else:
            x_0 = tokens

        # random times
        dtype = values.dtype if values is not None else torch.float32
        time_steps = torch.rand((x_1.shape[0],), dtype=dtype, device=x_1.device)
        t = rearrange(time_steps, "b -> b 1 1")

        # sample x_t
        x_t = x_0.clone()
        mask = torch.rand_like(x_1.float()) < cubic_scheduler(t)
        x_t[mask] = x_1[mask]

        y_t = self.tokenizer.decode_values(x_t, token_type=self.token_keys, normalize=True).float()
        if self.token_indices is not None:
            tokens = tokens[..., self.token_indices]
            values = values[..., self.token_indices]
        x_t[~unmask], y_t[~unmask] = tokens[~unmask], values[~unmask]

        # predict

        out: TupleTransformerOutput = self.model(
            tokens=x_t,
            values=y_t,
            time_steps=time_steps,
            **kwargs
        )
        p_1t = out.logits

        with torch.no_grad():
            pred_tokens = []
            for key, key_logits in p_1t.items():
                pred_tokens.append(key_logits.argmax(dim=-1))
            pred_tokens = torch.stack(pred_tokens, dim=-1)
            pred_tokens[~unmask] = tokens[~unmask]

        loss, losses = None, None
        if compute_loss and out.logits is not None and labels is not None:
            if self.token_indices is not None:
                labels = labels[..., self.token_indices]
            label_mask = labels != self.ignore_index
            dfm_losses = {
                f"{key}/dfm_ce": masked_batch_mean(
                    F.cross_entropy(
                        out.logits[key].transpose(1, 2),
                        labels[..., i],
                        reduction="none"
                    ),
                    mask=label_mask[..., i]
                )
                for i, (key, logits) in enumerate(out.logits.items())
                if torch.any(label_mask[..., i])
            }

            loss = sum(dfm_losses.values()) / len(dfm_losses)
            loss = loss + 0. * sum(p.sum() for p in self.model.lm_head.parameters())

            losses = {"dfm_ce": loss}
            losses.update(**dfm_losses)

        return TupleTransformerDFMOutput(
            pred_tokens=pred_tokens,
            loss=loss,
            losses=losses,
            **out.__dict__
        )

    def generate(
            self,
            tokens: torch.Tensor,
            values: torch.Tensor,
            tokenizer: SyMuPe,
            mask: torch.Tensor | None = None,

            seq_len: int = 256,
            steps: int = 4,
            step_factor: float = 1.,
            method: str | None = None,

            context: torch.Tensor | None = None,
            context_mask: torch.Tensor | None = None,
            context_tokens: torch.Tensor | None = None,
            context_values: torch.Tensor | None = None,
            context_scale: float = 1.,

            disable_tqdm: bool = False,
            **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, DFMIntermediates]:
        tokenizer = tokenizer or self.tokenizer

        was_training = self.model.training
        if was_training:
            self.model.eval()

        num_dims = len(tokens.shape)
        if num_dims == 2:
            tokens = tokens[None]
            values = values[None]

        tokens_padded, values_padded = tokens, values
        if tokens.shape[1] < seq_len:
            tokens_padded = F.pad(tokens, (0, 0, 0, seq_len - tokens.shape[1]), value=self.mask_token_id)
            values_padded = F.pad(values, (0, 0, 0, seq_len - tokens.shape[1]),
                                  value=SPECIAL_TOKENS_VALUE - self.mask_token_id)

        if mask is None:
            mask = torch.full_like(values_padded[..., 0], True, dtype=torch.bool, device=tokens.device)

        unmask = tokens_padded == self.mask_token_id

        if self.distribution == "uniform":
            x_0 = (torch.rand_like(tokens_padded) * self.token_nums).long()
            x_0[tokens_padded == 0] = 0
            x_0[~unmask] = tokens_padded[~unmask]
        else:
            x_0 = tokens_padded.clone()

        method = method or self.method
        assert method in ("euler",)

        if step_factor == 1.:
            time_steps = torch.linspace(0, 1, steps + 1)
        else:
            assert step_factor < 1.
            time_steps = -torch.diff(torch.logspace(0, steps, steps + 1, base=step_factor))
            time_steps = torch.cat([torch.tensor([0.]), time_steps])
            time_steps = torch.cumsum(time_steps / time_steps.sum(), dim=0)

        x_t = x_0.clone()
        p_1t = {}

        if context_scale != 1.:
            if context is not None:
                context = torch.cat([context, torch.zeros_like(context)], dim=0)
            if context_tokens is not None:
                context_tokens = torch.cat([
                    context_tokens,
                    torch.full_like(context_tokens, fill_value=self.mask_token_id)
                ], dim=0)
            if context_values is not None:
                context_values = torch.cat([
                    context_values,
                    torch.full_like(context_values, fill_value=SPECIAL_TOKENS_VALUE - self.mask_token_id)
                ], dim=0)

        states, probs = [], []
        pbar = time_steps[:-1] if disable_tqdm else tqdm(time_steps[:-1], leave=False)
        for i, t in enumerate(pbar):
            dt = time_steps[i + 1] - time_steps[i]

            delta_t = {key: x2prob(x_t[..., i], num) for i, (key, num) in enumerate(self.num_tokens.items())}

            y_t = tokenizer.decode_values(x_t, token_type=self.token_types, normalize=True).float()
            x_t[~unmask], y_t[~unmask] = tokens_padded[~unmask], values_padded[~unmask]

            with torch.inference_mode():
                _x_t = torch.cat([x_t, x_t], dim=0) if context_scale != 1. else x_t
                _y_t = torch.cat([y_t, y_t], dim=0) if context_scale != 1. else y_t

                outputs: TupleTransformerOutput = self.model(
                    tokens=_x_t,
                    values=_y_t,
                    time_steps=t,
                    mask=mask,
                    context=context,
                    context_mask=context_mask,
                    context_tokens=context_tokens,
                    context_values=context_values,
                    **kwargs
                )

            if t < time_steps[-2]:
                kappa_coeff = cubic_scheduler_derivative(t) / (1 - cubic_scheduler(t))

                if context_scale == 1.:
                    p_1t = outputs.logits
                else:
                    p_1t = {}
                    for key, p1t_i in outputs.logits.items():
                        p1t_i, null_p1t_i = torch.chunk(p1t_i, 2, dim=0)
                        p_1t[key] = null_p1t_i + (p1t_i - null_p1t_i) * context_scale

                p_t, x_t = {}, {}
                for i, (key, num) in enumerate(self.num_tokens.items()):
                    p_t[key] = delta_t[key] + dt * kappa_coeff * (p_1t[key].softmax(-1) - delta_t[key])
                    x_t[key] = sample_p(p_t[key])

                x_t = torch.stack(list(x_t.values()), dim=-1)

                states.append(x_t)
                probs.append(p_t)

        x_t = {}
        for key, p1t_i in p_1t.items():
            x_t[key] = sample_p(p1t_i.softmax(-1))
        x_t = torch.stack(list(x_t.values()), dim=-1)

        states.append(x_t)
        probs.append(p_1t)

        y_t = tokenizer.decode_values(x_t, token_type=self.token_types, normalize=True).float()
        x_t[~unmask], y_t[~unmask] = tokens_padded[~unmask], values_padded[~unmask]

        out_tokens, out_values = x_t, y_t

        if num_dims == 2:
            out_tokens = out_tokens.squeeze(0)
            out_values = out_values.squeeze(0)

        intermediates = DFMIntermediates(
            time_steps=time_steps[:-1],
            states=states,
            probs=probs
        )

        if was_training:
            self.model.train(was_training)

        return out_tokens, out_values, intermediates


@dataclass
class TupleTransformerFMOutput(TupleTransformerOutput):
    xt_1_pred: torch.Tensor | None = None
    xv_1_pred: torch.Tensor | None = None
    pred_tokens: torch.Tensor | None = None
    pred_values: torch.Tensor | None = None
    loss: torch.Tensor | None = None
    losses: dict[str, torch.Tensor] | None = None


@dataclass
class FMIntermediates:
    time_steps: list[torch.Tensor] | None = None
    states: list[torch.Tensor] | None = None
    points: list[torch.Tensor] | None = None
    probs: list[dict[str, torch.Tensor]] | None = None


class TupleTransformerFMWrapper(ModelWrapper):
    def __init__(
            self,
            model: TupleTransformer,
            tokenizer: SyMuPe,
            token_keys: list[str],
            value_keys: list[str],
            sigma: float = 1e-5,
            distribution: str = "uniform",
            method: str = "euler",
            context_vectors: bool = False,
            causal_time: float | None = None,
            multi_mode: bool = False,
            two_step: bool = False,
            two_step_time: float = 0.25,
            separate_times: bool = False,
            value_mean: torch.Tensor | None = None,
            value_std: torch.Tensor | None = None,
            mask_token_id: int = 1,
            ignore_index: int = -100
    ):
        super().__init__(model=model)

        self.sigma = sigma

        assert distribution in ("mask", "uniform")
        self.distribution = distribution

        assert method in ("euler",)
        self.method = method

        self.context_vectors = context_vectors
        self.causal_time = causal_time
        self.multi_mode = multi_mode
        assert not two_step or multi_mode
        self.two_step = two_step
        self.two_step_time = two_step_time
        self.separate_times = separate_times

        self.tokenizer = tokenizer
        self.token_keys = token_keys
        self.value_keys = value_keys

        self.mask_token_id = mask_token_id
        self.eos_token_id = None
        self.eod_token_id = None
        self.ignore_index = ignore_index

        self.token_types = list(self.model.token_emb.embs.keys())
        self.num_tokens = self.model.token_emb.num_tokens
        self.register_buffer("token_nums", torch.tensor(list(self.num_tokens.values())), persistent=False)

        self.register_buffer("value_mean", value_mean, persistent=False)
        self.register_buffer("value_std", value_std, persistent=False)

    @property
    def num_token_features(self):
        return len(self.token_keys)

    @property
    def num_value_features(self):
        return len(self.value_keys)

    def normalize_values(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.value_mean) / self.value_std

    def denormalize_values(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.value_std + self.value_mean

    def set_tokenizer(self, tokenizer: SyMuPe):
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer[0, EOS_TOKEN] if EOS_TOKEN in tokenizer.special_tokens else None
        self.eod_token_id = tokenizer[0, EOD_TOKEN] if EOD_TOKEN in tokenizer.special_tokens else None

    def forward(
            self,
            tokens: torch.Tensor,
            values: torch.Tensor | None = None,
            vectors: torch.Tensor | None = None,
            labels: torch.Tensor | None = None,
            targets: torch.Tensor | None = None,
            full_labels: torch.Tensor | None = None,
            compute_loss: bool = True,
            ema_model: nn.Module | None = None,
            **kwargs
    ) -> TupleTransformerFMOutput:
        batch, seq_len = tokens.shape[:2]
        device = tokens.device
        dtype = values.dtype if values is not None else torch.float32

        # main conditional flow logic
        xt_1 = full_labels.clone()
        xv_1 = self.normalize_values(vectors.clone())
        xv_1[vectors <= SPECIAL_TOKENS_VALUE] = 0.

        d_tokens, v_tokens = tokens[..., :self.num_token_features], tokens[..., self.num_token_features:]

        unmask = tokens == self.mask_token_id
        d_unmask, v_unmask = unmask[..., :self.num_token_features], unmask[..., self.num_token_features:]

        # x_0 is noisy/masked input
        if self.distribution == "uniform":
            low = self.tokenizer.ignore_token if self.two_step else self.tokenizer.zero_token
            xt_0 = (
                    torch.rand_like(tokens[..., :self.num_token_features])
                    * (self.token_nums[:self.num_token_features] - low)
            ).long() + low
            xt_0[d_tokens == 0] = 0
            xt_0[~d_unmask] = d_tokens[~d_unmask]

            xt_0 = torch.cat([xt_0, v_tokens], dim=-1)
        else:
            xt_0 = tokens

        xv_0 = torch.randn_like(xv_1)

        # random times
        if self.separate_times:
            tt = torch.rand((xt_1.shape[0],), dtype=dtype, device=device)[:, None, None]
            tv = torch.rand((xt_1.shape[0],), dtype=dtype, device=device)[:, None, None]
        else:
            tt = tv = torch.rand((xt_1.shape[0],), dtype=dtype, device=device)[:, None, None]

        is_causal = kwargs.get("causal", torch.zeros(batch, dtype=torch.bool, device=device))
        if self.multi_mode or self.causal_time is not None:
            if self.multi_mode:
                is_causal = (torch.rand(batch, device=device) > 0.5).bool()
            else:
                is_causal = tt.squeeze() < self.causal_time

        if self.two_step:
            # random times
            t_causal = self.two_step_time * torch.rand((xt_1.shape[0],), dtype=dtype, device=device)[:, None, None]

            # sample x_t
            xt_t = xt_0.clone()
            mask = torch.rand_like(xt_1.float()) < cubic_scheduler(t_causal)
            xt_t[mask] = xt_1[mask]

            yt_t = self.tokenizer.decode_values(xt_t, token_type=self.token_types, normalize=True).float()
            xt_t[~unmask], yt_t[~unmask] = tokens[~unmask], values[~unmask]
            xt_t[..., self.num_token_features:] = tokens[..., self.num_token_features:]
            yt_t[..., self.num_token_features:] = values[..., self.num_token_features:]

            xv_t = (1 - (1 - self.sigma) * t_causal) * xv_0 + t_causal * xv_1

            xv_t_in = xv_t
            if self.context_vectors:
                values_ctx = values.clone()[..., self.num_token_features:]
                xv_ctx = self.normalize_values(values_ctx)
                xv_ctx[values_ctx <= SPECIAL_TOKENS_VALUE] = 0.
                xv_t_in = torch.cat([xv_ctx, xv_t], dim=-1)

            if self.model.is_multiseq:
                masked_tokens = kwargs.pop("masked_tokens", None)
                if masked_tokens is not None:
                    xt_t = [xt_t, masked_tokens]

                masked_values = kwargs.pop("masked_values", None)
                if masked_values is not None:
                    yt_t = [yt_t, masked_values]

            out0: TupleTransformerOutput = self.model(
                tokens=xt_t,
                values=yt_t,
                vectors=xv_t_in,
                time_steps=torch.stack(
                    [t_causal.squeeze(), t_causal.squeeze()], dim=1
                ) if self.separate_times else t_causal.squeeze(),
                causal=torch.ones(batch, dtype=torch.bool, device=device),
                **kwargs
            )

            out0.hidden_state = out0.hidden_state.detach()
            out0.memory_state = out0.memory_state.detach() if out0.memory_state is not None else None
            out0.task_state = out0.task_state.detach() if out0.task_state is not None else None
            out0.mode_state = out0.mode_state.detach() if out0.mode_state is not None else None
            out0.logits = {k: v.detach() for k, v in out0.logits.items()}
            out0.values = out0.values.float().detach()

            p_1t, u_t = out0.logits, out0.values.float()

            xt_1_pred = torch.stack([sample_p(p_1t_i.softmax(-1)) for _, p_1t_i in p_1t.items()], dim=-1)
            xt_1_pred[d_tokens == 0] = 0
            xt_1_pred[~d_unmask] = d_tokens[~d_unmask]
            xt_0[~is_causal, ..., :self.num_token_features] = xt_1_pred[~is_causal]

            xv_1_pred = xv_t + u_t * (1 - t_causal)
            xv_0[~is_causal] = xv_1_pred[~is_causal]

        # sample x_t
        xt_t = xt_0.clone()
        mask = torch.rand_like(xt_1.float()) < cubic_scheduler(tt)
        xt_t[mask] = xt_1[mask]

        yt_t = self.tokenizer.decode_values(xt_t, token_type=self.token_types, normalize=True).float()
        xt_t[~unmask], yt_t[~unmask] = tokens[~unmask], values[~unmask]
        xt_t[..., self.num_token_features:] = tokens[..., self.num_token_features:]
        yt_t[..., self.num_token_features:] = values[..., self.num_token_features:]

        xv_t = (1 - (1 - self.sigma) * tv) * xv_0 + tv * xv_1
        flow = xv_1 - (1 - self.sigma) * xv_0

        xv_t_in = xv_t
        if self.context_vectors:
            values_ctx = values.clone()[..., self.num_token_features:]
            xv_ctx = self.normalize_values(values_ctx)
            xv_ctx[values_ctx <= SPECIAL_TOKENS_VALUE] = 0.
            xv_t_in = torch.cat([xv_ctx, xv_t], dim=-1)

        if self.model.is_multiseq:
            masked_tokens = kwargs.pop("masked_tokens", None)
            if masked_tokens is not None:
                xt_t = [xt_t, masked_tokens]

            masked_values = kwargs.pop("masked_values", None)
            if masked_values is not None:
                yt_t = [yt_t, masked_values]

        # predict

        out: TupleTransformerOutput = self.model(
            tokens=xt_t,
            values=yt_t,
            vectors=xv_t_in,
            time_steps=torch.stack([tt.squeeze(), tv.squeeze()], dim=1) if self.separate_times else tt.squeeze(),
            causal=is_causal,
            **kwargs
        )
        p_1t = out.logits
        u_t = out.values.float()

        with torch.no_grad():
            xv_1_pred = xv_t + u_t * (1 - tv)

            pred_values = self.denormalize_values(xv_1_pred)
            pred_values[~v_unmask] = values[..., self.num_token_features:][~v_unmask]

            pred_tokens = []
            for key, key_logits in p_1t.items():
                pred_tokens.append(key_logits.argmax(dim=-1))
            pred_tokens = torch.stack(pred_tokens, dim=-1)
            xt_1_pred = pred_tokens.clone()
            pred_tokens[~d_unmask] = d_tokens[~d_unmask]

            # process pedal and bar line token values
            pred_tokens, pred_values = self._process_pitch_tokens(
                pred_tokens, pred_values, self.tokenizer,
                process_tokens=True, process_values=True
            )

            pred_tokens = torch.cat([
                pred_tokens,
                self.tokenizer.encode_tokens(pred_values, token_type=self.value_keys, denormalize=True)
            ], dim=-1)

            pred_values = torch.cat([
                self.tokenizer.decode_values(
                    pred_tokens[..., :self.num_token_features], token_type=self.token_keys, normalize=True
                ),
                pred_values
            ], dim=-1)

        loss, losses = None, None
        if compute_loss:
            loss_mask = labels[..., self.num_token_features:] != self.ignore_index
            cfm_losses = {
                f"{key}/cfm_mse": masked_batch_mean(
                    F.mse_loss(
                        u_t[..., i],
                        flow[..., i],
                        reduction="none"
                    ),
                    mask=loss_mask[..., i]
                )
                for i, key in enumerate(self.value_keys)
                if torch.any(loss_mask[..., i])
            }
            cfm_loss = None
            if torch.any(loss_mask):
                cfm_loss = masked_batch_mean(F.mse_loss(u_t, flow, reduction="none"), loss_mask)

            label_mask = labels[..., :self.num_token_features] != self.ignore_index
            dfm_losses = {
                f"{key}/dfm_ce": masked_batch_mean(
                    F.cross_entropy(
                        out.logits[key].transpose(1, 2),
                        labels[..., i],
                        reduction="none"
                    ),
                    mask=label_mask[..., i]
                )
                for i, (key, logits) in enumerate(out.logits.items())
                if torch.any(label_mask[..., i])
            }
            dfm_loss = sum(dfm_losses.values()) / len(dfm_losses) if len(dfm_losses) > 0 else None

            loss = 0. * sum(p.mean() for p in self.model.lm_head.parameters())

            losses = {}
            if cfm_loss is not None:
                loss = loss + cfm_loss
                losses["cfm_mse"] = cfm_loss
            if dfm_loss is not None:
                loss = loss + dfm_loss
                losses["dfm_ce"] = dfm_loss
            losses.update(**cfm_losses)
            losses.update(**dfm_losses)

        return TupleTransformerFMOutput(
            xv_1_pred=xv_1_pred,
            xt_1_pred=xt_1_pred,
            pred_tokens=pred_tokens,
            pred_values=pred_values,
            loss=loss,
            losses=losses,
            **out.__dict__
        )

    def generate(
            self,
            tokens: torch.Tensor,
            values: torch.Tensor,
            tokenizer: SyMuPe,
            mask: torch.Tensor | None = None,

            seq_len: int = 256,
            steps: int = 4,
            step_factor: float = 1.,
            method: str | None = None,
            xt_0: torch.Tensor | None = None,
            xv_0: torch.Tensor | None = None,
            causal_time: float | None = None,

            context: torch.Tensor | None = None,
            context_mask: torch.Tensor | None = None,
            context_tokens: torch.Tensor | None = None,
            context_values: torch.Tensor | None = None,
            context_scale: float = 1.,

            filter_key_ids: dict[str, list] | None = None,
            ignore_non_special: list[str] | None = None,

            disable_tqdm: bool = False,
            **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, FMIntermediates]:
        tokenizer = tokenizer or self.tokenizer

        was_training = self.model.training
        if was_training:
            self.model.eval()

        ignore_non_special = ignore_non_special or []
        filter_key_ids = filter_key_ids or dict()
        mask_value = -float("Inf")
        causal_time = causal_time or self.causal_time

        num_dims = len(tokens.shape)
        if num_dims == 2:
            tokens = tokens[None]
            values = values[None]

        vectors = values[..., self.num_token_features:]
        xv_1 = self.normalize_values(vectors)
        xv_1[vectors <= SPECIAL_TOKENS_VALUE] = 0.

        masked_tokens = kwargs.pop("masked_tokens", None)
        if masked_tokens is not None:
            masked_tokens = masked_tokens[None] if masked_tokens.ndim == 2 else masked_tokens

        masked_values = kwargs.pop("masked_values", None)
        if masked_values is not None:
            masked_values = masked_values[None] if masked_values.ndim == 2 else masked_values

        tokens_padded, values_padded = tokens, values
        masked_tokens_padded, masked_values_padded = masked_tokens, masked_values
        fill_len = seq_len - tokens.shape[1]
        if fill_len > 0:
            tokens_padded = F.pad(tokens, (0, 0, 0, fill_len), value=self.mask_token_id)
            values_padded = F.pad(values, (0, 0, 0, fill_len), value=SPECIAL_TOKENS_VALUE - self.mask_token_id)

            if masked_tokens_padded is not None:
                masked_tokens_padded = F.pad(masked_tokens_padded, (0, 0, 0, fill_len), value=0.)
            if masked_values_padded is not None:
                masked_values_padded = F.pad(masked_values_padded, (0, 0, 0, fill_len), value=SPECIAL_TOKENS_VALUE)

            for i, key in enumerate(self.num_tokens):
                if key in ignore_non_special:
                    tokens_padded[:, -fill_len:, i] = tokenizer.ignore_token
                    values_padded[:, -fill_len:, i] = tokenizer.ignore_value

            xv_1 = F.pad(xv_1, (0, 0, 0, fill_len), value=0.) if xv_1 is not None else xv_1
            if xv_0 is not None:
                xv_0 = torch.cat(
                    [xv_0, torch.randn((xv_0.shape[0], seq_len - xv_0.shape[1], xv_0.shape[2]), device=xv_0.device)],
                    dim=1
                )

        if mask is None:
            mask = torch.full_like(tokens_padded[..., 0], True, dtype=torch.bool, device=tokens.device)

        d_tokens_padded = tokens_padded[..., :self.num_token_features]
        v_tokens_padded = tokens_padded[..., self.num_token_features:]

        unmask = tokens_padded == self.mask_token_id
        d_unmask, v_unmask = d_tokens_padded == self.mask_token_id, v_tokens_padded == self.mask_token_id

        # x_0 is noisy/masked input
        if self.distribution == "uniform":
            low = self.tokenizer.ignore_token if self.two_step else self.tokenizer.zero_token
            xt_0_rnd = (
                    torch.rand_like(tokens_padded[..., :self.num_token_features])
                    * (self.token_nums[:self.num_token_features] - low)
            ).long() + low

            if xt_0 is not None:
                xt_0_rnd[:, :xt_0.shape[1], :self.num_token_features] \
                    = xt_0[:, :xt_0.shape[1], :self.num_token_features]
            xt_0 = xt_0_rnd

            xt_0[d_tokens_padded == 0] = 0
            xt_0[~d_unmask] = d_tokens_padded[~d_unmask]

            xt_0 = torch.cat([xt_0, v_tokens_padded], dim=-1)
        else:
            xt_0 = tokens_padded.clone() if xt_0 is None else F.pad(xt_0, (0, 0, 0, fill_len), value=self.mask_token_id)

        xv_0 = torch.randn_like(xv_1) if xv_0 is None else xv_0.view_as(v_tokens_padded.float())

        method = method or self.method
        assert method in ("euler", "jump")

        if step_factor == 1.:
            # time_steps = torch.linspace(0, 1, steps + 1)
            time_steps = cubic_scheduler(torch.linspace(0, 1, steps + 1))
        else:
            assert step_factor < 1.
            time_steps = -torch.diff(torch.logspace(0, steps, steps + 1, base=step_factor))
            time_steps = torch.cat([torch.tensor([0.]), time_steps])
            time_steps = torch.cumsum(time_steps / time_steps.sum(), dim=0)

        xt_t = xt_0.clone()
        xv_t = xv_0.clone()
        p_t = {}

        if context_scale != 1.:
            if context is not None:
                context = torch.cat([context, torch.zeros_like(context)], dim=0)
            if context_tokens is not None:
                context_tokens = torch.cat([
                    context_tokens,
                    torch.full_like(context_tokens, fill_value=self.mask_token_id)
                ], dim=0)
            if context_values is not None:
                context_values = torch.cat([
                    context_values,
                    torch.full_like(context_values, fill_value=SPECIAL_TOKENS_VALUE - self.mask_token_id)
                ], dim=0)

        states, points, probs = [], [], []
        pbar = time_steps[:-1] if disable_tqdm else tqdm(time_steps[:-1], leave=False)
        for i, t in enumerate(pbar):
            dt = time_steps[i + 1] - time_steps[i]

            yt_t = tokenizer.decode_values(xt_t, token_type=self.token_types, normalize=True).float()
            xt_t[~unmask], yt_t[~unmask] = tokens_padded[~unmask], values_padded[~unmask]
            xt_t[..., self.num_token_features:] = tokens_padded[..., self.num_token_features:]
            yt_t[..., self.num_token_features:] = values_padded[..., self.num_token_features:]

            xv_t_known = (1 - (1 - self.sigma) * t) * xv_0 + t * xv_1
            xv_t[~v_unmask] = xv_t_known[~v_unmask]

            xt_t, xv_t = self._process_pitch_tokens(
                xt_t, xv_t, tokenizer, process_tokens=True, process_values=False
            )

            states.append(xt_t[..., :self.num_token_features])
            probs.append(p_t)
            points.append(xv_t)

            with torch.inference_mode():
                _xv_t = xv_t
                if self.context_vectors:
                    values_ctx = values_padded.clone()[..., self.num_token_features:]
                    xv_ctx = self.normalize_values(values_ctx)
                    xv_ctx[values_ctx <= SPECIAL_TOKENS_VALUE] = 0.
                    _xv_t = torch.cat([xv_ctx, _xv_t], dim=-1)

                def maybe_duplicate_tensor(tensor):
                    return torch.cat([tensor, tensor], dim=0) if context_scale != 1. else tensor

                _xt_t = maybe_duplicate_tensor(xt_t)
                _yt_t = maybe_duplicate_tensor(yt_t)
                _xv_t = maybe_duplicate_tensor(_xv_t)

                device = tokens.device
                batch, seq_len = _xt_t.shape[:2]
                if causal_time is not None and t.item() < causal_time:
                    is_causal = torch.ones(batch, dtype=torch.bool, device=device)
                else:
                    is_causal = torch.zeros(batch, dtype=torch.bool, device=device)

                if self.model.is_multiseq:
                    if masked_tokens_padded is not None:
                        _xt_t = [_xt_t, maybe_duplicate_tensor(masked_tokens_padded)]

                    if masked_values_padded is not None:
                        _yt_t = [_yt_t, maybe_duplicate_tensor(masked_values_padded)]

                outputs: TupleTransformerOutput = self.model(
                    tokens=_xt_t,
                    values=_yt_t,
                    vectors=_xv_t,
                    time_steps=torch.stack([t.view(-1), t.view(-1)], dim=1) if self.separate_times else t,
                    mask=mask,
                    context=context,
                    context_mask=context_mask,
                    context_tokens=context_tokens,
                    context_values=context_values,
                    causal=is_causal,
                    **kwargs
                )

            if context_scale == 1.:
                p_1t = outputs.logits
            else:
                p_1t = {}
                for key, p1t_i in outputs.logits.items():
                    p1t_i, null_p1t_i = torch.chunk(p1t_i, 2, dim=0)
                    p_1t[key] = null_p1t_i + (p1t_i - null_p1t_i) * context_scale

            for key in p_1t.keys():
                filter_ids = filter_key_ids.get(key, None)
                if filter_ids is not None:
                    p_1t[key][..., filter_ids] = mask_value

                if key in ignore_non_special:
                    p_1t[key][..., tokenizer.zero_token:] = mask_value

            if t < time_steps[-2]:
                # kappa_coeff = 1 / (1 - t)
                kappa_coeff = cubic_scheduler_derivative(t) / (1 - cubic_scheduler(t))

                delta_t = {
                    key: x2prob(xt_t[..., idx], num) for idx, (key, num) in enumerate(self.num_tokens.items())
                    if idx < self.num_token_features
                }

                if method == "jump":
                    p_t = {key: logits.softmax(-1) for key, logits in p_1t.items()}
                    xt_t = {
                        key: xt_t[..., idx] for idx, (key, num) in enumerate(self.num_tokens.items())
                        if idx < self.num_token_features
                    }
                    for idx, (key, num) in enumerate(self.num_tokens.items()):
                        if idx == self.num_token_features:
                            break

                        xt_1_i = sample_p(p_t[key])
                        delta_1_i = x2prob(xt_1_i, num)

                        u = kappa_coeff * delta_1_i
                        u = torch.where(delta_t[key].to(dtype=torch.bool), torch.zeros_like(u), u)

                        intensity = u.sum(dim=-1)
                        mask_jump = torch.rand(size=xt_1_i.shape, device=xt_1_i.device) < 1 - torch.exp(-dt * intensity)

                        if mask_jump.sum() > 0:
                            xt_t[key][mask_jump] = sample_p(u[mask_jump])
                else:
                    p_t, xt_t = {}, {}
                    for idx, (key, num) in enumerate(self.num_tokens.items()):
                        if idx == self.num_token_features:
                            break
                        p_t[key] = delta_t[key] + dt * kappa_coeff * (p_1t[key].softmax(-1) - delta_t[key])
                        # p_t[key] = delta_t[key] + dt * kappa_coeff * (top_p_filtering(p_1t[key], thres=0.9).softmax(-1) - delta_t[key])
                        xt_t[key] = sample_p(p_t[key])
            else:
                p_t = {key: top_p_filtering(p1t_i, threshold=0.95).softmax(-1) for key, p1t_i in p_1t.items()}
                xt_t = {key: sample_p(p_t_i) for key, p_t_i in p_t.items()}

            xt_t = torch.stack(list(xt_t.values()), dim=-1)

            if t < time_steps[-2]:
                xt_t = torch.cat([xt_t, v_tokens_padded], dim=-1)

            if context_scale == 1.:
                u_t = outputs.values
            else:
                u_t, null_u_t = torch.chunk(outputs.values, 2, dim=0)
                u_t = null_u_t + (u_t - null_u_t) * context_scale

            xv_t = xv_t + u_t * dt

        pred_tokens = xt_t
        pred_tokens[~d_unmask] = d_tokens_padded[~d_unmask]

        states.append(xt_t)
        probs.append(p_t)

        xv_t[~v_unmask] = xv_1[~v_unmask]
        points.append(xv_t)

        pred_values = self.denormalize_values(xv_t)
        pred_values[~v_unmask] = values_padded[..., self.num_token_features:][~v_unmask]

        # process pedal and bar line token values
        pred_tokens, pred_values = self._process_pitch_tokens(
            pred_tokens, pred_values, tokenizer, process_tokens=True, process_values=True
        )

        out_tokens = torch.cat([
            pred_tokens,
            tokenizer.encode_tokens(pred_values, token_type=self.value_keys, denormalize=True)
        ], dim=-1)

        out_values = torch.cat([
            tokenizer.decode_values(pred_tokens, token_type=self.token_keys, normalize=True),
            pred_values
        ], dim=-1)

        if self.eod_token_id is not None:
            cut_idx = torch.where(out_tokens == self.eod_token_id)[1]
            if len(cut_idx) > 0:
                cut_idx = cut_idx.min() + 1
                out_tokens, out_values = out_tokens[:, :cut_idx], out_values[:, :cut_idx]

        if num_dims == 2:
            out_tokens = out_tokens.squeeze(0)
            out_values = out_values.squeeze(0)

        intermediates = FMIntermediates(
            time_steps=time_steps[:-1],
            states=states,
            points=points,
            probs=probs
        )

        if was_training:
            self.model.train(was_training)

        return out_tokens, out_values, intermediates

    def _process_pitch_tokens(
            self,
            tokens: torch.Tensor,
            values: torch.Tensor,
            tokenizer: SyMuPe,
            process_tokens: bool = False,
            process_values: bool = False,
            fill_with_zero: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not process_tokens and not process_values:
            return tokens, values

        bar_line_id = tokenizer.bar_line_id
        pedal_on_id, pedal_off_id = tokenizer.pedal_ids

        self.eos_token_id = tokenizer[0, EOS_TOKEN] if EOS_TOKEN in tokenizer.special_tokens else None
        self.eod_token_id = tokenizer[0, EOD_TOKEN] if EOD_TOKEN in tokenizer.special_tokens else None

        if bar_line_id is None or pedal_on_id is None or self.eod_token_id is None:
            return tokens, values

        assert self.token_keys[0].startswith("Pitch")
        pitches = tokens[..., 0]

        for token_id in [self.eos_token_id, self.eod_token_id]:
            if token_id is None:
                continue

            is_eos = torch.where(pitches == token_id)
            if process_tokens:
                tokens[is_eos[0], is_eos[1], :self.num_token_features] = token_id
            if process_values:
                for i, key in enumerate(self.value_keys):
                    values[is_eos[0], is_eos[1], i] = 0. if fill_with_zero else SPECIAL_TOKENS_VALUE - token_id

        if process_values and bar_line_id is not None:
            is_bar_line = torch.where(pitches == bar_line_id)
            for i, key in enumerate(self.value_keys):
                if key != "PositionShift":
                    values[is_bar_line[0], is_bar_line[1], i] = 0. if fill_with_zero else tokenizer.ignore_value

        if process_values and pedal_on_id is not None:
            is_pedal = torch.where(torch.logical_or(pitches == pedal_on_id, pitches == pedal_off_id))
            for i, key in enumerate(self.value_keys):
                if key != "TimeShift":
                    values[is_pedal[0], is_pedal[1], i] = 0. if fill_with_zero else tokenizer.ignore_value

        return tokens, values
