""" TupleTransformer wrappers for Language Modeling tasks. """
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from symupe.data.tokenizers import OctupleM
from symupe.data.tokenizers.constants import EOD_TOKEN
from symupe.modules.sampling import filter_and_sample
from symupe.modules.tuple_transformer import TupleTransformer, TupleTransformerOutput, TupleTransformerCache
from symupe.utils import fill_by_mask_and_indices
from ..classes import ModelWrapper
from ..metrics import masked_batch_mean, weighted_distance, emd_cdf_loss, Reduction


@dataclass
class TupleTransformerLMOutput(TupleTransformerOutput):
    loss: torch.Tensor | None = None
    losses: dict[str, torch.Tensor] | None = None


class TupleTransformerLMWrapper(ModelWrapper):
    def __init__(
            self,
            model: TupleTransformer,
            token_keys: list[str] | None = None,
            ignore_index: int = -100,
            distance_loss: bool = False,
            task_loss: bool = False,
            emd_loss: bool = False
    ):
        super().__init__(model=model)

        self.max_seq_len = self.model.max_seq_len
        self.ignore_index = ignore_index

        self.token_types = list(self.model.num_tokens.keys())
        self.token_keys = token_keys or self.token_types
        self.token_indices = None
        if self.token_keys:
            self.token_indices = [idx for idx, key in enumerate(self.token_types) if key in token_keys]

        self.distance_loss = distance_loss
        self.task_loss = task_loss
        self.emd_loss = emd_loss

    def forward(
            self,
            tokens: torch.Tensor,
            values: torch.Tensor | None = None,
            mask: torch.Tensor | None = None,
            labels: torch.Tensor | None = None,
            targets: torch.Tensor | None = None,
            task_ids: torch.Tensor | None = None,
            type_ids: torch.Tensor | None = None,
            **kwargs
    ) -> TupleTransformerLMOutput:
        out = self.model(
            tokens=tokens, values=values, mask=mask,
            task_ids=task_ids, type_ids=type_ids, **kwargs
        )

        if labels is not None and self.token_indices is not None:
            labels = labels[..., self.token_indices]

        loss, losses = None, {}
        if out.logits is not None and labels is not None:
            label_mask = labels != self.ignore_index

            losses = {
                key: masked_batch_mean(
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

            loss = sum(losses.values()) / len(losses)

            num_first_discrete = self.model.token_emb.num_first_discrete

            if self.distance_loss:
                _label_mask = label_mask & labels >= num_first_discrete
                dist_losses = {
                    f"{key}/distance": weighted_distance(
                        probs=out.logits[key].softmax(dim=-1),
                        targets=targets[..., i],
                        token_values=self.model.token_emb.embs[key].token_values,
                        mask=_label_mask[..., i],
                        reduction=Reduction.BATCH_MEAN
                    )
                    for i, (key, logits) in enumerate(out.logits.items())
                    if self.model.token_emb.embs[key].token_values is not None and torch.any(_label_mask[..., i])
                }
                loss += sum(dist_losses.values()) / len(dist_losses)
                losses.update(**dist_losses)

            if self.emd_loss:
                num_first_discrete = self.model.token_emb.num_first_discrete
                emd_losses = {
                    f"{key}/emd": emd_cdf_loss(
                        probs=out.logits[key].softmax(dim=-1)[..., num_first_discrete:],
                        labels=labels[..., i] - num_first_discrete,
                        mask=label_mask[..., i],
                        reduction=Reduction.BATCH_MEAN
                    )
                    for i, (key, logits) in enumerate(out.logits.items())
                    if torch.any(label_mask[..., i])
                }
                loss += sum(emd_losses.values()) / len(emd_losses)
                losses.update(**emd_losses)

            loss = loss + 0. * sum(p.sum() for p in self.model.lm_head.parameters())

        if out.values is not None and targets is not None:
            target_mask = targets > self.ignore_index  # no loss on special tokens

            reg_losses = {}
            for i, key in enumerate(self.model.token_emb.embs.keys()):
                if key not in out.values:
                    continue

                if not torch.any(target_mask[..., i]):
                    continue

                reg_losses[f"{key}/l1"] = masked_batch_mean(
                    F.l1_loss(
                        out.values[key],
                        targets[..., i],
                        reduction="none"
                    ),
                    mask=target_mask[..., i]
                )

            reg_loss = sum(reg_losses.values()) / len(reg_losses)
            loss = reg_loss if loss is None else loss + reg_loss
            losses.update(**reg_losses)

        if out.task_logits is not None and task_ids is not None:
            task_loss = masked_batch_mean(
                F.cross_entropy(
                    out.task_logits.transpose(1, 2),
                    task_ids[:, None].expand(-1, out.task_logits.shape[1]),
                    reduction="none"
                ),
                mask=mask
            )
            loss = loss + task_loss
            losses["task"] = task_loss

        if out.type_logits is not None and type_ids is not None:
            type_loss = masked_batch_mean(
                F.cross_entropy(out.type_logits.transpose(1, 2), type_ids, reduction="none"),
                mask=mask
            )
            loss = loss + type_loss
            losses["type"] = type_loss

        return TupleTransformerLMOutput(
            loss=loss,
            losses=losses,
            **out.__dict__
        )


class TupleTransformerMLMWrapper(TupleTransformerLMWrapper):
    def __init__(
            self,
            model: TupleTransformer,
            token_keys: list[str] | None = None,
            mask_token_id: int = 1,
            ignore_index: int = -100,
            distance_loss: bool = False,
            emd_loss: bool = False
    ):
        super().__init__(
            model=model,
            token_keys=token_keys,
            ignore_index=ignore_index,
            distance_loss=distance_loss,
            emd_loss=emd_loss
        )
        self.mask_token_id = mask_token_id
        self.num_special_tokens = self.model.token_emb.num_first_discrete

    @torch.inference_mode()
    def unmask(
            self,
            tokens: torch.Tensor,
            values: torch.Tensor | None = None,
            single_run: bool = True,
            temperature: float = 1.,
            top_k: float | int = -1,
            top_p: float = 0.8,
            filter_key_ids: dict[str, list] | None = None,
            ignore_non_special: list[str] | None = None,
            tokenizer: OctupleM = None,
            disable_tqdm: bool = False,
            **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert values is None or tokenizer is not None

        was_training = self.model.training
        if was_training:
            self.model.eval()

        num_dims = len(tokens.shape)
        if num_dims == 2:
            tokens = tokens[None]
            values = values[None] if values is not None else None

        out_tokens = tokens.clone().detach()
        out_values = values.clone().detach() if values is not None else None
        mask = kwargs.pop("mask", None)

        if mask is None:
            mask = torch.full_like(out_tokens[..., 0], True, dtype=torch.bool, device=out_tokens.device)

        filter_key_ids = filter_key_ids or dict()
        mask_value = -float("Inf")

        def _unmask_step():
            outputs = self(
                tokens=out_tokens,
                values=out_values if out_values is not None else None,
                mask=mask, return_embeddings=True, **kwargs
            )

            if self.model.lm_head is not None:
                logits_i, pred_tokens_i, pred_values_i = self.model.lm_head.infer(
                    outputs.hidden_state[:, idx], keys=self.token_keys,
                    temperature=temperature, top_k=top_k, top_p=top_p,
                    num_first_mask=self.num_special_tokens,
                    filter_key_ids=filter_key_ids, ignore_non_special=ignore_non_special,
                    tokenizer=tokenizer
                )

                return pred_tokens_i, pred_values_i

            elif self.model.value_head is not None:
                token_types, pred_tokens_i, pred_values_i = self.model.value_head.infer(
                    outputs.hidden_state[:, idx], keys=self.token_keys
                )
                return pred_tokens_i, pred_values_i

            return None, None

        unmask = out_tokens == self.mask_token_id
        if self.token_indices is not None:
            unmask = unmask[..., self.token_indices]

        if single_run:
            outputs = self.model(out_tokens, values=out_values, mask=mask, **kwargs)
            if outputs.logits is not None:
                for key_logits in outputs.logits.values():
                    key_logits[..., :self.num_special_tokens] = mask_value
                pred_tokens = torch.cat(
                    list(map(
                        lambda l_i: filter_and_sample(l_i, temperature=temperature, top_k=top_k, top_p=top_p),
                        outputs.logits.values()
                    )),
                    dim=-1
                )
                out_tokens = fill_by_mask_and_indices(
                    out_tokens, pred_tokens, mask=unmask, dims=self.token_indices
                )

                if out_values is not None:
                    shape = pred_tokens.shape[:2]
                    pred_values = tokenizer.decode_values(
                        pred_tokens.view(-1, pred_tokens.shape[-1]),
                        token_type=list(outputs.logits.keys()), normalize=True
                    ).view(*shape, pred_tokens.shape[-1]).float()

                    out_values = fill_by_mask_and_indices(
                        out_values, pred_values, mask=unmask, dims=self.token_indices
                    )

            if outputs.values is not None:
                pred_values = torch.cat(list(outputs.values.values()), dim=-1)
                if out_values is not None:
                    out_values = fill_by_mask_and_indices(
                        out_values, pred_values, mask=unmask, dims=self.token_indices
                    )

                pred_tokens = tokenizer.encode_tokens(
                    pred_values, token_type=list(outputs.values.keys()), denormalize=True
                )
                out_tokens = fill_by_mask_and_indices(
                    out_tokens, pred_tokens, mask=unmask, dims=self.token_indices
                )
        else:
            unmask_ids = torch.where(torch.any(unmask, dim=2))[1]

            pbar = unmask_ids if disable_tqdm else tqdm(unmask_ids, leave=False)
            for idx in pbar:
                # get samples
                type_mask = unmask[:, idx][0]

                pred_tokens, pred_values = _unmask_step()
                # fill input/output sequence
                if self.token_indices is not None:
                    out_token = out_tokens[:, idx, self.token_indices]
                    out_token[:, type_mask] = pred_tokens
                    out_tokens[:, idx, self.token_indices] = out_token
                    if out_values is not None:
                        out_value = out_values[:, idx, self.token_indices]
                        out_value[:, type_mask] = pred_values
                        out_values[:, idx, self.token_indices] = out_value
                else:
                    out_tokens[:, idx, type_mask] = pred_tokens
                    if out_values is not None:
                        out_values[:, idx, type_mask] = pred_values

        if num_dims == 2:
            out_tokens = out_tokens.squeeze(0)
            out_values = out_values.squeeze(0) if out_values is not None else None

        if was_training:
            self.model.train(was_training)

        return out_tokens, out_values


class TupleTransformerARWrapper(TupleTransformerLMWrapper):
    def __init__(
            self,
            model: TupleTransformer,
            token_keys: list[str] | None = None,
            pad_token_id: int = 0,
            mask_token_id: int = 1,
            eos_token_id: int = 3,
            ignore_index: int = -100,
            distance_loss: bool = False,
            emd_loss: bool = False
    ):
        super().__init__(
            model=model,
            token_keys=token_keys,
            ignore_index=ignore_index,
            distance_loss=distance_loss,
            emd_loss=emd_loss
        )
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.eos_token_id = eos_token_id

    @torch.inference_mode()
    def generate(
            self,
            tokens: torch.Tensor,
            values: torch.Tensor | None = None,
            known_tokens: torch.Tensor | None = None,
            known_values: torch.Tensor | None = None,
            context: torch.Tensor | None = None,
            context_mask: torch.Tensor | None = None,
            context_tokens: torch.Tensor | None = None,
            context_values: torch.Tensor | None = None,
            score_tokens: torch.Tensor | None = None,
            score_values: torch.Tensor | None = None,
            seq_len: int = 256,
            max_bar: int | None = None,
            temperature: float = 1.,
            top_k: float | int = -1,
            top_p: float = 0.8,
            cache: TupleTransformerCache | None = None,
            return_cache: bool = False,
            filter_key_ids: dict[str, list] | None = None,
            ignore_non_special: list[str] | None = None,
            tokenizer: OctupleM = None,
            fix_errors: bool = False,
            disable_tqdm: bool = False,
            record_attention: bool = False,
            **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, TupleTransformerCache]:
        was_training = self.model.training
        if was_training:
            self.model.eval()

        num_dims = len(tokens.shape)
        if num_dims == 2:
            tokens = tokens[None]
            values = values[None] if values is not None else None

        out_tokens = tokens.clone().detach()
        out_values = values.clone().detach() if values is not None else None
        mask = kwargs.pop("mask", None)

        if mask is None:
            mask = torch.full_like(out_tokens[..., 0], True, dtype=torch.bool, device=out_tokens.device)

        b, t = tokens.shape[:2]
        mask_value = -float("Inf")
        filter_key_ids = filter_key_ids or dict()
        ignore_non_special = ignore_non_special or []

        known_mask = known_tokens != self.mask_token_id if known_tokens is not None else None
        total_cache = None

        eod_token_id = None
        if tokenizer is not None and EOD_TOKEN in tokenizer.special_tokens:
            eod_token_id = tokenizer[0, EOD_TOKEN]

        bar_line_id = tokenizer.bar_line_id
        pedal_on_id, pedal_off_id = tokenizer.pedal_ids

        pbar = range(t, seq_len + 1)
        pbar = pbar if disable_tqdm else tqdm(pbar, leave=False)
        for idx in pbar:
            outputs = self(
                # these are cut inside forward
                tokens=out_tokens[:, -self.max_seq_len:],
                values=out_values[:, -self.max_seq_len:] if out_values is not None else None,
                mask=mask[:, -self.max_seq_len:], cache=cache, cut_last=False,
                context=context,
                context_tokens=context_tokens[:, :idx + 1] if context_tokens is not None else None,
                context_values=context_values[:, :idx + 1] if context_values is not None else None,
                score_tokens=score_tokens[:, :idx + 1] if score_tokens is not None else None,
                score_values=score_values[:, :idx + 1] if score_values is not None else None,
                return_embeddings=True, return_cache=return_cache, **kwargs
            )
            cache = outputs.cache
            if record_attention:
                total_cache = cache if total_cache is None else cache.add_past_cache(total_cache)

            if self.model.lm_head is not None:
                logits_i, pred_tokens_i, pred_values_i = self.model.lm_head.infer(
                    outputs.hidden_state[:, -1],
                    temperature=temperature, top_k=top_k, top_p=top_p,
                    num_first_mask=2, filter_key_ids=filter_key_ids, ignore_non_special=ignore_non_special,
                    tokenizer=tokenizer
                )

                if fix_errors and tokenizer is not None:
                    token_types = list(logits_i.keys())
                    if "Pitch" in logits_i:
                        pitch = pred_tokens_i[..., token_types.index("Pitch")]
                    elif "PitchClass" in logits_i:
                        pitch = pred_tokens_i[..., token_types.index("PitchClass")]
                    else:
                        raise ValueError("Pitch tokens should be predicted by the model")

                    resample_keys = []
                    if pitch == pedal_on_id or pitch == pedal_off_id:
                        resample_keys = ["TimeShift"]
                    elif pitch == bar_line_id:
                        resample_keys = ["Bar", "Position", "PositionShift"]
                    elif pitch >= tokenizer.zero_token:
                        resample_keys = list(logits_i.keys())

                    fill_token_id = pitch.item() if pitch <= eod_token_id else tokenizer.ignore_token

                    for i, (key, key_logits) in enumerate(logits_i.items()):
                        if key.startswith("Pitch"):
                            continue
                        elif key in resample_keys and key not in ignore_non_special:
                            key_logits[:, :tokenizer.zero_token] = mask_value
                            pred_tokens_i[..., i] = filter_and_sample(
                                key_logits, temperature=temperature, top_k=top_k, top_p=top_p
                            )
                        else:
                            pred_tokens_i[..., i] = torch.full_like(pred_tokens_i[..., i], fill_value=fill_token_id)

                if (fix_errors and tokenizer is not None) or known_tokens is not None:
                    token_types = list(logits_i.keys())
                    pred_values_i = tokenizer.decode_values(pred_tokens_i, token_type=token_types, normalize=True).float()

            else:
                assert self.model.value_head is not None

                token_types, pred_tokens_i, pred_values_i = self.model.value_head.infer(outputs.hidden_state[:, -1])

            if known_tokens is not None:
                pred_tokens_i_full = known_tokens[:, idx].clone()
                if self.token_indices is not None:
                    pred_tokens_i_full[..., self.token_indices] = pred_tokens_i
                else:
                    pred_tokens_i_full[..., ~known_mask[0, idx]] = pred_tokens_i[..., ~known_mask[0, idx]]
                pred_tokens_i = pred_tokens_i_full[:, None]

            if known_values is not None:
                pred_values_i_full = known_values[:, idx].clone()
                if self.token_indices is not None:
                    pred_values_i_full[..., self.token_indices] = pred_values_i
                else:
                    pred_values_i_full[..., ~known_mask[0, idx]] = pred_values_i[..., ~known_mask[0, idx]]
                pred_values_i = pred_values_i_full[:, None]

            out_tokens = torch.cat((out_tokens, pred_tokens_i), dim=1)
            out_values = torch.cat((out_values, pred_values_i), dim=1) if out_values is not None else None
            mask = F.pad(mask, (0, 1), value=True)

            eos_token_id = eod_token_id or self.eos_token_id
            if eos_token_id is not None:  # eos only in `Bar`
                is_eos_tokens = (out_tokens[..., -1, 0] == eos_token_id) | (out_tokens[..., -1, 0] == self.eos_token_id)
                if is_eos_tokens.any(dim=-1):
                    out_tokens[:, -1, 1:] = out_tokens[:, -1, 0]
                    pbar.close()
                    break
            elif max_bar is not None:
                is_max_bar_tokens = (out_tokens[..., -1, 0] > max_bar)
                if is_max_bar_tokens.any(dim=-1):
                    out_tokens = out_tokens[:, :-1, :]
                    pbar.close()
                    break

        if num_dims == 2:
            out_tokens = out_tokens.squeeze(0)
            out_values = out_values.squeeze(0) if out_values is not None else None

        if was_training:
            self.model.train(was_training)

        if return_cache:
            return out_tokens, out_values, cache
        return out_tokens, out_values

    def forward(
            self,
            tokens: torch.Tensor,
            values: torch.Tensor | None = None,
            labels: torch.Tensor | None = None,
            targets: torch.Tensor | None = None,
            cut_last: bool = True,
            **kwargs
    ) -> TupleTransformerLMOutput:
        tokens = tokens[:, :-1] if cut_last else tokens
        labels = labels[:, 1:] if labels is not None else None

        values = values[:, :-1] if cut_last and values is not None else values
        targets = targets[:, 1:] if targets is not None else None

        full_labels = kwargs.get("full_labels", None)
        if full_labels is not None and full_labels.shape[1] == labels.shape[1] + 1:
            kwargs["full_labels"] = full_labels[:, 1:]

        context = kwargs.get("context", None)
        if context is not None and self.model.context_embedding == "cat":
            kwargs["context"] = context[:, 1:]

        context_tokens, context_values = kwargs.get("context_tokens", None), kwargs.get("context_values", None)
        if context_tokens is not None and self.model.context_embedding == "cat":
            kwargs["context_tokens"] = context_tokens[:, 1:]
            kwargs["context_values"] = context_values[:, 1:] if context_values is not None else None

        score_tokens, score_values = kwargs.get("score_tokens", None), kwargs.get("score_values", None)
        if score_tokens is not None and self.model.context_embedding == "cat":
            kwargs["score_tokens"] = score_tokens[:, 1:]
            kwargs["score_values"] = score_values[:, 1:] if score_values is not None else None

        style_embeddings = kwargs.get("style_embeddings", None)
        if style_embeddings is not None:
            kwargs["style_embeddings"] = style_embeddings[:, 1:]

        mask = kwargs.get("mask", None)
        if cut_last and mask is not None and mask.shape[1] == tokens.shape[1] + 1:
            kwargs["mask"] = mask[:, :-1]

        type_ids = kwargs.get("type_ids", None)
        if cut_last and type_ids is not None and type_ids.shape[1] == tokens.shape[1] + 1:
            kwargs["type_ids"] = type_ids[:, :-1]

        return super().forward(tokens, values=values, labels=labels, targets=targets, **kwargs)


class TupleTransformerMixedLMWrapper(TupleTransformerLMWrapper):
    def __init__(
            self,
            model: TupleTransformer,
            pad_token_id: int = 0,
            mask_token_id: int = 1,
            ignore_index: int = -100,
            distance_loss: bool = False
    ):
        super().__init__(model=model, ignore_index=ignore_index, distance_loss=distance_loss)
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id

    @torch.inference_mode()
    def unmask(
            self,
            tokens: torch.Tensor,
            masked_tokens: torch.Tensor | None = None,
            values: torch.Tensor | None = None,
            masked_values: torch.Tensor | None = None,
            temperature: float = 1.,
            top_k: float | int = -1,
            top_p: float = 0.8,
            filter_key_ids: dict[str, list] | None = None,
            ignore_non_special: list[str] | None = None,
            tokenizer: OctupleM = None,
            cache: TupleTransformerCache | None = None,
            return_cache: bool = False,
            disable_tqdm: bool = False,
            **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, TupleTransformerCache]:
        assert values is None or tokenizer is not None

        was_training = self.model.training
        if was_training:
            self.model.eval()

        num_dims = len(tokens.shape)
        if num_dims == 2:
            tokens = tokens[None, :]
            masked_tokens = masked_tokens[None, :] if masked_tokens is not None else None
            values = values[None] if values is not None else None
            masked_values = masked_values[None] if masked_values is not None else None

        out_tokens = tokens.clone().detach()
        out_values = values.clone().detach() if values is not None else None

        mask = kwargs.pop("mask", None)
        if mask is None:
            mask = torch.full_like(out_tokens[..., 0], True, dtype=torch.bool, device=out_tokens.device)

        filter_key_ids = filter_key_ids or dict()

        def _unmask_step():
            # shifting is handled in forward
            token_inputs = out_tokens[:, :idx + 1]
            masked_token_inputs = masked_tokens[:, :idx + 1] if masked_tokens is not None else None

            value_inputs = out_values[:, :idx + 1] if out_values is not None else None
            masked_value_inputs = masked_values[:, :idx + 1] if masked_values is not None else None

            outputs = self(
                tokens=token_inputs, values=value_inputs,
                masked_tokens=masked_token_inputs, masked_values=masked_value_inputs,
                mask=mask[:, :idx + 1], return_embeddings=True, return_cache=True, **kwargs
            )

            if self.model.lm_head is not None:
                logits_i, pred_tokens_i, pred_values_i = self.model.lm_head.infer(
                    outputs.hidden_state[:, idx - 1], keys=output_keys,
                    temperature=temperature, top_k=top_k, top_p=top_p,
                    num_first_mask=self.num_special_tokens,
                    filter_key_ids=filter_key_ids, ignore_non_special=ignore_non_special,
                    tokenizer=tokenizer
                )
            else:
                token_types, pred_tokens_i, pred_values_i = self.model.value_head.infer(
                    outputs.hidden_state[:, idx], keys=output_keys
                )

            return pred_tokens_i, pred_values_i, outputs.cache

        unmask = out_tokens == self.mask_token_id
        unmask_ids = torch.where(torch.any(unmask, dim=2))[1]

        pbar = unmask_ids if disable_tqdm else tqdm(unmask_ids, leave=False)
        for idx in pbar:
            # get samples
            type_mask = unmask[:, idx][0]
            output_keys = torch.where(type_mask)[0].tolist()

            pred_tokens, pred_values, cache = _unmask_step()

            # fill input/output sequence
            out_tokens[:, idx, type_mask] = pred_tokens
            if out_values is not None:
                out_values[:, idx, type_mask] = pred_values

        if num_dims == 2:
            out_tokens = out_tokens.squeeze(0)
            out_values = out_values.squeeze(0) if out_values is not None else None

        if was_training:
            self.model.train(was_training)

        if return_cache:
            return out_tokens, out_values, cache
        return out_tokens, out_values

    def forward(
            self,
            tokens: torch.Tensor,
            values: torch.Tensor | None = None,
            labels: torch.Tensor | None = None,
            targets: torch.Tensor | None = None,
            **kwargs
    ) -> TupleTransformerLMOutput:
        seq_len = tokens.shape[1]
        tokens = tokens[:, :-1]
        labels = labels[:, 1:] if labels is not None else None

        values = values[:, :-1] if values is not None else None
        targets = targets[:, 1:] if targets is not None else None

        if self.model.is_multiseq:
            masked_tokens = kwargs.pop("masked_tokens", None)
            if masked_tokens is not None:
                masked_tokens = masked_tokens[:, 1:]
                tokens = [tokens, masked_tokens]

            masked_values = kwargs.pop("masked_values", None)
            if masked_values is not None:
                masked_values = masked_values[:, 1:]
                values = [values, masked_values]

        context = kwargs.get("context", None)
        if context is not None and self.model.context_embedding == "cat":
            kwargs["context"] = context[:, 1:]

        style_embeddings = kwargs.get("style_embeddings", None)
        if style_embeddings is not None:
            kwargs["style_embeddings"] = style_embeddings[:, 1:]

        mask = kwargs.get("mask", None)
        if mask is not None and mask.shape[1] == seq_len:
            kwargs["mask"] = mask[:, :-1]

        return super().forward(tokens, values=values, labels=labels, targets=targets, **kwargs)
