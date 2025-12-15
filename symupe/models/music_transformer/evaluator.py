""" MusicTransformer models' evaluator modules for computing and logging metrics during training. """
from __future__ import annotations

import torch

from symupe.modules.metrics import Reduction, accuracy, distance, weighted_distance
from symupe.data.collators import LMSequenceInputs, LMScorePerformanceInputs
from symupe.data.tokenizers import OctupleM
from symupe.modules.classes import LanguageModelingMode
from symupe.modules.tuple_transformer import (
    TupleTransformerOutput, TupleTransformerCFMOutput, TupleTransformerFMOutput
)
from .model import MusicTransformerOutput, CFMMusicTransformerOutput, FMMusicTransformerOutput


class MusicTransformerEvaluator:
    def __init__(
            self,
            model,
            tokenizer: OctupleM,
            label_pad_token_id: int = -100,
            normalized_targets: bool = True,
            ignore_keys: list[str] | None = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id
        self.normalized_targets = normalized_targets
        self.ignore_keys = ignore_keys

        self.token_keys = self.model.token_keys

        self.token_types = list(self.model.num_tokens.keys())
        self.token_values = {
            key: torch.from_numpy(values)
            for key, values in self.tokenizer.token_values(normalize=False).items()
        }

    @torch.no_grad()
    def __call__(
            self,
            inputs: dict | LMSequenceInputs | LMScorePerformanceInputs,
            outputs: TupleTransformerOutput | MusicTransformerOutput,
            ignore_keys: list[str] | None = None
    ) -> dict[str, torch.Tensor]:
        metrics = {}
        ignore_keys = ignore_keys or self.ignore_keys

        if isinstance(inputs, (LMSequenceInputs, LMScorePerformanceInputs)):
            labels = inputs.labels.tokens.to(outputs.hidden_state.device)
            targets = inputs.labels.values.to(outputs.hidden_state.device)
        else:
            labels, targets = inputs["labels"], inputs["targets"]

        if self.model.mode in (LanguageModelingMode.CLM, LanguageModelingMode.MixedLM):
            labels, targets = labels[:, 1:], targets[:, 1:]

        label_mask = labels != self.label_pad_token_id

        token_keys = self.token_keys
        pred_tokens, pred_values = None, None
        if outputs.logits is not None:
            pred_tokens = torch.cat(list(
                map(lambda l: torch.argmax(l, dim=-1, keepdim=True), outputs.logits.values())
            ), dim=-1)

            pred_values = self.tokenizer.decode_values(pred_tokens, token_type=token_keys)

        elif outputs.values is not None:
            token_keys = None
            if isinstance(pred_values, dict):
                token_keys = list(pred_values.keys())
                pred_values = torch.cat(list(pred_values.values()), dim=-1)[None].float()

            pred_tokens = self.tokenizer.encode_tokens(pred_values, token_type=token_keys, denormalize=True)
            pred_values = self.tokenizer.denormalize_values(outputs.values, token_type=token_keys)

        used_ids = [idx for idx, key in enumerate(self.token_types) if key in token_keys]
        labels, targets = labels[..., used_ids], targets[..., used_ids]
        label_mask = label_mask[..., used_ids]

        assert pred_tokens.shape[-1] == labels.shape[-1] == pred_values.shape[-1]

        if pred_tokens is not None:
            metrics["accuracy"] = accuracy(pred_tokens, labels, mask=label_mask, reduction=Reduction.BATCH_MEAN)
            for i, key in enumerate(token_keys):
                if i == labels.shape[-1]:
                    break
                if ignore_keys and key in ignore_keys:
                    continue

                if torch.any(label_mask[..., i]):
                    metrics[f"accuracy/{key}"] = accuracy(
                        pred_tokens[..., i], labels[..., i], mask=label_mask[..., i], reduction=Reduction.BATCH_MEAN
                    )

            if self.normalized_targets:
                targets = self.tokenizer.denormalize_values(targets, token_type=token_keys)

            for i, key in enumerate(token_keys):
                if i == labels.shape[-1]:
                    break
                if ignore_keys and key in ignore_keys:
                    continue

                if torch.any(label_mask[..., i]):
                    pred_values_i, targets_i = pred_values[..., i], targets[..., i]

                    if outputs.values is not None:
                        metrics[f"distance/{key}"] = distance(
                            pred_values_i, targets_i, mask=label_mask[..., i], reduction=Reduction.BATCH_MEAN
                        )
                    else:
                        probs = outputs.logits[key].softmax(dim=-1)
                        metrics[f"distance/{key}"] = weighted_distance(
                            probs, targets_i, self.token_values[key],
                            mask=label_mask[..., i], reduction=Reduction.BATCH_MEAN
                        )

        if isinstance(inputs, LMSequenceInputs):
            task_ids = inputs.task_ids.to(outputs.hidden_state.device)
            type_ids = inputs.type_ids.to(outputs.hidden_state.device)
        else:
            task_ids = inputs.get("task_ids", None)
            type_ids = inputs.get("type_ids", None)

        if task_ids is not None and outputs.task_logits is not None:
            pred_tasks = torch.argmax(outputs.task_logits, dim=-1)
            target_tasks = task_ids[:, None].expand(-1, pred_tasks.shape[1])
            metrics["accuracy/task"] = accuracy(
                pred_tasks, target_tasks, mask=torch.any(label_mask, dim=-1), reduction=Reduction.BATCH_MEAN
            )

        if type_ids is not None and outputs.type_logits is not None:
            pred_types = torch.argmax(outputs.type_logits, dim=-1)
            target_types = type_ids
            if self.model.mode in (LanguageModelingMode.CLM, LanguageModelingMode.MixedLM):
                target_types = target_types[:, 1:]

            metrics["accuracy/type"] = accuracy(
                pred_types, target_types, mask=torch.any(label_mask, dim=-1), reduction=Reduction.BATCH_MEAN
            )

        return metrics


class CFMMusicTransformerEvaluator:
    def __init__(
            self,
            model,
            tokenizer: OctupleM,
            label_pad_token_id: int = -100,
            normalized_targets: bool = True,
            ignore_keys: list[str] | None = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id
        self.normalized_targets = normalized_targets
        self.ignore_keys = ignore_keys

        self.value_keys = self.model.value_keys

        self.token_types = list(self.model.num_tokens.keys())

    @torch.no_grad()
    def __call__(
            self,
            inputs: dict | LMScorePerformanceInputs,
            outputs: TupleTransformerCFMOutput | CFMMusicTransformerOutput,
            ignore_keys: list[str] | None = None
    ) -> dict[str, torch.Tensor]:
        metrics = {}
        ignore_keys = ignore_keys or self.ignore_keys

        pedals = None
        if isinstance(inputs, LMScorePerformanceInputs):
            labels = inputs.labels.tokens.to(outputs.hidden_state.device)
            targets = inputs.labels.values.to(outputs.hidden_state.device)
        else:
            labels, targets = inputs["labels"], inputs["targets"]
            pedals = inputs.get("pedals", None)

        label_mask = labels != self.label_pad_token_id

        value_keys = self.value_keys
        pred_tokens, pred_values = None, None
        if outputs.pred_values is not None:
            pred_values = self.tokenizer.denormalize_values(outputs.pred_values, token_type=value_keys)
            pred_tokens = self.tokenizer.encode_tokens(pred_values, token_type=value_keys)

        used_ids = [idx for idx, key in enumerate(self.token_types) if key in value_keys]
        labels, targets = labels[..., used_ids], targets[..., used_ids]
        label_mask = label_mask[..., used_ids]

        assert pred_tokens.shape[-1] == labels.shape[-1] == pred_values.shape[-1]

        if pred_tokens is not None:
            metrics["accuracy"] = accuracy(pred_tokens, labels, mask=label_mask, reduction=Reduction.BATCH_MEAN)
            for i, key in enumerate(value_keys):
                if ignore_keys and key in ignore_keys:
                    continue

                if torch.any(label_mask[..., i]):
                    metrics[f"accuracy/{key}"] = accuracy(
                        pred_tokens[..., i], labels[..., i], mask=label_mask[..., i], reduction=Reduction.BATCH_MEAN
                    )

            if self.normalized_targets:
                targets = self.tokenizer.denormalize_values(targets, token_type=value_keys)

            for i, key in enumerate(value_keys):
                if ignore_keys and key in ignore_keys:
                    continue

                if torch.any(label_mask[..., i]):
                    metrics[f"distance/{key}"] = distance(
                        pred_values[..., i], targets[..., i], mask=label_mask[..., i], reduction=Reduction.BATCH_MEAN
                    )

        if pedals is not None and outputs.pred_pedals is not None:
            mask = inputs.get("mask", torch.ones_like(pedals[:, 0], dtype=torch.bool))

            metrics["accuracy/Pedal"] = accuracy(
                outputs.pred_pedals[..., 0], pedals[..., 0], mask=mask, reduction=Reduction.BATCH_MEAN
            )

            metrics["distance/PedalShift"] = distance(
                outputs.pred_pedals[..., 1], pedals[..., 1], mask=mask, reduction=Reduction.BATCH_MEAN
            )


        return metrics


class FMMusicTransformerEvaluator:
    def __init__(
            self,
            model,
            tokenizer: OctupleM,
            label_pad_token_id: int = -100,
            normalized_targets: bool = True,
            ignore_keys: list[str] | None = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id
        self.normalized_targets = normalized_targets
        self.ignore_keys = ignore_keys

        self.token_types = list(self.model.num_tokens.keys())

    @torch.no_grad()
    def __call__(
            self,
            inputs: dict | LMScorePerformanceInputs,
            outputs: TupleTransformerFMOutput | FMMusicTransformerOutput,
            ignore_keys: list[str] | None = None
    ) -> dict[str, torch.Tensor]:
        metrics = {}
        ignore_keys = ignore_keys or self.ignore_keys

        if isinstance(inputs, LMScorePerformanceInputs):
            labels = inputs.labels.tokens.to(outputs.hidden_state.device)
            targets = inputs.labels.values.to(outputs.hidden_state.device)
        else:
            labels, targets = inputs["labels"], inputs["targets"]

        label_mask = labels != self.label_pad_token_id

        pred_tokens = outputs.pred_tokens
        pred_values = self.tokenizer.denormalize_values(outputs.pred_values, token_type=self.token_types)

        if pred_tokens is not None:
            metrics["accuracy"] = accuracy(pred_tokens, labels, mask=label_mask, reduction=Reduction.BATCH_MEAN)
            for i, key in enumerate(self.token_types):
                if ignore_keys and key in ignore_keys:
                    continue

                if torch.any(label_mask[..., i]):
                    metrics[f"accuracy/{key}"] = accuracy(
                        pred_tokens[..., i], labels[..., i], mask=label_mask[..., i], reduction=Reduction.BATCH_MEAN
                    )

            if self.normalized_targets:
                targets = self.tokenizer.denormalize_values(targets, token_type=self.token_types)

            for i, key in enumerate(self.token_types):
                if ignore_keys and key in ignore_keys:
                    continue

                if torch.any(label_mask[..., i]):
                    metrics[f"distance/{key}"] = distance(
                        pred_values[..., i], targets[..., i], mask=label_mask[..., i], reduction=Reduction.BATCH_MEAN
                    )

        return metrics
