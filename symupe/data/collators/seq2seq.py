""" Sequence-to-Sequence data collators. """
from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from omegaconf import DictConfig

from symupe.utils import asdict
from .base import SeqInputs, SeqSegments
from .sequence import (
    LMSequenceCollator,
    MaskLevel,
    SEGMENT_MASKS, LAST_NOTES_MASKS, REGION_MASKS,
    sample_mask, mask_with_token_dims, mask_with_tokens,
)
from ..datasets import SequenceSample
from ..tokenizers.constants import SPECIAL_TOKENS_VALUE


@dataclass
class Seq2SeqInputs:
    input_sequences: SeqInputs
    output_sequences: SeqInputs
    segments: SeqSegments | None = None
    type_ids: torch.Tensor | None = None
    encoding_ids: torch.Tensor | None = None
    task_ids: torch.Tensor | None = None
    task_sequences: SeqInputs | None = None
    score_sequences: SeqInputs | None = None
    context_sequences: SeqInputs | None = None
    emotion_labels: torch.Tensor | None = None
    emotion_embeddings: torch.Tensor | None = None
    random_sequences: SeqInputs | None = None


class Seq2SeqCollator(LMSequenceCollator):
    def __init__(
            self,
            pad_token_id: int = 0,
            pad_to_maximum: bool = False,
            pad_is_input: bool = False,
            pad_to_multiple_of: int = 1
    ):
        super().__init__(pad_token_id, pad_to_multiple_of)
        self.pad_to_maximum = pad_to_maximum
        self.pad_is_input = pad_is_input

    def get_max_lengths(self, batch: Sequence[SequenceSample], inference: bool = False):
        max_lens = super().get_max_lengths(batch, inference)
        lens_target = np.array(list(map(lambda sample: len(sample.target_seq), batch))).T
        max_lens["output_sequence"] = np.max(lens_target) if inference else self.pad_len(np.max(lens_target))
        if self.pad_to_maximum:
            max_lens["sequence"] = max_lens["output_sequence"] = max(max_lens["sequence"], max_lens["output_sequence"])
        return max_lens

    def _init_seq_data(self, batch_size: int, max_len: int, compound_factor: int = 0):
        seq_data = super()._init_seq_data(batch_size=batch_size, max_len=max_len, compound_factor=compound_factor)
        if self.pad_is_input:
            seq_data.mask = ~seq_data.mask
        return seq_data

    def init_data(self, batch: Sequence[SequenceSample], inference: bool = False):
        data = super().init_data(batch, inference)

        max_lens = self.get_max_lengths(batch, inference=inference)

        sample, b = batch[0], len(batch)
        return Seq2SeqInputs(
            input_sequences=data.sequences,
            output_sequences=self._init_seq_data(
                b, max_lens["output_sequence"],
                compound_factor=sample.target_seq.ids.shape[-1]
            ),
            segments=data.segments,
            type_ids=data.type_ids,
            encoding_ids=torch.zeros(b, 2, dtype=torch.long),
            task_ids=data.task_ids,
            task_sequences=data.task_sequences,
            score_sequences=data.score_sequences,
            context_sequences=data.context_sequences,
            emotion_labels=data.emotion_labels,
            emotion_embeddings=data.emotion_embeddings,
            random_sequences=data.random_sequences
        )

    def process_sample(self, i: int, sample: SequenceSample, data: Seq2SeqInputs, inference: bool = False):
        # process source sequence
        self._process_sequence(i, seq=sample.seq, seq_data=data.input_sequences)

        # process target sequence
        self._process_sequence(i, seq=sample.target_seq, seq_data=data.output_sequences)

        # process note segments if present
        self._process_segments(i, sample=sample, seg_data=data.segments, seq_len=len(sample.seq))

        # process auxiliary
        self._process_auxiliary(i, sample=sample, data=data)

    def __call__(self, batch: Sequence[SequenceSample], inference: bool = False, return_dict: bool = True):
        data = self.init_data(batch, inference=inference)
        for i, sample in enumerate(batch):
            self.process_sample(i, sample, data)

        return asdict(data) if return_dict else data


# FOR LANGUAGE MODELING
@dataclass
class LMSeq2SeqInputs(Seq2SeqInputs):
    labels: SeqInputs | None = None
    full_labels: SeqInputs | None = None


class LMSeq2SeqCollator(Seq2SeqCollator, LMSequenceCollator):
    def __init__(
            self,
            pad_token_id: int = 0,
            pad_to_multiple_of: int = 1,

            mlm: bool = True,
            morph: bool = False,
            mask_level: str | MaskLevel | dict[str | MaskLevel, float] = MaskLevel.NOTE,
            mask_level_exceptions: dict[str, list[str]] | None = None,
            mask_compound: bool = True,
            mask_prob: float | tuple[float, float] = 0.15,
            replace_prob: float = 0.9,
            random_token_prob: float = 0.,
            copy_sequence_prob: float = 0.,
            output_mask_level: str | MaskLevel | dict[str | MaskLevel, float] | None = None,
            output_mask_prob: float | tuple[float, float] | None = None,
            output_unmask_inputs: bool = False,
            mask_token_id: int = 1,
            ignore_token_id: int = 4,
            mask_ignore_token_ids: list[int] | None = None,
            mask_token_dims: dict[str, list[int]] | list[int] | None = None,
            output_mask_ignore_token_ids: list[int] | None = None,
            output_replace_token_ids: list[tuple[int, int]] | None = None,
            output_predict_token_ids: list[int] | None = None,
            output_mask_token_dims: dict[str, list[int]] | list[int] | None = None,
            label_pad_token_id: int = -100
    ):
        Seq2SeqCollator.__init__(
            self,
            pad_token_id=pad_token_id,
            pad_to_maximum=morph,
            pad_is_input=morph,
            pad_to_multiple_of=pad_to_multiple_of,
        )

        LMSequenceCollator.__init__(
            self,
            pad_token_id=pad_token_id,
            pad_to_multiple_of=pad_to_multiple_of,
            mlm=mlm,
            mask_level=mask_level,
            mask_level_exceptions=mask_level_exceptions,
            mask_compound=mask_compound,
            mask_prob=mask_prob,
            replace_prob=replace_prob,
            random_token_prob=random_token_prob,
            copy_sequence_prob=copy_sequence_prob,
            mask_token_id=mask_token_id,
            ignore_token_id=ignore_token_id,
            mask_ignore_token_ids=mask_ignore_token_ids,
            mask_token_dims=mask_token_dims,
            label_pad_token_id=label_pad_token_id
        )

        self.morph = morph

        if output_mask_level is not None:
            mask_levels = list(output_mask_level) if isinstance(output_mask_level, (dict, DictConfig)) else [output_mask_level]
            for mask_level in mask_levels:
                assert MaskLevel.has_value(mask_level), \
                    f"`{mask_level}` is not a valid `mask_level`, available modes: {MaskLevel.list()}"
        self.output_mask_level = output_mask_level

        self.output_unmask_inputs = output_unmask_inputs

        output_mask_prob = output_mask_prob or mask_prob
        self.output_mask_prob = tuple(output_mask_prob) if isinstance(output_mask_prob, Sequence) else output_mask_prob

        output_mask_ignore_token_ids = output_mask_ignore_token_ids or mask_ignore_token_ids
        self.output_mask_ignore_token_ids = list(set(
            (output_mask_ignore_token_ids or []) + ([] if self.morph else [pad_token_id])
        ))
        self.output_mask_token_dims = output_mask_token_dims

        self.output_replace_token_ids = list(output_replace_token_ids or [])
        self.output_predict_token_ids = list(output_predict_token_ids or [])

    def _mask_output_sequence(
            self,
            mask_level: str | MaskLevel,
            seq: torch.Tensor,
            values: torch.Tensor | None = None,
            task_ids: torch.Tensor | None = None,
            segments: SeqSegments | None = None,
            sequence_task_types: dict[str, int] | None = None
    ):
        b, t = seq.shape[:2]

        masked_seq = seq.clone().detach()
        masked_values = values.clone().detach() if values is not None else None

        for source_id, target_id in self.output_replace_token_ids:
            switch_mask = masked_seq == source_id
            masked_seq[switch_mask] = target_id
            if values is not None:
                masked_values[switch_mask] = SPECIAL_TOKENS_VALUE - target_id

        # per-dimension mask for possibly compound tokens (dimensions to mask)
        if (self.output_mask_token_dims is not None and len(self.output_mask_token_dims) > 0
                and isinstance(self.output_mask_token_dims, (dict, DictConfig))
                and task_ids is not None):
            assert sequence_task_types is not None
            dim_masks = torch.concatenate([
                mask_with_token_dims(seq, self.output_mask_token_dims.get(task_type, []))
                for task_type in sequence_task_types.keys()
            ], dim=0)
            dim_mask = dim_masks[task_ids]
        else:
            dim_mask = mask_with_token_dims(seq, self.output_mask_token_dims)

        # do not mask [pad] tokens, or any other tokens in the tokens designated to be excluded ([sos], [eos])
        # also do not include these special tokens in the tokens chosen at random
        no_mask = mask_with_tokens(seq, self.output_mask_ignore_token_ids)
        no_mask_note = mask_with_tokens(seq, self.output_mask_ignore_token_ids, squeeze=False)

        force_mask_note = mask_with_tokens(seq, self.output_predict_token_ids, squeeze=False)

        if mask_level in (MaskLevel.NONE, MaskLevel.ALL, MaskLevel.ALL_IGNORE):  # predict everything
            label_mask = (~no_mask_note).clone().detach() * dim_mask

            if mask_level in (MaskLevel.ALL, MaskLevel.ALL_IGNORE):  # also mask everything
                mask_token_id = self.mask_token_id if mask_level == MaskLevel.ALL else self.ignore_token_id
                masked_seq.masked_fill_(label_mask, mask_token_id)
                if masked_values is not None:
                    masked_values.masked_fill_(label_mask, self.label_pad_token_id - mask_token_id)

            label_mask = label_mask | force_mask_note
            labels = seq.clone().detach().masked_fill(~label_mask, self.label_pad_token_id)

            targets = None
            if values is not None:
                targets = values.clone().detach().masked_fill(~label_mask, self.label_pad_token_id)

            return (masked_seq, labels), (masked_values, targets), label_mask

        # process segment masks
        segment_ids = torch.arange(t)[None].expand(b, t)
        if mask_level in SEGMENT_MASKS:
            assert segments is not None, "`segments` must be provided to compute segment-level mask"

            if mask_level in (MaskLevel.ONSET, MaskLevel.LAST_ONSET):
                segments = segments.onset
            elif mask_level in (MaskLevel.BEAT, MaskLevel.LAST_BEAT):
                segments = segments.beat
            else:
                segments = segments.bar

            max_values = segments.max(dim=1).values
            segment_ids = torch.arange(max_values.max() + 1)[None].expand(b, -1)
            no_mask = mask_with_tokens(segment_ids, self.output_mask_ignore_token_ids)
            no_mask = no_mask | (segment_ids > max_values[:, None])
        elif mask_level == MaskLevel.ELEMENT:
            no_mask = no_mask_note.clone() * (~dim_mask)

        def _maybe_distribute_segment_mask(segment_mask):
            if mask_level in SEGMENT_MASKS:
                segment_mask = segment_mask[(torch.arange(b).repeat_interleave(t), segments.view(-1))].view(b, t, -1)
            else:
                segment_mask = segment_mask.view(b, t, -1)
            return segment_mask if seq.ndim == 3 else segment_mask.squeeze(-1)

        mask_prob = self.output_mask_prob
        if isinstance(mask_prob, tuple):
            mask_prob = torch.zeros((b,)).uniform_(*mask_prob)

        if mask_level in LAST_NOTES_MASKS:
            last_segment_ids = torch.max(segment_ids * (~no_mask), dim=1).values
            mask = segment_ids == last_segment_ids[:, None]
        elif mask_level in REGION_MASKS:
            mask = sample_mask(~no_mask, mask_prob, mask_level)
        else:
            mask = sample_mask(~no_mask, mask_prob)

        # possibly expand mask
        if seq.ndim == 3 and mask.ndim == 2:
            mask = mask[..., None].expand(-1, -1, seq.size(2))

        # mask with [mask] tokens
        mask = _maybe_distribute_segment_mask(mask)
        label_mask = mask.clone().detach() * dim_mask * (~no_mask_note)

        masked_seq.masked_fill_(label_mask, self.mask_token_id)
        if masked_values is not None:
            masked_values.masked_fill_(label_mask, self.label_pad_token_id - self.mask_token_id)

        label_mask = label_mask | force_mask_note
        labels = seq.clone().detach().masked_fill(~label_mask, self.label_pad_token_id)

        targets = None
        if values is not None:
            targets = values.clone().detach().masked_fill(~label_mask, self.label_pad_token_id)

        return (masked_seq, labels), (masked_values, targets), label_mask

    def mask_output_sequence(
            self,
            seq: torch.Tensor,
            values: torch.Tensor | None = None,
            task_ids: torch.Tensor | None = None,
            segments: SeqSegments | None = None,
            sequence_task_types: dict[str, int] | None = None
    ):
        mask_levels = self.output_mask_level
        mask_levels = {mask_levels: 1.} if isinstance(mask_levels, (str, MaskLevel)) else mask_levels

        mask_levels, probs = list(mask_levels.keys()), list(mask_levels.values())
        mask_level_ids = torch.tensor(random.choices(list(range(len(probs))), weights=probs, k=len(seq)))

        masked_seq = seq.clone().detach()
        masked_values = values.clone().detach() if values is not None else None
        labels = seq.clone().detach()
        targets = values.clone().detach() if values is not None else None
        label_mask = torch.zeros_like(labels).bool()

        for i, mask_level in enumerate(mask_levels):
            (masked_seq_i, labels_i), (masked_values_i, targets_i), label_mask_i = self._mask_output_sequence(
                mask_level,
                seq,
                values=values,
                task_ids=task_ids,
                segments=segments,
                sequence_task_types=sequence_task_types
            )
            level_mask = mask_level_ids == i

            masked_seq[level_mask] = masked_seq_i[level_mask]
            labels[level_mask] = labels_i[level_mask]
            if values is not None:
                masked_values[level_mask] = masked_values_i[level_mask]
                targets[level_mask] = targets_i[level_mask]
            label_mask[level_mask] = label_mask_i[level_mask]

        return (masked_seq, labels), (masked_values, targets), label_mask

    def mask_and_compute_labels(
            self,
            sequences: SeqInputs,
            output_sequences: SeqInputs | None = None,
            random_sequences: SeqInputs | None = None,
            segments: SeqSegments | None = None,
            task_ids: torch.Tensor | None = None,
            score_sequences: SeqInputs | None = None,
            context_sequences: SeqInputs | None = None,
            emotion_embeddings: torch.Tensor | None = None,
            num_tokens: int | dict[str, int] | None = None,
            sequence_task_types: dict[str, int] | None = None,
            encoding_ids: torch.Tensor | None = None
    ):
        output_sequences = output_sequences or sequences
        labels, targets, label_mask = None, None, None

        if self.mlm:
            label_mask = output_sequences.mask.clone().detach()

            # mask tokens in `input_sequences`
            (masked_seq, _), (masked_values, _), input_label_mask = self.mask_sequence(
                sequences.tokens,
                values=sequences.values,
                task_ids=task_ids,
                random_seq=random_sequences.tokens,
                random_values=random_sequences.values,
                segments=segments,
                num_tokens=num_tokens,
                sequence_task_types=sequence_task_types
            )
            sequences.tokens = masked_seq
            sequences.values = masked_values

            # mask tokens in `output_sequences`
            if self.output_unmask_inputs:
                labels = output_sequences.tokens.clone().detach()
                labels[~input_label_mask] = self.label_pad_token_id

                if output_sequences.values is not None:
                    targets = output_sequences.values.clone().detach()
                    targets[~input_label_mask] = self.label_pad_token_id

            elif self.output_mask_level is not None:
                (masked_seq, labels), (masked_values, targets), label_mask = self.mask_output_sequence(
                    output_sequences.tokens,
                    values=output_sequences.values,
                    task_ids=task_ids,
                    segments=segments,
                    sequence_task_types=sequence_task_types
                )
                output_sequences.tokens = masked_seq
                output_sequences.values = masked_values
        else:
            # remove copies from input sequences
            same_encoding = encoding_ids[:, 0] == encoding_ids[:, 1]
            sequences.tokens[same_encoding] = self.pad_token_id
            sequences.values[same_encoding] = SPECIAL_TOKENS_VALUE - self.pad_token_id
            sequences.mask[same_encoding] = False

            label_mask = (
                    (output_sequences.tokens != self.pad_token_id) & (output_sequences.tokens != self.mask_token_id)
            )

        if labels is None:  # compute labels
            labels = output_sequences.tokens.clone().detach()
            labels[~label_mask] = self.label_pad_token_id

            if output_sequences.values is not None:
                targets = output_sequences.values.clone().detach()
                targets[~label_mask] = self.label_pad_token_id

        self.dropout_context(
            sequences=output_sequences,
            label_mask=label_mask,
            score_sequences=score_sequences,
            context_sequences=context_sequences,
            emotion_embeddings=emotion_embeddings
        )

        return SeqInputs(
            tokens=labels,
            values=targets,
            mask=label_mask
        )

    def __call__(self, batch: Sequence[SequenceSample], inference: bool = False, return_dict: bool = True):
        data = super().__call__(batch, inference=inference, return_dict=False)

        full_labels = SeqInputs(
            tokens=data.output_sequences.tokens,
            values=data.output_sequences.values,
            mask=data.output_sequences.mask
        )

        labels = self.mask_and_compute_labels(
            sequences=data.input_sequences,
            output_sequences=data.output_sequences,
            random_sequences=data.random_sequences,
            segments=data.segments,
            task_ids=data.task_ids,
            num_tokens=batch[0].meta.token_sizes,
            sequence_task_types=batch[0].meta.sequence_task_types,
            encoding_ids=data.encoding_ids
        )

        data = vars(data)
        data = LMSeq2SeqInputs(
            **data,
            labels=labels,
            full_labels=full_labels
        )

        return asdict(data) if return_dict else data
