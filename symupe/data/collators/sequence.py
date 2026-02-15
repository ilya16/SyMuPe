""" Base sequence data collators. """
from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce

import numpy as np
import torch
from omegaconf import DictConfig

from symupe.utils import ExplicitEnum, asdict
from .base import DataCollator, SeqInputs, SeqSegments
from ..datasets import SequenceSample
from ..tokenizers import TokSequence
from ..tokenizers.constants import SPECIAL_TOKENS_VALUE


@dataclass
class SequenceInputs:
    sequences: SeqInputs
    segments: SeqSegments | None = None
    pedals: torch.Tensor = None
    sequence_labels: torch.Tensor = None
    type_ids: torch.Tensor = None
    encoding_ids: torch.Tensor | None = None
    task_ids: torch.Tensor = None
    task_sequences: SeqInputs | None = None
    score_sequences: SeqInputs | None = None
    context_sequences: SeqInputs | None = None
    emotion_labels: torch.Tensor | None = None
    emotion_embeddings: torch.Tensor | None = None
    random_sequences: SeqInputs | None = None


class SequenceCollator(DataCollator):
    def __init__(
            self,
            pad_token_id: int = 0,
            pad_to_multiple_of: int = 1,
            **kwargs
    ):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def pad_len(self, length: int):
        # pad to a multiple of `pad_to_multiple`
        if self.pad_to_multiple_of > 0:
            pad_size = self.pad_to_multiple_of - length % self.pad_to_multiple_of
            length += pad_size if 0 < pad_size < self.pad_to_multiple_of else 0
        return length

    def get_max_lengths(self, batch: Sequence[SequenceSample], inference: bool = False):
        lens_seq = np.array(list(map(lambda sample: len(sample.seq), batch))).T
        max_lens = {
            "sequence": np.max(lens_seq) if inference else self.pad_len(np.max(lens_seq))
        }
        if batch[0].task_seq is not None:
            lens_task = np.array(list(map(lambda sample: len(sample.task_seq), batch))).T
            max_lens["task_sequence"] = np.max(lens_task)
        return max_lens

    def _init_seq_data(self, batch_size: int, max_len: int, compound_factor: int = 0):
        seq_shape = (batch_size, max_len, compound_factor) if compound_factor > 0 else (batch_size, max_len)
        seq_data = SeqInputs(
            tokens=torch.full(seq_shape, self.pad_token_id, dtype=torch.long),
            values=torch.full(seq_shape, self.pad_token_id, dtype=torch.float),
            mask=torch.zeros(*seq_shape[:2], dtype=torch.bool)
        )
        return seq_data

    def init_data(self, batch: Sequence[SequenceSample], inference: bool = False):
        max_lens = self.get_max_lengths(batch, inference=inference)

        sample, b = batch[0], len(batch)
        return SequenceInputs(
            sequences=self._init_seq_data(
                b, max_lens["sequence"],
                compound_factor=sample.seq.ids.shape[-1]
            ),
            segments=SeqSegments(
                bar=torch.zeros(b, max_lens["sequence"], dtype=torch.long),
                beat=torch.zeros(b, max_lens["sequence"], dtype=torch.long),
                onset=torch.zeros(b, max_lens["sequence"], dtype=torch.long)
            ) if sample.segments is not None else None,
            pedals=torch.full((b, max_lens["sequence"], 2), fill_value=-1.),
            sequence_labels=torch.zeros(b, dtype=torch.long),
            type_ids=torch.zeros(b, max_lens["sequence"], dtype=torch.long),
            encoding_ids=torch.zeros(b, dtype=torch.long),
            task_ids=torch.zeros(b, dtype=torch.long),
            task_sequences=self._init_seq_data(
                b, max_lens["task_sequence"],
                compound_factor=sample.task_seq.ids.shape[-1]
            ) if sample.task_seq is not None else None,
            score_sequences=self._init_seq_data(
                b, max_lens["sequence"],
                compound_factor=sample.score_seq.ids.shape[-1]
            ) if sample.score_seq is not None else None,
            context_sequences=self._init_seq_data(
                b, max_lens["sequence"],
                compound_factor=sample.context_seq.ids.shape[-1]
            ) if sample.context_seq is not None else None,
            emotion_labels=torch.zeros(b, dtype=torch.long) if sample.emotion_labels is not None else None,
            emotion_embeddings=torch.zeros(
                b, max_lens["sequence"], sample.emotion_embeddings.shape[-1]
            ) if sample.emotion_embeddings is not None else None,
            random_sequences=self._init_seq_data(
                b, max_lens["sequence"],
                compound_factor=sample.seq.ids.shape[-1]
            ) if sample.random_seq is not None else None
        )

    @staticmethod
    def _process_sequence(
            i: int,
            seq: TokSequence,
            seq_data: SeqInputs
    ):
        seq_len = len(seq)

        seq_data.tokens[i, :seq_len] = torch.from_numpy(seq.ids)
        if seq.values is not None:
            seq_data.values[i, :seq_len] = torch.from_numpy(seq.values)

        seq_data.mask[i, :seq_len] = True

    @staticmethod
    def _process_segments(i: int, sample: SequenceSample, seg_data: SeqSegments, seq_len: int):
        if sample.segments is not None:
            seg_data.bar[i, :seq_len] = torch.from_numpy(sample.segments.bar)
            seg_data.beat[i, :seq_len] = torch.from_numpy(sample.segments.beat)
            seg_data.onset[i, :seq_len] = torch.from_numpy(sample.segments.onset)

    def _process_auxiliary(self, i: int, sample: SequenceSample, data: SequenceInputs):
        seq_len = len(sample.seq)

        # process pedals
        if hasattr(data, "pedals") and sample.pedals is not None:
            pedals = sample.pedals[:seq_len]  # in most cases number of pedals is less than the number of notes
            data.pedals[i, :len(pedals)] = torch.from_numpy(pedals)

        # process sequence labels
        if sample.seq_label is not None:
            data.sequence_labels[i] = sample.seq_label

        # process type ids
        if sample.type_ids is not None:
            data.type_ids[i, :seq_len] = torch.from_numpy(sample.type_ids)

        # process encoding id
        if data.encoding_ids.ndim == 2:
            data.encoding_ids[i] = torch.tensor(list(sample.encoding_ids))
        else:
            data.encoding_ids[i] = sample.encoding_ids[0]

        # process task id
        data.task_ids[i] = sample.task_idx

        # process task sequence
        if sample.task_seq is not None:
            seq = sample.task_seq
            task_seq_len = len(seq.ids)
            data.task_sequences.tokens[i, -task_seq_len:] = torch.from_numpy(seq.ids)
            if seq.values is not None:
                data.task_sequences.values[i, -task_seq_len:] = torch.from_numpy(seq.values)
            data.task_sequences.mask[i, -task_seq_len:] = True

        # process context sequence
        if hasattr(data, "score_sequences") and sample.score_seq is not None:
            self._process_sequence(i, seq=sample.score_seq, seq_data=data.score_sequences)

        # process context sequence
        if sample.context_seq is not None:
            self._process_sequence(i, seq=sample.context_seq, seq_data=data.context_sequences)

        # process emotion labels
        if sample.emotion_labels is not None:
            data.emotion_labels[i] = sample.emotion_labels.argmax()

        # process emotion embeddings
        if sample.emotion_embeddings is not None:
            data.emotion_embeddings[i, :seq_len] = torch.from_numpy(sample.emotion_embeddings)

        # process random performance is present
        if sample.random_seq is not None:
            self._process_sequence(i, seq=sample.random_seq, seq_data=data.random_sequences)

    def process_sample(self, i: int, sample: SequenceSample, data: SequenceInputs, inference: bool = False):
        # process sequence
        self._process_sequence(i, seq=sample.seq, seq_data=data.sequences)

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

def prob_mask_like(t: torch.Tensor, prob: float):
    if t.ndim <= 2:
        return torch.zeros_like(t).float().uniform_(0, 1) < prob
    else:
        mask = torch.zeros(*t.shape[:2], dtype=torch.float32, device=t.device).uniform_(0, 1) < prob
        return mask[..., None]


def mask_with_tokens(t: torch.Tensor, token_ids: Sequence[int], squeeze: bool = True):
    if t.ndim == 2 or not squeeze:
        init_no_mask = torch.full_like(t, False, dtype=torch.bool)
        mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    else:
        init_no_mask = torch.full(t.shape[:2], False, dtype=torch.bool, device=t.device)
        mask = reduce(lambda acc, el: acc | torch.any(t == el, dim=-1), token_ids, init_no_mask)
    return mask


def mask_with_token_dims(seq: torch.Tensor, token_dims: Sequence[int] | None = None):
    if seq.ndim == 2:
        return torch.full((1, 1), True, dtype=torch.bool, device=seq.device)
    elif token_dims is None:
        return torch.full((1, 1, seq.shape[-1]), True, dtype=torch.bool, device=seq.device)
    else:
        mask = torch.full((1, 1, seq.shape[-1]), False, dtype=torch.bool, device=seq.device)
        if token_dims:
            mask[..., token_dims] = True
        return mask


def sample_mask(mask: torch.Tensor, prob: float | list[float] | torch.Tensor, level: MaskLevel | None = None):
    b, t_init, device = *mask.shape[:2], mask.device
    prob = torch.tensor(prob, device=device) if not isinstance(prob, torch.Tensor) else prob
    prob = prob.view(-1)

    is_compound = mask.ndim == 3
    if is_compound:
        mask = mask.contiguous().view(b, -1)

    t = mask.shape[1]
    max_masked = torch.ceil(prob * t).long()

    num_tokens = mask.sum(dim=-1, keepdim=True)
    num_mask_tokens = (num_tokens * prob[:, None]).ceil().long()
    mask_excess = torch.arange(t, device=device)[None] >= num_mask_tokens
    mask_excess = mask_excess[:, :max_masked.max()]

    t_range = torch.arange(t, device=device)
    if level == MaskLevel.START:
        values = torch.arange(t, 0, -1, device=device)[None].repeat(b, 1)
    elif level == MaskLevel.MIDDLE:
        low = (mask * (torch.arange(t, 0, -1, device=device)[None] + 1)).argmax(dim=-1)
        mask_range = torch.maximum(torch.tensor(1.), (num_tokens - num_mask_tokens).flatten())
        start_ids = torch.randint(2 ** 63 - 1, size=(b,)) % mask_range + low
        values = torch.arange(t, 0, -1, device=device)[None].repeat(b, 1)
        values[t_range[None] < start_ids[:, None]] = -1e9
    elif level == MaskLevel.END:
        values = t_range[None].repeat(b, 1)
    else:
        values = torch.rand((b, t), device=device)

    values = values.masked_fill(~mask, -1e9)
    _, sampled_indices = values.topk(max_masked.max(), dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

    new_mask = torch.zeros((b, t + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    new_mask = new_mask[:, 1:].bool()

    if is_compound:
        new_mask = new_mask.view(b, t_init, -1)

    return new_mask


@dataclass
class LMSequenceInputs(SequenceInputs):
    labels: SeqInputs | None = None
    full_labels: SeqInputs | None = None


class MaskLevel(ExplicitEnum):
    ELEMENT = "element"
    NOTE = "note"
    ONSET = "onset"
    BEAT = "beat"
    BAR = "bar"
    START = "start"
    MIDDLE = "middle"
    END = "end"
    LAST_NOTE = "last_note"
    LAST_ONSET = "last_onset"
    LAST_BEAT = "last_beat"
    LAST_BAR = "last_bar"
    ALL = "all"
    ALL_IGNORE = "all_ignore"
    NONE = "none"


SEGMENT_MASKS = [
    MaskLevel.ONSET, MaskLevel.BEAT, MaskLevel.BAR,
    MaskLevel.LAST_ONSET, MaskLevel.LAST_BEAT, MaskLevel.LAST_BAR
]
REGION_MASKS = [
    MaskLevel.START, MaskLevel.MIDDLE, MaskLevel.END
]
LAST_NOTES_MASKS = [
    MaskLevel.LAST_NOTE, MaskLevel.LAST_ONSET, MaskLevel.LAST_BEAT, MaskLevel.LAST_BAR
]


class LMSequenceCollator(SequenceCollator):
    def __init__(
            self,
            pad_token_id: int = 0,
            pad_to_multiple_of: int = 1,

            mlm: bool = False,
            mask_level: str | MaskLevel | dict[str | MaskLevel, float] = MaskLevel.NOTE,
            mask_level_exceptions: dict[str, list[str]] | None = None,
            mask_compound: bool = True,
            mask_prob: float | tuple[float, float] = 0.15,
            replace_prob: float = 0.9,
            random_token_prob: float = 0.,
            copy_sequence_prob: float = 0.,
            mask_token_id: int = 1,
            ignore_token_id: int = 4,
            mask_ignore_token_ids: list[int] | None = None,
            mask_token_dims: dict[str, list[int]] | list[int] | None = None,
            mask_token_dims_by_type_id: dict[int, list[int]] | None = None,
            label_pad_token_id: int = -100,
            context_known_dropout: float = 0.,
            context_label_dropout: float = 0.
    ):
        super().__init__(pad_token_id=pad_token_id, pad_to_multiple_of=pad_to_multiple_of)

        self.mlm = mlm
        self.mask_level = mask_level
        self.mask_level_exceptions = mask_level_exceptions
        self.mask_compound = mask_compound

        mask_levels = list(mask_level) if isinstance(mask_level, (dict, DictConfig)) else [mask_level]
        for mask_level in mask_levels:
            assert MaskLevel.has_value(mask_level), \
                f"`{mask_level}` is not a valid `mask_level`, available modes: {MaskLevel.list()}"

        self.mask_prob = tuple(mask_prob) if isinstance(mask_prob, Sequence) else mask_prob
        self.replace_prob = replace_prob
        self.random_token_prob = random_token_prob
        assert self.replace_prob + self.random_token_prob <= 1.

        self.copy_sequence_prob = copy_sequence_prob

        self.mask_token_id = mask_token_id
        self.ignore_token_id = ignore_token_id
        self.mask_token_value = SPECIAL_TOKENS_VALUE - mask_token_id
        self.mask_ignore_token_ids = list({*(mask_ignore_token_ids or []), pad_token_id})
        self.mask_token_dims = mask_token_dims
        self.mask_token_dims_by_type_id = mask_token_dims_by_type_id
        self.label_pad_token_id = label_pad_token_id

        self.context_known_dropout = context_known_dropout
        self.context_label_dropout = context_label_dropout

    def _mask_sequence(
            self,
            mask_level: str | MaskLevel,
            seq: torch.Tensor,
            values: torch.Tensor | None = None,
            task_ids: torch.Tensor | None = None,
            random_seq: torch.Tensor | None = None,
            random_values: torch.Tensor | None = None,
            segments: SeqSegments | None = None,
            num_tokens: int | dict[str, int] | None = None,
            sequence_task_types: dict[str, int] | None = None,
            type_ids: torch.Tensor | None = None
    ):
        b, t = seq.shape[:2]
        replace_prob, random_token_prob = self.replace_prob, self.random_token_prob

        masked_seq = seq.clone().detach()
        masked_values = values.clone().detach() if values is not None else None

        # per-dimension mask for possibly compound tokens (dimensions to mask)
        if (self.mask_token_dims is not None and len(self.mask_token_dims) > 0
                and isinstance(self.mask_token_dims, (dict, DictConfig))
                and task_ids is not None):
            assert sequence_task_types is not None
            dim_masks = torch.concatenate([
                mask_with_token_dims(seq, self.mask_token_dims.get(task_type, []))
                for task_type in sequence_task_types.keys()
            ], dim=0)
            dim_mask = dim_masks[task_ids]
        else:
            dim_mask = mask_with_token_dims(seq, self.mask_token_dims)

        # do not mask [pad] tokens, or any other tokens in the tokens designated to be excluded ([sos], [eos])
        # also do not include these special tokens in the tokens chosen at random
        no_mask = mask_with_tokens(seq, self.mask_ignore_token_ids)
        no_mask_note = mask_with_tokens(seq, self.mask_ignore_token_ids, squeeze=False)

        if mask_level in (MaskLevel.NONE, MaskLevel.ALL, MaskLevel.ALL_IGNORE):  # predict everything
            label_mask = (~no_mask_note).clone().detach() * dim_mask
            labels = seq.clone().detach().masked_fill(~label_mask, self.label_pad_token_id)

            targets = None
            if values is not None:
                targets = values.clone().detach().masked_fill(~label_mask, self.label_pad_token_id)

            if mask_level in (MaskLevel.ALL, MaskLevel.ALL_IGNORE):  # also mask everything
                mask_token_id = self.mask_token_id if mask_level == MaskLevel.ALL else self.ignore_token_id
                masked_seq.masked_fill_(label_mask, mask_token_id)
                if masked_values is not None:
                    masked_values.masked_fill_(label_mask, self.label_pad_token_id - mask_token_id)

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
            no_mask = mask_with_tokens(segment_ids, self.mask_ignore_token_ids)
            no_mask = no_mask | (segment_ids > max_values[:, None])
        elif mask_level == MaskLevel.ELEMENT:
            no_mask = no_mask_note.clone() * (~dim_mask)

        def _maybe_distribute_segment_mask(segment_mask):
            if mask_level in SEGMENT_MASKS:
                segment_mask = segment_mask[(torch.arange(b).repeat_interleave(t), segments.view(-1))].view(b, t, -1)
            else:
                segment_mask = segment_mask.view(b, t, -1)
            return segment_mask if seq.ndim == 3 else segment_mask.squeeze(-1)

        mask_prob = self.mask_prob
        if isinstance(mask_prob, tuple):
            mask_prob = torch.zeros((b,)).uniform_(*mask_prob)

        if mask_level in LAST_NOTES_MASKS:
            last_segment_ids = torch.max(segment_ids * (~no_mask), dim=1).values
            mask = segment_ids == last_segment_ids[:, None]
            replace_prob, random_token_prob = 1., 0.
        elif mask_level in REGION_MASKS:
            mask = sample_mask(~no_mask, mask_prob, mask_level)
        else:
            mask = sample_mask(~no_mask, mask_prob)

        # possibly expand mask
        if not self.mask_compound and seq.ndim == 3 and mask.ndim == 2:
            mask = mask[..., None].expand(-1, -1, seq.size(2))
            mask = mask * dim_mask

        # mask input with [mask] tokens with probability of `replace_prob`
        replace_mask = sample_mask(mask, replace_prob)
        no_replace_mask = mask & (~replace_mask)

        # possibly expand mask
        if self.mask_compound and seq.ndim == 3 and mask.ndim == 2:
            mask = mask[..., None].expand(-1, -1, seq.size(2))

        # mask input with [mask] tokens with probability of `replace_mask`
        replace_mask = _maybe_distribute_segment_mask(replace_mask)
        replace_mask = replace_mask * dim_mask * (~no_mask_note)
        masked_seq.masked_fill_(replace_mask, self.mask_token_id)
        if masked_values is not None:
            masked_values.masked_fill_(replace_mask, self.label_pad_token_id - self.mask_token_id)

        # if random token probability > 0 for mlm
        if random_token_prob > 0:
            random_token_mask = sample_mask(no_replace_mask, random_token_prob / (1 - replace_prob))
            random_token_mask = _maybe_distribute_segment_mask(random_token_mask)
            random_token_mask = random_token_mask * dim_mask * (~no_mask_note)

            if random_seq is None:
                assert num_tokens is not None, "`num_tokens` must be provided for computing random tokens"
                if seq.ndim == 3:
                    random_seq = torch.stack([
                        torch.randint_like(seq[..., i], low=2, high=num)
                        for i, num in enumerate(num_tokens.values())
                    ], dim=-1)
                else:
                    random_seq = torch.randint_like(seq, low=2, high=num_tokens)

            masked_seq[random_token_mask] = random_seq[random_token_mask]
            if masked_values is not None:
                if random_values is not None:
                    masked_values[random_token_mask] = random_values[random_token_mask]
                else:
                    masked_values[random_token_mask] = self.label_pad_token_id - self.mask_token_id

        # mask note features by type id (input mask and no loss)
        type_id_mask = None
        if self.mask_token_dims_by_type_id is not None and type_ids is not None:
            for type_id, mask_token_dims in self.mask_token_dims_by_type_id.items():
                type_id_dim_mask = mask_with_token_dims(seq, mask_token_dims)
                type_id_dim_mask = ((type_ids == type_id)[..., None] & type_id_dim_mask)
                type_id_mask = type_id_dim_mask if type_id_mask is None else type_id_mask | type_id_dim_mask

            masked_seq.masked_fill_(type_id_mask, self.mask_token_id)
            if masked_values is not None:
                masked_values.masked_fill_(type_id_mask, self.label_pad_token_id - self.mask_token_id)

        # derive labels to predict
        if self.copy_sequence_prob > 0:
            copy_mask = prob_mask_like(seq[:, 0, 0] if seq.ndim == 3 else seq[:, 0], self.copy_sequence_prob)
            mask = mask | (copy_mask[:, None, None] if seq.ndim == 3 else copy_mask[:, None])
            dim_mask[copy_mask] = True
            if type_id_mask is not None:
                type_id_mask = type_id_mask & (~copy_mask[:, None, None])

        mask = _maybe_distribute_segment_mask(mask)
        mask = mask & (seq != self.ignore_token_id) & dim_mask
        label_mask = mask.clone().detach() * (~no_mask_note)
        if type_id_mask is not None:
            label_mask = label_mask & (~type_id_mask)
        labels = seq.clone().detach().masked_fill(~label_mask, self.label_pad_token_id)

        targets = None
        if values is not None:
            targets = values.clone().detach().masked_fill(~label_mask, self.label_pad_token_id)

        return (masked_seq, labels), (masked_values, targets), label_mask

    def mask_sequence(
            self,
            seq: torch.Tensor,
            values: torch.Tensor | None = None,
            task_ids: torch.Tensor | None = None,
            random_seq: torch.Tensor | None = None,
            random_values: torch.Tensor | None = None,
            segments: SeqSegments | None = None,
            num_tokens: int | dict[str, int] | None = None,
            sequence_task_types: dict[str, int] | None = None,
            type_ids: torch.Tensor | None = None,
    ):
        mask_levels = self.mask_level
        mask_levels = {mask_levels: 1.} if isinstance(mask_levels, (str, MaskLevel)) else mask_levels

        if self.mask_level_exceptions is None:
            mask_levels, probs = list(mask_levels.keys()), list(mask_levels.values())
            mask_level_ids = torch.tensor(random.choices(list(range(len(probs))), weights=probs, k=len(seq)))
        else:
            sequence_task_map = {idx: task for task, idx in sequence_task_types.items()}
            mask_level_ids = []
            for task_idx in task_ids:
                task = sequence_task_map[int(task_idx)]
                _mask_levels = dict(mask_levels)
                for mask_level, tasks in self.mask_level_exceptions.items():
                    if task in tasks:
                        _mask_levels[mask_level] = 0.

                probs = list(_mask_levels.values())
                mask_level_ids.append(random.choices(list(range(len(probs))), weights=probs, k=1)[0])
            mask_level_ids = torch.tensor(mask_level_ids)

        masked_seq = seq.clone().detach()
        masked_values = values.clone().detach() if values is not None else None
        labels = seq.clone().detach()
        targets = values.clone().detach() if values is not None else None
        label_mask = torch.zeros_like(labels).bool()

        for i, mask_level in enumerate(mask_levels):
            (masked_seq_i, labels_i), (masked_values_i, targets_i), label_mask_i = self._mask_sequence(
                mask_level,
                seq,
                values=values,
                task_ids=task_ids,
                random_seq=random_seq,
                random_values=random_values,
                segments=segments,
                num_tokens=num_tokens,
                sequence_task_types=sequence_task_types,
                type_ids=type_ids
            )
            level_mask = mask_level_ids == i

            masked_seq[level_mask] = masked_seq_i[level_mask]
            labels[level_mask] = labels_i[level_mask]
            if values is not None:
                masked_values[level_mask] = masked_values_i[level_mask]
                targets[level_mask] = targets_i[level_mask]
            label_mask[level_mask] = label_mask_i[level_mask]

        return (masked_seq, labels), (masked_values, targets), label_mask

    def dropout_context(
            self,
            sequences: SeqInputs,
            label_mask: torch.Tensor,
            score_sequences: SeqInputs | None = None,
            context_sequences: SeqInputs | None = None,
            emotion_embeddings: torch.Tensor | None = None,
    ):
        label_note_mask = torch.any(label_mask, dim=-1) if label_mask.ndim == 3 else label_mask  # predicted notes

        def compute_dropout_mask(tensor):
            mask = torch.zeros_like(tensor, dtype=torch.bool)

            if self.context_known_dropout > 0:  # dropout context for known notes
                drop_context_mask = prob_mask_like(sequences.tokens[:, 0, 0], self.context_known_dropout)[:, None]
                mask = mask | (~label_note_mask & drop_context_mask)[..., None]
            if self.context_label_dropout > 0:  # dropout context for predicted notes
                drop_label_mask = prob_mask_like(sequences.tokens[:, 0, 0], self.context_label_dropout)[:, None]
                mask = mask | (label_note_mask & drop_label_mask)[..., None]

            return mask

        if score_sequences is not None:
            drop_mask = compute_dropout_mask(score_sequences.tokens)
            score_sequences.tokens.masked_fill_(drop_mask, self.mask_token_id)

        if context_sequences is not None:
            drop_mask = compute_dropout_mask(context_sequences.tokens)
            context_sequences.tokens.masked_fill_(drop_mask, self.mask_token_id)

        if emotion_embeddings is not None:
            drop_mask = compute_dropout_mask(emotion_embeddings)
            emotion_embeddings[drop_mask] = 0.

        return score_sequences, context_sequences, emotion_embeddings

    def mask_and_compute_labels(
            self,
            sequences: SeqInputs,
            random_sequences: SeqInputs | None = None,
            segments: SeqSegments | None = None,
            type_ids: torch.Tensor | None = None,
            task_ids: torch.Tensor | None = None,
            score_sequences: SeqInputs | None = None,
            context_sequences: SeqInputs | None = None,
            emotion_embeddings: torch.Tensor | None = None,
            num_tokens: int | dict[str, int] | None = None,
            sequence_task_types: dict[str, int] | None = None
    ):
        targets = None
        if self.mlm:
            (masked_seq, labels), (masked_values, targets), label_mask = self.mask_sequence(
                sequences.tokens,
                values=sequences.values,
                task_ids=task_ids,
                random_seq=random_sequences.tokens,
                random_values=random_sequences.values,
                segments=segments,
                num_tokens=num_tokens,
                sequence_task_types=sequence_task_types,
                type_ids=type_ids
            )
            sequences.tokens, sequences.values = masked_seq, masked_values
        else:
            labels = sequences.tokens.clone().detach()
            labels[labels == self.pad_token_id] = self.label_pad_token_id
            label_mask = sequences.mask.clone().detach()

            if sequences.values is not None:
                targets = sequences.values.clone().detach()
                targets[~label_mask] = self.label_pad_token_id

        if self.mlm:
            self.dropout_context(
                sequences=sequences,
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
            tokens=data.sequences.tokens,
            values=data.sequences.values,
            mask=data.sequences.mask
        )

        labels = self.mask_and_compute_labels(
            sequences=data.sequences,
            random_sequences=data.random_sequences,
            segments=data.segments,
            type_ids=data.type_ids,
            task_ids=data.task_ids,
            score_sequences=data.score_sequences,
            context_sequences=data.context_sequences,
            emotion_embeddings=data.emotion_embeddings,
            num_tokens=batch[0].meta.token_sizes,
            sequence_task_types=batch[0].meta.sequence_task_types
        )

        data = vars(data)
        data = LMSequenceInputs(
            **data,
            labels=labels,
            full_labels=full_labels
        )

        return asdict(data) if return_dict else data


@dataclass
class MultiSequenceInputs(SequenceInputs):
    sequence_mask: torch.Tensor | None = None


class LMMultiSequenceCollator(SequenceCollator):
    def __init__(
            self,
            task_sequences: bool = False,
            eod_token_single_seq: bool = False,

            pad_token_id: int = 0,
            pad_to_multiple_of: int = 1,
            label_pad_token_id: int = -100,
            **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, pad_to_multiple_of=pad_to_multiple_of)

        self.task_sequences = task_sequences
        self.eod_token_single_seq = eod_token_single_seq

        self.label_pad_token_id = label_pad_token_id

    def get_max_lengths(self, batch: Sequence[SequenceSample], inference: bool = False):
        lens_seq = np.array(list(map(
            lambda sample: len(sample.seq) if sample.meta.encoding_type[0] != sample.meta.encoding_type[1] else 0,
            batch
        ))).T
        lens_target_seq = np.array(list(map(lambda sample: len(sample.target_seq), batch))).T
        lens = lens_seq + lens_target_seq

        lens_enc = np.array(list(map(
            lambda sample: len(sample.enc_seq)  if sample.meta.encoding_type[0] != sample.meta.encoding_type[1] else 0,
            batch
        ))).T
        lens_task = np.array(list(map(lambda sample: len(sample.task_seq), batch))).T
        if self.task_sequences:
            lens = lens + lens_enc + lens_task

        max_lens = {
            "sequence": np.max(lens) if inference else self.pad_len(np.max(lens)),
            "task_sequence": np.max(lens_task)  # backward compatibility
        }
        return max_lens

    def init_data(self, batch: Sequence[SequenceSample], inference: bool = False):
        data = super().init_data(batch, inference)
        return MultiSequenceInputs(
            **vars(data),
            sequence_mask=torch.zeros_like(data.sequences.mask, dtype=torch.int)
        )

    def _process_multi_sequence(self, i: int, sample: SequenceSample, seq_data: SeqInputs, seq_mask: torch.Tensor):
        def _process_sequence_and_encoding(seq: TokSequence, enc_seq: TokSequence, cur_len: int):
            if self.task_sequences:
                seq_data.tokens[i, cur_len:cur_len + len(enc_seq)] = torch.from_numpy(enc_seq.ids)
                if enc_seq.values is not None:
                    seq_data.values[i, cur_len:cur_len + len(enc_seq)] = torch.from_numpy(enc_seq.values)
                seq_mask[i, cur_len:cur_len + len(enc_seq)] = 0
                cur_len += len(enc_seq)

            seq_len = len(seq)
            seq_data.tokens[i, cur_len:cur_len + seq_len] = torch.from_numpy(seq.ids)
            if seq.values is not None:
                seq_data.values[i, cur_len:cur_len + seq_len] = torch.from_numpy(seq.values)
            # if sample.type_ids is not None:
            #     seq_data.type_ids[i, cur_len:cur_len + seq_len] = torch.from_numpy(sample.type_ids)
            seq_mask[i, cur_len:cur_len + seq_len] = 1
            cur_len += seq_len
            return cur_len

        source_encoding, target_encoding = sample.meta.encoding_type

        cur_len = 0
        if source_encoding != target_encoding:
            cur_len = _process_sequence_and_encoding(sample.seq, sample.enc_seq, cur_len)
            seq_mask[i, cur_len - 1] = 0  # eod token for the first sequence

        cur_len = _process_sequence_and_encoding(sample.target_seq, sample.task_seq, cur_len)

        if source_encoding == target_encoding:
            seq_mask[i, cur_len - 1] = 0  # eod token for a single sequence

        seq_data.mask[i, :cur_len] = True

    def process_sample(self, i: int, sample: SequenceSample, data: MultiSequenceInputs, inference: bool = False):
        # process sequence
        self._process_multi_sequence(i, sample=sample, seq_data=data.sequences, seq_mask=data.sequence_mask)

        # process note segments if present
        self._process_segments(i, sample=sample, seg_data=data.segments, seq_len=len(sample.seq))

        # process auxiliary
        self._process_auxiliary(i, sample=sample, data=data)

    def compute_labels(self, sequences: SeqInputs, sequence_mask: torch.Tensor):
        labels = sequences.tokens.clone().detach()
        label_mask = sequences.mask.clone().detach() & (sequence_mask != 0)
        labels[~label_mask] = self.label_pad_token_id

        targets = None
        if sequences.values is not None:
            targets = sequences.values.clone().detach()
            targets[~label_mask] = self.label_pad_token_id

        return SeqInputs(
            tokens=labels,
            values=targets,
            mask=label_mask
        )

    def __call__(self, batch: Sequence[SequenceSample], inference: bool = False, return_dict: bool = True):
        data = super().__call__(batch, inference=inference, return_dict=False)

        full_labels = SeqInputs(
            tokens=data.sequences.tokens,
            values=data.sequences.values,
            mask=data.sequences.mask
        )

        labels = self.compute_labels(
            sequences=data.sequences,
            sequence_mask=data.sequence_mask
        )

        data = vars(data)
        data.pop("sequence_mask")
        data = LMSequenceInputs(
            **data,
            labels=labels,
            full_labels=full_labels
        )

        return asdict(data) if return_dict else data


@dataclass
class MixedLMSequenceInputs(LMSequenceInputs):
    masked_sequences: SeqInputs | None = None


class MixedLMSequenceCollator(SequenceCollator):
    def __init__(
            self,
            pad_token_id: int = 0,
            pad_to_multiple_of: int = 1,

            mask_token_id: int = 1,
            mask_ignore_token_ids: list[int] | None = None,
            mask_token_dims: list[int] | None = None,
            label_pad_token_id: int = -100
    ):
        super().__init__(pad_token_id=pad_token_id, pad_to_multiple_of=pad_to_multiple_of)

        self.mask_token_id = mask_token_id
        self.mask_token_value = SPECIAL_TOKENS_VALUE - mask_token_id
        self.mask_ignore_token_ids = {*(mask_ignore_token_ids or []), pad_token_id}
        self.mask_token_dims = mask_token_dims
        self.label_pad_token_id = label_pad_token_id

    def mask_sequence(self, seq: torch.Tensor, values: torch.Tensor | None):
        # do not mask positions for ignored tokens
        no_mask = mask_with_tokens(seq, self.mask_ignore_token_ids, squeeze=False)

        # mask only non-ignored token dimension
        dim_mask = mask_with_token_dims(seq, self.mask_token_dims)

        label_mask = (~no_mask) * dim_mask
        masked_seq = seq.clone().detach().masked_fill(label_mask, self.mask_token_id)

        masked_values = None
        if values is not None:
            masked_values = values.clone().detach().masked_fill(
                label_mask, self.label_pad_token_id - self.mask_token_id
            )

        # derive labels to predict
        labels = seq.clone().detach().masked_fill(~label_mask, self.label_pad_token_id)

        targets = None
        if values is not None:
            targets = values.clone().detach().masked_fill(~label_mask, self.label_pad_token_id)

        return (masked_seq, labels), (masked_values, targets), label_mask

    def mask_and_compute_labels(self, sequences: SeqInputs):
        (masked_tokens, labels), (masked_values, targets), label_mask = self.mask_sequence(
            sequences.tokens,
            values=sequences.values
        )

        masked_sequences = SeqInputs(
            tokens=masked_tokens,
            values=masked_values,
            mask=sequences.mask.clone().detach()
        )

        labels = SeqInputs(
            tokens=labels,
            values=targets,
            mask=label_mask
        )

        return masked_sequences, labels

    def __call__(self, batch: Sequence[SequenceSample], inference: bool = False, return_dict: bool = True):
        data = super().__call__(batch, inference=inference, return_dict=False)

        masked_sequences, labels = self.mask_and_compute_labels(sequences=data.sequences)

        data = MixedLMSequenceInputs(
            sequences=data.sequences,
            segments=data.segments,
            masked_sequences=masked_sequences,
            type_ids=data.type_ids,
            task_ids=data.task_ids,
            task_sequences=data.task_sequences,
            labels=labels
        )

        return asdict(data) if return_dict else data
