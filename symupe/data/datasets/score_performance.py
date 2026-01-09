""" Score-Performance token sequence datasets. """
from __future__ import annotations

import copy
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass, replace
from functools import partial
from itertools import chain
from pathlib import Path

import numpy as np
from omegaconf import ListConfig, DictConfig
from torch.utils.data import Dataset

from symupe.utils import prob2bool, load_json, dump_json, tqdm_iterator
from .common import NoteSegments, DATA_SPLITS, SequenceTask
from .token_sequence import load_and_process_token_sequence, TokenSequenceDataset, LocalTokenSequenceDataset
from .utils import get_num_bars, compute_sample_positions, get_end_bar, load_token_sequence
from ..helpers import (
    TupleTokenSequenceProcessor,
    TokenSequenceBarIndexer,
    TokenSequenceAugmentations
)
from ..tokenizers import TOKENIZERS, SyMuPe, TokSequence, EncodingType


@dataclass
class ScorePerformanceSampleMeta:
    idx: int | None
    score_idx: int
    perf_idx: int
    start_bar: int
    end_bar: int | None
    start_idx: int | None = None
    end_idx: int | None = None
    position_shifts: dict[str, int | float] | None = None
    note_shifts: tuple[int, int] = (0, 0)
    context_offsets: tuple[int, int] = (0, -1)
    augmentations: TokenSequenceAugmentations | None = None
    noisy_augmentations: TokenSequenceAugmentations | None = None
    encoding_type: str | EncodingType = EncodingType.PERFORMANCE
    is_deadpan: bool = False
    token_sizes: dict[str, int] | None = None


@dataclass
class ScorePerformanceSample:
    meta: ScorePerformanceSampleMeta | None
    score: TokSequence
    perf: TokSequence
    noisy_perf: TokSequence | None = None
    segments: NoteSegments | None = None
    task_idx: int = 0
    context_offsets: tuple[int, int] = (0, -1)
    directions: dict[str, dict[tuple[int, str], np.ndarray]] | None = None
    is_deadpan: bool = False
    random_perf: TokSequence | None = None


class ScorePerformanceDataset(Dataset):
    def __init__(
            self,
            scores: TokenSequenceDataset,
            performances: TokenSequenceDataset,
            metadata: dict[str, list[str]],
            tokenizer: SyMuPe | dict[str, object],
            score_token_types: list[str] | None = None,
            performance_token_types: list[str] | None = None,
            alignments: dict[str, np.ndarray] | None = None,
            auxiliary_data: dict[str, object] | None = None,
            performance_directions: str | Path | list[str] | dict[str, list[str]] | None = None,
            score_directions_dict: str | Path | dict[str, list[dict]] | None = None,

            max_seq_len: int = 512,
            max_bar: int = 256,
            bar_sliding_window: int = 16,

            sample_bars: bool | float = False,
            sample_note_shift: bool | float = False,
            max_note_shift_ratio: float = 0.2,
            force_max_seq_len: bool | float = False,

            fit_to_max_bar: bool = False,
            shift_bar_to_zero: bool = False,
            sample_bar_shift: bool | float = False,

            context_prev_bars: int = 0,
            context_next_bars: int = 0,

            add_sos_eos: bool = False,

            sample: bool = False,
            seed: int = 23,

            augment_performance: bool | float = False,
            pitch_shift_range: tuple[int, int] = (-3, 3),
            velocity_shift_range: tuple[int, int] = (-4, 4),
            tempo_stretch_range: tuple[float, float] = (-0.1, 0.1),

            noisy_performance: bool = False,
            noise_strength: float = 0.5,
            noisy_random_bars: bool | float = 0.5,

            deadpan_performance: bool | float = False,

            quantize_values: bool | float = False,
            clip_values: bool = False,
            normalize_values: bool = False,

            **kwargs
    ):
        self.metadata = metadata

        self.performance_names = list(sorted(set(chain.from_iterable(self.metadata.values()))))
        self.score_names = list(sorted(self.metadata.keys()))

        self._performance_map = {
            perf: (score, idx)
            for score, performances in self.metadata.items()
            for idx, perf in enumerate(performances)
        }

        self.scores = scores
        self.performances = performances

        # perf-to-score alignments
        self.alignments = alignments

        # load tokenizer
        if isinstance(tokenizer, dict):
            encoding = TOKENIZERS[tokenizer["tokenization"]]
            self.tokenizer = encoding(params=tokenizer)
        else:
            self.tokenizer = tokenizer
        self.encoding = self.tokenizer.__class__.__name__

        self.score_token_types = score_token_types or list(self.tokenizer.score_sizes.keys())
        self.performance_token_types = performance_token_types or list(self.tokenizer.performance_sizes.keys())

        self.score_token_sizes = {
            key: num for key, num in self.tokenizer.sizes.items()
            if key in self.score_token_types
        }
        self.performance_token_sizes = {
            key: num for key, num in self.tokenizer.sizes.items()
            if key in self.performance_token_types
        }

        # augmentations
        self.augment_performance = augment_performance
        self.noisy_performance = noisy_performance

        if self.augment_performance == 0. and not self.noisy_performance:
            pitch_shift_range = velocity_shift_range = tempo_stretch_range = (0, 0)

        self.noise_strength = noise_strength
        self.noisy_random_bars = noisy_random_bars

        # sequence processor
        self.processor = TupleTokenSequenceProcessor(
            tokenizer=self.tokenizer,
            pitch_shift_range=pitch_shift_range,
            velocity_shift_range=velocity_shift_range,
            tempo_stretch_range=tempo_stretch_range,
        )

        # set up auxiliary data
        if auxiliary_data is not None:
            for key, data in auxiliary_data.items():
                setattr(self, key, data)

        # configurations
        self.max_seq_len = max_seq_len
        self.max_bar = max_bar
        self.bar_sliding_window = bar_sliding_window
        self.add_sos_eos = add_sos_eos
        assert max_bar <= self.tokenizer.config.additional_params["max_bar_embedding"]

        # bar indexer and indices arrays
        self.indexer = TokenSequenceBarIndexer(self.tokenizer)
        self._score_indices = [None] * len(self.scores)
        self._perf_indices = [None] * len(self.performances)

        # load or compute number of bars in performances used to build samples
        self.bars = getattr(self, "bars", {})
        for perf_idx, perf in enumerate(tqdm_iterator(self.performance_names, desc="Precomputing bars...")):
            if perf not in self.bars:
                self.bars[perf] = get_num_bars(self.performances[perf_idx], tokenizer=self.tokenizer)
        _perf_num_bars = np.array([self.bars[perf] for perf in self.performance_names])

        # compute sample positions
        self._length, self._sample_positions, self._sample_ids = compute_sample_positions(
            seq_num_elements=_perf_num_bars, sliding_window=self.bar_sliding_window
        )

        # random effects they do not advertise
        self.sample = sample
        if self.sample:
            random.seed(seed)
            np.random.seed(seed)

        # sequence sampling
        self.sample_bars = sample_bars
        self.sample_note_shift = sample_note_shift
        self.max_note_shift_ratio = max_note_shift_ratio
        self.force_max_seq_len = force_max_seq_len

        # bar of the first note
        assert not (fit_to_max_bar and shift_bar_to_zero), \
            "Only one of `fit_to_max_bar`/`fit_to_zero_bar` could be set to True"
        self.fit_to_max_bar = fit_to_max_bar
        self.shift_bar_to_zero = shift_bar_to_zero
        self.sample_bar_shift = sample_bar_shift

        # context performance
        self.context_prev_bars = context_prev_bars
        self.context_next_bars = context_next_bars

        # occasional score-based deadpan performances
        self.deadpan_performance = deadpan_performance

        # values processing
        self.quantize_values = quantize_values
        self.clip_values = clip_values
        self.normalize_values = normalize_values

        # performance directions data
        if isinstance(performance_directions, (str, Path)):
            with open(performance_directions, "r") as f:
                performance_directions = json.load(f)

        performance_direction_sizes = None
        if performance_directions is not None:
            assert score_directions_dict is not None, \
                "`score_directions_dict` should be provided with `performance_directions`"
            if isinstance(performance_directions, (list, ListConfig)):
                performance_directions = {"directions": list(performance_directions)}
            elif isinstance(performance_directions, DictConfig):
                performance_directions = dict(performance_directions)

            performance_direction_sizes = {
                key: len(performance_directions[key]) + 1 for key in performance_directions
            }

        self.performance_directions = performance_directions
        self.performance_direction_sizes = performance_direction_sizes

        # score-direction maps
        if isinstance(score_directions_dict, (str, Path)):
            with open(score_directions_dict, "r") as f:
                score_directions_dict = json.load(f)

        self.score_direction_maps = None
        if score_directions_dict is not None:
            from .directions import build_score_direction_maps
            performance_directions = [
                item for group_keys in self.performance_directions.values() for item in group_keys
            ]
            self.score_direction_maps = build_score_direction_maps(
                self, score_directions_dict, direction_keys=performance_directions
            )["score"]["note"]

    def get_direction_class_weights(self):
        directions_nums = {}
        for group_name, group_directions in self.performance_directions.items():
            directions_nums[group_name] = defaultdict(int)

        none_key = (0, "none")
        total_notes = 0
        for score_idx, score in enumerate(self.score_names):
            score_direction_note_maps = self.score_direction_maps[score_idx]
            total_notes += len(self.scores[score_idx]) * len(self.metadata[score])
            for group_name, group_directions in self.performance_directions.items():
                directions_nums[group_name][none_key] += len(self.scores[score_idx]) * len(self.metadata[score])
                for i, key in enumerate(group_directions):
                    if key in score_direction_note_maps:
                        num_notes = int(score_direction_note_maps[key].sum())
                    else:
                        num_notes = 0
                    directions_nums[group_name][(i + 1, key)] += num_notes * len(self.metadata[score])

        weights = {}
        for group_name, group_directions in self.performance_directions.items():
            not_empty = sum(directions_nums[group_name].values()) - directions_nums[group_name][none_key]
            directions_nums[group_name][none_key] = (total_notes - not_empty) / total_notes

            for i, key in enumerate(group_directions):
                directions_nums[group_name][(i + 1, key)] /= total_notes

            weights[group_name] = list(directions_nums[group_name].values())

        return directions_nums, weights

    def _get_augmentations(
            self, meta: ScorePerformanceSampleMeta, is_noisy_perf: bool = False
    ) -> TokenSequenceAugmentations | None:
        if meta is None:
            if self.sample and prob2bool(self.augment_performance) and not is_noisy_perf:
                return self.processor.sample_augmentations()
            elif self.sample and self.noisy_performance and is_noisy_perf:
                return self.processor.sample_augmentations(multiplier=self.noise_strength)
            else:
                return None
        elif is_noisy_perf:
            return meta.noisy_augmentations
        else:
            return meta.augmentations

    def _augment_sequence(
            self,
            seq: TokSequence,
            augmentations: TokenSequenceAugmentations | None = None,
            is_perf: bool = True
    ) -> tuple[TokSequence, np.ndarray]:
        if augmentations is None:
            return seq, np.ones_like(seq.ids[:, 0]).astype(bool)

        if not is_perf:
            augmentations = copy.deepcopy(augmentations)
            augmentations.velocity_shift = 0
            augmentations.tempo_shift = 0

        seq = self.processor.augment_sequence(seq, augmentations)
        mask = self.processor.compute_valid_pitch_mask(seq)

        seq.ids = seq.ids[mask]
        seq.values = seq.values[mask] if seq.values is not None else None
        return seq, mask

    def get(self, idx: int | None = None, meta: ScorePerformanceSampleMeta | None = None) -> ScorePerformanceSample:
        assert idx is not None or meta is not None, "one of `idx`/`meta` should be provided as an argument"

        # get performance
        if meta is None:
            perf_idx = np.where(idx >= self._sample_ids)[0][-1]
        else:
            idx, perf_idx = meta.idx, meta.perf_idx
        perf = self.performance_names[perf_idx]
        perf_tok_seq = self.performances[perf_idx]

        # get score
        score, score_perf_idx = self._performance_map[perf]
        score_idx = self.scores._name_to_idx[score]
        score_tok_seq = self.scores[score_idx]

        score_indices = self._score_indices[score_idx]
        if score_indices is None:
            score_indices = self._score_indices[score_idx] = self.indexer.compute_bar_indices(score_tok_seq)
        perf_indices = self._perf_indices[perf_idx]
        if perf_indices is None:
            perf_indices = self._perf_indices[perf_idx] = self.indexer.compute_bar_indices(perf_tok_seq)

        score_total_bars = score_indices.shape[0] - 1
        perf_total_bars = perf_indices.shape[0] - 1
        score_total_notes = score_tok_seq.ids.shape[0]

        # compute start bar index
        if meta is None:
            start_bar = self._sample_positions[idx]
            start_bar = min(start_bar, perf_indices.shape[0] - self.bar_sliding_window // 2)  # bars of silent notes
            if self.sample and prob2bool(self.sample_bars):
                low = max(0, start_bar - self.bar_sliding_window // 2)
                high = min(min(score_total_bars, perf_total_bars) - self.bar_sliding_window // 4,
                           start_bar + self.bar_sliding_window // 2)
                high = max(low + 1, high)
                start_bar = np.random.randint(low, high)
        else:
            start_bar = meta.start_bar

        # compute end bar index
        if meta is None or meta.end_bar is None:
            end_bar = get_end_bar(score_indices, start_bar, self.max_seq_len, self.max_bar)
        else:
            end_bar = meta.end_bar

        # context boundary bars
        ctx_prev_bars, ctx_next_bars = 0, 0
        if self.context_prev_bars != 0 or self.context_next_bars != 0:
            ctx_prev_bars = max(0, min(self.context_prev_bars, start_bar - self.context_prev_bars))
            ctx_next_bars = max(0, min(self.context_next_bars, score_total_bars - end_bar - 1))

            if self.sample:
                ctx_prev_bars = np.random.randint(ctx_prev_bars, self.context_prev_bars + 1)
                ctx_next_bars = np.random.randint(ctx_next_bars, self.context_next_bars + 1)

        # compute start and end indices
        score_start, score_end = score_indices[start_bar], score_indices[end_bar + 1]
        perf_start, perf_end = perf_indices[start_bar], perf_indices[min(end_bar + 1, perf_total_bars)]

        # if bar does not fit or overfits `max_seq_len`
        if score_start == score_end or score_end - score_start > self.max_seq_len:
            score_end = min(score_end, score_start + self.max_seq_len)
            perf_end = min(perf_end, perf_start + self.max_seq_len)

        # sample note shifts to avoid sequence starting from the 1st bar note
        if meta is None:
            start_note_shift = end_note_shift = 0
            if self.sample and prob2bool(self.sample_note_shift):
                max_note_shift = int(self.max_note_shift_ratio * self.max_seq_len)
                score_next_start = score_indices[min(start_bar + 1, score_total_bars)]
                if score_next_start > score_start:
                    start_note_shift = np.random.randint(0, min(score_next_start - score_start, max_note_shift))

                score_prev_end = max(score_indices[max(0, end_bar)], score_start + start_note_shift)
                if score_prev_end - score_end + 1 < 0:
                    end_note_shift = np.random.randint(max(-max_note_shift, score_prev_end - score_end + 1), 1)

            # force `max_seq_len` even if the sequence is shorter (note: for the sequence tail might be no-op)
            if prob2bool(self.force_max_seq_len):
                end_note_shift += min(
                    self.max_seq_len - (score_end + end_note_shift - (score_start + start_note_shift)),
                    score_total_notes - (score_end + end_note_shift),
                    score_indices[min(start_bar + self.max_bar - 1, score_total_bars)] - score_end
                )
        else:
            start_note_shift, end_note_shift = meta.note_shifts

        score_start, perf_start = map(lambda x: x + start_note_shift, (score_start, perf_start))
        score_end, perf_end = map(lambda x: x + end_note_shift, (score_end, perf_end))

        # get token sequences
        score_seq = score_tok_seq[score_start:score_end]

        if self.alignments is not None:
            alignment = self.alignments[perf]
            perf_indices = alignment[np.arange(score_start, score_end)]
            perf_seq = replace(
                perf_tok_seq,
                ids=copy.copy(perf_tok_seq.ids[perf_indices]),
                values=copy.copy(perf_tok_seq.values[perf_indices]) if perf_tok_seq.values is not None else None,
            )
        else:
            perf_seq = perf_tok_seq[perf_start:perf_end]

        _bar_index = self.tokenizer.vocab_types_idx["Bar"]

        min_bar = perf_seq.ids[:, _bar_index].min() - self.tokenizer.zero_token
        min_bar = min(min_bar, score_seq.ids[:, _bar_index].min() - self.tokenizer.zero_token)

        max_bar = perf_seq.ids[:, _bar_index].max() - self.tokenizer.zero_token
        max_bar = max(max_bar, score_seq.ids[:, _bar_index].max() - self.tokenizer.zero_token)

        # bar/beat/onset note maps
        bars, beats, onsets = self.tokenizer.compute_bar_beat_onset_indices(score_tok_seq)
        bars, beats, onsets = map(
            lambda s: s[score_start:score_end] - s[score_start] + self.tokenizer.zero_token,
            (bars, beats, onsets)
        )

        # shift bar indices
        shifts, shift_to_zero = None, False
        if meta is None:
            if self.fit_to_max_bar and score_start != 0:  # do not move for the starting note as it has SOS token
                # to make bar index distribute in [0, bar_max)
                if self.sample and prob2bool(self.sample_bar_shift):
                    shifts = {
                        "Bar": np.random.randint(-min_bar, -min_bar + max(1, self.max_bar - (max_bar - min_bar + 1)))
                    }
                elif max_bar >= self.max_bar:
                    # move in proportion to `score_total_bars`
                    _end_bar = int((self.max_bar - 1) * max_bar / score_total_bars)
                    shifts = {"Bar": _end_bar - max_bar}
            elif self.shift_bar_to_zero:
                shift_to_zero = True
        else:
            shifts = meta.position_shifts

        score_seq, shifts = self.tokenizer.shift_positions(score_seq, shifts=shifts, shift_to_zero=shift_to_zero)
        perf_seq, shifts = self.tokenizer.shift_positions(perf_seq, shifts=shifts)

        # augmentations
        augmentations = self._get_augmentations(meta)
        score_seq, mask = self._augment_sequence(score_seq, augmentations, is_perf=False)
        perf_seq, _ = self._augment_sequence(perf_seq, augmentations, is_perf=True)

        # select subset of segments for left notes
        bars, beats, onsets = map(lambda s: s[mask], (bars, beats, onsets))

        # performance encoding type
        if meta is None:
            use_deadpan = self.sample and prob2bool(self.deadpan_performance)
        else:
            use_deadpan = meta.is_deadpan

        # deadpan performance
        if use_deadpan:
            # all previous performance processing made no sense, we love some deadpan performance
            perf_seq = self.tokenizer.score_tokens_as_performance(score_seq)

        # noisy performance
        noisy_perf_seq = noisy_augmentations = None
        if self.noisy_performance:
            noisy_augmentations = self._get_augmentations(meta, is_noisy_perf=True)
            noisy_perf_seq = replace(perf_seq)
            noisy_perf_seq, _ = self._augment_sequence(noisy_perf_seq, noisy_augmentations, is_perf=True)
            if len(noisy_perf_seq) < len(perf_seq):
                # pitch overflow, omit by reverting changes for now
                noisy_perf_seq = replace(perf_seq)

            if prob2bool(self.noisy_random_bars):
                bar_ids = np.arange(self.max_bar)
                np.random.shuffle(bar_ids)
                bar_0 = self.tokenizer.zero_token
                noisy_perf_seq[:, _bar_index] = bar_ids[noisy_perf_seq[:, _bar_index] - bar_0] + bar_0

        # just a random sequence that might be used during MLM
        random_perf = replace(
            perf_seq,
            ids=np.stack([
                np.random.randint(low=self.tokenizer.zero_token, high=num, size=(len(perf_seq) + 2,))
                for i, num in enumerate(self.tokenizer.performance_sizes.values())
            ], axis=-1),
            values=None
        )

        # compute values if required
        quantize_values = self.sample and prob2bool(self.quantize_values)
        for seq in (score_seq, perf_seq, noisy_perf_seq, random_perf):
            if seq is None:
                continue
            if seq.values is None or quantize_values:
                seq.values = self.tokenizer.decode_values(seq.ids)

        # process values
        for seq in (score_seq, perf_seq, noisy_perf_seq, random_perf):
            if seq is None:
                continue
            if self.clip_values:
                self.tokenizer.clip_values(seq)
            if self.normalize_values:
                self.tokenizer.normalize_values(seq)

        zeros = np.zeros((score_seq.ids.shape[0], perf_tok_seq.ids.shape[1] - score_seq.ids.shape[1]))
        score_seq.ids = np.concatenate([score_seq.ids, zeros.astype(int)], axis=1)
        if score_seq.values is not None:
            score_seq.values = np.concatenate([score_seq.values, zeros], axis=1)
        score_seq = self.processor.to_score_encoding(score_seq)

        # sequence boundaries
        if self.add_sos_eos:
            if score_start == 0:
                score_seq = self.tokenizer.add_sos_token(score_seq)
                perf_seq = self.tokenizer.add_sos_token(perf_seq)
                noisy_perf_seq = self.tokenizer.add_sos_token(noisy_perf_seq) if exists(noisy_perf_seq) else None
                bars, beats, onsets = map(
                    lambda s: np.concatenate([[s[0] if len(s) else self.tokenizer.zero_token], s]),
                    (bars, beats, onsets)
                )
            if score_end == score_total_notes:
                score_seq = self.tokenizer.add_eos_token(score_seq)
                perf_seq = self.tokenizer.add_eos_token(perf_seq)
                noisy_perf_seq = self.tokenizer.add_eos_token(noisy_perf_seq) if exists(noisy_perf_seq) else None
                bars, beats, onsets = map(
                    lambda s: np.concatenate([s, [s[-1] if len(s) else self.tokenizer.zero_token]]),
                    (bars, beats, onsets)
                )

        # note performance direction labels
        directions = {}
        if self.performance_directions is not None:
            score_direction_note_maps = self.score_direction_maps[score_idx]
            for group_name, group_directions in self.performance_directions.items():
                directions[group_name] = {}
                for i, key in enumerate(group_directions):
                    if key in score_direction_note_maps:
                        note_map = copy.copy(score_direction_note_maps[key][score_start:score_end])[mask]
                        if self.add_sos_eos:
                            note_map = np.concatenate([[0], note_map]) if score_start == 0 else note_map
                            note_map = np.concatenate([note_map, [0]]) if score_end == score_total_notes else note_map
                    else:
                        note_map = np.zeros(len(score_seq))
                    directions[group_name][(i + 1, key)] = note_map.astype(int)  # 0 is for None

        # context on sequence boundaries (might be omitted by the models)
        context_offsets = (0, len(score_seq))
        if meta is None:
            if ctx_prev_bars != 0 or ctx_next_bars != 0:
                bars = score_seq.ids[:, _bar_index].copy()
                if score_start == 0:
                    bars[0] = bars[1]
                if score_end == score_total_notes:
                    bars[-1] = bars[-2]

                ctx_start_offset = np.where(bars >= bars[0] + ctx_prev_bars)[0]
                ctx_start_offset = ctx_start_offset[0] if len(ctx_start_offset) else 0
                ctx_start_offset = min(ctx_start_offset, len(score_seq) - 1)

                ctx_end_offset = np.where(bars <= bars[-1] - ctx_next_bars)[0]
                ctx_end_offset = ctx_end_offset[-1] - len(score_seq) + 1 if len(ctx_end_offset) else 0
                ctx_end_offset = len(score_seq) + ctx_end_offset
                ctx_end_offset = min(max(ctx_end_offset, ctx_start_offset + 1), len(score_seq))

                context_offsets = (ctx_start_offset, ctx_end_offset)
        else:
            context_offsets = meta.context_offsets

        # filter token types
        if len(self.score_token_types) != score_seq.ids.shape[-1]:
            self.tokenizer.compress(score_seq, token_types=self.score_token_types)

        for seq in (perf_seq, noisy_perf_seq, random_perf):
            if seq is None:
                continue
            if len(self.performance_token_types) != seq.ids.shape[-1]:
                self.tokenizer.compress(seq, token_types=self.performance_token_types)

        random_perf.ids = random_perf.ids[:len(perf_seq)]
        random_perf.values = random_perf.values[:len(perf_seq)]

        # build sample metadata
        meta = ScorePerformanceSampleMeta(
            idx=idx,
            score_idx=score_idx,
            perf_idx=perf_idx,
            start_bar=start_bar,
            end_bar=end_bar,
            start_idx=score_start,
            end_idx=score_end,
            position_shifts=shifts,
            note_shifts=(start_note_shift, end_note_shift),
            context_offsets=context_offsets,
            augmentations=augmentations,
            noisy_augmentations=noisy_augmentations,
            encoding_type=EncodingType.PERFORMANCE,
            is_deadpan=use_deadpan,
            token_sizes=self.performance_token_sizes
        )

        return ScorePerformanceSample(
            meta=meta,
            score=score_seq,
            perf=perf_seq,
            noisy_perf=noisy_perf_seq,
            segments=NoteSegments(
                bar=bars,
                beat=beats,
                onset=onsets
            ),
            task_idx=SequenceTask.list().index(SequenceTask.PERFORMANCE),
            context_offsets=context_offsets,
            directions=directions,
            is_deadpan=use_deadpan,
            random_perf=random_perf
        )

    def __getitem__(self, idx: int) -> ScorePerformanceSample:
        return self.get(idx=idx)

    def __len__(self) -> int:
        return self._length


class LocalScorePerformanceDataset(ScorePerformanceDataset):
    def __init__(
            self,
            root: str,
            metadata: str = "metadata.json",
            split: str = "train",
            extension: str = ".json",
            tokenizer: str = "config.json",
            score_token_types: list[str] | None = None,
            performance_token_types: list[str] | None = None,
            use_alignments: bool = False,
            auxiliary_data_keys: list[str] | None = None,
            save_auxiliary_data: bool = True,
            performance_directions: str | Path | list[str] | dict[str, list[str]] | None = None,
            score_directions_dict: str | Path | None = None,

            max_seq_len: int = 512,
            max_bar: int = 256,
            bar_sliding_window: int = 16,

            sample_bars: bool | float = False,
            sample_note_shift: bool | float = False,
            max_note_shift_ratio: float = 0.2,
            force_max_seq_len: bool | float = False,

            fit_to_max_bar: bool = False,
            shift_bar_to_zero: bool = False,
            sample_bar_shift: bool | float = False,

            context_prev_bars: int = 0,
            context_next_bars: int = 0,

            add_sos_eos: bool = False,

            sample: bool = False,
            seed: int = 23,

            augment_performance: bool | float = False,
            pitch_shift_range: tuple[int, int] = (-3, 3),
            velocity_shift_range: tuple[int, int] = (-4, 4),
            tempo_stretch_range: tuple[float, float] = (-0.1, 0.1),

            noisy_performance: bool = False,
            noise_strength: float = 0.5,
            noisy_random_bars: bool | float = 0.5,

            deadpan_performance: bool | float = False,

            quantize_values: bool | float = False,
            clip_values: bool = False,
            normalize_values: bool = False,

            zero_out_silent_durations: bool = True,
            delete_silent_notes: bool = False,

            preload: bool = False,
            cache: bool = True,
            **kwargs
    ):

        self.root = root
        self.split = split

        # load metadata
        metadata_file = os.path.join(self.root, metadata)
        metadata = load_json(metadata_file)

        if any(key in metadata for key in DATA_SPLITS):
            metadata = metadata[self.split]

        self.performance_names = list(sorted(set(chain.from_iterable(metadata.values()))))
        self.score_names = list(sorted(metadata.keys()))

        self._performance_map = {
            perf: (score, idx)
            for score, performances in metadata.items()
            for idx, perf in enumerate(performances)
        }

        # perf-to-score alignments
        alignments = None
        if use_alignments:
            alignment_file = os.path.join(self.root, "alignments.json")
            if os.path.exists(alignment_file):
                alignments = {
                    key: np.array(values) for key, values in load_json(alignment_file).items()
                    if key in self._performance_map
                }

        # load tokenizer
        params_path = os.path.join(self.root, tokenizer)
        with open(params_path) as f:
            params = json.load(f)
        encoding = TOKENIZERS[params["tokenization"]]
        tokenizer = encoding(params=params_path)

        # sequence processor for sequence loading
        processor = TupleTokenSequenceProcessor(tokenizer=tokenizer)

        # load sequences
        load_tokens_fn = partial(load_token_sequence, tokenizer=tokenizer)
        seq_proc_funcs, perf_seq_proc_funcs = [], []
        if zero_out_silent_durations:  # silent notes have non-zero duration
            seq_proc_funcs.append(processor.zero_out_durations)
        if delete_silent_notes:  # remove silent notes from performances
            perf_seq_proc_funcs.append(processor.remove_silent_notes)

        score_load_fn = partial(
            load_and_process_token_sequence,
            load_fn=load_tokens_fn,
            processing_funcs=seq_proc_funcs
        )
        scores = LocalTokenSequenceDataset(
            root=self.root,
            files=self.score_names,
            extension=extension,
            load_fn=score_load_fn,
            preload=preload,
            cache=cache
        )

        perf_load_fn = partial(
            load_and_process_token_sequence,
            load_fn=load_tokens_fn,
            processing_funcs=seq_proc_funcs + perf_seq_proc_funcs
        )
        performances = LocalTokenSequenceDataset(
            root=self.root,
            files=self.performance_names,
            extension=extension,
            load_fn=perf_load_fn,
            preload=preload,
            cache=cache
        )

        # load auxiliary data
        auxiliary_data = {}
        if auxiliary_data_keys is not None:
            for key in auxiliary_data_keys:
                data_file = os.path.join(self.root, f"{key}.json")
                if os.path.exists(data_file):
                    auxiliary_data[key] = load_json(data_file)

        super().__init__(
            scores=scores,
            performances=performances,
            metadata=metadata,
            tokenizer=tokenizer,
            score_token_types=score_token_types,
            performance_token_types=performance_token_types,
            alignments=alignments,
            auxiliary_data=auxiliary_data,
            performance_directions=performance_directions,
            score_directions_dict=score_directions_dict,
            max_seq_len=max_seq_len,
            max_bar=max_bar,
            bar_sliding_window=bar_sliding_window,
            sample_bars=sample_bars,
            sample_note_shift=sample_note_shift,
            max_note_shift_ratio=max_note_shift_ratio,
            force_max_seq_len=force_max_seq_len,
            fit_to_max_bar=fit_to_max_bar,
            shift_bar_to_zero=shift_bar_to_zero,
            sample_bar_shift=sample_bar_shift,
            context_prev_bars=context_prev_bars,
            context_next_bars=context_next_bars,
            add_sos_eos=add_sos_eos,
            sample=sample,
            seed=seed,
            augment_performance=augment_performance,
            pitch_shift_range=pitch_shift_range,
            velocity_shift_range=velocity_shift_range,
            tempo_stretch_range=tempo_stretch_range,
            noisy_performance=noisy_performance,
            noise_strength=noise_strength,
            noisy_random_bars=noisy_random_bars,
            deadpan_performance=deadpan_performance,
            quantize_values=quantize_values,
            clip_values=clip_values,
            normalize_values=normalize_values
        )

        if save_auxiliary_data:
            for key in auxiliary_data_keys:
                data_file = os.path.join(self.root, f"{key}.json")
                data = getattr(self, key, None)
                if data is not None:
                    old_data = load_json(data_file) if os.path.exists(data_file) else {}
                    data.update(**old_data)
                    if len(data) != len(old_data):
                        dump_json(data, data_file)

        for score in self.score_names:
            assert score in self.scores._name_to_idx, score
