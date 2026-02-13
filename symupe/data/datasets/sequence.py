""" Any token sequence dataset. """
from __future__ import annotations

import copy
import itertools
import json
import os
import random
from dataclasses import dataclass, replace
from functools import partial

import numpy as np
from torch.utils.data import Dataset

from symupe.utils import prob2bool, load_json, dump_json, tqdm_iterator, find_closest
from .classifiers import EmotionPredictionsDataset, LocalEmotionPredictionsDataset
from .common import NoteSegments, SequenceTask, TASK_TO_ENCODINGS, DATA_SPLITS, TASK_SEQUENCE_SORTING
from .token_sequence import load_and_process_token_sequence, LocalTokenSequenceDataset, TokenSequenceDataset
from .utils import compute_sample_positions, load_token_sequence
from ..helpers import (
    TupleTokenSequenceProcessor,
    TokenSequenceAugmentations
)
from ..tokenizers import (
    TokSequence, SequenceType, EncodingType, SortingType, ENCODING_SORTING,
    TOKENIZERS, SyMuPe, SyMuPeTransformer
)
from ..tokenizers.constants import EOD_TOKEN, BAR_LINE_TOKEN


@dataclass
class SequenceSampleMeta:
    idx: int | None
    seq_idx: int
    name: str
    start: int
    end: int | None
    position_shifts: dict[str, int | float] | None = None
    augmentations: TokenSequenceAugmentations | None = None
    sequence_type: str | SequenceType = SequenceType.SCORE
    encoding_type: tuple[str, str | None] | tuple[EncodingType, EncodingType | None] = EncodingType.SCORE
    task_type: str | SequenceTask = SequenceTask.SCORE
    sort_by_time: tuple[bool, bool] = False
    token_sizes: dict[str, int] | None = None
    sequence_task_types: dict[str, int] | None = None


@dataclass
class SequenceSample:
    meta: SequenceSampleMeta
    seq: TokSequence
    target_seq: TokSequence | None = None
    segments: NoteSegments | None = None
    pedals: np.ndarray | None = None
    seq_label: int | None = None
    type_ids: np.ndarray | None = None
    encoding_ids: tuple[int, int] | None = None
    task_idx: int = 0
    task_seq: TokSequence | None = None
    enc_seq: TokSequence | None = None
    score_seq: TokSequence | None = None
    context_seq: TokSequence | None = None
    emotion_labels: np.ndarray | None = None
    emotion_embeddings: np.ndarray | None = None
    emotion_template_idx: int | None = None
    random_seq: TokSequence | None = None


class SequenceDataset(Dataset):
    def __init__(
            self,
            sequences: TokenSequenceDataset,
            metadata: dict[str, list[str]],
            tokenizer: SyMuPe | dict[str, object],
            token_types: list[str] | None = None,
            context_token_types: list[str] | None = None,
            score_sequences: TokenSequenceDataset | None = None,
            score_token_types: list[str] | None = None,
            emotion_dataset: EmotionPredictionsDataset | None = None,
            auxiliary_data: dict[str, object] | None = None,

            max_seq_len: int = 512,
            min_seq_len: int = 16,
            note_sliding_window: int = 128,
            max_bar: int = 512,
            max_time: float = 1_000.,
            max_len_single_seq: bool = True,

            sample_start_index: bool | float = False,
            sample_length: bool | float = False,

            shift_time_to_zero: bool | float = False,

            sort_by_time: bool | float = False,
            sample_sort: bool = False,

            add_sos_eos: bool = False,
            add_eod_token: bool = False,
            remove_special_tokens: bool = False,
            enc_seq_minimal: bool = False,
            eod_same_encoding: bool = False,

            sample: bool = False,
            seed: int = 23,
            shuffle: bool = False,

            augment_sequence: bool = False,
            pitch_shift_range: tuple[int, int] = (-3, 3),
            velocity_shift_range: tuple[int, int] = (-4, 4),
            tempo_stretch_range: tuple[float, float] = (-0.1, 0.1),

            sequence_tasks: dict[str, dict[str, float]] | None = None,

            quantize_values: bool | float = False,
            clip_values: bool = False,
            normalize_values: bool = False,

            compute_note_types: bool = False,

            **kwargs
    ):
        self.metadata = metadata

        self.sequence_names = list(self.metadata)
        self.sequences = sequences

        # load tokenizer
        if isinstance(tokenizer, dict):
            encoding = TOKENIZERS[tokenizer["tokenization"]]
            self.tokenizer = encoding(params=tokenizer)
        else:
            self.tokenizer = tokenizer
        self.encoding = self.tokenizer.__class__.__name__

        self.token_types = token_types or list(self.tokenizer.sizes.keys())
        self.token_sizes = {key: self.tokenizer.sizes[key] for key in self.token_types}

        self.context_token_types = context_token_types or None
        self.context_token_sizes = {
            key: num for key, num in self.tokenizer.sizes.items()
            if key in self.context_token_types
        } if self.context_token_types is not None else None

        self.score_token_types = (score_token_types or None) if score_sequences is not None else None
        self.score_token_sizes = {
            key: num for key, num in self.tokenizer.sizes.items()
            if key in self.score_token_types
        } if self.score_token_types is not None else None

        # augmentations
        self.augment_sequence = augment_sequence

        # sequence processors
        self.processor = TupleTokenSequenceProcessor(
            tokenizer=self.tokenizer,
            pitch_shift_range=pitch_shift_range,
            velocity_shift_range=velocity_shift_range,
            tempo_stretch_range=tempo_stretch_range
        )
        self.transformer = SyMuPeTransformer(
            tokenizer=self.tokenizer
        )

        # score sequence dataset
        self.score_sequences = score_sequences
        if score_sequences is not None:
            assert remove_special_tokens

        # emotion predictions dataset
        self.emotion_dataset = emotion_dataset

        # set up auxiliary data
        if auxiliary_data is not None:
            for key, data in auxiliary_data.items():
                setattr(self, key, data)

        # configurations
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.note_sliding_window = note_sliding_window
        self.max_bar = max_bar
        self.max_time = max_time
        self.max_len_single_seq = max_len_single_seq
        assert max_bar <= self.tokenizer.config.additional_params["max_bar_embedding"]
        self.tokenizer.config.additional_params["time_position_segment"] = max_time

        self.add_sos_eos = add_sos_eos
        self.add_eod_token = add_eod_token
        self.remove_special_tokens = remove_special_tokens
        self.enc_seq_minimal = enc_seq_minimal
        self.eod_same_encoding = eod_same_encoding

        # load or compute number of notes in sequences used to build samples
        self.lengths = getattr(self, "lengths", {})
        for seq_idx, name in enumerate(tqdm_iterator(self.sequence_names, desc="Precomputing lengths...")):
            if name not in self.lengths:
                self.lengths[name] = len(self.sequences[seq_idx])
        _num_notes = np.array([self.lengths[name] for name in self.sequence_names])

        # compute sample positions
        self._length, self._sample_positions, self._sample_ids = compute_sample_positions(
            seq_num_elements=_num_notes, sliding_window=self.note_sliding_window, backward=False
        )

        # random effects they do not advertise
        self.sample = sample
        if self.sample:
            random.seed(seed)
            np.random.seed(seed)

        # dataset shuffling
        self._shuffle = None
        if shuffle:
            self.shuffle(seed=seed)

        # sequence sampling
        self.sample_start_index = sample_start_index
        self.sample_length = sample_length

        # time of the first note
        self.shift_time_to_zero = shift_time_to_zero

        # sort by
        self.sort_by_time = sort_by_time
        self.sample_sort = sample_sort

        # sequence task probabilities
        sequence_task_types, sequence_encoding_types = [], []
        if sequence_tasks is not None:
            for seq_type, task_types in sequence_tasks.items():
                for task_type in task_types:
                    assert SequenceTask.has_value(task_type)
                    if task_type not in sequence_task_types:
                        sequence_task_types.append(task_type)
                    sequence_encoding_types.extend(list(TASK_TO_ENCODINGS[task_type]))

        self.sequence_tasks = sequence_tasks
        self.sequence_task_types = {task_type: i for i, task_type in enumerate(sequence_task_types)}

        sequence_encoding_types = sorted(
            filter(lambda x: x is not None, list(set(sequence_encoding_types))),
            key=lambda x: EncodingType.list().index(x.value)
        )
        self.sequence_encoding_types = {encoding.value: i for i, encoding in enumerate(sequence_encoding_types)}

        # values processing
        self.quantize_values = quantize_values
        self.clip_values = clip_values
        self.normalize_values = normalize_values

        self.compute_note_types = compute_note_types

    def _get_augmentations(
            self,
            meta: SequenceSampleMeta,
            min_pitch: int | None = None,
            max_pitch: int | None = None
    ) -> TokenSequenceAugmentations | None:
        if meta is None:
            if self.sample and prob2bool(self.augment_sequence):
                return self.processor.sample_augmentations(min_pitch=min_pitch, max_pitch=max_pitch)
            else:
                return None
        else:
            return meta.augmentations

    def _augment_sequence(
            self,
            seq: TokSequence,
            augmentations: TokenSequenceAugmentations | None = None
    ) -> tuple[TokSequence, np.ndarray]:
        if augmentations is None:
            return seq, np.ones_like(seq.ids[:, 0]).astype(bool)

        seq = self.processor.augment_sequence(seq, augmentations)
        mask = self.processor.compute_valid_pitch_mask(seq)

        seq.ids = seq.ids[mask]
        seq.values = seq.values[mask] if seq.values is not None else None
        seq.interpolated = seq.interpolated[mask] if seq.interpolated is not None else None
        return seq, mask

    def get(self, idx: int | None = None, meta: SequenceSampleMeta | None = None) -> SequenceSample:
        assert idx is not None or meta is not None, "one of `idx`/`meta` should be provided as an argument"

        if self._shuffle is not None:
            idx = self._shuffle[idx]

        _bar_index = self.tokenizer.vocab_types_idx["Bar"]
        # use_time_positions = self.tokenizer.config.additional_params.get("use_time_positions", False)
        # _time_index = self.tokenizer.vocab_types_idx["TimePosition"] if use_time_positions else None

        # get sequence
        if meta is None:
            seq_idx = np.where(idx >= self._sample_ids)[0][-1]
        else:
            idx, seq_idx = meta.idx, meta.seq_idx

        name = self.sequence_names[seq_idx]
        tok_seq = self.sequences[seq_idx]
        seq_type = tok_seq.type

        # get score sequence if available
        score_tok_seq = None
        if self.score_token_types is not None and self.score_sequences is not None:
            score_name = os.path.dirname(name) + ("/mini" if "_mini" in name else "")
            score_seq_idx = self.score_sequences._name_to_idx.get(score_name, None)
            if score_seq_idx is not None:
                score_tok_seq = self.score_sequences[score_seq_idx]

        # sample sequence encoding if needed
        if meta is None:
            if self.sequence_tasks is not None:
                assert seq_type in self.sequence_tasks
                sequence_tasks = self.sequence_tasks[seq_type]
                probs = list(sequence_tasks.values())
                task_type = list(sequence_tasks)[random.choices(list(range(len(probs))), weights=probs)[0]]
            else:
                task_type = SequenceTask.SCORE
                if seq_type == SequenceType.PERFORMANCE:
                    task_type = SequenceTask.PERFORMANCE
                elif seq_type == SequenceType.TIME_PERFORMANCE:
                    task_type = SequenceTask.TIME_PERFORMANCE
        else:
            task_type = meta.task_type

        # get sequence encodings
        source_encoding, target_encoding = TASK_TO_ENCODINGS[task_type]
        target_encoding = source_encoding if target_encoding is None else target_encoding

        # decompress token sequence
        tok_seq = self.tokenizer.decompress(copy.deepcopy(tok_seq))
        seq_total_notes = len(tok_seq)

        # compute start index
        if meta is None:
            start = self._sample_positions[idx]
            start = max(0, min(start, seq_total_notes - self.min_seq_len))
            sample_start_index = self.sample_start_index if start > 0 else float(self.sample_start_index) / 2
            if self.sample and prob2bool(sample_start_index):
                low = max(0, start - self.note_sliding_window // 2)
                high = min(seq_total_notes - self.note_sliding_window // 4, start + self.note_sliding_window // 2)
                high = max(low + 1, high)
                start = np.random.randint(low, high)
        else:
            start = meta.start

        has_bars = tok_seq.type not in (SequenceType.TIME_PERFORMANCE, SequenceType.TIME_PERFORMANCE_SUSTAIN)
        # has_time = _time_index is not None and tok_seq.type != SequenceType.SCORE

        # compute end index
        if meta is None or meta.end is None:
            seq_len = self.max_seq_len

            if source_encoding == target_encoding and not self.max_len_single_seq:
                seq_len *= 2

            # adjust maximum sequence length by the removed special tokens
            seq_len_adjusted = self.transformer.adjust_seq_len_for_special_tokens(
                tok_seq, encodings=(EncodingType.SCORE_TIME_PERFORMANCE, source_encoding, target_encoding),
                offset=start, seq_len=seq_len,
            )

            if self.remove_special_tokens:
                seq_len = seq_len_adjusted[EncodingType.SCORE_TIME_PERFORMANCE]
            else:
                seq_len = min(seq_len_adjusted[source_encoding], seq_len_adjusted[target_encoding])

            if has_bars:
                max_seq_len = np.where(
                    tok_seq.ids[start:, _bar_index] > tok_seq.ids[start, _bar_index] + self.max_bar - 1
                )[0]
                seq_len = min(seq_len, max_seq_len[0]) if len(max_seq_len) > 0 else seq_len
            # if has_time:
            #     max_seq_len = np.where(
            #         tok_seq.ids[start:, _time_index] > tok_seq.ids[start, _time_index] + self.max_time - 1
            #     )[0]
            #     seq_len = min(seq_len, max_seq_len[0]) if len(max_seq_len) > 0 else seq_len

            if self.sample and prob2bool(self.sample_length) and seq_len >= self.min_seq_len:
                seq_len = np.random.randint(self.min_seq_len, seq_len + 1)
            end = min(seq_total_notes, start + seq_len)
        else:
            end = meta.end

        bar_end = (
                end == seq_total_notes
                or tok_seq.ids[min(seq_total_notes - 1, end - 1), _bar_index] != tok_seq.ids[end, _bar_index]
        )

        # get subsequence
        seq = tok_seq[start:end]
        seq.pedals = None

        # remove bar line and pedal tokens if present
        pedals = None
        if self.remove_special_tokens:
            seq = self.tokenizer.add_time_position_tokens(seq, segment_tokens=False)
            seq = self.transformer.remove_special_tokens(seq, force=True)
            pedals = seq.pedals

            if pedals is not None:
                pedals[:, 1] = np.diff(np.concatenate([[0.], pedals[:, 1]]))
        else:
            seq = self.tokenizer.add_time_position_tokens(seq)

        if len(seq) == 0:
            rand_idx = np.random.randint(0, len(self))
            return self.get(rand_idx)

        # slice score sequence
        score_seq = None
        if score_tok_seq is not None:
            seq_index = seq.ids[:, :3]
            seq_index = 1_000_000 * seq_index[:, 0] + 1_000 * seq_index[:, 1] + seq_index[:, 2]
            score_index = score_tok_seq.ids[:, :3]
            score_index = 1_000_000 * score_index[:, 0] + 1_000 * score_index[:, 1] + score_index[:, 2]
            index = find_closest(score_index, seq_index)

            score_seq = replace(score_tok_seq, ids=score_tok_seq.ids[index], values=score_tok_seq.values[index])

            if len(score_seq) != len(seq):
                rand_idx = np.random.randint(0, len(self))
                return self.get(rand_idx)

        # bars that might be used after the sequence encoding
        bars_init = None
        if self.emotion_dataset is not None and seq_type in (SequenceType.PERFORMANCE, SequenceType.PERFORMANCE_SUSTAIN):
            bars_init, _, _ = self.tokenizer.compute_bar_beat_onset_indices(seq)

        # transform sequence encodings
        source_seq = self.transformer(copy.deepcopy(seq), encoding=source_encoding)
        target_seq = self.transformer(copy.deepcopy(seq), encoding=target_encoding)

        pitches = self.tokenizer.get_values(source_seq, "Pitch", from_ids=True)
        pitches = pitches[pitches >= 0]
        if len(pitches) == 0 or len(source_seq) == 0 or len(target_seq) == 0:
            rand_idx = np.random.randint(0, len(self))
            return self.get(rand_idx)

        if len(source_seq) != len(seq):
            bars_init = None

        def sort_sequence(tok_sequence, encoding_type):
            if task_type in TASK_SEQUENCE_SORTING[SortingType.SCORE] or encoding_type in ENCODING_SORTING[SortingType.SCORE]:
                sort_by_time = False
            elif task_type in TASK_SEQUENCE_SORTING[SortingType.TIME] or encoding_type in ENCODING_SORTING[SortingType.TIME]:
                sort_by_time = True
            elif self.sample and self.sample_sort:
                sort_by_time = prob2bool(self.sort_by_time)
            else:
                sort_by_time = self.sort_by_time > 0.5

            is_sort_by_time = seq_type in (SequenceType.TIME_PERFORMANCE, SequenceType.TIME_PERFORMANCE_SUSTAIN)
            if is_sort_by_time != sort_by_time:
                tok_sequence = self.tokenizer.sort_tokens(tok_sequence, by_time=sort_by_time)

            return tok_sequence, sort_by_time

        source_seq, source_sort_by_time = sort_sequence(source_seq, source_encoding)
        target_seq, target_sort_by_time = sort_sequence(target_seq, target_encoding)

        # check open/closed pedals
        if source_encoding in ENCODING_SORTING[SortingType.TIME]:
            source_seq = self.tokenizer.add_artificial_pedal_on(source_seq)
        if target_encoding in ENCODING_SORTING[SortingType.TIME]:
            target_seq = self.tokenizer.add_artificial_pedal_on(target_seq)

        # shift bar indices
        shifts, shift_to_zero = None, False
        if meta is None:
            shift_to_zero = self.shift_time_to_zero or start != 0
        else:
            shifts = meta.position_shifts

        source_seq, shifts = self.tokenizer.shift_positions(source_seq, shifts=shifts, shift_to_zero=shift_to_zero)
        target_seq, _ = self.tokenizer.shift_positions(target_seq, shifts=None, shift_to_zero=shift_to_zero)

        # augmentations
        pitches = self.tokenizer.get_values(source_seq, "Pitch", from_ids=True)
        pitches = pitches[pitches >= 0]
        min_pitch, max_pitch = (pitches.min(), pitches.max()) if np.any(pitches) else (None, None)
        augmentations = self._get_augmentations(meta, min_pitch=min_pitch, max_pitch=max_pitch)
        source_seq, mask = self._augment_sequence(source_seq, augmentations)
        target_seq, _ = self._augment_sequence(target_seq, augmentations)

        # bar/beat/onset note maps
        has_bars = source_encoding != EncodingType.TIME_PERFORMANCE
        if bars_init is not None:
            bars_init = bars_init[mask]
        elif has_bars and not source_sort_by_time:
            bars_init, _, _ = self.tokenizer.compute_bar_beat_onset_indices(source_seq)
            bars_init -= shifts["Bar"]

        # # map score sequence
        # if score_seq is not None:
        #     score_seq.ids = score_seq.ids[mask]
        #     score_seq.values = score_seq.values[mask] if score_seq.values is not None else None

        # emotion embeddings
        emotion_embeddings, emotion_template_idx = None, 0
        if self.emotion_dataset is not None:
            label_embeddings = self.emotion_dataset.label_embeddings

            emotion_template_idx = 0
            if label_embeddings.ndim == 3:
                emotion_template_idx = random.randint(0, label_embeddings.shape[0] - 1)
                label_embeddings = label_embeddings[emotion_template_idx]

            emotion_bar_probs = self.emotion_dataset[name] if bars_init is not None else None
            if emotion_bar_probs is not None:
                emotion_probs = emotion_bar_probs[bars_init]
                emotion_embeddings = emotion_probs @ label_embeddings
            else:
                emotion_template_idx = -1
                emotion_embeddings = np.zeros((len(source_seq), label_embeddings.shape[-1]))

        # compute values if required
        quantize_values = self.sample and prob2bool(self.quantize_values)
        for _seq in (source_seq, target_seq):
            if _seq.values is None or quantize_values:
                _seq.values = self.tokenizer.decode_values(_seq.ids)

        # bar start tokens for scores
        if BAR_LINE_TOKEN in self.tokenizer.special_tokens and not self.remove_special_tokens:
            if not bar_end and not source_sort_by_time and source_encoding in ENCODING_SORTING[SortingType.SCORE]:
                source_seq = source_seq[:-1]
            if not bar_end and not target_sort_by_time and target_encoding in ENCODING_SORTING[SortingType.SCORE]:
                target_seq = target_seq[:-1]

        if has_bars and not source_sort_by_time:
            bars, beats, onsets = self.tokenizer.compute_bar_beat_onset_indices(source_seq)
        else:
            bars, beats, onsets = self.tokenizer.compute_time_bar_beat_onset_indices(source_seq)

        # slice and shift bar/beat/onset maps
        bars, beats, onsets = map(lambda s: s - s.min() + self.tokenizer.zero_token, (bars, beats, onsets))

        # process values
        for _seq in (source_seq, target_seq, score_seq):
            if _seq is None:
                continue
            if self.clip_values:
                _seq.values = self.tokenizer.clip_values(_seq.values)
            if self.normalize_values:
                _seq.values = self.tokenizer.normalize_values(_seq.values)

        if pedals is not None and self.normalize_values:
            pedals[:, 1] = self.tokenizer.normalize_values(pedals[:, 1], "TimeShift")

        # process interpolated masks
        for _seq in (source_seq, target_seq):
            if _seq.interpolated is None:
                _seq.interpolated = np.zeros_like(_seq.values[:, 0])

        # sequence boundaries
        num_sos, num_eos = 0, 0

        if self.add_sos_eos:
            if start == 0:
                source_seq = self.tokenizer.add_sos_token(source_seq)
                target_seq = self.tokenizer.add_sos_token(target_seq)
                if score_seq is not None:
                    score_seq = self.tokenizer.add_sos_token(score_seq)
                num_sos += 1

            if end == seq_total_notes:
                source_seq = self.tokenizer.add_eos_token(source_seq)
                target_seq = self.tokenizer.add_eos_token(target_seq)
                if score_seq is not None:
                    score_seq = self.tokenizer.add_eos_token(score_seq)
                num_eos += 1

        if self.add_eod_token and (self.eod_same_encoding or source_encoding != target_encoding):
            source_seq = self.tokenizer.add_eos_token(source_seq, token_name=EOD_TOKEN)
            target_seq = self.tokenizer.add_eos_token(target_seq, token_name=EOD_TOKEN)
            if score_seq is not None:
                score_seq = self.tokenizer.add_eos_token(score_seq, token_name=EOD_TOKEN)
            num_eos += 1

        # process segments and emotions
        if num_sos + num_eos > 0:
            bars, beats, onsets = map(
                lambda s: np.concatenate([
                    [s[0] if len(s) > 0 else self.tokenizer.zero_token] * num_sos,
                    s,
                    [s[-1] if len(s) > 0 else self.tokenizer.zero_token] * num_eos,
                ]),
                (bars, beats, onsets)
            )
            if emotion_embeddings is not None:
                emotion_embeddings = np.concatenate([
                    np.zeros((num_sos, emotion_embeddings.shape[1])),
                    emotion_embeddings,
                    np.zeros((num_eos, emotion_embeddings.shape[1]))
                ], axis=0)

        # filter score token types
        if score_seq is not None and self.score_token_types:
            score_seq = self.tokenizer.compress(replace(score_seq), token_types=self.score_token_types)

        # filter token types
        context_seq = None
        if self.context_token_types is not None:
            context_seq = self.tokenizer.compress(replace(target_seq), token_types=self.context_token_types)

        # just a random sequence that might be used during MLM
        random_seq = replace(
            source_seq,
            ids=np.stack([
                np.random.randint(low=self.tokenizer.zero_token, high=num, size=(len(source_seq),))
                for i, num in enumerate(self.tokenizer.sizes.values())
            ], axis=-1),
            values=None
        )
        random_seq.values = self.tokenizer.decode_values(random_seq.ids)

        for _seq in (source_seq, target_seq, random_seq):
            if len(self.token_types) != _seq.ids.shape[-1]:
                self.tokenizer.compress(_seq, token_types=self.token_types)

        # task seq
        enc_seq = self.transformer.get_encoding_template(
            encoding=source_encoding, token_types=self.token_types, minimal=self.enc_seq_minimal
        )
        task_seq = self.transformer.get_encoding_template(
            encoding=target_encoding, token_types=self.token_types, minimal=self.enc_seq_minimal
        )

        # type ids
        label = self.labels.get(name, None) if hasattr(self, "labels") else None
        seq_label = self.processor.compute_seq_type_id(target_seq, name=name, label=label)

        type_ids = self.processor.compute_note_type_ids(
            target_seq, name=name, is_interpolated=not self.compute_note_types
        )

        # build sample metadata
        meta = SequenceSampleMeta(
            idx=idx,
            seq_idx=seq_idx,
            name=name,
            start=start,
            end=end,
            position_shifts=shifts,
            augmentations=augmentations,
            sequence_type=seq_type,
            encoding_type=(source_encoding, target_encoding),
            task_type=task_type,
            sort_by_time=(source_sort_by_time, target_sort_by_time),
            token_sizes=self.token_sizes,
            sequence_task_types=self.sequence_task_types
        )

        emotion_labels = source_seq.meta.get("label_probs", None)
        emotion_labels = np.array(emotion_labels).squeeze() if emotion_labels is not None else None

        return SequenceSample(
            meta=meta,
            seq=source_seq,
            target_seq=target_seq,
            segments=NoteSegments(
                bar=bars,
                beat=beats,
                onset=onsets
            ),
            pedals=pedals,
            seq_label=seq_label,
            type_ids=type_ids,
            encoding_ids=(
                self.sequence_encoding_types[source_encoding],
                self.sequence_encoding_types[target_encoding]
            ),
            task_idx=self.sequence_task_types[task_type],
            task_seq=task_seq,
            enc_seq=enc_seq,
            score_seq=score_seq,
            context_seq=context_seq,
            emotion_labels=emotion_labels,
            emotion_embeddings=emotion_embeddings,
            emotion_template_idx=emotion_template_idx,
            random_seq=random_seq
        )

    def __getitem__(self, idx: int) -> SequenceSample:
        return self.get(idx=idx)

    def __len__(self) -> int:
        return self._length

    def shuffle(self, seed: int | None = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._shuffle = np.arange(len(self))
        np.random.shuffle(self._shuffle)


class LocalSequenceDataset(SequenceDataset):
    def __init__(
            self,
            root: str,
            metadata: str = "metadata.json",
            split: str = "train",
            extension: str = ".json",
            tokenizer: str = "config.json",
            token_types: list[str] | None = None,
            context_token_types: list[str] | None = None,

            scores_root: str | None = None,
            scores_metadata: str | None = None,
            score_token_types: list[str] | None = None,

            emotions_root: str | None = None,
            emotion_embeddings: str = "label_embeddings.npy",

            auxiliary_data_keys: list[str] | None = None,
            save_auxiliary_data: bool = True,

            max_seq_len: int = 512,
            min_seq_len: int = 16,
            note_sliding_window: int = 128,
            max_bar: int = 512,
            max_time: float = 10_000.,
            max_len_single_seq: bool = True,

            sample_start_index: bool | float = False,
            sample_length: bool | float = False,

            shift_time_to_zero: bool = False,

            sort_by_time: bool | float = False,
            sample_sort: bool = False,

            add_sos_eos: bool = False,
            add_eod_token: bool = False,
            remove_special_tokens: bool = False,
            enc_seq_minimal: bool = False,

            sample: bool = False,
            seed: int = 23,
            shuffle: bool = False,

            augment_sequence: bool | float = False,
            pitch_shift_range: tuple[int, int] = (-3, 3),
            velocity_shift_range: tuple[int, int] = (-6, 6),
            tempo_stretch_range: tuple[float, float] = (-0., 0.),

            sequence_tasks: dict[str, float] | None = None,

            quantize_values: bool | float = False,
            clip_values: bool = False,
            normalize_values: bool = False,

            zero_out_silent_durations: bool = True,
            delete_silent_notes: bool = False,
            compute_note_types: bool = False,

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
            if isinstance(metadata, dict):
                metadata = list(metadata.keys()) + list(itertools.chain.from_iterable(metadata.values()))

        self.sequence_names = list(metadata)

        # load tokenizer
        params_path = os.path.join(self.root, tokenizer)
        with open(params_path) as f:
            params = json.load(f)
        encoding = TOKENIZERS[params["tokenization"]]
        tokenizer = encoding(params=params_path)

        # sequence processor for sequence loading
        processor = TupleTokenSequenceProcessor(tokenizer=tokenizer)

        # load sequences
        token_types = token_types or []
        context_token_types = context_token_types or []
        load_tokens_fn = partial(
            load_token_sequence,
            tokenizer=tokenizer
        )
        seq_proc_funcs, seq_proc_funcs = [], []
        if zero_out_silent_durations:  # silent notes have non-zero duration
            seq_proc_funcs.append(processor.zero_out_durations)
        if delete_silent_notes:  # remove silent notes from sequences
            seq_proc_funcs.append(processor.remove_silent_notes)

        seq_load_fn = partial(
            load_and_process_token_sequence,
            load_fn=load_tokens_fn,
            processing_funcs=seq_proc_funcs
        )
        sequences = LocalTokenSequenceDataset(
            root=self.root,
            files=self.sequence_names,
            extension=extension,
            load_fn=seq_load_fn,
            preload=preload,
            cache=cache
        )

        # build score sequence dataset if available
        score_sequences = None
        if scores_root is not None:
            scores_metadata_file = os.path.join(scores_root, scores_metadata)
            scores_metadata = load_json(scores_metadata_file)

            if any(key in scores_metadata for key in DATA_SPLITS):
                scores_metadata = scores_metadata[self.split]

            score_sequence_names = list(scores_metadata)

            score_sequences = LocalTokenSequenceDataset(
                root=scores_root,
                files=score_sequence_names,
                extension=extension,
                load_fn=seq_load_fn,
                name_map=lambda x: os.path.dirname(x) + ("/mini" if "_mini" in os.path.basename(x) else ""),
                preload=preload,
                cache=cache
            )

        # load emotion predictions if available
        emotion_dataset = None
        if emotions_root is not None:
            emotion_dataset = LocalEmotionPredictionsDataset(
                root=emotions_root,
                files=self.sequence_names,
                label_embeddings_file=emotion_embeddings,
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
            sequences=sequences,
            metadata=metadata,
            tokenizer=tokenizer,
            token_types=token_types,
            context_token_types=context_token_types,
            score_sequences=score_sequences,
            score_token_types=score_token_types,
            emotion_dataset=emotion_dataset,
            auxiliary_data=auxiliary_data,
            max_seq_len=max_seq_len,
            min_seq_len=min_seq_len,
            note_sliding_window=note_sliding_window,
            max_bar=max_bar,
            max_time=max_time,
            max_len_single_seq=max_len_single_seq,
            sample_start_index=sample_start_index,
            sample_length=sample_length,
            shift_time_to_zero=shift_time_to_zero,
            sort_by_time=sort_by_time,
            sample_sort=sample_sort,
            add_sos_eos=add_sos_eos,
            add_eod_token=add_eod_token,
            remove_special_tokens=remove_special_tokens,
            enc_seq_minimal=enc_seq_minimal,
            sample=sample,
            seed=seed,
            shuffle=shuffle,
            augment_sequence=augment_sequence,
            pitch_shift_range=pitch_shift_range,
            velocity_shift_range=velocity_shift_range,
            tempo_stretch_range=tempo_stretch_range,
            sequence_tasks=sequence_tasks,
            quantize_values=quantize_values,
            clip_values=clip_values,
            normalize_values=normalize_values,
            compute_note_types=compute_note_types
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
