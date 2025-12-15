""" MusicTransformer models' inference modules. """
from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from symupe.data.datasets import SequenceDataset, SequenceTask
from symupe.data.tokenizers import TokSequence, TokSequenceContext, EncodingType, SyMuPe, SyMuPeTransformer
from symupe.data.tokenizers.constants import SOS_TOKEN, EOS_TOKEN, MASK_TOKEN, SPECIAL_TOKENS_VALUE
from symupe.modules.tuple_transformer.flow_matching import resample
from .model import MusicTransformer, CFMMusicTransformer


@dataclass
class NoteMetadata:
    index: np.ndarray | None = None
    onset: np.ndarray | None = None
    beat: np.ndarray | None = None
    bar: np.ndarray | None = None
    tempo: np.ndarray | None = None

    def __getitem__(self, val: slice | int) -> NoteMetadata | dict:
        """
        Return the ``idx``th element or slice of the sequence.

        If an integer is providing, it checks by order: ids, tokens, events, bytes.

        :param val: index of the element to retrieve.
        :return: ``idx``th element.
        """
        if isinstance(val, slice):
            return self.__slice(val)

        attributes = ["index", "onset", "beat", "bar", "tempo"]
        data_dict = {}
        for attr in attributes:
            data = getattr(self, attr)
            if data is not None and len(data) > 0:
                data_dict[attr] = data[val]
            else:
                data_dict[attr] = None
        return data_dict

    def __slice(self, sli: slice) -> NoteMetadata:
        meta = replace(self)
        attributes = ["index", "onset", "beat", "bar", "tempo"]
        for attr in attributes:
            data = getattr(self, attr)
            if data is not None and len(data) > 0:
                setattr(meta, attr, data[sli])
        return meta

    def __add__(self, other: NoteMetadata) -> NoteMetadata:
        """
        Concatenate two ``TokSequence`` objects.

        The `ìds``, ``tokens``, ``events`` and ``bytes`` will be concatenated.

        :param other: other ``TokSequence``.
        :return: the two sequences concatenated.
        """
        meta = replace(self)

        if not isinstance(other, NoteMetadata):
            msg = (
                "Addition to a `NoteMetadata` object can only be performed with other"
                f"`NoteMetadata` objects. Received: {other.__class__.__name__}"
            )
            raise ValueError(msg)

        attributes = ["index", "onset", "beat", "bar", "tempo"]
        for attr in attributes:
            self_attr, other_attr = getattr(self, attr), getattr(other, attr)
            if self_attr is not None and other_attr is not None:
                new_attr = np.concatenate([self_attr, other_attr])
                setattr(meta, attr, new_attr)

        return meta

    def __len__(self) -> int:
        return len(self.index) if self.index is not None else 0


@dataclass
class SequenceData:
    seq: TokSequence | None = None
    init_seq: TokSequence | None = None
    cond_embeddings: torch.Tensor | None = None
    cond_seq: TokSequence | None = None
    score_seq: TokSequence | None = None
    gen_seq: TokSequence | None = None
    seq_context: TokSequenceContext | None = None
    cached_note_times: list[float] | None = None
    task: SequenceTask | str | None = None
    context_len: int = 0
    onset_token_dims: list[int] | None = None
    has_sos_eos: bool = False
    reached_eos: bool = False
    note_meta: NoteMetadata | None = None
    ticks_data: dict | None = None
    beat_ids: np.ndarray | None = None
    tempos: np.ndarray | None = None


class MusicTransformerGenerator:
    def __init__(
            self,
            model: MusicTransformer | CFMMusicTransformer,
            tokenizer: SyMuPe,
            dataset: SequenceDataset | None,
            used_token_types: list[str] | None = None,
            mask_token_dims: dict[str, list[int]] | list[int] | None = None,
            used_context_token_types: list[str] | None = None,
            used_score_token_types: list[str] | None = None,

            device: str | torch.device | None = None
    ):
        self.model = model
        self.context_dim = self.model.unwrapped_transformer.context_embedding_dim

        self.tokenizer = tokenizer
        self.dataset = dataset

        assert isinstance(self.tokenizer, SyMuPe)
        self.token_transformer = SyMuPeTransformer(tokenizer=self.tokenizer)

        self.data = SequenceData()

        self.device = device
        self.sos_token_id = self.tokenizer[0, SOS_TOKEN]
        self.eos_token_id = self.tokenizer[0, EOS_TOKEN]
        self.mask_token_id = self.tokenizer[0, MASK_TOKEN]
        self.mask_token_value = SPECIAL_TOKENS_VALUE - self.mask_token_id

        self.used_token_types = used_token_types
        self.mask_token_dims = mask_token_dims or {}
        self.used_context_token_types = used_context_token_types
        self.used_score_token_types = used_score_token_types

    def reset(self):
        self.data = SequenceData()

    def prepare_sequence(
            self,
            seq_idx: int | None = None,
            seq: TokSequence | None = None,
            score_seq: TokSequence | None = None,
            task: SequenceTask | str | None = None,
            context_len: int = 0,
            add_sos_eos: bool = False,
            has_pedals: bool = False
    ) -> SequenceData:
        assert seq_idx is not None or seq is not None, "One of `seq_idx` or `seq` must be provided."

        if seq is None:
            assert self.dataset is not None, "`dataset` must be present when `seq_idx` is used."
            seq = self.dataset.sequences[seq_idx]

        # prepare sequence
        seq = copy.deepcopy(seq)
        seq.interpolated = None
        if seq.values is None:
            seq.values = self.tokenizer.decode_values(seq)

        if seq.encoding != EncodingType.TIME_PERFORMANCE:
            self.data.ticks_data = self.tokenizer.compute_ticks(
                seq, time_division=self.tokenizer.config.max_num_pos_per_beat
            )
            bars, beats, onsets = self.tokenizer.compute_bar_beat_onset_indices(seq)

            self.data.note_meta = NoteMetadata(
                index=np.arange(len(seq)),
                onset=onsets,
                beat=beats,
                bar=bars,
                tempo=np.full(len(seq), fill_value=-1.)
            )

            note_ticks = self.data.ticks_data["note_on"]
            note_ticks = np.concatenate([note_ticks, [note_ticks[-1] + self.tokenizer.config.max_num_pos_per_beat]])

            beat_ticks = self.data.ticks_data["beat"]
            beat_ids = np.searchsorted(note_ticks, beat_ticks, side="right") - 1
            next_beat_ids = beat_ids[1:][
                np.minimum(np.searchsorted(beat_ids[1:], beat_ids[:-1], side="right"), len(beat_ids) - 2)
            ]
            self.data.beat_ids = (beat_ids[:-1], next_beat_ids)

        else:
            self.data.note_meta = NoteMetadata(index=np.arange(len(seq)))

        task = task or SequenceTask.PERFORMANCE
        if task in (
                SequenceTask.TIME_PERFORMANCE,
                SequenceTask.TIME_PERFORMANCE_TO_SCORE,
                SequenceTask.TIME_PERFORMANCE_TO_SCORE_DECOUPLED
        ):
            seq = self.tokenizer.sort_tokens(seq, by_time=True)

        if task in (SequenceTask.TIME_PERFORMANCE, SequenceTask.TIME_PERFORMANCE_TO_SCORE_DECOUPLED):
            seq = self.token_transformer(seq, encoding=EncodingType.TIME_PERFORMANCE)
        elif task in (SequenceTask.SCORE, SequenceTask.PLAIN_SCORE):
            seq = self.token_transformer(seq, encoding=task)
        elif task == SequenceTask.SCORE_TO_TIME_PERFORMANCE_DECOUPLED:
            seq = self.token_transformer(seq, encoding=EncodingType.PLAIN_SCORE)

        seq = self.tokenizer.normalize_values(self.tokenizer.clip_values(seq))

        # prepare condition token sequence
        cond_seq = None
        if self.used_context_token_types is not None:
            cond_seq = self.tokenizer.compress(replace(seq), token_types=self.used_context_token_types)
            cond_seq.ids[context_len:] = self.mask_token_id
            cond_seq.values[context_len:] = self.mask_token_value
            cond_seq = cond_seq.torch(device=self.device)

            if add_sos_eos:
                cond_seq = self.tokenizer.add_sos_token(cond_seq)
                cond_seq = self.tokenizer.add_eos_token(cond_seq)

        self.data.cond_seq = cond_seq

        # prepare context token sequence
        if score_seq is not None and self.used_score_token_types is not None:
            score_seq = self.tokenizer.compress(replace(score_seq), token_types=self.used_score_token_types)
            score_seq = self.tokenizer.normalize_values(self.tokenizer.clip_values(score_seq))
            # score_seq.ids[context_len:] = self.mask_token_id
            # score_seq.values[context_len:] = self.mask_token_value
            score_seq = score_seq.torch(device=self.device)

            if add_sos_eos:
                score_seq = self.tokenizer.add_sos_token(score_seq)
                score_seq = self.tokenizer.add_eos_token(score_seq)

            self.data.score_seq = score_seq

        # prepare sequence
        seq = self.tokenizer.compress(seq, token_types=self.used_token_types)

        self.data.seq = seq
        self.data.task = task

        # prepare initial (masked) sequence and move it to the device
        init_seq = copy.deepcopy(seq).torch(device=self.device)
        mask_token_dims = self.mask_token_dims.get(task, None)
        if mask_token_dims is not None:
            init_seq.ids[context_len:, mask_token_dims] = self.mask_token_id
            init_seq.values[context_len:, mask_token_dims] = self.mask_token_value

        # add SOS/EOS tokens
        if add_sos_eos:
            init_seq = self.tokenizer.add_sos_token(init_seq)
            init_seq = self.tokenizer.add_eos_token(init_seq)
        self.data.has_sos_eos = add_sos_eos

        # save initial and prepare generated sequence
        self.data.init_seq = init_seq
        self.data.gen_seq = self.data.init_seq[:int(add_sos_eos) + context_len]
        self.data.context_len = context_len

        if context_len > 0 and task in (SequenceTask.PERFORMANCE, SequenceTask.TIME_PERFORMANCE) and not has_pedals:
            context_seq = self.data.gen_seq[int(add_sos_eos):]
            context_seq = self.tokenizer.denormalize_values(context_seq)
            token_times, _ = self.tokenizer.tokens_to_midi_messages(
                context_seq, note_attributes=False, note_off_events=False, sort=False
            )
            self.data.cached_note_times = token_times.tolist()

        # prepare condition embeddings
        if self.context_dim > 0:
            self.data.cond_embeddings = torch.zeros((len(init_seq), self.context_dim), device=self.device)

        if self.tokenizer.has_token_types(init_seq, ["Bar", "Position"]):
            self.data.onset_token_dims = [init_seq.vocab["Bar"], init_seq.vocab["Position"]]
        elif self.tokenizer.has_token_types(init_seq, ["PositionShift"]):
            self.data.onset_token_dims = [init_seq.vocab["PositionShift"]]

        return self.data

    def _generate(
            self,
            input_tokens: torch.Tensor,
            input_values: torch.Tensor | None,
            context: torch.Tensor | None = None,
            context_tokens: torch.Tensor | None = None,
            context_values: torch.Tensor | None = None,
            score_tokens: torch.Tensor | None = None,
            score_values: torch.Tensor | None = None,
            disable_tqdm: bool = True,
            **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        gen_tokens, gen_values, gen_pedals = self.model.generate(
            input_tokens,
            values=input_values,
            context=context,
            context_tokens=context_tokens,
            context_values=context_values,
            score_tokens=score_tokens,
            score_values=score_values,
            tokenizer=self.tokenizer,
            disable_tqdm=disable_tqdm,
            return_intermediates=False,
            **kwargs
        )

        return gen_tokens, gen_values, gen_pedals

    def generate(
            self,
            start_time: float = 0.,
            time_window: float = 0.2,
            time_window_overflow: float = 0.1,
            max_new_notes: int = 16,
            max_seq_len: int = 512,
            cond_control: torch.Tensor | None = None,
            cond_seq_control: dict[str, float] | None = None,
            interpolated: bool = False,
            group_onset_notes: bool = True,
            drop_cached_notes: bool = False,
            force_keep_cached_notes: bool = False,
            drop_known_seq: bool = False,
            sort_messages: bool = False,
            disable_tqdm: bool = True,
            **model_kwargs
    ) -> tuple[TokSequence | None, list | np.ndarray, NoteMetadata]:
        init_seq, gen_seq = self.data.init_seq, self.data.gen_seq
        current_note_idx = len(gen_seq)

        has_condition = self.data.cond_embeddings is not None
        cond_embeddings = self.data.cond_embeddings.clone().detach() if has_condition else None

        has_condition_seq = self.data.cond_seq is not None
        cond_seq = replace(self.data.cond_seq) if has_condition_seq else None
        score_seq = self.data.score_seq

        # reset the cache for generated notes if asked explicitly or control is provided
        cached_num_notes = len(self.data.cached_note_times or [])
        drop_cached_notes = (
                drop_cached_notes
                or (has_condition and cond_control is not None)
                or (has_condition_seq and cond_seq_control is not None)
        ) and not force_keep_cached_notes
        if drop_cached_notes and cached_num_notes > 0:
            if has_condition:
                start, end = len(gen_seq) - cached_num_notes, len(gen_seq)
                self.data.cond_embeddings[start:end] = 0.
                cond_embeddings = self.data.cond_embeddings.clone().detach()

            if has_condition_seq:
                start, end = len(gen_seq) - cached_num_notes, len(gen_seq)
                self.data.cond_seq.ids[start:end] = self.mask_token_id
                self.data.cond_seq.values[start:end] = self.mask_token_value
                cond_seq = replace(self.data.cond_seq)

            self.data.gen_seq = self.data.gen_seq[:-cached_num_notes]
            gen_seq = gen_seq[:-cached_num_notes]
            current_note_idx = len(gen_seq)
            self.data.cached_note_times = []

        # prepare position counters
        start_idx = 0
        if current_note_idx >= max_seq_len - max_new_notes:
            start_idx = current_note_idx - max_seq_len + max_new_notes

        if drop_known_seq:
            start_idx = current_note_idx

        # take known sequence as context
        input_seq = gen_seq[start_idx:]
        known_input_len = len(input_seq)  # cut by `max_seq_len`

        # process sos
        has_sos = known_input_len > 0 and input_seq.ids[0, 0] == self.sos_token_id
        first_note_idx = int(has_sos)

        # move context control to device if present
        cond_control = cond_control.to(self.device) if has_condition and cond_control is not None else None

        # process cached note times and cut out generated notes for these cached notes
        all_token_times = self.data.cached_note_times or []
        cached_num_notes = len(all_token_times)
        all_gen_seq = None
        if cached_num_notes > 0:
            all_gen_seq = gen_seq[-cached_num_notes:]
            all_gen_seq = self.tokenizer.denormalize_values(all_gen_seq)

        # process sequence context
        seq_context = self.data.seq_context
        if all_gen_seq is not None and len(all_gen_seq) > 0:
            _, seq_context = self.tokenizer.tokens_to_midi_messages(all_gen_seq, context=seq_context)

        # generate next notes
        while not self.data.reached_eos:
            # but maybe we don't need to
            if all_token_times and max(all_token_times) >= start_time + time_window + time_window_overflow:
                break

            # add notes to predict
            cut_idx = current_note_idx + max_new_notes

            odims = self.data.onset_token_dims
            if group_onset_notes and odims is not None and cut_idx < len(self.data.init_seq):
                positions = None
                if len(odims) == 2:
                    positions = init_seq.ids[:, odims]
                elif len(odims) == 1:
                    positions = torch.cumsum(init_seq.values[:, odims], dim=-1)

                if positions is not None:
                    change_ids = (positions[cut_idx + 1:] != positions[cut_idx]).any(dim=-1).nonzero(as_tuple=True)[0]
                    if change_ids.numel() > 0:
                        cut_idx += change_ids[0].item()

            new_seq = init_seq[current_note_idx:cut_idx]
            num_new_notes = len(new_seq)

            if num_new_notes == 0:
                self.data.reached_eos = True
                break

            has_eos = new_seq.ids[-1, 0] == self.eos_token_id
            if has_eos and (num_new_notes == 1 or not group_onset_notes):
                self.data.reached_eos = True
                break

            input_seq = input_seq + new_seq
            input_len = len(input_seq)
            last_note_idx = input_len - int(has_eos)

            # cut input sequence if exceeds `max_seq_len`
            if input_len >= max_seq_len:
                shift = input_len - max_seq_len

                input_seq = input_seq[shift:]
                known_input_len -= shift
                last_note_idx -= shift
                start_idx += shift
                has_sos, first_note_idx = False, 0

            # shift positions to zero before computation
            input_tokens, input_values = input_seq.ids.clone().detach(), input_seq.values.clone().detach()
            input_note_seq = replace(
                self.data.seq,
                ids=input_tokens[first_note_idx:last_note_idx],
                values=input_values[first_note_idx:last_note_idx]
            )  # notes only, used for position shifting
            input_note_seq, shifts = self.tokenizer.shift_positions(
                input_note_seq, shifts=None, shift_to_zero=True, normalized_values=True
            )

            # add context control if present
            note_slice = slice(start_idx, current_note_idx + num_new_notes)
            new_note_slice = slice(current_note_idx, current_note_idx + num_new_notes)
            context = None
            if has_condition:
                if cond_control is not None:
                    cond_embeddings[new_note_slice] = cond_control
                context = cond_embeddings[note_slice][None]

            context_tokens, context_values = None, None
            if has_condition_seq:
                if cond_seq_control is not None:
                    cond_seq = self.tokenizer.denormalize_values(cond_seq)
                    for key, control_value in cond_seq_control.items():
                        if key not in self.used_context_token_types:
                            continue
                        cond_seq.values[new_note_slice, cond_seq.vocab[key]] = control_value
                    cond_seq.ids = self.tokenizer.encode_tokens(cond_seq)
                    cond_seq = self.tokenizer.normalize_values(cond_seq)

                input_cond_seq = cond_seq[note_slice]
                context_tokens = input_cond_seq.ids.clone().detach()[None]
                context_values = input_cond_seq.values.clone().detach()[None]

            score_tokens, score_values = None, None
            if score_seq is not None:
                input_score_seq = score_seq[note_slice]
                score_tokens = input_score_seq.ids.clone().detach()[None]
                score_values = input_score_seq.values.clone().detach()[None]

            # generate next notes
            gen_tokens, gen_values, gen_pedals = self._generate(
                input_tokens=input_tokens,
                input_values=input_values,
                context=context,
                context_tokens=context_tokens,
                context_values=context_values,
                context_tokens_dropout=1. if cond_seq_control is None else 0.,
                score_tokens=score_tokens,
                score_values=score_values,
                disable_tqdm=disable_tqdm,
                context_len=known_input_len,
                type_ids=torch.ones_like(input_tokens[..., 0]) if interpolated else None,
                **model_kwargs
            )

            # shift back inplace for `input_tokens/values`
            self.tokenizer.shift_positions(input_note_seq, shifts=shifts, inverse_shifts=True, normalized_values=True)

            gen_seq = replace(
                self.data.seq,
                ids=gen_tokens[known_input_len:last_note_idx],
                values=gen_values[known_input_len:last_note_idx],
            )
            gen_seq = self.tokenizer.denormalize_values(gen_seq)
            gen_seq = self.tokenizer.clip_values(gen_seq)
            self.tokenizer.shift_positions(gen_seq, shifts=shifts, inverse_shifts=True)

            # cut generated sequence
            gen_seq = gen_seq[-num_new_notes:]

            # get token times and stop if needed
            token_times, seq_context = self.tokenizer.tokens_to_midi_messages(
                gen_seq, context=seq_context, note_attributes=False, note_off_events=False, sort=False
            )

            all_token_times.extend(token_times.tolist())
            all_gen_seq = copy.copy(gen_seq) if all_gen_seq is None else all_gen_seq + gen_seq

            if token_times.max() >= start_time + time_window + time_window_overflow:
                break

            # add generated notes
            gen_seq = self.tokenizer.normalize_values(gen_seq)
            input_seq.ids[known_input_len:last_note_idx] = gen_seq.ids
            input_seq.values[known_input_len:last_note_idx] = gen_seq.values
            current_note_idx += num_new_notes
            known_input_len += num_new_notes

        # cut notes fitting `time_window`
        cut_idx = np.where(np.array(all_token_times) <= start_time + time_window)[0]
        cut_idx = 0 if len(cut_idx) == 0 else cut_idx[-1] + 1

        # compute sequence and messages inside the current `time_window`
        gen_seq, messages = None, []
        if cut_idx > 0 and len(all_gen_seq) > 0:
            gen_seq = all_gen_seq[:cut_idx]
            messages, self.data.seq_context = self.tokenizer.tokens_to_midi_messages(
                gen_seq, context=self.data.seq_context, sort=sort_messages
            )
            gen_seq = self.tokenizer.normalize_values(gen_seq)

        start_idx = len(self.data.gen_seq) - cached_num_notes - int(self.data.has_sos_eos)
        end_idx = start_idx + (len(gen_seq) if gen_seq is not None else 0)

        self.data.cached_note_times = all_token_times[cut_idx:]

        # update total generated sequence
        save_seq = all_gen_seq[cached_num_notes:]
        if len(save_seq) > 0:
            # update context embeddings for the generated notes
            total_len = len(self.data.gen_seq)
            start, end = total_len, total_len + len(save_seq)

            if has_condition:
                self.data.cond_embeddings[start:end] = cond_embeddings[start:end]

            # if has_condition_seq:
            #     self.data.cond_seq.ids[start:end] = cond_seq.ids[start:end]
            #     self.data.cond_seq.values[start:end] = cond_seq.values[start:end]

            save_seq = self.tokenizer.normalize_values(save_seq)
            self.data.gen_seq += save_seq

            self.update_tempos()

        self.data.reached_eos = (
                self.data.reached_eos
                and len(self.data.gen_seq) - int(self.data.has_sos_eos) == len(self.data.seq)
                and len(self.data.cached_note_times) == 0
        )

        meta = self.data.note_meta[start_idx:end_idx]

        return gen_seq, messages, meta

    def generate_batch(
            self,
            num_sequences: int = 1,
            max_new_notes: int = 16,
            max_seq_len: int = 512,
            cond_control: torch.Tensor | None = None,
            cond_seq_control: dict[str, float] | None = None,
            interpolated: bool = False,
            group_onset_notes: bool = True,
            disable_tqdm: bool = True,
            **model_kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        init_seq, gen_seq = self.data.init_seq, self.data.gen_seq

        gen_tokens, gen_values = gen_seq.ids, gen_seq.values
        gen_tokens = gen_tokens.expand(num_sequences, -1, -1)
        gen_values = gen_values.expand(num_sequences, -1, -1)

        current_note_idx = gen_tokens.shape[1]

        has_condition = self.data.cond_embeddings is not None
        cond_embeddings = self.data.cond_embeddings.clone().detach() if has_condition else None

        has_condition_seq = self.data.cond_seq is not None
        cond_seq = replace(self.data.cond_seq) if has_condition_seq else None

        has_score_seq = self.data.score_seq is not None
        score_seq = replace(self.data.score_seq) if has_score_seq else None

        # prepare position counters
        start_idx = 0
        if current_note_idx >= max_seq_len - max_new_notes:
            start_idx = current_note_idx - max_seq_len + max_new_notes

        # take known sequence as context
        input_tokens, input_values = gen_tokens[:, start_idx:], gen_values[:, start_idx:]
        known_input_len = input_tokens.shape[1]  # cut by `max_seq_len`

        # process sos
        has_sos = known_input_len > 0 and input_tokens[0, 0, 0] == self.sos_token_id
        first_note_idx = int(has_sos)

        # move context control to device if present
        cond_control = cond_control.to(self.device) if has_condition and cond_control is not None else None

        all_gen_tokens, all_gen_values = gen_tokens[:, first_note_idx:], gen_values[:, first_note_idx:]

        # generate all notes till the end
        if not disable_tqdm:
            pbar = tqdm(total=len(init_seq))
            pbar.update(all_gen_tokens.shape[1])
        while not self.data.reached_eos:
            # add notes to predict
            cut_idx = current_note_idx + max_new_notes

            odims = self.data.onset_token_dims
            if group_onset_notes and odims is not None and cut_idx < len(self.data.init_seq):
                positions = None
                if len(odims) == 2:
                    positions = init_seq.ids[:, odims]
                elif len(odims) == 1:
                    positions = torch.cumsum(init_seq.values[:, odims], dim=-1)

                if positions is not None:
                    change_ids = (positions[cut_idx + 1:] != positions[cut_idx]).any(dim=-1).nonzero(as_tuple=True)[0]
                    if change_ids.numel() > 0:
                        cut_idx += change_ids[0].item()

            new_seq = init_seq[current_note_idx:cut_idx]
            num_new_notes = len(new_seq)

            if num_new_notes == 0:
                self.data.reached_eos = True
                break

            has_eos = new_seq.ids[-1, 0] == self.eos_token_id
            if has_eos and (num_new_notes == 1 or not group_onset_notes):
                self.data.reached_eos = True
                break

            input_tokens = torch.cat([input_tokens, new_seq.ids[None].expand(num_sequences, -1, -1)], dim=1)
            input_values = torch.cat([input_values, new_seq.values[None].expand(num_sequences, -1, -1)], dim=1)
            input_len = input_tokens.shape[1]
            last_note_idx = input_len - int(has_eos)

            # cut input sequence if exceeds `max_seq_len`
            if input_len >= max_seq_len:
                shift = input_len - max_seq_len

                input_tokens = input_tokens[:, shift:]
                input_values = input_values[:, shift:]
                known_input_len -= shift
                last_note_idx -= shift
                start_idx += shift
                # has_sos, first_note_idx = False, 0

            # shift positions to zero before computation
            # input_note_seq = replace(
            #     self.data.seq,
            #     ids=input_tokens[0, first_note_idx:last_note_idx],
            #     values=input_values[0, first_note_idx:last_note_idx]
            # )  # notes only, used for position shifting
            # input_note_seq, shifts = self.tokenizer.shift_positions(
            #     input_note_seq, shifts=None, shift_to_zero=True, normalized_values=True
            # )

            # add context control if present
            note_slice = slice(start_idx, current_note_idx + num_new_notes)
            new_note_slice = slice(current_note_idx, current_note_idx + num_new_notes)
            context = None
            if has_condition:
                if cond_control is not None:
                    cond_embeddings[new_note_slice] = cond_control
                context = cond_embeddings[note_slice][None].expand(num_sequences, -1, -1)

            context_tokens, context_values = None, None
            if has_condition_seq:
                if cond_seq_control is not None:
                    cond_seq = self.tokenizer.denormalize_values(cond_seq)
                    for key, control_value in cond_seq_control.items():
                        if key not in self.used_context_token_types:
                            continue
                        cond_seq.values[new_note_slice, cond_seq.vocab[key]] = control_value
                    cond_seq.ids = self.tokenizer.encode_tokens(cond_seq)
                    cond_seq = self.tokenizer.normalize_values(cond_seq)

                input_cond_seq = cond_seq[note_slice]
                context_tokens = input_cond_seq.ids.clone().detach()[None].expand(num_sequences, -1, -1)
                context_values = input_cond_seq.values.clone().detach()[None].expand(num_sequences, -1, -1)

            score_tokens, score_values = None, None
            if has_score_seq:
                input_score_seq = score_seq[note_slice]
                score_tokens = input_score_seq.ids.clone().detach()[None].expand(num_sequences, -1, -1)
                score_values = input_score_seq.values.clone().detach()[None].expand(num_sequences, -1, -1)

            # generate next notes
            gen_tokens, gen_values, gen_pedals = self._generate(
                input_tokens=input_tokens,
                input_values=input_values,
                context=context,
                context_tokens=context_tokens,
                context_values=context_values,
                score_tokens=score_tokens,
                score_values=score_values,
                context_tokens_dropout=1. if cond_seq_control is None else 0.,
                disable_tqdm=disable_tqdm,
                context_len=known_input_len,
                type_ids=torch.ones_like(input_tokens[..., 0]) if interpolated else None,
                **model_kwargs
            )

            # shift back inplace for `input_tokens/values`
            # self.tokenizer.shift_positions(input_note_seq, shifts=shifts, inverse_shifts=True, normalized_values=True)

            # gen_seq = replace(
            #     self.data.seq,
            #     ids=gen_tokens[:, known_input_len:last_note_idx],
            #     values=gen_values[known_input_len:last_note_idx],
            # )
            # gen_seq = self.tokenizer.denormalize_values(gen_seq)
            # gen_seq = self.tokenizer.clip_values(gen_seq)
            # self.tokenizer.shift_positions(gen_seq, shifts=shifts, inverse_shifts=True)

            # cut generated sequence
            gen_tokens = gen_tokens[:, known_input_len:last_note_idx][:, -num_new_notes:]
            gen_values = gen_values[:, known_input_len:last_note_idx][:, -num_new_notes:]

            all_gen_tokens = gen_tokens.clone().detach() if all_gen_tokens is None else torch.cat(
                [all_gen_tokens, gen_tokens], dim=1
            )
            all_gen_values = gen_values.clone().detach() if all_gen_values is None else torch.cat(
                [all_gen_values, gen_values], dim=1
            )

            # add generated notes
            input_tokens[:, known_input_len:last_note_idx] = gen_tokens
            input_values[:, known_input_len:last_note_idx] = gen_values
            current_note_idx += num_new_notes
            known_input_len += num_new_notes

            if not disable_tqdm:
                pbar.update(num_new_notes)

        self.data.reached_eos = (
                self.data.reached_eos
                and len(self.data.gen_seq[0]) - int(self.data.has_sos_eos) == len(self.data.seq)
        )

        return all_gen_tokens, all_gen_values

    def update_tempos(self) -> None:
        if len(self.data.gen_seq) <= int(self.data.has_sos_eos):
            self.data.tempos = None
            return

        time_shifts = self.tokenizer.get_values(
            self.data.gen_seq[int(self.data.has_sos_eos):].numpy(), "TimeShift"
        )
        note_times = np.cumsum(time_shifts)
        note_ticks = self.data.ticks_data["note_on"]

        beat_ids, next_beat_ids = self.data.beat_ids
        cut_idx = np.where(next_beat_ids < len(note_times))[0][-1] + 1
        beat_ids, next_beat_ids = beat_ids[:cut_idx], next_beat_ids[:cut_idx]

        delta_times = note_times[next_beat_ids] - note_times[beat_ids]
        delta_ticks = note_ticks[next_beat_ids] - note_ticks[beat_ids]

        tempos = 60 * delta_ticks / delta_times / self.tokenizer.config.max_num_pos_per_beat
        self.data.tempos = tempos

        if self.data.note_meta.tempo is not None:
            beats = self.data.note_meta.beat
            end_idx = len(self.data.gen_seq) - int(self.data.has_sos_eos)

            self.data.note_meta.tempo[:end_idx] = self.data.tempos[
                np.minimum(beats[:end_idx], len(self.data.tempos) - 1)
            ]

    def generated_sequence(
            self,
            postprocess: bool = True,
            encoding: EncodingType | str | None = None
    ) -> TokSequence:
        gen_seq = self.data.gen_seq[int(self.data.has_sos_eos):].numpy()
        gen_seq.type = "performance"

        if postprocess:
            gen_seq = self.tokenizer.denormalize_values(gen_seq)
            gen_seq = self.tokenizer.decompress(copy.deepcopy(gen_seq))

            if encoding is not None:
                gen_seq = self.token_transformer(gen_seq, encoding)

        return gen_seq

    def reset_position(
            self,
            time: float | None = None,
            note_idx: int | None = None,
    ) -> int:
        has_sos = int(self.data.has_sos_eos)
        if len(self.data.gen_seq) <= has_sos:
            return

        gen_seq = replace(self.data.gen_seq)[has_sos:]
        gen_seq = self.tokenizer.denormalize_values(gen_seq)

        if time is not None:
            token_times, _ = self.tokenizer.tokens_to_midi_messages(
                gen_seq, note_attributes=False, note_off_events=False, sort=False
            )
            cut_ids = np.where(token_times >= time)[0]
            note_idx = len(gen_seq) if len(cut_ids) == 0 else cut_ids[0]

        if note_idx is not None:
            self.data.gen_seq = self.data.gen_seq[:has_sos + note_idx]
            self.data.cached_note_times = []

        self.data.seq_context = None
        if len(self.data.gen_seq) > has_sos:
            _, self.data.seq_context = self.tokenizer.tokens_to_midi_messages(
                self.data.gen_seq[has_sos:], note_attributes=False, note_off_events=False, sort=False
            )

        self.data.reached_eos = (
                len(self.data.gen_seq) - int(self.data.has_sos_eos) == len(self.data.seq)
                and len(self.data.cached_note_times) == 0
        )

        return note_idx


class CFMMusicTransformerGenerator(MusicTransformerGenerator):
    def _generate(
            self,
            input_tokens: torch.Tensor,
            input_values: torch.Tensor | None,
            context: torch.Tensor | None = None,
            context_tokens: torch.Tensor | None = None,
            context_values: torch.Tensor | None = None,
            score_tokens: torch.Tensor | None = None,
            score_values: torch.Tensor | None = None,
            loss_fn: nn.Module | None = None,
            context_len: int = 0,
            gamma: float = 1,
            norm_fn: Callable | None = None,
            schedule_fn: Callable | None = None,
            num_resample: int = 1,
            resample_period: int = 5,
            resample_fn: Callable = resample,
            disable_tqdm: bool = True,
            **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        gen_tokens, gen_values, gen_pedals = self.model.generate(
            input_tokens,
            values=input_values,
            context=context,
            context_tokens=context_tokens,
            context_values=context_values,
            score_tokens=score_tokens,
            score_values=score_values,
            tokenizer=self.tokenizer,
            loss_fn=loss_fn,
            context_len=context_len,
            gamma=gamma,
            norm_fn=norm_fn,
            schedule_fn=schedule_fn,
            num_resample=num_resample,
            resample_period=resample_period,
            resample_fn=resample_fn,
            disable_tqdm=disable_tqdm,
            return_intermediates=False,
            **kwargs
        )

        return gen_tokens, gen_values, gen_pedals
