""" MusicTransformer's inference modules. """
from __future__ import annotations

import copy
from dataclasses import dataclass, replace

import torch
from tqdm.auto import tqdm

from symupe.data.datasets import SequenceDataset, SequenceTask
from symupe.data.datasets.common import TASK_SEQUENCE_SORTING, TASK_TO_ENCODINGS
from symupe.data.tokenizers import (
    TokSequence, TokSequenceContext,
    SequenceType, EncodingType, SortingType, ENCODING_SORTING,
    SyMuPe, SyMuPeTransformer
)
from symupe.data.tokenizers.constants import SOS_TOKEN, EOS_TOKEN, MASK_TOKEN, SPECIAL_TOKENS_VALUE
from .model import Seq2SeqMusicTransformer


@dataclass
class Seq2SeqData:
    source_seq: TokSequence | None = None
    target_seq: TokSequence | None = None
    init_seq: TokSequence | None = None
    gen_seq: TokSequence | None = None

    cond_seq: TokSequence | None = None
    score_seq: TokSequence | None = None
    cond_embeddings: torch.Tensor | None = None
    seq_context: TokSequenceContext | None = None

    task: SequenceTask | str | None = None
    context_len: int = 0
    parallel_task: bool = False
    onset_token_dims: list[int] | None = None
    has_sos_eos: bool = False
    reached_eos: bool = False


class Seq2SeqMusicTransformerGenerator:
    def __init__(
            self,
            model: Seq2SeqMusicTransformer,
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

        self.data = Seq2SeqData()

        self.device = device
        self.sos_token_id = self.tokenizer[0, SOS_TOKEN]
        self.eos_token_id = self.tokenizer[0, EOS_TOKEN]
        self.mask_token_id = self.tokenizer[0, MASK_TOKEN]
        self.mask_token_value = SPECIAL_TOKENS_VALUE - self.mask_token_id

        self.used_token_types = used_token_types
        self.mask_token_dims = mask_token_dims or {}
        self.used_context_token_types = used_context_token_types
        self.used_score_token_types = used_score_token_types

    def reset(self) -> None:
        self.data = Seq2SeqData()

    def prepare_sequence(
            self,
            seq_idx: int | None = None,
            seq: TokSequence | None = None,
            score_seq: TokSequence | None = None,
            task: SequenceTask | str | None = None,
            context_len: int = 0,
            add_sos_eos: bool = False
    ) -> Seq2SeqData:
        assert seq_idx is not None or seq is not None, "One of `seq_idx` or `seq` must be provided."

        if seq is None:
            assert self.dataset is not None, "`dataset` must be present when `seq_idx` is used."
            seq = self.dataset.sequences[seq_idx]

        # prepare sequence
        seq = copy.deepcopy(seq)
        seq.interpolated = None
        if seq.values is None:
            seq.values = self.tokenizer.decode_values(seq)

        seq_type = seq.type

        # process encodings
        source_encoding, target_encoding = TASK_TO_ENCODINGS[task]
        target_encoding = source_encoding if target_encoding is None else target_encoding

        source_seq = self.token_transformer(copy.deepcopy(seq), encoding=source_encoding)
        target_seq = self.token_transformer(copy.deepcopy(seq), encoding=target_encoding)

        def sort_sequence(tok_sequence, encoding_type):
            sort_by_time = False
            if task in TASK_SEQUENCE_SORTING[SortingType.TIME] or encoding_type in ENCODING_SORTING[SortingType.TIME]:
                sort_by_time = True

            is_sort_by_time = seq_type in (SequenceType.TIME_PERFORMANCE, SequenceType.TIME_PERFORMANCE_SUSTAIN)
            if is_sort_by_time != sort_by_time:
                tok_sequence = self.tokenizer.sort_tokens(tok_sequence, by_time=sort_by_time)

            return tok_sequence, sort_by_time

        source_seq, source_sort_by_time = sort_sequence(source_seq, source_encoding)
        target_seq, target_sort_by_time = sort_sequence(target_seq, target_encoding)

        source_seq = self.tokenizer.normalize_values(self.tokenizer.clip_values(source_seq))
        target_seq = self.tokenizer.normalize_values(self.tokenizer.clip_values(target_seq))

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
            score_seq = score_seq.torch(device=self.device)

            if add_sos_eos:
                score_seq = self.tokenizer.add_sos_token(score_seq)
                score_seq = self.tokenizer.add_eos_token(score_seq)

            self.data.score_seq = score_seq

        # prepare sequence
        source_seq = self.tokenizer.compress(source_seq, token_types=self.used_token_types)
        target_seq = self.tokenizer.compress(target_seq, token_types=self.used_token_types)

        self.data.source_seq = source_seq
        self.data.target_seq = target_seq
        self.data.task = task

        self.data.parallel_task = source_encoding == target_encoding

        if not self.data.parallel_task:
            raise NotImplementedError("Non-parallel decoding is not supported yet.")

        # prepare initial (masked) sequences and move them to the device
        init_seq = copy.deepcopy(source_seq).torch(device=self.device)
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

        # prepare condition embeddings
        if self.context_dim > 0:
            self.data.cond_embeddings = torch.zeros((len(init_seq), self.context_dim), device=self.device)

        if self.tokenizer.has_token_types(init_seq, ["Bar", "Position"]):
            self.data.onset_token_dims = [init_seq.vocab["Bar"], init_seq.vocab["Position"]]
        elif self.tokenizer.has_token_types(init_seq, ["PositionShift"]):
            self.data.onset_token_dims = [init_seq.vocab["PositionShift"]]

        return self.data

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
            # if has_condition:
            #     if cond_control is not None:
            #         cond_embeddings[new_note_slice] = cond_control
            #     context = cond_embeddings[note_slice][None]

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
            gen_tokens, gen_values = self.model.generate(
                enc_tokens=input_tokens,
                enc_values=input_values,
                dec_tokens=input_tokens[:, :-num_new_notes],
                dec_values=input_values[:, :-num_new_notes],
                dec_context=context,
                dec_context_tokens=context_tokens,
                dec_context_values=context_values,
                dec_score_tokens=score_tokens,
                dec_score_values=score_values,
                context_tokens_dropout=1. if cond_seq_control is None else 0.,
                disable_tqdm=disable_tqdm,
                context_len=known_input_len,
                type_ids=torch.ones_like(input_tokens[..., 0]) if interpolated else None,
                tokenizer=self.tokenizer,
                force_known_tokens=True,
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
                and len(self.data.gen_seq[0]) - int(self.data.has_sos_eos) == len(self.data.source_seq)
        )

        return all_gen_tokens, all_gen_values

    def generated_sequence(
            self,
            postprocess: bool = True,
            encoding: EncodingType | str | None = None
    ) -> TokSequence:
        gen_seq = self.data.gen_seq[int(self.data.has_sos_eos):].numpy()

        if postprocess:
            gen_seq = self.tokenizer.denormalize_values(gen_seq)
            gen_seq = self.tokenizer.decompress(copy.deepcopy(gen_seq))

            if encoding is not None:
                gen_seq = self.token_transformer(gen_seq, encoding)

        return gen_seq
