""" ScorePerformer's inference modules. """
from __future__ import annotations

import copy
from dataclasses import dataclass, replace

import numpy as np
import torch

from symupe.data.collators import MixedLMScorePerformanceCollator
from symupe.data.datasets import ScorePerformanceDataset, ScorePerformanceSampleMeta
from symupe.data.helpers import TokenSequenceAugmentations
from symupe.data.tokenizers import TokSequence, TokSequenceContext, SyMuPeLocal
from symupe.data.tokenizers.constants import SOS_TOKEN, EOS_TOKEN
from symupe.modules.transformer import AttentionIntermediates, TransformerLayerIntermediates, TransformerIntermediates
from symupe.modules.tuple_transformer import TupleTransformerCache
from .model import ScorePerformer


@dataclass
class PerformanceData:
    perf_seq: TokSequence | None = None
    init_seq: TokSequence | None = None
    perf_embeddings: torch.Tensor | None = None
    score_embeddings: torch.Tensor | None = None
    gen_seq: TokSequence | None = None
    seq_context: TokSequenceContext | None = None
    cache: TupleTransformerCache | None = None
    reached_eos: bool = False


class ScorePerformerGenerator:
    def __init__(
            self,
            model: ScorePerformer,
            dataset: ScorePerformanceDataset,
            collator: MixedLMScorePerformanceCollator,
            device: str | torch.device | None = None
    ):
        self.model = model
        assert model.perf_decoder is not None

        self.dataset = dataset
        self.tokenizer = dataset.tokenizer
        self.collator = collator

        self.data = PerformanceData()

        self.device = device
        self.sos_token_id = self.tokenizer[0, SOS_TOKEN]
        self.eos_token_id = self.tokenizer[0, EOS_TOKEN]

        num_dims = len(self.tokenizer.sizes)
        mask_token_dims = range(num_dims) if self.collator.mask_token_dims is None else self.collator.mask_token_dims
        self.mask_token_dims = torch.tensor(list(mask_token_dims))

    def reset(self) -> None:
        self.data = PerformanceData()

    def prepare_performance(
            self,
            perf_idx: int,
            score_embeddings: torch.Tensor | None = None,
            perf_embeddings: torch.Tensor | None = None,
            overlay_bars: float = 0.5
    ) -> PerformanceData:
        # get performance sequence (and notes) from dataset
        perf_seq = copy.deepcopy(self.dataset.performances[perf_idx])
        if perf_seq.values is None:
            perf_seq.values = self.tokenizer.decode_values(perf_seq)
        perf_seq = self.tokenizer.normalize_values(self.tokenizer.clip_values(perf_seq))

        self.data.perf_seq = perf_seq

        # prepare sequence and move to device
        init_seq = copy.deepcopy(perf_seq).torch(device=self.device)
        init_seq.ids[:, self.mask_token_dims] = self.collator.mask_token_id
        init_seq.values[:, self.mask_token_dims] = self.collator.mask_token_value

        # add SOS/EOS tokens
        init_seq = self.tokenizer.add_sos_token(init_seq)
        init_seq = self.tokenizer.add_eos_token(init_seq)

        self.data.init_seq = init_seq
        self.data.gen_seq = self.data.init_seq[:1]

        # compute performance embeddings if not provided
        compute_embeddings = self.model.perf_encoder is not None and perf_embeddings is None
        compute_embeddings = compute_embeddings or (self.model.score_encoder is not None and score_embeddings is None)
        if compute_embeddings:
            score_embeddings, perf_embeddings, _ = self.encode_embeddings(perf_idx, overlay_bars=overlay_bars)

        self.data.perf_embeddings = perf_embeddings
        self.data.score_embeddings = score_embeddings

        if isinstance(self.tokenizer, SyMuPeLocal):
            initial_tempo = perf_seq.meta.get("initial_tempo", self.tokenizer.default_tempo)
            self.data.seq_context = TokSequenceContext(initial_tempo=initial_tempo)

        return self.data

    def generate_performance_notes(
            self,
            start_time: float = 0.,
            time_window: float = 0.2,
            time_window_overflow: float = 0.1,
            delta_embedding: torch.Tensor | None = None,
            max_seq_len: int = 512,
            group_onset_notes: bool = True,
            sort_messages: bool = False,
            temperature: float = 1.,
            top_k: float | int = -1,
            top_p: float = 0.8,
            disable_tqdm: bool = True,
            disable_cache: bool = False
    ) -> tuple[TokSequence | None, list | np.ndarray]:
        init_seq, gen_seq = self.data.init_seq, self.data.gen_seq
        current_note_idx = len(gen_seq)

        has_perf_emb = self.data.perf_embeddings is not None
        has_score_emb = self.data.score_embeddings is not None
        perf_embeddings = self.data.perf_embeddings.clone().detach() if has_perf_emb else None
        score_embeddings = self.data.score_embeddings.clone().detach() if has_score_emb else None

        # prepare position counters
        start_idx = 0
        if current_note_idx >= max_seq_len - 1:
            next_bar_idx = torch.where(torch.diff(gen_seq.ids[1:, 0]))[0]
            if len(next_bar_idx) > 0:
                fits_context = torch.where(current_note_idx - (next_bar_idx + 1) < max_seq_len)[0]
                start_idx = 0 if len(fits_context) == 0 else next_bar_idx[fits_context[0]] + 2

        # take known sequence as context
        input_seq = gen_seq[start_idx:]
        known_input_len = len(input_seq)  # cut by `max_seq_len`

        # process sos
        has_sos = input_seq.ids[0, 0] == self.sos_token_id
        first_note_idx = int(has_sos)

        # move delta embedding to device if present
        delta_embedding = None if delta_embedding is None else delta_embedding.to(self.device)

        # add notes one by one, predict and check the timing, break if went behind window
        gen_tokens, gen_values = None, None
        cache, seq_context = self.data.cache, self.data.seq_context
        all_token_times, all_gen_seq = [], None
        while not self.data.reached_eos:
            # add notes to predict
            cut_idx = current_note_idx + 1

            if group_onset_notes and cut_idx < len(self.data.init_seq):
                positions = init_seq.ids[:, 2]
                change_ids = (positions[cut_idx + 1:] != positions[cut_idx]).any(dim=-1).nonzero(as_tuple=True)[0]
                if change_ids.numel() > 0:
                    cut_idx += change_ids[0].item()

            new_seq = init_seq[current_note_idx:cut_idx]
            num_new_notes = len(new_seq)

            if num_new_notes == 0:
                self.data.reached_eos = True
                break

            has_eos = new_seq.ids[-1, 0] == self.eos_token_id
            if has_eos and not group_onset_notes:
                self.data.reached_eos = True
                break

            tempo_key = self.tokenizer.performance_tempo_token
            tempo_idx = self.tokenizer.vocab_types_idx[tempo_key]
            if isinstance(self.tokenizer, SyMuPeLocal) and tempo_idx not in self.mask_token_dims:
                # update tempo tokens with current tempos only if tempos are not predicted
                tempo = seq_context.tempos[0][-1] if seq_context.tempos is not None else seq_context.initial_tempo
                tempo = np.array([tempo])
                new_seq.ids[:, tempo_idx] = self.tokenizer.encode_tokens(tempo, tempo_key)[0]
                new_seq.values[:, tempo_idx] = self.tokenizer.normalize_values(tempo, tempo_key)[0]

            input_seq = input_seq + new_seq
            input_len = len(input_seq)
            last_note_idx = input_len - int(has_eos)

            # cut input sequence if exceeds `max_seq_len`
            if input_len >= max_seq_len:
                next_bar_idx = torch.where(torch.diff(input_tokens[first_note_idx:last_note_idx, 0]))[0]
                shift = 1
                if len(next_bar_idx) > 0:
                    fits_context = torch.where(input_len - (next_bar_idx + first_note_idx) < max_seq_len)[0]
                    if len(fits_context) > 0 and next_bar_idx[fits_context[0]] + 1 + first_note_idx != input_len - 1:
                        shift = next_bar_idx[fits_context[0]] + 1 + first_note_idx

                input_seq = input_seq[shift:]
                known_input_len -= shift
                last_note_idx -= shift
                start_idx += shift
                has_sos, first_note_idx = False, 0
                cache = None

            # shift bars to zero before computation
            input_tokens, input_values = input_seq.ids.clone().detach(), input_seq.values.clone().detach()
            input_seq = replace(
                self.data.perf_seq,
                ids=input_tokens[first_note_idx:last_note_idx],
                values=input_values[first_note_idx:last_note_idx]
            )
            input_seq, shifts = self.tokenizer.shift_positions(
                input_seq, shifts=None, shift_to_zero=True, normalized_values=True
            )

            # add masked sequence
            masked_input_tokens = input_tokens.clone().detach()
            masked_input_tokens[first_note_idx:last_note_idx, self.mask_token_dims] = self.collator.mask_token_id

            masked_input_values = input_values.clone().detach().float()
            masked_input_values[first_note_idx:last_note_idx, self.mask_token_dims] = self.collator.mask_token_value

            # add delta embedding if present
            if has_perf_emb and delta_embedding is not None:
                perf_embeddings[current_note_idx:current_note_idx + num_new_notes] += delta_embedding

            # get score and performance embeddings
            score_embs = None
            if has_score_emb:
                score_embs = score_embeddings[start_idx:current_note_idx + num_new_notes].unsqueeze(0)
            perf_embs = None
            if has_perf_emb:
                perf_embs = perf_embeddings[start_idx:current_note_idx + num_new_notes].unsqueeze(0)

            if cache is not None:
                if input_tokens.shape[0] - 1 - num_new_notes != cache.token_emb.shape[1] \
                        or cache.token_emb.shape[1] == 0 \
                        or len(cache.transformer.layers) == 0:
                    cache = None

            # generate notes
            with torch.inference_mode():
                gen_tokens, gen_values, cache = self.model.perf_decoder.unmask(
                    input_tokens,
                    masked_tokens=masked_input_tokens,
                    values=input_values,
                    masked_values=masked_input_values,
                    context=score_embs,
                    style_embeddings=perf_embs,
                    cache=cache if not disable_cache else None,
                    return_cache=True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    tokenizer=self.tokenizer,
                    disable_tqdm=disable_tqdm
                )

                # shift back inplace for `input_tokens/values`
                self.tokenizer.shift_positions(input_seq, shifts=shifts, inverse_shifts=True, normalized_values=True)

                gen_seq = replace(
                    self.data.perf_seq,
                    ids=gen_tokens[known_input_len:last_note_idx],
                    values=gen_values[known_input_len:last_note_idx],
                )
                gen_seq = self.tokenizer.denormalize_values(gen_seq)
                self.tokenizer.shift_positions(gen_seq, shifts=shifts, inverse_shifts=True)

            # cut generated sequence
            gen_seq = gen_seq[-num_new_notes:]
            # gen_seq = self.data.perf_seq[current_note_idx - 1:current_note_idx - 1 + num_new_notes]

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
            input_tokens[-num_new_notes:] = gen_seq.ids
            input_values[-num_new_notes:] = gen_seq.values
            current_note_idx += num_new_notes
            known_input_len += num_new_notes

        if gen_tokens is None:
            return None, []

        # cut notes fitting `time_window`
        cut_idx = np.where(np.array(all_token_times) <= start_time + time_window)[0]
        cut_idx = 0 if len(cut_idx) == 0 else cut_idx[-1] + 1

        if cut_idx == 0:
            return None, []

        # compute new messages
        gen_seq = all_gen_seq[:cut_idx]
        messages, self.data.seq_context = self.tokenizer.tokens_to_midi_messages(
            gen_seq, context=self.data.seq_context, sort=sort_messages
        )

        # update performance embeddings for the generated notes
        if has_perf_emb and delta_embedding is not None:
            total_len = len(self.data.gen_seq)
            self.data.perf_embeddings[total_len:total_len + cut_idx] = perf_embeddings[total_len:total_len + cut_idx]

        # update total generated sequence
        gen_seq = self.tokenizer.normalize_values(gen_seq)
        self.data.gen_seq += gen_seq

        # update cache
        if cache is not None:
            cut_len = cache.token_emb.shape[1] - (len(all_token_times) - cut_idx)
            cache = self.cut_cache(cache, right_idx=cut_len)
        self.data.cache = cache

        return gen_seq, messages

    def generated_sequence(
            self,
            postprocess: bool = True
    ) -> TokSequence:
        gen_seq = self.data.gen_seq[1:].numpy()

        if postprocess:
            gen_seq = self.tokenizer.denormalize_values(gen_seq)

        return gen_seq

    def predict_number_of_notes(
            self,
            start_time: float = 0.,
            time_window: float = 0.2,
            max_notes: int = 32,
    ) -> np.ndarray:
        num_gen_notes = len(self.data.gen_seq) - 1 if self.data.gen_seq is not None else 0
        future_tokens = self.data.perf_seq.ids[num_gen_notes:num_gen_notes + max_notes]
        if len(future_tokens) == 0:
            return 0.
        future_values = self.data.perf_seq.values[num_gen_notes:num_gen_notes + max_notes]

        if self.data.seq_context is not None:  # adjust tempos
            tempo_key = self.tokenizer.performance_tempo_token
            tempo_idx = self.tokenizer.vocab_types_idx[tempo_key]
            tempo = np.array([self.data.seq_context.tempos[0][-1]])

            future_tokens[:, tempo_idx] += (
                    self.tokenizer.encode_tokens(tempo, tempo_key)[0]
                    - self.data.perf_seq.ids[num_gen_notes - 1, tempo_idx]
            )
            future_values[:, tempo_idx] += tempo - self.data.perf_seq.values[num_gen_notes - 1, tempo_idx]

        times = self.tokenizer.tokens_to_midi_messages(
            replace(self.data.perf_seq, ids=future_tokens, values=future_values),
            context=self.data.seq_context, note_attributes=False, note_off_events=False, sort=False
        )
        return (times <= start_time + time_window).sum()

    def encode_embeddings(
            self,
            perf_idx: int,
            compute_latents: bool = False,
            overlay_bars: float = 0.,
            augmentations: TokenSequenceAugmentations | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # get score sequence and its data
        perf = self.dataset.performance_names[perf_idx]
        score, _ = self.dataset._performance_map[perf]
        score_idx = self.dataset.scores._name_to_idx[score]
        score_indices = self.dataset._score_indices[score_idx]
        if score_indices is None:
            score_indices = self.dataset.indexer.compute_bar_indices(self.dataset.scores[score_idx])
            self.dataset._score_indices[score_idx] = score_indices

        # get initial meta and sample
        from symupe.data.datasets.utils import get_end_bar
        start_bar = 0
        end_bar = get_end_bar(score_indices, start_bar, self.dataset.max_seq_len, self.dataset.max_bar)
        meta = ScorePerformanceSampleMeta(
            idx=None, score_idx=score_idx, perf_idx=perf_idx,
            start_bar=start_bar, end_bar=end_bar,
            augmentations=augmentations,
        )
        sample = self.dataset.get(meta=meta)

        # get current last bar and total number of bars
        bar_idx = self.tokenizer.vocab_types_idx["Bar"]
        _bar_0 = self.tokenizer.zero_token
        score_seq = self.dataset.scores[score_idx]
        has_sos = sample.score.ids[0, 0] == self.sos_token_id
        has_eos = sample.score.ids[-1, 0] == self.eos_token_id
        first_note_idx, last_note_idx = int(has_sos), len(sample.score) - int(has_eos)
        last_perf_note_idx = len(sample.perf) - int(has_eos)
        last_bar = sample.score.ids[-1 - int(has_eos), bar_idx] - _bar_0
        total_bars = score_seq.ids[-1, bar_idx] - _bar_0

        emb_start_bar = start_bar
        score_embeddings, perf_embeddings = [], []
        while last_bar <= total_bars:
            _inputs = self.collator((sample,))
            inputs = self.model.allocate_inputs(self.model.prepare_inputs(_inputs), self.device)

            # move bars to zero
            bar_shift_to_zero = inputs["score"][:, first_note_idx, bar_idx] - _bar_0
            inputs["score"][:, first_note_idx:last_note_idx, bar_idx] -= bar_shift_to_zero
            inputs["perf"][:, first_note_idx:last_perf_note_idx, bar_idx] -= bar_shift_to_zero

            bar_value_shift_to_zero = 0.
            if inputs["score_values"] is not None:
                bar_value_shift_to_zero = (
                        inputs["score_values"][:, first_note_idx, 0]
                        - 1 / self.tokenizer.config.additional_params["max_bar_embedding"]
                )
                inputs["score_values"][:, first_note_idx:last_perf_note_idx, bar_idx] -= bar_value_shift_to_zero
            if inputs["perf_values"] is not None:
                inputs["perf_values"][:, first_note_idx:last_perf_note_idx, bar_idx] -= bar_value_shift_to_zero

            with torch.inference_mode():
                # get encoder embeddings
                enc_out = self.model.forward_encoders(
                    score=inputs["score"], score_values=inputs["score_values"], score_mask=inputs["score_mask"],
                    perf=inputs["perf"], perf_values=inputs["perf_values"], perf_mask=inputs["perf_mask"],
                    bars=inputs["bars"], beats=inputs["beats"], onsets=inputs["onsets"],
                    deadpan_mask=inputs["deadpan_mask"],
                    compute_loss=False
                )

            # append new note embeddings
            note_cut_idx = 0
            if overlay_bars:
                note_cut_idx = np.where(sample.score.ids[:, bar_idx] - _bar_0 >= emb_start_bar)[0][0] - first_note_idx

            if enc_out.score_embeddings is not None:
                score_embeddings.append(enc_out.score_embeddings[0, note_cut_idx:])
            if enc_out.perf_embeddings is not None:
                perf_embeddings.append(enc_out.perf_embeddings[0, note_cut_idx:])

            if has_eos:
                break

            # move to the next bars
            if overlay_bars:
                start_bar = sample.score.ids[int(len(sample.score) * (1 - overlay_bars)), 0] - _bar_0
                emb_start_bar = end_bar + 1
            else:
                emb_start_bar = start_bar = end_bar + 1
            end_bar = get_end_bar(score_indices, start_bar, self.dataset.max_seq_len, self.dataset.max_bar)

            # get next sample
            meta.start_bar, meta.end_bar = start_bar, end_bar
            sample = self.dataset.get(meta=meta)

            # process EOS, get new last bar
            has_sos = sample.score.ids[0, bar_idx] == self.sos_token_id
            has_eos = sample.score.ids[-1, bar_idx] == self.eos_token_id
            first_note_idx, last_note_idx = int(has_sos), len(sample.score) - int(has_eos)
            last_perf_note_idx = len(sample.perf) - int(has_eos)
            last_bar = sample.score.ids[last_note_idx - 1, bar_idx] - _bar_0

        score_embeddings = torch.cat(score_embeddings, dim=0) if score_embeddings else None
        perf_embeddings = torch.cat(perf_embeddings, dim=0) if perf_embeddings else None

        latents = None
        if perf_embeddings is not None and compute_latents:
            bars, beats, onsets = self.tokenizer.compute_bar_beat_onset_indices(score_seq)

            bars, beats, onsets = map(
                lambda s: torch.from_numpy(np.concatenate([[s[0]], s, [s[-1]]]))[None].to(self.device),
                (bars, beats, onsets)
            )
            latents = self.model.perf_encoder.embeddings_to_latents(
                embeddings=perf_embeddings[None], bars=bars, beats=beats, onsets=onsets
            )

        return score_embeddings, perf_embeddings, latents

    @staticmethod
    def cut_cache(
            cache: TupleTransformerCache,
            left_idx: int = 0,
            right_idx: int | None = None
    ) -> TupleTransformerCache:
        right_idx = cache.token_emb.shape[-1] if right_idx is None else right_idx
        cache.token_emb = cache.token_emb[:, left_idx:right_idx]
        cache.transformer = TransformerIntermediates(
            output=cache.transformer.output[..., left_idx:right_idx, :],
            layers=[
                TransformerLayerIntermediates(
                    attention=AttentionIntermediates(
                        keys=layer_cache.attention.keys[..., left_idx:right_idx, :],
                        values=layer_cache.attention.values[..., left_idx:right_idx, :]
                    ),
                    output=layer_cache.output[..., left_idx:right_idx, :],
                )
                for layer_cache in cache.transformer.layers
            ],
        )
        return cache


class ScorePerformerInpainter:
    def __init__(
            self,
            model: ScorePerformer,
            dataset: ScorePerformanceDataset,
            collator: MixedLMScorePerformanceCollator,
            device: str | torch.device | None = None
    ):
        self.model = model

        self.dataset = dataset
        self.tokenizer = dataset.tokenizer
        self.collator = collator

        self.sos_token_id = self.tokenizer[0, SOS_TOKEN]
        self.eos_token_id = self.tokenizer[0, EOS_TOKEN]

        self.device = device
        self._init_variables()

    def _init_variables(self):
        num_dims = len(self.tokenizer.vocab_types_idx)
        mask_dims = range(num_dims) if self.collator.mask_token_dims is None else self.collator.mask_token_dims
        self.mask_dims = torch.tensor(list(mask_dims))
        self.dims_mask = torch.zeros(num_dims, dtype=torch.bool, device=self.device)
        self.dims_mask[self.mask_dims] = True

    def fill_silent_notes(
            self,
            perf_idx: int,
            bar_window: int = 4,
            temperature: float = 1.,
            top_k: float | int = 1,
            top_p: float = 1.,
            filter_key_ids: dict[str, list] | None = None,
            disable_tqdm: bool = False,
            verbose: bool = False
    ) -> torch.Tensor:
        # silent note related data
        bar_idx = self.tokenizer.vocab_types_idx["Bar"]
        vel_idx = self.tokenizer.vocab_types_idx["Velocity"]
        try:
            zero_velocity_id = self.tokenizer[vel_idx, "Velocity_0"]
        except KeyError:
            zero_velocity_id = self.tokenizer.zero_token
        self.collator.mlm = 0.  # we mask what we need to unmask

        # get score sequence and its data
        perf = self.dataset.performance_names[perf_idx]
        score, _ = self.dataset._performance_map[perf]
        score_idx = self.dataset.scores._name_to_idx[score]
        score_indices = self.dataset._score_indices[score_idx]
        if score_indices is None:
            score_indices = self.dataset.indexer.compute_bar_indices(self.dataset.scores[score_idx])
            self.dataset._score_indices[score_idx] = score_indices

        # get initial meta and sample
        from symupe.data.datasets.utils import get_end_bar
        start_bar = 0
        end_bar = get_end_bar(score_indices, start_bar, self.dataset.max_seq_len, self.dataset.max_bar)
        meta = ScorePerformanceSampleMeta(
            idx=None, score_idx=score_idx, perf_idx=perf_idx,
            start_bar=start_bar, end_bar=end_bar, bar_shift=0
        )
        sample = self.dataset.get(meta=meta)

        # get current last bar and total number of bars
        _bar_0 = self.tokenizer.zero_token
        score_seq = self.dataset.scores[score_idx]
        has_sos = sample.score.ids[0, 0] == self.sos_token_id
        has_eos = sample.score.ids[-1, 0] == self.eos_token_id
        first_note_idx, last_note_idx = int(has_sos), len(sample.score) - int(has_eos)
        last_perf_note_idx = len(sample.perf) - int(has_eos)
        last_bar = sample.score.ids[-1 - int(has_eos), bar_idx] - _bar_0
        total_bars = score_seq.ids[-1, bar_idx] - _bar_0

        # get the first input sequence
        input_seq = torch.from_numpy(sample.perf.ids).to(self.device)
        total_seq = input_seq[:first_note_idx]
        known_input_len = first_note_idx

        step = new_start_idx = 0
        while last_bar <= total_bars:
            step += 1
            _inputs = self.collator((sample,))
            inputs = self.model.allocate_inputs(self.model.prepare_inputs(_inputs), self.device)

            # apply token mask on unperformed notes
            silent_mask = input_seq[:, vel_idx] == zero_velocity_id
            silent_mask[:known_input_len] = False
            silent_mask = silent_mask[:, None] * self.dims_mask[None]

            # silent_mask = silent_mask[:, self.mask_dims] * DIMS_MASK[None]
            input_seq[silent_mask] = self.collator.mask_token_id

            if verbose:
                print(f"#{step} filling {silent_mask[:, -1].sum().item()} notes in bar range: "
                      f"[{max(0, input_seq[known_input_len - 1, bar_idx] + 1 - _bar_0)}, "
                      f"{inputs['score'][..., bar_idx].max().item() - _bar_0}]\n"
                      f"score_len: {inputs['score'].shape[1]}, input_len: {input_seq.shape[0]}")

            # move bars to 0
            bar_shift_to_zero = inputs["score"][:, first_note_idx, bar_idx] - _bar_0
            inputs["score"][:, first_note_idx:last_note_idx, bar_idx] -= bar_shift_to_zero
            input_seq[first_note_idx:last_perf_note_idx, bar_idx] -= bar_shift_to_zero

            with torch.inference_mode():
                enc_out = self.model.forward_encoders(
                    score=inputs["score"], score_mask=inputs["score_mask"]
                )

                gen_seq, _ = self.model.perf_decoder.unmask(
                    input_seq,
                    single_run=False,
                    context=enc_out.score_embeddings,
                    context_mask=enc_out.score_mask,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    filter_key_ids=filter_key_ids,
                    tokenizer=self.tokenizer,
                    disable_tqdm=disable_tqdm
                )

                # move bars back to absolute values
                input_seq[first_note_idx:last_perf_note_idx, bar_idx] += bar_shift_to_zero
                gen_seq[first_note_idx:last_perf_note_idx, bar_idx] += bar_shift_to_zero

            # append unmasked notes
            total_seq = torch.cat([total_seq, gen_seq[known_input_len:]], dim=0)

            if has_eos:
                break

            # shift bars from left side until `bar_window` fit on the right side
            max_gen_bar = torch.unique(total_seq[..., 0] - _bar_0).max()
            while end_bar < total_bars and end_bar <= meta.end_bar + bar_window and start_bar < max_gen_bar - 1:
                start_bar += 1
                end_bar = get_end_bar(score_indices, start_bar, self.dataset.max_seq_len, self.dataset.max_bar)

            if start_bar - meta.start_bar == 0:  # nothing new
                break

            new_start_idx += np.where(sample.score.ids[..., bar_idx] - _bar_0 >= start_bar)[0][0].item()

            # get the next sample
            meta.start_bar, meta.end_bar = start_bar, end_bar
            sample = self.dataset.get(meta=meta)

            # update note and bar pointers
            has_sos = sample.score.ids[0, bar_idx] == self.sos_token_id
            has_eos = sample.score.ids[-1, bar_idx] == self.eos_token_id
            first_note_idx, last_note_idx = int(has_sos), len(sample.score) - int(has_eos)
            last_perf_note_idx = len(sample.perf) - int(has_eos)
            last_bar = sample.score.ids[last_note_idx - 1, bar_idx] - _bar_0

            # cut input sequence matching by bars
            input_seq = total_seq[new_start_idx:].clone()
            known_input_len = len(input_seq)

            # append performance notes to be processed
            perf_mapped = torch.from_numpy(sample.perf).to(self.device)
            input_seq = torch.cat([input_seq, perf_mapped[known_input_len:]], dim=0)

        return total_seq
