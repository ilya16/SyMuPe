from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np

from ..tokenizers import TokSequence, OctupleM, SyMuPe, SequenceType, EncodingType
from ..tokenizers.constants import MASK_TOKEN, SPECIAL_TOKENS_VALUE


def sample_integer_shift(low: int = -6, high: int = 6):
    return np.random.randint(low, high + 1)


def sample_float_stretch(low: float = -0.5, high: float = 0.5):
    return np.random.uniform(low, high)


@dataclass
class TokenSequenceAugmentations:
    pitch_shift: int | None = 0
    velocity_shift: int | None = 0
    tempo_shift: float | None = 0.


class TokenSequenceProcessor:
    def __init__(
            self,
            pitch_shift_range: tuple[int, int] = (0, 0),
            velocity_shift_range: tuple[int, int] = (0, 0),
            tempo_stretch_range: tuple[float, float] = (0., 0.)
    ):
        self.pitch_shift_fn = partial(sample_integer_shift, *pitch_shift_range)
        self.velocity_shift_fn = partial(sample_integer_shift, *velocity_shift_range)
        self.tempo_shift_fn = partial(sample_float_stretch, *tempo_stretch_range)

    def sample_augmentations(self, multiplier: float = 1.0):
        return TokenSequenceAugmentations(
            pitch_shift=int(multiplier * self.pitch_shift_fn()),
            velocity_shift=int(multiplier * self.velocity_shift_fn()),
            tempo_shift=round(multiplier * self.tempo_shift_fn(), 3)
        )

    def augment_sequence(
            self,
            seq: TokSequence,
            augmentations: TokenSequenceAugmentations
    ) -> TokSequence:
        ...


class TupleTokenSequenceProcessor(TokenSequenceProcessor):
    def __init__(
            self,
            tokenizer: OctupleM,
            pitch_shift_range: tuple[int, int] = (0, 0),
            velocity_shift_range: tuple[int, int] = (0, 0),
            tempo_stretch_range: tuple[float, float] = (0., 0.)
    ):
        super().__init__(pitch_shift_range, velocity_shift_range, tempo_stretch_range)

        self.tokenizer = tokenizer

    def sample_augmentations(self, multiplier: float = 1.0, min_pitch: int | None = None, max_pitch: int | None = None):
        augmentations = super().sample_augmentations(multiplier=multiplier)
        if min_pitch is not None:
            augmentations.pitch_shift = max(augmentations.pitch_shift, self.tokenizer.config.pitch_range[0] - min_pitch)
        if max_pitch is not None:
            augmentations.pitch_shift = min(augmentations.pitch_shift, self.tokenizer.config.pitch_range[1] - max_pitch)
        return augmentations

    def augment_sequence(
            self,
            seq: TokSequence,
            augmentations: TokenSequenceAugmentations
    ) -> TokSequence:
        vocab = seq.vocab or self.tokenizer.vocab_types_idx

        pitch_index = vocab["Pitch"]
        note_mask = seq.ids[:, pitch_index] >= self.tokenizer.zero_token

        if augmentations.pitch_shift != 0:
            seq.ids[note_mask, pitch_index] += augmentations.pitch_shift
            if seq.values is not None:
                seq.values[note_mask, pitch_index] += augmentations.pitch_shift

            if isinstance(self.tokenizer, SyMuPe):
                seq = self.tokenizer.fill_extra_pitch_tokens(seq, force=True)

        if augmentations.velocity_shift != 0 and "Velocity" in vocab:
            type_index = vocab["Velocity"]
            token_mask = seq.ids[:, type_index] > self.tokenizer.zero_token

            if np.any(token_mask):
                velocities = self.tokenizer.get_values(seq, "Velocity")
                velocities[token_mask] += augmentations.velocity_shift

                vel_min, vel_max = self.tokenizer.velocities[0], self.tokenizer.velocities[-1]
                velocities[token_mask] = np.maximum(vel_min, np.minimum(vel_max, velocities[token_mask]))

                if seq.values is not None:
                    seq.values[token_mask, type_index] = velocities[token_mask]
                seq.ids[token_mask, type_index] = self.tokenizer.encode_tokens(velocities[token_mask], "Velocity")

        if augmentations.tempo_shift != 0.:
            if self.tokenizer.config.use_tempos and "Tempo" in vocab:
                type_index = vocab["Tempo"]
                token_mask = seq.ids[:, type_index] >= self.tokenizer.zero_token

                if np.any(token_mask):
                    tempos = self.tokenizer.get_values(seq, "Tempo")
                    tempos[token_mask] *= (1 + augmentations.tempo_shift)

                    tempo_min, tempo_max = self.tokenizer.tempos[0], self.tokenizer.tempos[-1]
                    tempos[token_mask] = np.maximum(tempo_min, np.minimum(tempo_max, tempos[token_mask]))

                    if seq.values is not None:
                        seq.values[token_mask, type_index] = tempos[token_mask]
                    seq.ids[token_mask, type_index] = self.tokenizer.encode_tokens(tempos[token_mask], "Tempo")

            if isinstance(self.tokenizer, SyMuPe) :
                for token_type in ["TimeShift", "TimeDuration", "TimeDurationSustain", "TimePosition"]:
                    if token_type not in vocab:
                        continue

                    type_index = self.tokenizer.vocab_types_idx[token_type]
                    token_mask = seq.ids[:, type_index] >= self.tokenizer.zero_token

                    if np.any(token_mask):
                        token_values = self.tokenizer.get_values(seq, token_type)
                        token_values[token_mask] *= (1 + augmentations.tempo_shift)

                        if seq.values is not None:
                            seq.values[token_mask, type_index] = token_values[token_mask]
                        seq.ids[token_mask, type_index] = self.tokenizer.encode_tokens(
                            token_values[token_mask], token_type
                        )

        return seq

    # Auxiliary processing functions

    def zero_out_durations(self, seq: TokSequence) -> TokSequence:
        vocab = seq.vocab or self.tokenizer.vocab_types_idx
        velocity_index = vocab["Velocity"]
        if "PerfDuration" in vocab and seq.ids.shape[-1] == len(vocab):
            duration_index = vocab["PerfDuration"]
        else:
            duration_index = vocab["Duration"]

        silent_mask = seq.ids[:, velocity_index] == self.tokenizer.zero_token
        seq.ids[silent_mask, duration_index] = self.tokenizer.zero_token
        if seq.values is not None:
            seq.values[silent_mask, duration_index] = self.tokenizer.zero_token

        return seq

    def remove_silent_notes(self, seq: TokSequence) -> TokSequence:
        vocab = seq.vocab or self.tokenizer.vocab_types_idx
        velocity_index = vocab["Velocity"]

        silent_mask = seq.values[:, velocity_index] == 0.
        seq.ids = seq.ids[~silent_mask]
        if seq.values is not None:
            seq.values = seq.values[~silent_mask]

        return seq

    def compute_valid_pitch_mask(self, seq: TokSequence) -> np.ndarray:
        vocab = seq.vocab or self.tokenizer.vocab_types_idx
        pitch_index, velocity_index = vocab["Pitch"], vocab["Velocity"]
        pitch_min, pitch_max = self.tokenizer.zero_token, len(self.tokenizer.vocab[pitch_index]) - 1
        mask = np.logical_or(
            np.logical_and(seq.ids[:, pitch_index] <= self.tokenizer.zero_token, seq.ids[:, velocity_index] <= self.tokenizer.zero_token),
            np.logical_and(seq.ids[:, pitch_index] >= pitch_min, seq.ids[:, pitch_index] <= pitch_max)
        )
        return mask

    @staticmethod
    def compute_seq_type_id(seq: TokSequence, name: str, label: str | None = None):
        return 0

    @staticmethod
    def compute_note_type_ids(seq: TokSequence, name: str, is_interpolated: bool = False) -> np.ndarray:
        if is_interpolated:
            return seq.interpolated.astype(int)

        type_idx = 0
        type_ids = np.full_like(seq.ids[:, 0], fill_value=type_idx)
        return type_ids
