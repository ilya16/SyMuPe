"""
SyMuPe (Symbolic Music Performance) encoding.

The tokenizer is based on the OctupleM encoding. It supports:
    - tokenization of score MIDI sequences;
    - tokenization of performance MIDI sequences using a time-only and a score-aligned encodings;
    - both real-valued features and discrete tokens for each note.
"""

from __future__ import annotations

import warnings
from dataclasses import replace, dataclass
from pathlib import Path

import numpy as np
from miditok.constants import TIME_SIGNATURE
from miditok.utils.utils import tempo_qpm_to_mspq
from symusic import Score, Tempo, TimeSignature, ControlChange

from symupe.utils import find_closest, forward_fill, backward_fill
from .classes import TokSequence, SequenceType, EncodingType, TokSequenceContext, backend
from .constants import (
    TICKS_PER_QUARTER,
    SPECIAL_TOKENS_VALUE,
    TIME_PERFORMANCE_KEYS,
    COMPRESSIBLE_TOKENS,
    BAR_LINE_TOKEN,
    PEDAL_ON_TOKEN,
    PEDAL_OFF_TOKEN,
    TIME_SEGMENT_TOKEN,
    SCORE_KEYS,
)
from .octuple_m import OctupleM
from ..midi.sync import sync_performance_midi, GridLevel
from ..midi.timing import MIDITimeMapper
from ..midi.utils import sort_notes, extract_track_pedals, cut_overlapping_notes
from ..music_constants import NOTES_WSHARP


@dataclass
class SyMuPeTokSequence(TokSequence):
    score_to_perf_token: np.ndarray | None = None


class SyMuPe(OctupleM):
    """Symbolic Music Performance (SyMuPe) encoding.

    Introduced in "SyMuPe: Affective and Controllable Symbolic Music Performance" [1]
    (https://arxiv.org/abs/2511.03425), this class implements a multi-dimensional
    compound encoding for music.

    It evolves the SPMuple encoding from ScorePerformer [2], building upon a modified OctupleM
    encoding (:class:`symupe.data.tokenizers.OctupleM`) to support both score-metrical
    and fine-grained performance features.

    The tokenizer supports three primary encoding modes:
    1. **Score**: Encodes score MIDI sequence.
    2. **Score-Aligned Performance**: Encodes performance nuances relative to a score-metrical grid.
    3. **Time-Only Performance**: Encodes performance using absolute temporal intervals,
       bypassing metrical structures.

    The tokenizer outputs both **real-valued features** and **discrete tokens**,
    which can be used for regression and categorical prediction of score and performance features.

    Each compound token is a tuple containing the following dimensions (if enabled):

    **Base Dimensions (OctupleM):**
    * 0: `Bar`
    * 1: `Position`
    * 2: `Pitch`
    * 3: `Duration`
    * 4: `Velocity`
    * (+ Optional) `Tempo`
    * (+ Optional) `TimeSignature` or (`BeatDuration`, `BeatsInBar` / `MaxBarPosition`)
    * (+ Optional) `Program`

    **Universal / Note Attribute Extensions:**
    * (+ Optional) `PitchClass`: Chroma or pitch class of the note (0–11).
    * (+ Optional) `PitchOctave`: MIDI octave index of the note.

    **Score-Side Auxiliary Dimensions:**
    * (+ Optional) `PositionShift`: Metrical delta between onsets.
    * (+ Optional) `NotesInOnset`: Polyphony count at current onset.
    * (+ Optional) `PositionInOnset`: Index of the note within a chord.

    **Performance-Side Dimensions:**
    * (+ Optional) `OnsetDev` / `RelOnsetDev`: Timing deviation from score onset.
    * (+ Optional) `PerfDuration` / `RelPerfDuration`: Actual duration/articulation.
    * (+ Optional) `TimeShift`: Absolute time between performance onsets (seconds).
    * (+ Optional) `TimeDuration`: Absolute performance duration (seconds).
    * (+ Optional) `TimePosition`: Cyclic temporal position within a segment.
    * (+ Optional) `TimeDurationSustain`: Duration considering sustain pedal state.
    * (+ Optional) `Sustained`: Boolean indicator for active sustain.

    References:
        [1]: Borovik, I., Gavrilev, D., and Viro, V. (2025). "SyMuPe: Affective and
        Controllable Symbolic Music Performance." In Proceedings of the 33rd ACM International
        Conference on Multimedia (ACM MM).
        [2]: Borovik, I., & Viro, V. (2023). "ScorePerformer: Expressive Piano Performance
        Rendering with Fine-Grained Control." In Proceedings of the 24th International Society
        for Music Information Retrieval Conference (ISMIR).
    """

    def _tweak_config_before_creating_voc(self):
        """Tweaks default :class:`TokenizerConfig`."""
        super()._tweak_config_before_creating_voc()

        additional_params = self.config.additional_params

        # midi postprocessing
        additional_params["cut_overlapping_notes"] = True

        # optional compound pitch tokens
        additional_params["use_pitch_classes"] = additional_params.get("use_pitch_classes", False)

        # optional score tokens and their value bins
        additional_params["use_position_shifts"] = additional_params.get(
            "use_position_shifts", False
        )
        additional_params["onset_position_shifts"] = additional_params.get(
            "onset_position_shifts", True
        )
        additional_params["negative_position_shifts"] = additional_params.get(
            "negative_position_shifts", False
        )
        self.position_shifts = None

        additional_params["use_onset_indices"] = additional_params.get("use_onset_indices", False)
        additional_params["max_notes_in_onset"] = additional_params.get("max_notes_in_onset", 12)

        # optional onset deviation performance tokens
        additional_params["use_onset_tokens"] = additional_params.get("use_onset_tokens", False)

        additional_params["rel_onset_dev"] = additional_params.get("rel_onset_dev", False)
        additional_params["num_onset_devs"] = additional_params.get("num_onset_devs", 129)
        self.rel_onset_deviations = additional_params.get("rel_onset_deviations", None)

        additional_params["rel_perf_duration"] = additional_params.get("rel_perf_duration", False)
        additional_params["num_perf_durations"] = additional_params.get("num_perf_durations", 65)
        self.rel_performed_durations = additional_params.get("rel_performed_durations", None)

        # optional time performance tokens
        additional_params["use_time_tokens"] = additional_params.get("use_time_tokens", False)
        additional_params["negative_time_shifts"] = additional_params.get(
            "negative_time_shifts", False
        )
        self.time_shifts = additional_params.get("time_shifts", None)
        self.time_durations = additional_params.get("time_durations", None)

        # optional time position tokens
        additional_params["use_time_positions"] = additional_params.get("use_time_positions", False)
        additional_params["time_position_segment"] = additional_params.get(
            "time_position_segment", 5.0
        )
        additional_params["time_position_step"] = additional_params.get("time_position_step", 0.01)
        self.time_positions = additional_params.get("time_positions", None)

        # optional sustain based tokens
        additional_params["use_sustain_tokens"] = additional_params.get("use_sustain_tokens", False)

        # performance tempo token configuration
        additional_params["bar_tempos"] = additional_params.get("bar_tempos", False)

        assert additional_params["use_onset_tokens"] or additional_params["use_time_tokens"]

    def preprocess_score(
        self,
        midi: Score,
        quantize_times: bool = True,
        quantize_velocities: bool = False,
        quantize_time_signatures: bool = True,
        quantize_tempos: bool = False,
    ) -> Score:
        """Preprocesses a score :class:`symusic.Score` object for the SyMuPe encoding.

        Args:
            midi: :class:`symusic.Score` object to process.
            quantize_times: Resample and quantize note times.
            quantize_velocities: Quantize velocity of each note.
            quantize_time_signatures: Resample and quantize time signature times.
            quantize_tempos: Quantize tempo values.

        Returns:
            Preprocessed :class:`symusic.Score` object.
        """
        return super().preprocess_score(
            midi,
            quantize_times=quantize_times,
            quantize_velocities=quantize_velocities,
            quantize_time_signatures=quantize_time_signatures,
            quantize_tempos=quantize_tempos,
        )

    def preprocess_performance(self, midi: Score) -> Score:
        """Preprocesses a performance :class:`symusic.Score` object for the SyMuPe encoding.

        Does not quantize any of the timing attributes.

        Args:
            midi: :class:`symusic.Score` object representing the performance.

        Returns:
            Preprocessed :class:`symusic.Score` object with maintained micro-timing.
        """
        return self.preprocess_score(
            midi,
            quantize_times=False,
            quantize_velocities=False,
            quantize_time_signatures=False,
            quantize_tempos=False,
        )

    def encode_score(self, midi: Score) -> TokSequence:
        """Converts a score MIDI into a sequence of score SyMuPe tokens.

        Adds optional score-side features like `PitchClass`/`PitchOctave`,
        `PositionShift` and `NotesInOnset`/`PositionInOnset`.

        Args:
            midi: :class:`symusic.Score` object to convert.

        Returns:
            :class:`TokSequence` containing the score representation and note alignments.
        """
        # Preprocess the MIDI file
        midi = self.preprocess_score(midi)

        # Sort notes and compute note order change
        token_to_note_alignments = []
        for track in midi.tracks:
            track.notes, track_token_to_note = sort_notes(track.notes)
            token_to_note_alignments.append(track_token_to_note)
        token_to_note = np.concatenate(token_to_note_alignments)  # note: incorrect when tracks > 1

        # Tokenize it
        tokens = self._score_to_tokens(midi)

        # Add alignment between notes and tokens
        tokens.token_to_note = token_to_note

        num_new_tokens = 2 * int(self.config.additional_params["use_pitch_classes"])
        num_new_tokens += int(self.config.additional_params["use_position_shifts"])
        num_new_tokens += 2 * int(self.config.additional_params["use_onset_indices"])

        if num_new_tokens > 0:
            # Add new score tokens if they are present in the encoding
            if tokens.ids is not None:
                tokens.ids = np.concatenate(
                    [
                        tokens.ids,
                        np.full(
                            (len(tokens.ids), num_new_tokens), fill_value=self.ignore_token
                        ).astype(int),
                    ],
                    axis=1,
                )

            if tokens.values is not None:
                tokens.values = np.concatenate(
                    [
                        tokens.values,
                        np.full((len(tokens.values), num_new_tokens), fill_value=self.ignore_value),
                    ],
                    axis=1,
                )

            tokens = self.fill_extra_pitch_tokens(tokens=tokens)
            tokens = self.fill_extra_score_tokens(tokens=tokens)

        tokens.vocab = {
            ttype: idx for ttype, idx in self.vocab_types_idx.items() if idx < len(tokens.ids[0])
        }

        return tokens

    def fill_pitch_tokens(self, tokens: TokSequence) -> TokSequence:
        """Populates `Pitch` tokens from `PitchClass` and `PitchOctave` components.

        Args:
            tokens: :class:`TokSequence` object to update.

        Returns:
            Updated :class:`TokSequence` object.
        """
        if self.has_token_types(tokens, "Pitch"):
            return tokens

        assert self.config.additional_params["use_pitch_classes"] and self.has_token_types(
            tokens, ["PitchClass", "PitchOctave"]
        )

        vocab = tokens.vocab or self.vocab_types_idx

        if "Pitch" not in vocab:
            tokens.ids = np.concatenate(
                [
                    tokens.ids,
                    np.full((len(tokens.ids), 1), fill_value=self.ignore_token).astype(int),
                ],
                axis=1,
            )
            tokens.values = np.concatenate(
                [tokens.values, np.full((len(tokens.values), 1), fill_value=self.ignore_value)],
                axis=1,
            )
            vocab["Pitch"] = len(vocab)

        pitch_classes = self.get_values(tokens, "PitchClass")
        pitch_octaves = self.get_values(tokens, "PitchOctave")
        pitches = 12 * (pitch_octaves + 1) + pitch_classes

        if tokens.ids is not None:
            tokens.ids[:, vocab["Pitch"]] = self.encode_tokens(pitches, "Pitch")
        if tokens.values is not None:
            tokens.values[:, vocab["Pitch"]] = pitches

        tokens.vocab = vocab

        return tokens

    def fill_extra_pitch_tokens(self, tokens: TokSequence, force: bool = False) -> TokSequence:
        """Decomposes standard `Pitch` tokens into `PitchClass` and `PitchOctave` components.

        Args:
            tokens: :class:`TokSequence` object to update.
            force: If ``True``, forces recomputation of existing pitch component tokens.

        Returns:
            Updated :class:`TokSequence` object.
        """
        new_values = {}

        if self.config.additional_params["use_pitch_classes"] and (
            force or not self.has_token_types(tokens, ["PitchClass", "PitchOctave"])
        ):
            # Divide pitches into two components
            pitches = self.get_values(tokens, "Pitch")
            pitch_mask = pitches >= 0
            new_values["PitchClass"] = np.where(pitch_mask, pitches % 12, pitches)
            new_values["PitchOctave"] = np.where(pitch_mask, pitches // 12 - 1, pitches)

        for token_type, values in new_values.items():
            if tokens.ids is not None:
                tokens.ids[:, self.vocab_types_idx[token_type]] = self.encode_tokens(
                    values, token_type
                )
            if tokens.values is not None:
                tokens.values[:, self.vocab_types_idx[token_type]] = values

        return tokens

    def fill_extra_score_tokens(self, tokens: TokSequence, force: bool = False) -> TokSequence:
        """Computes auxiliary score tokens (`PositionShift`, `NotesInOnset`, `PositionInOnset`)
        from existing Bar and Position information.

        Args:
            tokens: :class:`TokSequence` object to update.
            force: If ``True``, forces recomputation of existing pitch component tokens.

        Returns:
            Updated :class:`TokSequence` object.
        """
        # Add new score tokens if they are present in the encoding
        new_values = {}

        if not self.has_token_types(tokens, ["Bar", "Position"]):
            return tokens

        score_positions = None
        if self.config.additional_params["use_position_shifts"] and (
            force or not self.has_token_types(tokens, "PositionShift")
        ):
            if score_positions is None:
                score_positions = self.compute_ticks(
                    tokens, time_division=self.config.max_num_pos_per_beat
                )["note_on"]

            new_values["PositionShift"] = np.maximum(
                self.compute_position_shifts(score_positions), SPECIAL_TOKENS_VALUE + 1
            )
            new_values["PositionShift"][0] = self.get_values(tokens, "Position")[0]

        if self.config.additional_params["use_onset_indices"] and (
            force or not self.has_token_types(tokens, ["NotesInOnset", "PositionInOnset"])
        ):
            if score_positions is None:
                score_positions = self.compute_ticks(
                    tokens, time_division=self.config.max_num_pos_per_beat
                )["note_on"]

            _, notes_in_onset, pos_in_onset = self.compute_onset_values(score_positions)
            new_values["NotesInOnset"] = notes_in_onset
            new_values["PositionInOnset"] = pos_in_onset

        for token_type, values in new_values.items():
            if tokens.ids is not None:
                tokens.ids[:, self.vocab_types_idx[token_type]] = self.encode_tokens(
                    values, token_type
                )
            if tokens.values is not None:
                tokens.values[:, self.vocab_types_idx[token_type]] = values

        return tokens

    def encode_performance(
        self,
        midi: Score,
        *,
        score_tokens: TokSequence | None,
        note_alignment: np.ndarray | None = None,
    ) -> TokSequence:
        """Converts a performance MIDI into a sequence of performance SyMuPe tokens,
        optionally aligned to a score.

        Supports two distinct encoding modes:

        1. **Score-aligned** (`score_tokens` is not ``None``):
            Uses score and performance tokens (`symupe.data.tokenizers.constants.PERFORMANCE_KEYS`).
            If `config.additional_params["use_onset_tokens"]` is ``True``, the encoding also includes
            relative performance tokens (`symupe.data.tokenizers.constants.REL_PERFORMANCE_KEYS`)

            `note_alignment` should be used to provide the MIDI-level mapping between
            the score and performance notes. The alignment on the token level is computed inside
            using `score_tokens.token_to_note` (alignment between score tokens and notes)
            and is returned as a token sequence metadata.

        2. **Time-only** (`score_tokens` is ``None``):
            Uses performance-only tokens (`symupe.data.tokenizers.constants.TIME_PERFORMANCE_KEYS`).
            Score and relative performance tokens/values are set to `IGNORE_TOKEN`/`IGNORE_VALUE`

        Args:
            midi: Performance :class:`symusic.Score` object.
            score_tokens: Optional reference score tokens for aligned encoding.
            note_alignment: Optional precomputed mapping between score and performance notes.

        Returns:
            :class:`TokSequence` object containing performance features.
        """
        # Preprocess the MIDI file
        self.preprocess_performance(midi)

        # Sort notes and compute note order change
        token_to_note_alignments = []
        for track in midi.tracks:
            track.notes, track_token_to_note = sort_notes(track.notes)
            token_to_note_alignments.append(track_token_to_note)
        perf_token_to_perf_note = np.concatenate(token_to_note_alignments)

        score_token_to_perf_token = None
        if score_tokens is not None:
            # Compute the token-level alignment using the note-level alignment
            # and token-to-note alignments for the score and performance sequences
            #
            # Given: `alignment` = `score_note_to_perf_note`,
            #        `score_token_to_score_note`,
            #        `perf_note_to_perf_token`
            # Find:  `score_token_to_perf_token` alignment
            # Transitivity Rule: A->B = C->B[A->C]

            score_note_to_perf_note = (
                np.arange(len(score_tokens)) if note_alignment is None else note_alignment
            )
            score_token_to_score_note = score_tokens.token_to_note
            perf_note_to_perf_token = np.argsort(perf_token_to_perf_note)

            if score_token_to_score_note is not None:
                score_token_to_perf_note = score_note_to_perf_note[score_token_to_score_note]
            else:
                score_token_to_perf_note = score_note_to_perf_note
            score_token_to_perf_token = perf_note_to_perf_token[score_token_to_perf_note]

            perf_token_to_perf_note = score_token_to_perf_note

        # Tokenize it
        tokens = self._encode_performance(midi, score_tokens, score_token_to_perf_token)

        # Add alignment between notes and tokens
        tokens = vars(tokens)
        tokens.update(
            token_to_note=perf_token_to_perf_note,
            score_to_perf_token=score_token_to_perf_token,
        )
        tokens["meta"] = tokens.get("meta", {})
        tokens["meta"].update(time_division=midi.ticks_per_quarter)
        tokens = TokSequence(**tokens)

        return tokens

    def _encode_performance(
        self,
        midi: Score,
        score_tokens: TokSequence | None,
        note_alignment: np.ndarray | None = None,
    ) -> TokSequence:
        """Internal implementation of performance encoding logic.

        Calculates onset deviations and performance durations relative to the score grid
        or uses `self._encode_time_only_performance` to create a time-only encoding.

        Args:
            midi: Performance :class:`symusic.Score` object.
            score_tokens: Optional reference score tokens for aligned encoding.
            note_alignment: Optional precomputed mapping between score and performance notes.

        Returns:
            :class:`TokSequence` object containing performance features.
        """
        self._current_midi_metadata = {
            "time_division": midi.ticks_per_quarter,
            "tempos": midi.tempos,
        }

        if score_tokens is None:
            return self._encode_time_only_performance(midi)

        additional_params = self.config.additional_params

        # Prepare constants used for calculations
        time_division = midi.ticks_per_quarter
        ticks_per_sample = time_division / self.config.max_num_pos_per_beat

        # Merge track into one
        notes = midi.tracks[0].notes
        for track in midi.tracks[1:]:
            notes.extend(track.notes)

        # Sort by time, pitch, duration, velocity
        # Note: (?) sorting for multi-instrumental music should be provided in alignment
        notes, _ = sort_notes(notes, order="time")
        note_soa = notes.numpy()

        # Save performance position and duration ticks
        perf_positions = note_soa["time"] / ticks_per_sample
        perf_durations = note_soa["duration"] / ticks_per_sample

        # Compute time positions
        time_mapper = MIDITimeMapper(midi)
        perf_times = time_mapper.t2s(perf_positions * ticks_per_sample)
        perf_offset_times = time_mapper.t2s((perf_positions + perf_durations) * ticks_per_sample)
        perf_time_shifts = np.diff(np.concatenate([[0.0], perf_times]))
        perf_time_durations = perf_offset_times - perf_times

        # Find the closest tempo for each note
        tempo_soa = midi.tempos.numpy()
        tempo_positions = tempo_soa["time"] / ticks_per_sample
        perf_tempos = tempo_qpm_to_mspq(tempo_soa["mspq"])[
            np.minimum(
                np.searchsorted(tempo_positions, perf_positions, side="right") - 1,
                tempo_positions.shape[0] - 1,
            )
        ]

        # Construct an array of values
        values = np.full((len(perf_positions), len(self.score_sizes)), fill_value=self.ignore_value)

        # Fill in pitch, velocity, and tempo values
        values[:, self.vocab_types_idx["Pitch"]] = note_soa["pitch"]
        values[:, self.vocab_types_idx["Velocity"]] = note_soa["velocity"]

        # Compute NoteON, Time Signature, Bar and Beat ticks
        ticks_data = self.compute_ticks(score_tokens, time_division=time_division)
        note_on_ticks = ticks_data["note_on"]
        beat_ticks = (
            ticks_data["bar"] if self.config.additional_params["bar_tempos"] else ticks_data["beat"]
        )

        # Map note ticks to beats
        note_beats = beat_ticks[
            np.minimum(
                np.searchsorted(beat_ticks, note_on_ticks, side="right") - 1,
                beat_ticks.shape[0] - 1,
            )
        ]

        # Process tempos and their jumping positions
        # Record beat tempos before applying alignment
        if note_alignment is not None:
            note_beats = note_beats[np.argsort(note_alignment)]

        note_beat_tempo = np.stack([note_beats, perf_tempos], axis=1)
        un_beat_tempos, counts = np.unique(note_beat_tempo, return_counts=True, axis=0)
        beat_tempo_data = np.concatenate([un_beat_tempos, counts[:, None]], axis=1)

        beat_tempos = []
        while len(beat_tempo_data) > 0:
            beat_tempos_ = beat_tempo_data[beat_tempo_data[:, 0] == beat_tempo_data[0, 0]]
            beat_tempos.append(beat_tempos_[beat_tempos_[:, 2].argmax(), :2])
            beat_tempo_data = beat_tempo_data[len(beat_tempos_) :]
        beat_tempos = np.stack(beat_tempos)

        # Apply alignment
        if note_alignment is not None:
            (
                values,
                perf_positions,
                perf_durations,
                perf_times,
                perf_time_shifts,
                perf_time_durations,
            ) = map(
                lambda x: x[note_alignment],
                (
                    values,
                    perf_positions,
                    perf_durations,
                    perf_times,
                    perf_time_shifts,
                    perf_time_durations,
                ),
            )

        # Put back correct beat tempos
        values[:, self.vocab_types_idx["Tempo"]] = beat_tempos[
            np.searchsorted(beat_tempos[:, 0], note_beats)
        ][:, 1]

        # Copy score values to performance values
        for token_type in self.score_only_tokens:
            idx = self.vocab_types_idx[token_type]
            values[:, idx] = score_tokens.values[:, idx]

        if additional_params["use_onset_tokens"]:
            # Compute score positions and durations
            score_positions = ticks_data["note_on"] / ticks_per_sample
            score_durations = ticks_data["duration"] / ticks_per_sample

            # Compute OnsetDeviation and PerformanceDuration values
            onset_devs = perf_positions - score_positions

            # Scale onset deviations based on score durations if needed
            if additional_params["rel_onset_dev"]:
                pos_shifts = self.compute_position_shifts(score_positions, onset_shift=True)
                pos_shifts[pos_shifts == 0] = 1
                onset_dev_values = onset_devs / pos_shifts
            else:
                onset_dev_values = onset_devs

            # Scale performed durations based on score durations if needed
            if additional_params["rel_perf_duration"]:
                perf_duration_values = perf_durations / score_durations
            else:
                perf_duration_values = perf_durations

            # Append (Rel)OnsetDev and (Rel)PerfDuration values
            values = np.concatenate(
                [
                    values,
                    onset_dev_values[:, None],
                    perf_duration_values[:, None],
                ],
                axis=1,
            )

        # Append TimeShift/TimeDuration values
        if additional_params["use_time_tokens"]:
            if additional_params["negative_time_shifts"]:
                perf_time_shifts = np.diff(np.concatenate([[0.0], perf_times]))
                if perf_time_shifts[0] < 0.0:
                    perf_times -= perf_time_shifts[0]
                    perf_time_shifts -= perf_time_shifts[0]

            perf_time_shifts = np.maximum(perf_time_shifts, SPECIAL_TOKENS_VALUE + 1)

            values = np.concatenate(
                [
                    values,
                    perf_time_shifts[:, None],
                    perf_time_durations[:, None],
                ],
                axis=1,
            )

            # Append TimePosition values
            if additional_params["use_time_positions"]:
                values = np.concatenate(
                    [
                        values,
                        np.round(
                            perf_times[:, None] % additional_params["time_position_segment"], 6
                        ),
                    ],
                    axis=1,
                )

        # Append TimeDurationSustain and Sustained values
        if self.config.additional_params["use_sustain_tokens"]:
            values = np.concatenate(
                [
                    values,
                    perf_time_durations[:, None],
                    np.zeros_like(perf_time_durations[:, None]),
                ],
                axis=1,
            )

        # Convert values to tokens and build final TokSequence
        tokens = self.encode_tokens(values, clip=True)

        pedals = self._extract_pedals(midi=midi)
        if pedals is not None:
            pedals[:, 1] = time_mapper.t2s(pedals[:, 1])

        tokens = TokSequence(
            ids=tokens,
            values=values,
            pedals=pedals,
            type=SequenceType.PERFORMANCE,
            encoding=EncodingType.PERFORMANCE,
            vocab=self.vocab_types_idx,
            meta={
                "time_division": midi.ticks_per_quarter,
                "bars": int(tokens[-1, self.vocab_types_idx["Bar"]] - self.zero_token + 1),
            },
        )

        tokens = self.fill_extra_pitch_tokens(tokens=tokens)

        return tokens

    def _encode_time_only_performance(self, midi: Score) -> TokSequence:
        """Tokenizes a performance :class:`symusic.Score` into :class:`TokSequence`
        with performance-only (`Pitch`, `Velocity`, TimeShift, TimeDuration) tokens.

        Args:
            midi: performance :class:`symusic.Score` object.

        Returns:
            :class:`TokSequence` object containing performance features.
        """
        additional_params = self.config.additional_params
        assert additional_params["use_time_tokens"], (
            "TimeShift and TimeDuration tokens should be present in the tokenizer to compute "
            "a performance time only tokenization with a score metrical grid"
        )

        # Convert midi events to seconds
        midi = midi.to("second")

        # Merge track into one
        notes = midi.tracks[0].notes
        for track in midi.tracks[1:]:
            notes.extend(track.notes)

        # Sort by time, pitch, duration, velocity
        notes, _ = sort_notes(notes, order="time")
        note_soa = notes.numpy()

        # Construct an array of values
        values = np.full((len(notes), len(self.performance_sizes)), fill_value=self.ignore_value)

        # Fill in pitch, velocity, time shift, and time duration values
        values[:, self.vocab_types_idx["Pitch"]] = note_soa["pitch"]
        values[:, self.vocab_types_idx["Velocity"]] = note_soa["velocity"]

        if additional_params["use_pitch_classes"]:
            values[:, self.vocab_types_idx["PitchClass"]] = note_soa["pitch"] % 12
            values[:, self.vocab_types_idx["PitchOctave"]] = note_soa["pitch"] // 12 - 1

        values[:, self.vocab_types_idx["TimeShift"]] = np.diff(
            np.concatenate([[0.0], note_soa["time"]])
        )
        values[:, self.vocab_types_idx["TimeDuration"]] = note_soa["duration"]

        if additional_params["use_time_positions"]:
            time_segment = additional_params["time_position_segment"]
            values[:, self.vocab_types_idx["TimePosition"]] = np.round(
                note_soa["time"] % time_segment, 6
            )

        # Append TimeDurationSustain values
        if self.config.additional_params["use_sustain_tokens"]:
            values[:, self.vocab_types_idx["TimeDurationSustain"]] = note_soa["duration"]
            values[:, self.vocab_types_idx["Sustained"]] = np.zeros_like(note_soa["duration"])

        # Convert values to tokens and build final TokSequence
        tokens = self.encode_tokens(values)

        return TokSequence(
            ids=tokens,
            values=values,
            pedals=self._extract_pedals(midi=midi),
            type=SequenceType.TIME_PERFORMANCE,
            encoding=EncodingType.TIME_PERFORMANCE,
            vocab=self.vocab_types_idx,
            meta={"time_division": midi.ticks_per_quarter},
        )

    def _extract_pedals(self, midi: Score) -> np.ndarray | None:
        """Extracts sustain pedal (CC64) events from a MIDI file.

        Converts pedals into a standardized token-compatible array format.

        Args:
            midi: :class:`symusic.Score` object to process.

        Returns:
            Array of pedals (token_id, time)
        """
        controls = midi.tracks[0].controls
        for track in midi.tracks[1:]:
            controls.extend(track.controls)

        sustain_ons, sustain_offs = extract_track_pedals(midi.tracks[0])

        if PEDAL_ON_TOKEN in self.special_tokens and PEDAL_OFF_TOKEN in self.special_tokens:
            on_token_id, off_token_id = self[0, PEDAL_ON_TOKEN], self[0, PEDAL_OFF_TOKEN]
        else:
            on_token_id, off_token_id = 1, 0

        pedal_on_values = np.stack(
            (
                np.full(len(sustain_ons), fill_value=SPECIAL_TOKENS_VALUE - on_token_id),
                sustain_ons,
            ),
            axis=-1,
        )
        pedal_off_values = np.stack(
            (
                np.full(len(sustain_offs), fill_value=SPECIAL_TOKENS_VALUE - off_token_id),
                sustain_offs,
            ),
            axis=-1,
        )

        pedals = np.concatenate((pedal_on_values, pedal_off_values), axis=0)
        pedals = pedals[np.lexsort((pedals[:, 0], pedals[:, 1]))]

        return pedals

    def compress(
        self,
        tokens: TokSequence,
        minimal: bool = False,
        token_types: list[str] | None = None,
    ) -> TokSequence:
        """Reduces :class:`TokSequence` object to its minimal representation.

        Args:
            tokens: :class:`TokSequence` object to compress.
            minimal: If ``True``, removes all compressible/recomputable tokens.
            token_types: Specific token types to keep, if ``None``, uses defaults based on sequence type.

        Returns:
            Compressed :class:`TokSequence` object.
        """
        seq_type = tokens.type
        assert seq_type is not None

        assert tokens.ids is not None or tokens.values is not None
        data = tokens.ids if tokens.ids is not None else tokens.values
        vocab = tokens.vocab or self.vocab_types_idx

        if token_types is None:
            if seq_type in (SequenceType.TIME_PERFORMANCE, SequenceType.TIME_PERFORMANCE_SUSTAIN):
                token_types = self.time_performance_sizes.keys()
            elif seq_type in (SequenceType.SCORE, SequenceType.SYNC_PERFORMANCE):
                token_types = self.score_sizes.keys()
            else:
                token_types = list(self.vocab_types_idx.keys())

            if minimal:
                token_types = [key for key in token_types if key not in COMPRESSIBLE_TOKENS]

        if len(data) and len(data[0]) > len(token_types):
            ttype_ids = [vocab[key] for key in token_types]

            if tokens.ids is not None:
                tokens.ids = tokens.ids[:, ttype_ids]
            if tokens.values is not None:
                tokens.values = tokens.values[:, ttype_ids]

        tokens.vocab = {ttype: idx for idx, ttype in enumerate(token_types)}

        return tokens

    def decompress(self, tokens: TokSequence) -> TokSequence:
        """Restores a compressed :class:`TokSequence` object to the full-token representation.

        Recomputes missing values (like Pitch components or extra score tokens) where possible.

        Args:
            tokens: :class:`TokSequence` object to decompress.

        Returns:
            Decompressed :class:`TokSequence` object.
        """
        seq_type = tokens.type
        assert seq_type is not None

        assert tokens.ids is not None or tokens.values is not None
        data = tokens.ids if tokens.ids is not None else tokens.values

        if len(data) == 0:
            tokens.vocab = self.vocab_types_idx
        elif len(data[0]) < len(self.performance_sizes):
            ids, values = tokens.ids, tokens.values
            num_tokens = len(data[0])
            shape = (len(data), len(self.performance_sizes))

            token_types = list((tokens.vocab or {}).keys())
            if len(token_types) == 0:
                if seq_type in (
                    SequenceType.TIME_PERFORMANCE,
                    SequenceType.TIME_PERFORMANCE_SUSTAIN,
                ):
                    token_types = self.time_performance_sizes.keys()
                else:
                    token_types = self.score_sizes.keys()
                assert len(token_types) == num_tokens
            elif len(token_types) > num_tokens:
                token_types = token_types[:num_tokens]

            ttype_ids = [self.vocab_types_idx[key] for key in token_types]

            _backend = backend(data)
            if tokens.ids is not None:
                tokens.ids = _backend.full(shape, fill_value=self.ignore_token, dtype=int)
                tokens.ids[:, ttype_ids] = ids
            if tokens.values is not None:
                tokens.values = _backend.full(shape, fill_value=self.ignore_value)
                tokens.values[:, ttype_ids] = values

            tokens.vocab = self.vocab_types_idx

            tokens = self.fill_extra_pitch_tokens(tokens=tokens)
            if seq_type not in (
                SequenceType.TIME_PERFORMANCE,
                SequenceType.TIME_PERFORMANCE_SUSTAIN,
            ):
                tokens = self.fill_extra_score_tokens(tokens=tokens)

        return tokens

    def decode_score(
        self,
        tokens: TokSequence | list[list[int]] | np.ndarray,
        programs: list[tuple[int, bool]] | None = None,
        output_path: str | None = None,
    ) -> Score:
        """Decodes a sequence of score tokens into a ``symusic.Score``.

        Args:
            tokens: Sequence of tokens to convert.
            programs: Programs of the tracks. If ``None``, will default to piano (program 0).
            output_path: Path to save the MIDI file.

        Returns:
            :class:`symusic.Score` object.
        """
        return super().decode(tokens, programs=programs, output_path=output_path)

    def decode_note_positions(
        self,
        tokens: TokSequence,
        context: TokSequenceContext | None = None,
        time_division: int = TICKS_PER_QUARTER,
    ) -> tuple[dict[str, any], TokSequenceContext]:
        """Decodes temporal metadata from a token sequence to determine note onsets/offsets.

        Extracts `Bar`, `Position`, and `TimeShift` information to calculate absolute
        ticks and seconds. If `use_onset_tokens` is active, it applies
        `RelOnsetDev` and `RelPerfDuration` to the score-metrical grid.

        Args:
            tokens: :class:`TokSequence` object to decode.
            context: Optional :class:`TokSequenceContext` for incremental decoding.
            time_division: MIDI time division / resolution, in ticks/beat.

        Returns:
            Tuple containing a dictionary of position data (ticks, times, tempos)
            and the updated :class:`TokSequenceContext` object.
        """
        additional_params = self.config.additional_params
        time_division = time_division or self.time_division
        ticks_per_sample = time_division // self.config.max_num_pos_per_beat

        context = context or TokSequenceContext()
        prev_score_ticks = context.score_ticks if context.score_ticks is not None else np.zeros(1)
        prev_note_on_times = (
            context.note_on_times if context.note_on_times is not None else np.zeros(1)
        )
        prev_tempos, prev_tempo_ticks, prev_tempo_times = context.tempos or (None, None, None)

        ticks_data, score_ticks = None, None
        note_on_ticks, note_off_ticks = None, None
        note_on_times, note_off_times = None, None
        tempos, tempo_ticks, tempo_times = None, None, None
        pedals = None

        time_duration_token = (
            "TimeDuration" if "TimeDuration" in tokens.vocab else "TimeDurationSustain"
        )
        has_time_tokens = additional_params["use_time_tokens"] and self.has_token_types(
            tokens, ["TimeShift", time_duration_token]
        )
        if has_time_tokens:
            # Note times
            time_shifts = self.get_values(tokens, "TimeShift")
            note_on_times = np.cumsum(time_shifts) + prev_note_on_times[-1]
            note_on_times = np.maximum(0, note_on_times)

            perf_time_durations = self.get_values(tokens, time_duration_token)
            note_off_times = note_on_times + perf_time_durations

            pedals = tokens.pedals

        has_score = self.has_token_types(tokens, ["Bar", "Position"]) or (
            additional_params["use_position_shifts"]
            and self.has_token_types(tokens, ["PositionShift"])
        )
        if has_score and tokens.type not in (
            SequenceType.TIME_PERFORMANCE,
            SequenceType.TIME_PERFORMANCE_SUSTAIN,
        ):
            # Compute NoteON, Time Signature, Bar and Beat ticks
            ticks_data = self.compute_ticks(tokens, context=context, time_division=time_division)

            score_ticks = ticks_data["note_on"].round()
            score_durations = ticks_data["duration"].round()

            decode_from_onsets = additional_params["use_onset_tokens"]
            if decode_from_onsets:
                assert additional_params["use_onset_tokens"] and self.has_token_types(
                    tokens, [self.onset_deviation_token, self.performed_duration_token]
                )

                # Compute position shifts
                pos_shifts = self.compute_position_shifts(
                    np.concatenate([prev_score_ticks, score_ticks]), onset_shift=True
                )[len(prev_score_ticks) :]

                # Onset Deviations to ticks
                if additional_params["rel_onset_dev"]:
                    rel_onset_devs = self.get_values(tokens, "RelOnsetDev")
                    pos_shifts[pos_shifts == 0] = 1
                    onset_devs = rel_onset_devs * pos_shifts
                else:
                    onset_devs = self.get_values(tokens, "OnsetDev")
                    onset_devs = onset_devs * ticks_per_sample

                # Shift onsets
                note_on_ticks = score_ticks + onset_devs
                note_on_ticks = np.maximum(0, note_on_ticks).round().astype(int)

                # Performed Durations to ticks and NoteOFF ticks
                if additional_params["rel_perf_duration"]:
                    rel_perf_durations = self.get_values(tokens, "RelPerfDuration")
                    perf_durations = rel_perf_durations * score_durations
                else:
                    perf_durations = self.get_values(tokens, "PerfDuration")
                    perf_durations = perf_durations * ticks_per_sample

                perf_durations = perf_durations.round().astype(int)
                note_off_ticks = note_on_ticks + perf_durations

                # Process Tempo changes
                tempos, tempo_ticks, tempo_times = self._decode_tempos(
                    tempo_values=self.get_values(tokens, "Tempo"),
                    prev_tempos=prev_tempos,
                    prev_tempo_ticks=prev_tempo_ticks,
                    prev_tempo_times=prev_tempo_times,
                    beat_ticks=(
                        ticks_data["bar"] if additional_params["bar_tempos"] else ticks_data["beat"]
                    ),
                    score_ticks=score_ticks,
                    time_division=time_division,
                )

                # Build new context
                if prev_tempos is not None and tempos is not None:
                    tempos = np.concatenate([prev_tempos, tempos[1:]], axis=0)
                    tempo_ticks = np.concatenate([prev_tempo_ticks, tempo_ticks[1:]], axis=0)
                    tempo_times = np.concatenate([prev_tempo_times, tempo_times[1:]], axis=0)

                # Remove duplicates by ticks and tempos
                tempos, tempo_ticks, tempo_times = self._filter_equal_tempos(
                    tempos, tempo_ticks, tempo_times
                )

                _offset = np.where(note_on_ticks.min() >= tempo_ticks)[0][-1]
                note_on_times, note_off_times = self._decode_note_times(
                    note_on_ticks,
                    note_off_ticks,
                    tempos[_offset:],
                    tempo_ticks[_offset:],
                    tempo_times[_offset:],
                    time_division=time_division,
                )

        position_data = {
            "ticks_data": ticks_data,
            "note_on_ticks": note_on_ticks,
            "note_off_ticks": note_off_ticks,
            "note_on_times": note_on_times,
            "note_off_times": note_off_times,
            "tempos": (tempos, tempo_ticks, tempo_times),
            "pedals": pedals,
        }

        def extend_context(prev_data, new_data):
            return np.concatenate([prev_data, new_data]) if prev_data is not None else new_data

        new_context = TokSequenceContext(
            time_signatures=ticks_data["time_sig"] if ticks_data is not None else None,
            tempos=(tempos, tempo_ticks, tempo_times),
            score_ticks=extend_context(context.score_ticks, score_ticks),
            note_on_ticks=extend_context(context.note_on_ticks, note_on_ticks),
            note_on_times=extend_context(context.note_on_times, note_on_times),
            pedals=extend_context(context.pedals, pedals),
        )

        return position_data, new_context

    def decode_performance(
        self,
        tokens: TokSequence | list[list[int]] | np.ndarray,
        programs: list[tuple[int, bool]] | None = None,
        time_division: int = TICKS_PER_QUARTER,
        output_path: str | None = None,
        **kwargs,
    ) -> Score:
        """Decodes a sequence of performance tokens into a ``symusic.Score``.

        Args:
            tokens: Sequence of tokens to convert.
            programs: Programs of the tracks. If ``None``, will default to piano (program 0).
            time_division: MIDI time division / resolution, in ticks/beat.
            output_path: Path to save the MIDI file.

        Returns:
            :class`symusic.Score` object.
        """
        if not isinstance(tokens, (TokSequence, list)) or (
            isinstance(tokens, list) and any(not isinstance(seq, TokSequence) for seq in tokens)
        ):
            tokens = self._convert_sequence_to_tokseq(tokens)

        # Preprocess TokSequence(s)
        if isinstance(tokens, TokSequence):
            self._preprocess_tokseq_before_decoding(tokens)
        else:  # list[TokSequence]
            for seq in tokens:
                self._preprocess_tokseq_before_decoding(seq)

        midi = self._decode_performance(tokens, programs, time_division, **kwargs)

        # Create controls for pedals
        # This is required so that they are saved when the MIDI is dumped, as symusic
        # will only write the control messages.
        if self.config.use_sustain_pedals:
            for track in midi.tracks:
                for pedal in track.pedals:
                    track.controls.append(ControlChange(pedal.time, 64, 127))
                    track.controls.append(ControlChange(pedal.end, 64, 0))
                if len(track.pedals) > 0:
                    track.controls.sort()

        # Set default tempo and time signatures at tick 0 if not present
        if len(midi.tempos) == 0 or midi.tempos[0].time != 0:
            midi.tempos.insert(0, Tempo(0, self.default_tempo))
        if len(midi.time_signatures) == 0 or midi.time_signatures[0].time != 0:
            midi.time_signatures.insert(0, TimeSignature(0, *TIME_SIGNATURE))

        if self.config.additional_params["cut_overlapping_notes"]:
            for track in midi.tracks:
                track.notes = cut_overlapping_notes(track.notes, sort=True)

        # Write MIDI file
        if output_path:
            Path(output_path).mkdir(parents=True, exist_ok=True)
            midi.dump_midi(output_path)
        return midi

    def _decode_performance(
        self,
        tokens: TokSequence,
        context: TokSequenceContext | None = None,
        time_division: int = TICKS_PER_QUARTER,
        sync_midi: bool = True,
    ) -> Score:
        """Internal logic for performance decoding.

        Reconstructs note events, tempos, and control changes, optionally synchronizing to a metrical grid.

        Calculates onset deviations and performance durations relative to the score grid
        or uses `self._encode_time_only_performance` to create a time-only encoding.

        Args:
            tokens: Sequence of tokens to convert.
            context: :class:`TokSequenceContext` from the preceding notes.
            time_division: MIDI time division / resolution, in ticks/beat.
            sync_midi: If ``True``, synchronizes performance MIDI to score beat grid.

        Returns:
            :class:`symusic.Score` object.
        """
        additional_params = self.config.additional_params

        if time_division % self.config.max_num_pos_per_beat != 0:
            raise ValueError(
                f"Invalid time division, please give one divisible by"
                f"{self.config.max_num_pos_per_beat}"
            )

        # Note attributes
        pitches = self.get_values(tokens, "Pitch", from_ids=True)
        velocities = self.get_values(tokens, "Velocity")

        # Time attributes
        position_data, new_context = self.decode_note_positions(
            tokens=tokens, context=context, time_division=time_division
        )
        ticks_data = position_data["ticks_data"]
        note_on_ticks, note_off_ticks = (
            position_data["note_on_ticks"],
            position_data["note_off_ticks"],
        )
        note_on_times, note_off_times = (
            position_data["note_on_times"],
            position_data["note_off_times"],
        )
        pedals = position_data["pedals"]

        midi_is_score = False
        if ticks_data is None:  # time-performance encoding
            ttype = "second"
            midi_times, midi_durations = note_on_times, note_off_times - note_on_times
            tempos, time_signatures = None, None
        else:
            ttype = "tick"

            if additional_params["use_onset_tokens"] and note_on_ticks is not None:
                midi_times, midi_durations = note_on_ticks, note_off_ticks - note_on_ticks
            else:  # `midi` is score MIDI using which we might synchronize later
                midi_times, midi_durations = (
                    ticks_data["note_on"].round(),
                    ticks_data["duration"].round(),
                )
                midi_is_score = True

            # Build Tempo changes
            tempos, tempo_ticks, tempo_times = position_data["tempos"] or (None, None, None)
            if tempos is not None and not midi_is_score:
                tempos = Tempo.from_numpy(time=tempo_ticks, mspq=tempo_qpm_to_mspq(tempos))
            else:
                tempos = None  # SyMuPeLocal

            # Build Time Signature changes
            time_sigs, time_sig_ticks = ticks_data["time_sig"]
            time_signatures = TimeSignature.from_numpy(
                time=time_sig_ticks, numerator=time_sigs[:, 0], denominator=time_sigs[:, 1]
            )

        controls = (
            ControlChange.from_numpy(
                time=pedals[:, 1],
                number=np.full_like(pedals[:, 0], fill_value=64.0),
                value=(pedals[:, 0] > 0) * 127.0,
                ttype="second",
            )
            if pedals is not None and len(pedals) > 0
            else None
        )

        # Process Programs
        programs = (
            self.get_values(tokens, "Program", from_ids=True) if self.config.use_programs else None
        )

        midi = self._build_score(
            times=midi_times,
            durations=midi_durations,
            pitches=pitches,
            velocities=velocities,
            programs=programs,
            time_signatures=time_signatures,
            tempos=tempos,
            time_division=time_division,
            ttype=ttype,
        )

        if midi_is_score:
            # performance MIDI is not created from bar/beat onset deviations or can be synchronizable with the score

            onset_pairs = new_context.onset_pairs
            if onset_pairs is None:
                # Record performed score note ticks
                score_ticks = ticks_data["note_on"].round()
                is_performed = velocities != 0.0
                performed_ticks, perf_times = score_ticks[is_performed], note_on_times[is_performed]

                # Build onset pairs: a list of tuples (onset_score_tick, onset_perf_time)
                onset_pairs = self.compute_onset_pairs(
                    score_ticks=performed_ticks, perf_times=perf_times
                )

            midi_s = self._build_score(
                times=note_on_times,
                durations=note_off_times - note_on_times,
                pitches=pitches,
                velocities=velocities,
                programs=programs,
                time_signatures=None,
                tempos=None,
                time_division=time_division,
                ttype="second",
            )

            # Synchronize created MIDI by beats
            if sync_midi:
                midi, _ = sync_performance_midi(
                    score_midi=midi,
                    perf_midi=midi_s,
                    onset_pairs=onset_pairs[:, :2],
                    grid_level=GridLevel.BEAT,
                    inplace=True,
                    ticks_per_quarter=TICKS_PER_QUARTER,
                )
                if midi is None:
                    warnings.warn(
                        "Failed to synchronize the decoded performance MIDI, using a MIDI without synchronization"
                    )
                    midi = midi_s
            else:
                midi = midi_s

        if controls is not None:
            midi = midi.to("second")
            midi.tracks[0].controls = controls

        return midi.to("tick")

    def synchronize_performance_midi(
        self,
        perf_midi: Score,
        score_midi: Score,
        note_alignment: np.ndarray,
    ) -> Score:
        """Synchronizes a performance MIDI with a score metrical grid (Bar or Beat).

        Updates ticks based on bar/beat-level tempo estimation.

        Args:
            perf_midi: Raw performance :class:`symusic.Score` object.
            score_midi: Reference score :class:`symusic.Score` object.
            note_alignment: Note-level mapping indices.

        Returns:
            Synchronized performance :class:`symusic.Score` object.
        """
        score_note_soa = score_midi.tracks[0].notes.numpy()
        perf_midi_s = perf_midi.to("second")
        perf_note_soa = perf_midi_s.tracks[0].notes.numpy()

        score_ticks = score_note_soa["time"]
        perf_times = perf_note_soa["time"][note_alignment]

        onset_pairs = []
        for onset_tick in np.unique(score_ticks):
            onset_pairs.append((onset_tick, perf_times[score_ticks == onset_tick].mean()))

        onset_pairs = np.array(onset_pairs)

        midi, _ = sync_performance_midi(
            score_midi=score_midi,
            perf_midi=perf_midi,
            onset_pairs=onset_pairs,
            grid_level=GridLevel.BAR if getattr(self, "_bar_tempos", False) else GridLevel.BEAT,
            inplace=False,
            ticks_per_quarter=TICKS_PER_QUARTER,
        )
        return midi

    def score_tokens_as_performance(self, score_tokens: TokSequence) -> TokSequence:
        """Generates 'deadpan' performance tokens from score tokens.

        Converts a sequence of score tokens into a sequence of performance tokens.
        All onset deviations are set to zero and durations are set to the exact score length.

        Args:
            score_tokens: Score :class:`TokSequence` object to convert.

        Returns:
            Deadpan performance :class:`TokSequence` object.
        """
        if len(score_tokens.ids[0]) == len(self.performance_sizes):
            score_tokens = self.compress(score_tokens)

        values = score_tokens.values
        if values is None:
            values = self.decode_values(score_tokens.ids)

        if self.config.additional_params["use_onset_tokens"]:
            # Obtain and distribute zero onset deviation tokens
            onset_dev_values = np.zeros_like(values[:, 0])

            # Obtain and distribute no articulation performed duration tokens
            if self.config.additional_params["rel_perf_duration"]:
                perf_duration_values = np.ones_like(values[:, 0])
            else:
                perf_duration_values = values[:, self.vocab_types_idx["Duration"]]

            values = np.concatenate(
                [
                    values,
                    onset_dev_values[:, None],
                    perf_duration_values[:, None],
                ],
                axis=1,
            )

        if self.config.additional_params["use_time_tokens"]:
            ticks_data = self.compute_ticks(TokSequence(ids=None, values=values))

            note_on_ticks = ticks_data["note_on"].astype(float)
            durations = ticks_data["duration"].astype(float)
            note_off_ticks = note_on_ticks + durations

            tempo_indices = np.concatenate(
                [[0], np.where(np.diff(values[:, self.vocab_types_idx["Tempo"]]))[0] + 1]
            )
            tempos = values[tempo_indices, self.vocab_types_idx["Tempo"]]

            if len(tempos) > 0:
                # Get beat ticks to tie Tempo change to them
                beat_ticks = (
                    ticks_data["bar"]
                    if self.config.additional_params["bar_tempos"]
                    else ticks_data["beat"]
                )
                # Note: position at the start of the beat
                tempo_ticks = note_on_ticks[tempo_indices]
                tempo_ticks = beat_ticks[
                    np.minimum(
                        np.searchsorted(beat_ticks, tempo_ticks, side="right") - 1,
                        beat_ticks.shape[0] - 1,
                    )
                ]
                tempo_ticks[0] = 0
            else:
                tempo_ticks = [0]

            _time_scale = 60 / self.time_division
            tempo_times = np.cumsum(
                np.concatenate([[0.0], np.diff(tempo_ticks) * _time_scale / tempos[:-1]])
            )

            tempo_ids = np.searchsorted(tempo_ticks, note_on_ticks, side="right") - 1

            note_on_times = (
                tempo_times[tempo_ids]
                + (note_on_ticks - tempo_ticks[tempo_ids]) * _time_scale / tempos[tempo_ids]
            )

            tempo_ids = np.searchsorted(tempo_ticks, note_off_ticks, side="right") - 1
            note_off_times = (
                tempo_times[tempo_ids]
                + (note_off_ticks - tempo_ticks[tempo_ids]) * _time_scale / tempos[tempo_ids]
            )

            time_shifts = np.diff(np.concatenate([[0.0], note_on_times]))
            time_durations = note_off_times - note_on_times

            # Append TimeShift/TimeDuration values
            values = np.concatenate(
                [
                    values,
                    time_shifts[:, None],
                    time_durations[:, None],
                ],
                axis=1,
            )

            # Append TimePosition values
            if self.config.additional_params["use_time_positions"]:
                values = np.concatenate(
                    [
                        values,
                        np.round(
                            note_on_times[:, None]
                            % self.config.additional_params["time_position_segment"],
                            6,
                        ),
                    ],
                    axis=1,
                )

            # Append TimeDurationSustain values
            if self.config.additional_params["use_sustain_tokens"]:
                values = np.concatenate(
                    [
                        values,
                        time_durations[:, None],
                        np.zeros_like(time_durations[:, None]),
                    ],
                    axis=1,
                )

        tokens = self.encode_tokens(values)

        return TokSequence(
            ids=tokens,
            values=values,
            type=score_tokens.type,
            encoding=EncodingType.PERFORMANCE,
            vocab=self.vocab_types_idx,
            meta=score_tokens.meta or {},
        )

    def sort_tokens(
        self,
        tokens: TokSequence,
        by_time: bool = False,
        sort_ids: np.ndarray | None = None,
        ordered_shifts: bool = True,
        subsequence: bool = False,
        returns_sort_ids: bool = False,
    ) -> TokSequence | tuple[TokSequence, np.ndarray]:
        """Sorts :class:`TokSequence` object and recomputes relative temporal shifts.

        Ensures that `TimeShift` or `PositionShift` tokens accurately reflect
        the intervals between notes in their new order.

        Args:
            tokens: :class:`TokSequence` object to sort.
            by_time: If ``True``, sorts primarily by `TimeShift` (seconds);
                otherwise, sorts by `Bar` and `Position`.
            sort_ids: Optional precomputed sorting indices.
            ordered_shifts: If ``True``, updates the shift tokens relative
                to the preceding note in the sorted sequence.
            subsequence: Set to ``True`` if processing a partial sequence.
            returns_sort_ids: If ``True``, also returns the indices used for sorting.

        Returns:
            Sorted :class:`TokSequence` object.
        """
        additional_params = self.config.additional_params
        vocab = tokens.vocab or self.vocab_types_idx

        note_ticks, note_times = None, None

        if additional_params["use_time_tokens"] and self.has_token_types(tokens, "TimeShift"):
            time_shifts = self.get_values(tokens, "TimeShift")
            note_times = np.cumsum(time_shifts)

        if additional_params["use_position_shifts"] and self.has_token_types(
            tokens, "PositionShift"
        ):
            position_shifts = self.get_values(tokens, "PositionShift")
            note_ticks = np.cumsum(position_shifts)

        if sort_ids is None:
            if by_time:
                assert additional_params["use_time_tokens"] and note_times is not None
                sort_ids = np.lexsort(
                    (
                        np.where(
                            tokens.ids[:, vocab["Pitch"]] > self.zero_token,
                            tokens.ids[:, vocab["Pitch"]],
                            0,
                        ),
                        note_times,
                    )
                )
            else:
                sort_ids = np.lexsort(
                    (
                        np.where(
                            tokens.ids[:, vocab["Pitch"]] > self.zero_token,
                            tokens.ids[:, vocab["Pitch"]],
                            0,
                        ),
                        tokens.ids[:, vocab["Position"]],
                        tokens.ids[:, vocab["Bar"]],
                    )
                )

        tokens.ids = tokens.ids[sort_ids]
        if tokens.values is not None:
            tokens.values = tokens.values[sort_ids]
        if tokens.interpolated is not None:
            tokens.interpolated = tokens.interpolated[sort_ids]

        if ordered_shifts:
            if note_times is not None:
                if not self.config.additional_params["negative_time_shifts"]:
                    warnings.warn(
                        '`config.additional_params["negative_time_shifts"]` is set to False, '
                        "notes performed before the preceding notes will have 0 time shift tokens."
                    )

                new_note_times = note_times[sort_ids]
                new_time_shifts = np.diff(
                    np.concatenate([[min(0.0, new_note_times.min())], new_note_times])
                )

                tokens.ids[:, vocab["TimeShift"]] = self.encode_tokens(new_time_shifts, "TimeShift")
                if tokens.values is not None:
                    tokens.values[:, vocab["TimeShift"]] = new_time_shifts

            if BAR_LINE_TOKEN in self.special_tokens and "TimeShift" in vocab:
                type_idx = vocab["TimeShift"]
                bar_line_mask = tokens.ids[:, vocab["Pitch"]] == self[0, BAR_LINE_TOKEN]
                tokens.values[bar_line_mask, type_idx] = 1e-5
                tokens.ids[bar_line_mask, type_idx] = self.encode_tokens(
                    tokens.values[bar_line_mask, type_idx], token_type="TimeShift"
                )

            if note_ticks is not None and "PositionShift" in vocab:
                new_note_ticks = note_ticks[sort_ids]
                new_pos_shifts = np.diff(
                    np.concatenate([[min(0.0, new_note_ticks.min())], new_note_ticks])
                )
                new_pos_shifts = np.maximum(new_pos_shifts, SPECIAL_TOKENS_VALUE + 1)

                tokens.ids[:, vocab["PositionShift"]] = self.encode_tokens(
                    new_pos_shifts, "PositionShift"
                )
                if tokens.values is not None:
                    tokens.values[:, vocab["PositionShift"]] = new_pos_shifts

        if returns_sort_ids:
            return tokens, sort_ids
        return tokens

    def _create_base_vocabulary(self) -> list[list[str]]:
        """Creates vocabulary, as a list of lists of string token names.

        Returns:
            Stacked vocabulary of token names.
        """
        vocab = super()._create_base_vocabulary()

        # COMPOUND PITCH
        if self.config.additional_params["use_pitch_classes"]:
            min_pitch, max_pitch = self.config.pitch_range
            self._octave_range = min_pitch // 12 - 1, max_pitch // 12
            vocab.append([f"PitchClass_{i}" for i in NOTES_WSHARP])
            vocab.append([f"PitchOctave_{i}" for i in range(*self._octave_range)])

        # POSITION SHIFT
        if self.config.additional_params["use_position_shifts"]:
            self.position_shifts = self._create_position_shifts().astype(int)
            vocab.append([f"PositionShift_{i}" for i in self.position_shifts])

        # ONSET INDICES
        if self.config.additional_params["use_onset_indices"]:
            max_notes_in_onset = self.config.additional_params["max_notes_in_onset"]
            vocab.append([f"NotesInOnset_{i + 1}" for i in range(1, max_notes_in_onset)])
            vocab.append([f"PositionInOnset_{i}" for i in range(max_notes_in_onset)])

        # ONSET TOKENS
        if self.config.additional_params["use_onset_tokens"]:
            # (RELATIVE) ONSET (POSITION) DEVIATION
            if self.config.additional_params["rel_onset_dev"]:  # relative
                if self.rel_onset_deviations is None:
                    self.rel_onset_deviations = self._create_relative_onset_deviations()
                self.rel_onset_deviations = np.array(self.rel_onset_deviations)
                self.config.additional_params["rel_onset_deviations"] = (
                    self.rel_onset_deviations.tolist()
                )
                vocab.append([f"RelOnsetDev_{i}" for i in self.rel_onset_deviations])
            else:  # absolute
                num_positions = self.config.max_num_pos_per_beat * 2  # up to two quarter notes
                vocab.append([f"OnsetDev_{i}" for i in range(-num_positions, num_positions + 1)])

            # (RELATIVE) PERFORMED DURATION
            if self.config.additional_params["rel_perf_duration"]:  # relative
                if self.rel_performed_durations is None:
                    self.rel_performed_durations = self._create_relative_performed_durations()
                self.rel_performed_durations = np.array(self.rel_performed_durations)
                self.config.additional_params["rel_performed_durations"] = (
                    self.rel_performed_durations.tolist()
                )
                vocab.append([f"RelPerfDuration_{i}" for i in self.rel_performed_durations])
            else:
                vocab.append(vocab[self.vocab_types_idx["Duration"]])

        # TIME TOKENS
        if self.config.additional_params["use_time_tokens"]:
            # TIME SHIFT
            if self.time_shifts is None:
                self.time_shifts = self._create_time_tokens(
                    negative=self.config.additional_params["negative_time_shifts"]
                )
            self.time_shifts = np.array(self.time_shifts)
            self.config.additional_params["time_shifts"] = self.time_shifts.tolist()
            vocab.append([f"TimeShift_{i:.3f}" for i in self.time_shifts])

            # TIME DURATION
            if self.time_durations is None:
                self.time_durations = self._create_time_tokens()
            self.time_durations = np.array(self.time_durations)
            self.config.additional_params["time_durations"] = self.time_durations.tolist()
            vocab.append([f"TimeDuration_{i:.3f}" for i in self.time_durations])

            # TIME POSITION
            if self.config.additional_params["use_time_positions"]:
                if self.time_positions is None:
                    self.time_positions = self._create_time_positions()
                self.time_positions = np.array(self.time_positions)
                self.config.additional_params["time_positions"] = self.time_positions.tolist()
                vocab.append([f"TimePosition_{i:.3f}" for i in self.time_positions])

            if self.config.additional_params["use_sustain_tokens"]:
                vocab.append([f"TimeDuration_{i:.3f}" for i in self.time_durations])
                vocab.append(["Sustained_On", "Sustained_Off"])
                # vocab.append([f"PedalOnTimeShift_{i:.3f}" for i in self.time_shifts])
                # vocab.append([f"PedalOffTimeShift_{i:.3f}" for i in self.time_shifts])

        return vocab

    def _get_token_types(self) -> list[str]:
        """Creates an ordered list of available token types."""
        token_types = super()._get_token_types()

        # Universal tokens
        if self.config.additional_params["use_pitch_classes"]:
            token_types.extend(["PitchClass", "PitchOctave"])

        # Score tokens
        if self.config.additional_params["use_position_shifts"]:
            token_types.append("PositionShift")

        if self.config.additional_params["use_onset_indices"]:
            token_types.extend(["NotesInOnset", "PositionInOnset"])

        # Performance tokens
        if self.config.additional_params["use_onset_tokens"]:
            if self.config.additional_params["rel_onset_dev"]:
                token_types.append("RelOnsetDev")
            else:
                token_types.append("OnsetDev")

            if self.config.additional_params["rel_perf_duration"]:
                token_types.append("RelPerfDuration")
            else:
                token_types.append("PerfDuration")

        if self.config.additional_params["use_time_tokens"]:
            token_types.extend(["TimeShift", "TimeDuration"])

            if self.config.additional_params["use_time_positions"]:
                token_types.extend(["TimePosition"])

            if self.config.additional_params["use_sustain_tokens"]:
                token_types.extend(["TimeDurationSustain", "Sustained"])

        return token_types

    def _create_position_shifts(self) -> np.ndarray:
        """Creates possible position shifts in `max_bet_res`, an array of integers.

        Reuses duration tokens with fine-grained beat resolution defined in config.
        The more beats the position shift occupies, the smaller the resolution of position shift.

        Returns:
            Position shift bins.
        """
        pos_shifts = self.duration_values * self.config.max_num_pos_per_beat

        if self.config.additional_params["negative_position_shifts"]:
            assert -2 * self.config.max_num_pos_per_beat > SPECIAL_TOKENS_VALUE
            pos_shifts = np.concatenate(
                [-pos_shifts[pos_shifts <= 2 * self.config.max_num_pos_per_beat], pos_shifts]
            )
            pos_shifts = np.sort(np.unique(pos_shifts))

        return pos_shifts

    def _create_relative_onset_deviations(self) -> np.ndarray:
        """Creates relative onset deviation bins based on some heuristics.

        The larger the number of deviations (`self.config.additional_params["num_onset_devs"]`),
        the higher the resolution.

        Returns:
            Relative onset deviation bins.
        """
        onset_dev_quant = (self.config.additional_params["num_onset_devs"] - 1) // 8

        rel_onset_devs = np.concatenate(
            [
                # 25% from 0 to 1/24
                np.linspace(0.0, 1 / 24, onset_dev_quant + 1),
                # 25% from 1/24 to 1/8
                np.linspace(1 / 24, 1 / 8, onset_dev_quant + 1)[1:],
                # 25% from 1/8 to 1/3
                np.linspace(1 / 8, 1 / 3, onset_dev_quant + 1)[1:],
                # 12.5% from 1/3 to 3/5
                np.linspace(1 / 3, 3 / 5, onset_dev_quant // 2 + 1)[1:],
                # 6.25% from 3/5 to 1.0
                np.linspace(3 / 5, 1.0, onset_dev_quant // 4 + 1)[1:],
                # 6.25% from 1.0 to 4.0
                (2 ** (8 * np.arange(onset_dev_quant // 4 + 1) / onset_dev_quant))[1:],
            ]
        )
        rel_onset_devs = np.round(rel_onset_devs, 4)
        rel_onset_devs = np.sort(
            np.concatenate([-rel_onset_devs[1:], rel_onset_devs])
        )  # add negative deviations

        return rel_onset_devs

    def _create_relative_performed_durations(self) -> np.ndarray:
        """Creates relative performed duration bins based on some heuristics.

        The larger the number of deviations (`self.config.additional_params["num_perf_durations"]`),
        the higher the resolution.

        Returns:
            Relative onset deviation bins.
        """
        perf_dur_quant = (self.config.additional_params["num_perf_durations"] - 1) // 4

        rel_performed_durations = np.concatenate(
            [
                # 25% from 1/10 to 2/5
                np.linspace(1 / 10, 2 / 5, perf_dur_quant + 1),
                # 25% from 2/5 to 2/3
                np.linspace(2 / 5, 2 / 3, perf_dur_quant + 1)[1:],
                # 25% from 2/3 to 1.0
                np.linspace(2 / 3, 1.0, perf_dur_quant + 1)[1:],
                # 12.5% from 1.0 to 5/4
                np.linspace(1.0, 5 / 4, perf_dur_quant // 2 + 1)[1:],
                # 6.25% from 5/4 to 3/2
                np.linspace(5 / 4, 3 / 2, perf_dur_quant // 4 + 1)[1:],
                # 6.25% from 3/2 to 3.0
                (2 ** (4 * np.arange(perf_dur_quant // 4 + 1) / perf_dur_quant) * 3 / 2)[1:],
            ]
        )
        rel_performed_durations = np.round(rel_performed_durations, 4)

        return rel_performed_durations

    def _create_time_tokens(self, negative: bool = False) -> np.ndarray:
        """Creates time shift/duration bins.

        Returns:
            Time token bins (milliseconds).
        """
        points = [-400, -200, -100, -50, 250, 500, 1000, 2000, 5000, 10000 + 1]
        steps = [10, 5, 2, 1, 2, 5, 10, 50, 100]

        if not negative:
            points = [max(0, point) for point in points]

        return (
            np.concatenate(
                [
                    np.arange(start, stop, step)
                    for start, stop, step in zip(points[:-1], points[1:], steps)
                ]
            )
            / 1000.0
        )

    def _create_time_positions(self) -> np.ndarray:
        """Creates time positions bins.

        Returns:
            Time position bins (milliseconds).
        """
        time_segment = self.config.additional_params["time_position_segment"]
        time_step = self.config.additional_params["time_position_step"]
        return np.round(np.arange(0.0, time_segment, time_step), 3)

    def compute_position_shifts(
        self, score_positions: np.ndarray, onset_shift: bool | None = None
    ) -> np.ndarray:
        """Computes absolute position shifts between onsets from score positions.

        Args:
            score_positions: Array of score positions (ticks/beats).
            onset_shift: If ``True``, overwrites tokenizer setting for onset_shift position shift.

        Returns:
            Array of position shits.
        """
        onset_shift = (
            self.config.additional_params["onset_position_shifts"]
            if onset_shift is None
            else onset_shift
        )
        return super().compute_position_shifts(score_positions, onset_shift)

    def compute_onset_values(
        self, score_positions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes the number of notes and positions of notes in onsets.

        Used to built `NotesInOnset` and `PositionInOnset` tokens.

        Args:
            score_positions: Array of score positions (ticks/beats).

        Returns:
            Tuple of (score_position_indices, number_of_notes_in_onset, note_position_in_onset).
        """
        unique_score_pos, score_pos_counts = np.unique(score_positions, return_counts=True)
        score_pos_ids = np.arange(len(unique_score_pos)).repeat(score_pos_counts)

        notes_in_onset = score_pos_counts[score_pos_ids]
        notes_in_onset = np.minimum(
            notes_in_onset, self.config.additional_params["max_notes_in_onset"]
        )

        pos_in_onset = np.repeat(np.cumsum(-score_pos_counts) + score_pos_counts, score_pos_counts)
        pos_in_onset = pos_in_onset + np.arange(len(pos_in_onset))
        pos_in_onset = np.minimum(
            pos_in_onset, self.config.additional_params["max_notes_in_onset"] - 1
        )

        return score_pos_ids, notes_in_onset, pos_in_onset

    def compute_time_bar_beat_onset_indices(
        self, tokens: TokSequence
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimated time bar, beat and onset indices for each note in the token sequence.

        Uses a tempo of 120 BPM and a time signature of 4/4.

        Args:
            tokens: :class:`TokSequence` object to process.

        Returns:
            Tuple of (bar, beat, onset) indices.
        """
        additional_params = self.config.additional_params
        assert additional_params["use_time_tokens"], (
            "TimeShift and TimeDuration tokens should be present in the tokenizer to decode "
            "a performance time only tokenization with a score metrical grid"
        )

        time_shifts = self.get_values(tokens, "TimeShift")
        note_times = np.cumsum(time_shifts)

        bars, beats = note_times // 2.0, note_times // 0.5

        unique_onsets, onset_notes = np.unique(note_times // 0.01, return_counts=True)
        onsets = np.arange(len(unique_onsets)).repeat(onset_notes)

        bars, beats, onsets = map(lambda s: s - s[0], (bars, beats, onsets))
        return bars.astype(int), beats.astype(int), onsets.astype(int)

    @staticmethod
    def compute_onset_pairs(score_ticks: np.ndarray, perf_times: np.ndarray) -> np.ndarray:
        """Calculates a mapping between unique score onset ticks and their mean performance times.

        Args:
            score_ticks: Array of score ticks.
            perf_times: Array of performance times.

        Returns:
            Array of unique (score_tick, perf_time) pairs.
        """
        # Get unique performed score onsets
        sort_ids = np.argsort(score_ticks)
        score_onsets, onset_un_idx, onset_counts = np.unique(
            score_ticks[sort_ids], return_index=True, return_counts=True
        )

        # Build onset pairs: a list of tuples (onset_score_tick, onset_perf_time)
        onset_time_sums = np.add.reduceat(perf_times[sort_ids], onset_un_idx, axis=0)
        onset_times = onset_time_sums / onset_counts

        onset_pairs = np.stack([score_onsets, onset_times], axis=1)

        if score_onsets[0] != 0.0:
            onset_pairs = np.concatenate(([[0, 0]], onset_pairs))

        return onset_pairs

    def shift_positions(
        self,
        tokens: TokSequence,
        shifts: dict[str, int | float] | None = None,
        inverse_shifts: bool = False,
        normalized_values: bool = False,
        shift_to_zero: bool = False,
    ) -> tuple[TokSequence, dict[str, int | float]]:
        """Applies a global temporal shift (`Bar` or `Time`) to a token sequence.

        Args:
            tokens: :class:`TokSequence` object to process.
            shifts: Dictionary containing 'Bar' and 'Time' offsets.
            inverse_shifts: If ``True``, subtracts the shifts instead of adding.
            normalized_values: If ``True``, processes values in normalized space.
            shift_to_zero: If ``True``, shifts the first note of the sequence to zero.

        Returns:
            Shifted :class:`TokSequence` object and the shift values used.
        """
        assert not shift_to_zero or shifts is None

        vocab = tokens.vocab or self.vocab_types_idx

        has_bars = (
            tokens.encoding != EncodingType.TIME_PERFORMANCE
            and "Bar" in vocab
            and "Position" in vocab
            and self.has_token_types(tokens, ["Bar", "Position"])
        )

        has_time = (
            self.config.additional_params["use_time_positions"]
            and tokens.encoding
            not in (EncodingType.SCORE, EncodingType.PLAIN_SCORE, EncodingType.REL_PERFORMANCE)
            and "TimePosition" in vocab
            and self.has_token_types(tokens, ["TimePosition"])
        )

        shifts = shifts or {"Bar": 0, "Time": 0.0}
        if shift_to_zero and len(tokens) > 0:  # move the first note to zero time
            if has_bars:
                shifts["Bar"] = -int(self.get_values(tokens, "Bar", from_ids=True).min())

            if has_time:
                time_shifts = self.get_values(tokens, "TimeShift")

                note_times = np.cumsum(time_shifts)
                min_time_idx = note_times.argmin()

                shifts["Time"] = float(-note_times[min_time_idx])

        if inverse_shifts:
            shifts = {key: -value for key, value in shifts.items()}

        bar_shift = shifts.get("Bar", 0)
        if bar_shift != 0 and has_bars:
            bar_index = vocab["Bar"]

            tokens.ids[:, bar_index] += bar_shift
            if tokens.values is not None:
                if normalized_values:
                    tokens.values[:, bar_index] = self.denormalize_values(
                        tokens.values[:, bar_index], "Bar"
                    )
                    tokens.values[:, bar_index] += bar_shift
                    tokens.values[:, bar_index] = self.normalize_values(
                        tokens.values[:, bar_index], "Bar"
                    )
                else:
                    tokens.values[:, bar_index] += bar_shift

        time_shift = shifts.get("Time", 0.0)
        if time_shift != 0.0 and has_time:
            time_shifts = self.get_values(tokens, "TimeShift")
            note_times = np.cumsum(time_shifts)

            new_note_times = note_times + time_shift
            new_time_positions = np.round(
                new_note_times % self.config.additional_params["time_position_segment"], 6
            )

            tokens.ids[:, vocab["TimePosition"]] = self.encode_tokens(
                new_time_positions, token_type="TimePosition"
            )
            if tokens.values is not None:
                tokens.values[:, vocab["TimePosition"]] = new_time_positions

        return tokens, shifts

    def add_tempo_tokens(
        self, tokens: TokSequence, window: tuple[float, float] = (-2, 1)
    ) -> TokSequence:
        """Estimates local tempo from timing and injects `Tempo` tokens into the token sequence.

        Args:
            tokens: :class:`TokSequence` object to process.
            window: Temporal window size in seconds.

        Returns:
            :class:`TokSequence` object with updated `Tempo` tokens.
        """
        vocab = tokens.vocab or self.vocab_types_idx
        if "Tempo" not in vocab or not self.has_token_types(tokens, "TimeShift"):
            return tokens

        tempos = self.compute_local_tempos(tokens, window=window)

        tokens.ids[:, vocab["Tempo"]] = self.encode_tokens(tempos, "Tempo")
        if tokens.values is not None:
            tokens.values[:, vocab["Tempo"]] = tempos

        return tokens

    def compute_local_tempos(
        self, tokens: TokSequence, window: tuple[float, float] = (-2, 1)
    ) -> np.ndarray:
        """Calculates local BPM for each note using a sliding window of performance onsets.

        Args:
            tokens: :class:`TokSequence` object to process.
            window: Temporal window size in seconds.

        Returns:
            Array of local tempos.
        """
        position_data, _ = self.decode_note_positions(tokens, time_division=self.time_division)

        score_ticks = position_data["ticks_data"]["note_on"]
        note_on_times = position_data["note_on_times"]

        onset_pairs = self.compute_onset_pairs(score_ticks, note_on_times)
        max_times = np.maximum.accumulate(onset_pairs[:-1, 1])
        onset_pairs = onset_pairs[np.concatenate([[True], onset_pairs[1:, 1] > max_times])]
        onset_ticks, onset_times = onset_pairs[:, 0], onset_pairs[:, 1]

        left_ids = np.maximum(
            0,
            np.searchsorted(onset_ticks, onset_ticks + window[0] * self.time_division, side="right")
            - 1,
        )
        right_ids = np.minimum(
            onset_ticks.shape[-1] - 1,
            np.searchsorted(onset_ticks, onset_ticks + window[1] * self.time_division, side="right")
            - 1,
        )
        left_ids = np.minimum(left_ids, np.maximum(0, right_ids - 1))
        right_ids = np.maximum(right_ids, np.minimum(onset_ticks.shape[-1] - 1, left_ids + 1))

        tick_shifts = onset_ticks[right_ids] - onset_ticks[left_ids]
        time_shifts = onset_times[right_ids] - onset_times[left_ids]
        tempos = tick_shifts / time_shifts * 60 / self.time_division
        tempos = np.maximum(self.tempos[0], np.minimum(self.tempos[-1], tempos))

        tempos = tempos[find_closest(onset_ticks, score_ticks)]

        return tempos

    def add_bar_line_tokens(self, tokens: TokSequence, start: bool = True) -> TokSequence:
        """Injects `BAR_LINE` special tokens at the boundaries of each bar.

        Args:
            tokens: :class:`TokSequence` object to process.
            start: If ``True``, `BAR_LINE` tokens are inserted at the beginning of each bar.
                If ``False``, `BAR_LINE` tokens are inserted at the end of each bar.

        Returns:
            :class:`TokSequence` object with injected `BAR_LINE` tokens.
        """
        if (
            tokens.type in (SequenceType.TIME_PERFORMANCE, SequenceType.TIME_PERFORMANCE_SUSTAIN)
            or BAR_LINE_TOKEN not in self.special_tokens
        ):
            return tokens

        tokens = super().add_bar_line_tokens(tokens, start=start)

        vocab = tokens.vocab or self.vocab_types_idx

        if "TimeShift" in vocab:
            type_idx = vocab["TimeShift"]
            bar_line_mask = tokens.ids[:, vocab["Pitch"]] == self[0, BAR_LINE_TOKEN]
            tokens.values[bar_line_mask, type_idx] = 1e-5
            tokens.ids[bar_line_mask, type_idx] = self.encode_tokens(
                tokens.values[bar_line_mask, type_idx], token_type="TimeShift"
            )

        tokens = self.fill_extra_pitch_tokens(tokens=tokens, force=True)
        tokens = self.fill_extra_score_tokens(tokens, force=True)

        return tokens

    def remove_bar_line_tokens(self, tokens: TokSequence) -> TokSequence:
        """Removes `BAR_LINE` special tokens and recomputes surrounding position shifts.

        Args:
            tokens: :class:`TokSequence` object to process.

        Returns:
            :class:`TokSequence` object with deleted `BAR_LINE` tokens.
        """
        if BAR_LINE_TOKEN not in self.special_tokens:
            return tokens

        note_ticks = None
        if self.has_token_types(tokens, "PositionShift"):
            position_shifts = self.get_values(tokens, "PositionShift")
            note_ticks = np.cumsum(position_shifts)

        vocab = tokens.vocab or self.vocab_types_idx
        mask = tokens.ids[:, vocab["Pitch"]] != self[0, BAR_LINE_TOKEN]

        tokens.ids = tokens.ids[mask]
        tokens.values = tokens.values[mask]
        if tokens.interpolated is not None:
            tokens.interpolated = tokens.interpolated[mask]

        if note_ticks is not None and np.any(mask):
            note_ticks = note_ticks[mask]
            new_position_shifts = np.diff(
                np.concatenate([[min(0.0, note_ticks.min())], note_ticks])
            )

            tokens.ids[:, vocab["PositionShift"]] = self.encode_tokens(
                new_position_shifts, "PositionShift"
            )
            if tokens.values is not None:
                tokens.values[:, vocab["PositionShift"]] = new_position_shifts

        return tokens

    def add_pedal_tokens(self, tokens: TokSequence, ignore_redundant: bool = True) -> TokSequence:
        """Injects sustain pedal events as tokens into the sequence.

        Updates note durations to reflect sustain state if sustain tokens are enabled.

        Args:
            tokens: :class:`TokSequence` object to process.
            ignore_redundant: If ``True``, redundant pedal events are ignored.

        Returns:
            :class:`TokSequence` object with injected sustain pedal events.
        """
        if PEDAL_ON_TOKEN not in self.special_tokens or PEDAL_OFF_TOKEN not in self.special_tokens:
            return tokens

        if tokens.type == SequenceType.SCORE or tokens.pedals is None:
            return tokens

        vocab = tokens.vocab or self.vocab_types_idx
        assert "TimeShift" in vocab

        _backend = backend(tokens)

        pedals = tokens.pedals
        if pedals is None or len(pedals) == 0:
            return tokens

        if tokens.encoding != EncodingType.TIME_PERFORMANCE:
            tokens = self.sort_tokens(tokens, by_time=True)

        has_bars = (
            tokens.encoding != EncodingType.TIME_PERFORMANCE
            and "Bar" in vocab
            and "Position" in vocab
            and self.has_token_types(tokens, ["Bar", "Position"])
        )

        pedal_pitches, pedal_times = pedals[:, 0], pedals[:, 1]
        sustain_ons = pedal_times[pedal_pitches == SPECIAL_TOKENS_VALUE - self[0, PEDAL_ON_TOKEN]]
        sustain_offs = pedal_times[pedal_pitches == SPECIAL_TOKENS_VALUE - self[0, PEDAL_OFF_TOKEN]]

        note_pitches = self.get_values(tokens, "Pitch")
        note_time_shifts = self.get_values(tokens, "TimeShift")
        note_on_times = _backend.cumsum(note_time_shifts)
        note_durations = self.get_values(tokens, "TimeDuration").copy()
        note_off_times = note_on_times + note_durations

        note_mask = tokens.ids[:, vocab["Pitch"]] > self.zero_token

        # a note off during a pedal
        start_search = np.searchsorted(sustain_ons, note_off_times, side="right") - 1
        end_search = np.searchsorted(sustain_offs, note_off_times, side="left")
        note_off_sustain = note_mask & (start_search >= 0) & (end_search == start_search)
        note_ids, note_sustain_ids = np.where(note_off_sustain)[0], start_search[note_off_sustain]

        if ignore_redundant:
            start_search = np.searchsorted(sustain_ons, note_on_times, side="left")
            end_search = np.searchsorted(sustain_offs, note_off_times, side="right") - 1

            pedal_inside = note_mask & (start_search >= 0) & (end_search >= start_search)

            sustain_mask = np.zeros(len(sustain_ons), dtype=bool)
            for s, e in np.stack([start_search[pedal_inside], end_search[pedal_inside]], -1):
                sustain_mask[s : e + 1] = True
            sustain_ids = np.where(sustain_mask)[0]

            sustain_ids = np.concatenate([note_sustain_ids, sustain_ids])

            used_ids = np.unique(sustain_ids)
            used_ids = _backend.stack([2 * used_ids, 2 * used_ids + 1], axis=1).reshape(-1)
            pedal_pitches, pedal_times = pedal_pitches[used_ids], pedal_times[used_ids]

        if self.config.additional_params["use_sustain_tokens"]:
            note_durations[note_ids] = sustain_offs[note_sustain_ids] - note_on_times[note_ids]
            note_off_times = note_on_times + note_durations

            for pitch in np.unique(note_pitches[note_mask]):  # cut overlapping notes
                pitch_ids = np.where(note_pitches == pitch)[0]
                if len(pitch_ids) < 2:
                    continue

                prev_ids, next_ids = pitch_ids[:-1], pitch_ids[1:]

                overlap_mask = note_on_times[next_ids] < note_off_times[prev_ids]
                note_off_times[prev_ids] = np.where(
                    overlap_mask, note_on_times[next_ids], note_off_times[prev_ids]
                )

            note_durations = note_off_times - note_on_times
            note_duration_tokens = self.encode_tokens(note_durations, "TimeDurationSustain")

            if "TimeDurationSustain" in vocab:
                if tokens.values is not None:
                    tokens.values[:, vocab["TimeDurationSustain"]] = note_durations
                tokens.ids[:, vocab["TimeDurationSustain"]] = note_duration_tokens
            else:
                if tokens.values is not None:
                    tokens.values = _backend.concatenate(
                        [tokens.values, note_durations[:, None]], -1
                    )
                tokens.ids = _backend.concatenate([tokens.ids, note_duration_tokens[:, None]], -1)
                vocab["TimeDurationSustain"] = len(vocab)
                tokens.vocab = vocab

            sustained = np.zeros_like(note_durations)
            sustained[note_ids] = 1.0

            if "Sustained" in vocab:
                if tokens.values is not None:
                    tokens.values[:, vocab["Sustained"]] = sustained
                tokens.ids[:, vocab["Sustained"]] = self.zero_token + sustained.astype(int)

        new_values = _backend.full(
            (len(pedal_pitches), tokens.ids.shape[1]), fill_value=self.ignore_value
        )
        new_values[:, vocab["Pitch"]] = pedal_pitches

        new_tokens = replace(
            tokens,
            ids=self.encode_tokens(new_values),
            values=new_values,
            interpolated=(
                _backend.zeros_like(new_values[:, 0]) if tokens.interpolated is not None else None,
            ),
        )
        tokens = tokens + new_tokens

        pitches = _backend.concatenate([note_pitches, pedal_pitches])
        times = _backend.concatenate([note_on_times, pedal_times])
        sort_ids = np.lexsort((pitches, times))

        tokens = self.sort_tokens(tokens, sort_ids=sort_ids)
        new_times = times[sort_ids]

        new_time_shifts = _backend.diff(np.concatenate([[0.0], new_times]))
        tokens.ids[:, vocab["TimeShift"]] = self.encode_tokens(new_time_shifts, "TimeShift")
        if tokens.values is not None:
            tokens.values[:, vocab["TimeShift"]] = new_time_shifts

        if has_bars:
            for token_type in ["Bar", "Position"] + self.time_signature_tokens:
                type_idx = vocab[token_type]
                tokens.values[:, type_idx] = backward_fill(
                    tokens.values[:, type_idx], self.ignore_value
                )
                if np.any(tokens.values[:, type_idx] == self.ignore_value):
                    mask = tokens.values[:, type_idx] == self.ignore_value
                    tokens.values[:, type_idx] = forward_fill(
                        tokens.values[:, type_idx], self.ignore_value
                    )
                    if token_type == "Position":
                        tokens.values[mask, type_idx] = np.minimum(
                            tokens.values[:, type_idx].max(), tokens.values[mask, type_idx] + 1
                        )
                tokens.ids[:, type_idx] = self.encode_tokens(
                    tokens.values[:, type_idx], token_type=token_type
                )

            tokens = self.sort_tokens(tokens)
            tokens = self.fill_extra_score_tokens(tokens, force=True)

        tokens = self.fill_extra_pitch_tokens(tokens=tokens, force=True)

        return tokens

    def remove_pedal_tokens(self, tokens: TokSequence, save_pedals: bool = True) -> TokSequence:
        """Removes special pedal tokens and recomputes surrounding time shifts.

        Args:
            tokens: :class:`TokSequence` object to process.
            save_pedals: If ``True``, pedals are saved as an array inside :class:`TokSequence` object
                `pedals` attribute.

        Returns:
            :class:`TokSequence` object with deleted pedal tokens.
        """
        if PEDAL_ON_TOKEN not in self.special_tokens and PEDAL_OFF_TOKEN not in self.special_tokens:
            return tokens

        vocab = tokens.vocab or self.vocab_types_idx
        pitch_index = vocab["Pitch"]

        note_ticks = None
        if self.has_token_types(tokens, "PositionShift"):
            position_shifts = self.get_values(tokens, "PositionShift")
            note_ticks = np.cumsum(position_shifts)

        note_times = None
        if self.has_token_types(tokens, "TimeShift"):
            time_shifts = self.get_values(tokens, "TimeShift")
            note_times = np.cumsum(time_shifts)

        pedals = []
        for token_type in [PEDAL_ON_TOKEN, PEDAL_OFF_TOKEN]:
            token_id = self[0, token_type]
            mask = tokens.ids[:, pitch_index] == token_id
            tokens.ids, tokens.values = tokens.ids[~mask], tokens.values[~mask]
            if tokens.interpolated is not None:
                tokens.interpolated = tokens.interpolated[~mask]

            if note_times is not None:
                if token_type == PEDAL_ON_TOKEN:
                    sustain_ons = note_times[mask]
                    if np.any(sustain_ons):
                        pedals.append(np.stack([np.ones_like(sustain_ons), sustain_ons], axis=1))
                else:
                    sustain_offs = note_times[mask]
                    if np.any(sustain_offs):
                        pedals.append(np.stack([np.zeros_like(sustain_offs), sustain_offs], axis=1))

                note_times = note_times[~mask]

            if note_ticks is not None:
                note_ticks = note_ticks[~mask]

        if len(pedals) > 0 and save_pedals:
            pedals = np.concatenate(pedals)
            pedals = pedals[np.lexsort((-pedals[:, 0], pedals[:, 1]))]
            if pedals[0, 0] == 0:
                pedals = np.concatenate([np.array([[1, 0.0]]), pedals], axis=0)
            # if pedals[-1, 0] == 1:
            #     pedals = np.concatenate([pedals, np.array([[0, note_times.max()]])], axis=0)
            pedals = np.concatenate([pedals[:1], pedals[1:][np.diff(pedals[:, 0]) != 0.0]])
            tokens.pedals = pedals
        elif save_pedals:
            tokens.pedals = None

        if note_times is not None:
            new_time_shifts = np.diff(np.concatenate([[0.0], note_times]))

            tokens.ids[:, vocab["TimeShift"]] = self.encode_tokens(new_time_shifts, "TimeShift")
            if tokens.values is not None:
                tokens.values[:, vocab["TimeShift"]] = new_time_shifts

        if note_ticks is not None and len(note_ticks) > 0:
            new_position_shifts = np.diff(
                np.concatenate([[min(0.0, note_ticks.min())], note_ticks])
            )

            tokens.ids[:, vocab["PositionShift"]] = self.encode_tokens(
                new_position_shifts, "PositionShift"
            )
            if tokens.values is not None:
                tokens.values[:, vocab["PositionShift"]] = new_position_shifts

        return tokens

    def add_artificial_pedal_on(
        self, tokens: TokSequence, position_tokens: bool = False
    ) -> TokSequence:
        """Ensures a sequence starting with a pedal-off has a preceding pedal-on
        to maintain logical consistency for generative models.

        Args:
            tokens: :class:`TokSequence` object to process.
            position_tokens: If ``True``, zero position tokens are added.

        Returns:
            :class:`TokSequence` object with added pedal tokens.
        """
        if PEDAL_ON_TOKEN not in self.special_tokens and PEDAL_OFF_TOKEN not in self.special_tokens:
            return tokens

        vocab = tokens.vocab or self.vocab_types_idx
        pedal_ids = self.pedal_ids

        _backend = backend(tokens)

        pedal_on_ids = _backend.where(tokens.ids[:, vocab["Pitch"]] == pedal_ids[0])[0]
        pedal_off_ids = _backend.where(tokens.ids[:, vocab["Pitch"]] == pedal_ids[1])[0]
        zero_sustain_on = len(pedal_off_ids) > 0 and (
            len(pedal_on_ids) == 0
            or (len(pedal_on_ids) > 0 and pedal_off_ids.min() < pedal_on_ids.min())
        )

        if zero_sustain_on:
            new_pedal = _backend.full((1, tokens.ids.shape[1]), fill_value=self.ignore_value)
            new_pedal[:, vocab["Pitch"]] = SPECIAL_TOKENS_VALUE - pedal_ids[0]
            new_pedal[:, vocab["TimeShift"]] = 0.0

            if position_tokens:
                for token_type in ["Position", "PositionShift"]:
                    if token_type in vocab:
                        new_pedal[:, vocab[token_type]] = 0.0

                if "Bar" in vocab:
                    new_pedal[:, vocab["Bar"]] = tokens.values[0, vocab["Bar"]]

            tokens.ids = _backend.concatenate(
                (self.encode_tokens(new_pedal, vocab=vocab), tokens.ids), 0
            )

            if tokens.values is not None:
                tokens.values = _backend.concatenate([new_pedal, tokens.values])

            if tokens.interpolated is not None:
                tokens.interpolated = _backend.concatenate(
                    (_backend.zeros(1), tokens.interpolated), 0
                )

        return tokens

    def add_time_position_tokens(
        self,
        tokens: TokSequence,
        wrap: bool = False,
        segment_tokens: bool = False,
    ) -> TokSequence:
        """Calculates and injects cyclic `TimePosition` tokens and `Time_Segment` special tokens.

        Args:
            tokens: :class:`TokSequence` object to process.
            wrap: If ``True``, wrap around the time position tokens.
            segment_tokens: If ``True``, wrap around the `Time_Segment` special token markers.

        Returns:
            :class:`TokSequence` object with added time position tokens.
        """
        if (
            not self.config.additional_params["use_time_positions"]
            or tokens.type == SequenceType.SCORE
        ):
            return tokens

        vocab = tokens.vocab or self.vocab_types_idx
        assert "TimeShift" in vocab and "TimePosition" in vocab

        _backend = backend(tokens)

        has_bars = (
            tokens.encoding != EncodingType.TIME_PERFORMANCE
            and "Bar" in vocab
            and "Position" in vocab
            and self.has_token_types(tokens, ["Bar", "Position"])
        )

        note_pitches = self.get_values(tokens, "Pitch")
        note_time_shifts = self.get_values(tokens, "TimeShift")
        note_times = _backend.cumsum(note_time_shifts)

        time_segment = self.config.additional_params["time_position_segment"]

        segments = note_times // time_segment
        time_positions = note_times % time_segment if wrap else note_times

        min_segm, max_segm = map(int, (segments.min(), segments.max()))
        num_segments = max_segm - min_segm + 1

        tokens.values[:, vocab["TimePosition"]] = time_positions
        tokens.ids[:, vocab["TimePosition"]] = self.encode_tokens(
            tokens.values[:, vocab["TimePosition"]], "TimePosition"
        )

        if not segment_tokens or TIME_SEGMENT_TOKEN not in self.special_tokens:
            return tokens

        token_id = self[0, TIME_SEGMENT_TOKEN]

        new_values = _backend.full(
            (num_segments, tokens.ids.shape[1]), fill_value=self.ignore_value
        )
        new_values[:, vocab["Pitch"]] = SPECIAL_TOKENS_VALUE - token_id
        new_times = new_values[:, vocab["TimePosition"]] = _backend.arange(min_segm, max_segm + 1)

        new_tokens = replace(
            tokens,
            ids=self.encode_tokens(new_values),
            values=new_values,
            interpolated=(
                _backend.zeros_like(new_values[:, 0]) if tokens.interpolated is not None else None
            ),
        )
        tokens = tokens + new_tokens

        pitches = _backend.concatenate([note_pitches, self.get_values(new_tokens, "Pitch")])
        times = _backend.concatenate([note_times, new_times])
        sort_ids = np.lexsort((pitches, times))

        tokens = self.sort_tokens(tokens, sort_ids=sort_ids)
        new_times = times[sort_ids]

        new_time_shifts = _backend.diff(np.concatenate([[0.0], new_times]))
        tokens.ids[:, vocab["TimeShift"]] = self.encode_tokens(new_time_shifts, "TimeShift")
        if tokens.values is not None:
            tokens.values[:, vocab["TimeShift"]] = new_time_shifts

        if has_bars:
            for token_type in ["Bar", "Position"] + self.time_signature_tokens:
                type_idx = vocab[token_type]
                tokens.values[:, type_idx] = backward_fill(
                    tokens.values[:, type_idx], self.ignore_value
                )
                if np.any(tokens.values[:, type_idx] == self.ignore_value):
                    mask = tokens.values[:, type_idx] == self.ignore_value
                    tokens.values[:, type_idx] = forward_fill(
                        tokens.values[:, type_idx], self.ignore_value
                    )
                    if token_type == "Position":
                        tokens.values[mask, type_idx] = np.minimum(
                            tokens.values[:, type_idx].max(), tokens.values[mask, type_idx] + 1
                        )
                tokens.ids[:, type_idx] = self.encode_tokens(
                    tokens.values[:, type_idx], token_type=token_type
                )

            tokens = self.sort_tokens(tokens)
            tokens = self.fill_extra_score_tokens(tokens, force=True)

        return tokens

    def remove_time_segment_tokens(self, tokens: TokSequence) -> TokSequence:
        """Removes `Time_Segment` special tokens used to denote the start of temporal segments.

        Args:
            tokens: :class:`TokSequence` object to process.

        Returns:
            :class:`TokSequence` object with deleted time segment tokens.
        """
        if TIME_SEGMENT_TOKEN not in self.special_tokens:
            return tokens

        vocab = tokens.vocab or self.vocab_types_idx
        pitch_index = vocab["Pitch"]
        mask = tokens.ids[:, pitch_index] == self[0, TIME_SEGMENT_TOKEN]

        return self.remove_notes(tokens, mask=mask)

    def remove_notes(self, tokens: TokSequence, mask: np.ndarray) -> TokSequence:
        """Removes notes from a sequence based on a mask and recomputes relative shifts.

        Args:
            tokens: :class:`TokSequence` object to process.
            mask: Mask to apply to the tokens.

        Returns:
            :class:`TokSequence` object with notes removed.
        """
        if tokens.interpolated is None or tokens.interpolated.sum() == 0:
            return tokens

        vocab = tokens.vocab or self.vocab_types_idx

        note_ticks = None
        if self.has_token_types(tokens, "PositionShift"):
            position_shifts = self.get_values(tokens, "PositionShift")
            note_ticks = np.cumsum(position_shifts)

        note_times = None
        if self.has_token_types(tokens, "TimeShift"):
            time_shifts = self.get_values(tokens, "TimeShift")
            note_times = np.cumsum(time_shifts)

        tokens.ids, tokens.values = tokens.ids[~mask], tokens.values[~mask]
        tokens.interpolated = tokens.interpolated[~mask]

        if note_times is not None:
            note_times = note_times[~mask]

        if note_ticks is not None:
            note_ticks = note_ticks[~mask]

        if note_times is not None and len(note_times) > 0:
            new_time_shifts = np.diff(np.concatenate([[0.0], note_times]))

            tokens.ids[:, vocab["TimeShift"]] = self.encode_tokens(new_time_shifts, "TimeShift")
            if tokens.values is not None:
                tokens.values[:, vocab["TimeShift"]] = new_time_shifts

        if note_ticks is not None and len(note_ticks) > 0:
            new_position_shifts = np.diff(
                np.concatenate([[min(0.0, note_ticks.min())], note_ticks])
            )

            tokens.ids[:, vocab["PositionShift"]] = self.encode_tokens(
                new_position_shifts, "PositionShift"
            )
            if tokens.values is not None:
                tokens.values[:, vocab["PositionShift"]] = new_position_shifts

        return tokens

    def fill_bar_and_time_signature_tokens(self, tokens: TokSequence):
        """Infers and populates metrical tokens (`Bar`, `MaxPosition`, `BeatDuration`)
        from `BAR_LINE` tokens and existing positions.

        Args:
            tokens: :class:`TokSequence` object to process.

        Returns:
            :class:`TokSequence` object with bar and time signature tokens.
        """
        if self.has_token_types(tokens, ["Bar", "MaxPosition", "BeatDuration"]):
            return tokens

        vocab = tokens.vocab or self.vocab_types_idx
        bar_index = vocab["Bar"]
        max_pos_index = tokens.vocab["MaxPosition"]

        has_max_position = self.has_token_types(tokens, "MaxPosition")
        beat_res = self.config.max_num_pos_per_beat

        pitches = self.get_values(tokens, "Pitch", from_ids=True)
        is_bar_end = pitches == SPECIAL_TOKENS_VALUE - self[0, BAR_LINE_TOKEN]
        bar_end_pos = self.get_values(tokens, "Position")[is_bar_end]

        bar, start = -1, 0
        for bar, end in enumerate(np.where(is_bar_end)[0]):
            tokens.ids[start : end + 1, bar_index] = self.zero_token + bar
            tokens.values[start : end + 1, bar_index] = bar

            if not has_max_position:
                tokens.values[start : end + 1, max_pos_index] = bar_end_pos[bar]

            start = end + 1

        if start <= len(tokens):
            tokens.ids[start:, bar_index] = self.zero_token + bar + 1
            tokens.values[start:, bar_index] = bar + 1
            if not has_max_position:
                tokens.values[start:, max_pos_index] = (
                    4 * beat_res if len(bar_end_pos) == 0 else bar_end_pos[bar]
                )

        if not has_max_position:
            tokens.ids[:, max_pos_index] = self.encode_tokens(
                tokens.values[:, max_pos_index], token_type="MaxPosition"
            )

        if not self.has_token_types(tokens, "BeatDuration"):
            max_positions = tokens.values[:, max_pos_index]

            beat_index = tokens.vocab["BeatDuration"]
            tokens.values[max_positions % (beat_res / 4) == 0, beat_index] = 1 / 16  # 16th note
            tokens.values[max_positions % (beat_res / 2) == 0, beat_index] = 1 / 8  # 8th note
            tokens.values[max_positions % beat_res == 0, beat_index] = 1 / 4  # 4th note

            tokens.ids[:, beat_index] = self.encode_tokens(
                tokens.values[:, beat_index], token_type="BeatDuration"
            )

        return tokens

    def _values_to_tokens(
        self,
        values: np.ndarray,
        token_type: str,
        denormalize: bool = False,
        clip: bool = False,
    ) -> np.ndarray:
        """Encodes tokens from values for a specific `token_type`.

        Internal method used by `encode_tokens`.

        Args:
            values: Array of values to encode.
            token_type: Type of tokens (name) to encode.
            denormalize: Denormalize values before encoding.
            clip: Clip values to their quantized boundaries.

        Returns:
            Array of encoded tokens for provided `token_type`.
        """
        if denormalize:
            values = self._denormalize_values(values, token_type)

        is_special = values <= SPECIAL_TOKENS_VALUE
        special_values = values[is_special]

        if token_type == "PitchClass":
            tokens = values.astype(int)
        elif token_type == "PitchOctave":
            tokens = values - self._octave_range[0]
        elif token_type == "PositionShift":
            tokens = find_closest(self.position_shifts, values)
        elif token_type == "NotesInOnset":
            tokens = values - 1
        elif token_type == "PositionInOnset":
            tokens = values.astype(int)
        elif token_type == "OnsetDev":
            max_onset_dev = self.config.max_num_pos_per_beat * 2
            values = np.minimum(np.maximum(values.round(), -max_onset_dev), max_onset_dev)
            tokens = values + max_onset_dev
        elif token_type == "RelOnsetDev":
            tokens = find_closest(self.rel_onset_deviations, values)
        elif token_type == "PerfDuration":
            return super()._values_to_tokens(values, "Duration", denormalize=False)
        elif token_type == "RelPerfDuration":
            tokens = find_closest(self.rel_performed_durations, values)
        elif token_type == "TimeShift":
            tokens = find_closest(self.time_shifts, values)
        elif token_type in ("TimeDuration", "TimeDurationSustain"):
            tokens = find_closest(self.time_durations[1:], values) + 1
            tokens[values == self.time_durations[0]] = 0
        elif token_type == "Sustained":
            tokens = values.astype(int)
        elif token_type == "TimePosition":
            tokens = np.searchsorted(self.time_positions, values, side="right") - 1
        else:
            return super()._values_to_tokens(values, token_type, denormalize=False)

        tokens[is_special] = SPECIAL_TOKENS_VALUE - special_values  # special tokens
        tokens[~is_special] = tokens[~is_special] + self.zero_token

        return tokens

    def _tokens_to_values(
        self,
        tokens: np.ndarray,
        token_type: str,
        clip: bool = False,
        normalize: bool = False,
    ) -> np.ndarray:
        """Decodes values from tokens for a specific `token_type`.

        Internal method used by `decode_values`.

        Args:
            tokens: Array of tokens to decode.
            token_type: Type of tokens (name) to decode.
            clip: Clip values to their quantized boundaries.
            normalize: normalize values after decoding.

        Returns:
            Array of decoded values for provided `token_type`.
        """
        is_special = tokens < self.zero_token
        special_tokens = tokens[is_special]

        tokens = tokens - self.zero_token
        if token_type == "PitchClass":
            values = tokens
        elif token_type == "PitchOctave":
            values = tokens + self._octave_range[0]
        elif token_type == "PositionShift":
            values = self.position_shifts[tokens]
        elif token_type == "NotesInOnset":
            values = tokens + 1
        elif token_type == "PositionInOnset":
            values = tokens
        elif token_type == "OnsetDev":
            values = tokens - self.config.max_num_pos_per_beat * 2  # max_onset_dev
        elif token_type == "RelOnsetDev":
            values = self.rel_onset_deviations[tokens]
        elif token_type == "PerfDuration":
            return super()._tokens_to_values(
                tokens + self.zero_token, "Duration", normalize=normalize
            )
        elif token_type == "RelPerfDuration":
            values = self.rel_performed_durations[tokens]
        elif token_type == "TimeShift":
            values = self.time_shifts[tokens]
        elif token_type in ("TimeDuration", "TimeDurationSustain"):
            values = self.time_durations[tokens]
        elif token_type == "Sustained":
            values = tokens
        elif token_type == "TimePosition":
            values = self.time_positions[tokens]
        else:
            return super()._tokens_to_values(
                tokens + self.zero_token, token_type, normalize=normalize
            )

        values[is_special] = SPECIAL_TOKENS_VALUE - special_tokens  # special tokens

        if normalize:
            return self._normalize_values(values, token_type)
        return values

    def _clip_values(self, values: np.ndarray, token_type: str) -> np.ndarray:
        """Clamps values to the valid range for a specific `token_type`.

        Args:
            values: Array of values to clamp.
            token_type: Type of tokens (name) to clamp.

        Returns:
            Array of clamped values for provided `token_type`.
        """
        is_special = values <= SPECIAL_TOKENS_VALUE
        special_values = values[is_special]

        if token_type == "PitchClass":
            values = np.clip(values, 0, 11)
        elif token_type == "PitchOctave":
            values = np.clip(values, self._octave_range[0], self._octave_range[1] - 1)
        elif token_type == "PositionShift":
            values = np.clip(values, self.position_shifts[0], self.position_shifts[-1])
        elif token_type == "NotesInOnset":
            values = np.clip(values, 1, self.config.additional_params["max_notes_in_onset"] + 1)
        elif token_type == "PositionInOnset":
            values = np.clip(values, 0, self.config.additional_params["max_notes_in_onset"])
        elif token_type == "OnsetDev":
            max_onset_dev = self.config.max_num_pos_per_beat * 2
            values = np.clip(values, -max_onset_dev, max_onset_dev)
        elif token_type == "RelOnsetDev":
            values = np.clip(values, self.rel_onset_deviations[0], self.rel_onset_deviations[-1])
        elif token_type == "PerfDuration":
            return super()._clip_values(values, "Duration")
        elif token_type == "RelPerfDuration":
            values = np.clip(
                values, self.rel_performed_durations[0], self.rel_performed_durations[-1]
            )
        elif token_type == "TimeShift":
            values = np.clip(values, self.time_shifts[0], self.time_shifts[-1])
        elif token_type in ("TimeDuration", "TimeDurationSustain"):
            values = np.clip(values, self.time_durations[0], self.time_durations[-1])
        elif token_type == "Sustained":
            values = np.clip(values, 0.0, 1.0)
        elif token_type == "TimePosition":
            values = np.clip(values, self.time_positions[0], self.time_positions[-1])
        else:
            return super()._clip_values(values, token_type)

        values[is_special] = special_values
        return values

    def _normalize_values(self, values: np.ndarray, token_type: str) -> np.ndarray:
        """Scales values to a defined range for a specific `token_type`.

        Args:
            values: Array of values to normalize.
            token_type: Type of tokens (name) to normalize.

        Returns:
            Array of normalized values for provided `token_type`.
        """
        is_special = values <= SPECIAL_TOKENS_VALUE
        special_values = values[is_special]

        if token_type == "PitchClass":
            values = (values + 1) / 12
        elif token_type == "PitchOctave":
            values = (values + 1 - self._octave_range[0]) / 11
        elif token_type in ("PositionShift", "OnsetDev"):
            return super()._normalize_values(values, "Position")
        elif token_type == "NotesInOnset":
            values = values / self.config.additional_params["max_notes_in_onset"]
        elif token_type == "PositionInOnset":
            values = (values + 1) / self.config.additional_params["max_notes_in_onset"]
        elif token_type == "RelOnsetDev":
            return values
        elif token_type == "PerfDuration":
            return super()._normalize_values(values, "Duration")
        elif token_type == "RelPerfDuration":
            values = values.copy()
            non_zero = values > 0.0
            values[non_zero] = -np.log(values[non_zero]) / np.log(self.rel_performed_durations[0])
        elif token_type in (
            "TimeShift",
            "TimeDuration",
            "TimePosition",
            "TimeDurationSustain",
            "Sustained",
        ):
            return values
        else:
            return super()._normalize_values(values, token_type)

        values[is_special] = special_values
        return values

    def _denormalize_values(self, values: np.ndarray, token_type: str) -> np.ndarray:
        """Rescales values back to their original units for a specific `token_type`.

        Args:
            values: Array of values to denormalize.
            token_type: Type of tokens (name) to denormalize.

        Returns:
            Array of denormalized values for provided `token_type`.
        """
        is_special = values <= SPECIAL_TOKENS_VALUE
        special_values = values[is_special]

        if token_type == "PitchClass":
            values = np.round(values * 12) - 1
        elif token_type == "PitchOctave":
            values = np.round(values * 11) - 1 + self._octave_range[0]
        elif token_type in ("PositionShift", "OnsetDev"):
            return super()._denormalize_values(values, "Position")
        elif token_type == "NotesInOnset":
            values = values * self.config.additional_params["max_notes_in_onset"]
        elif token_type == "PositionInOnset":
            values = values * self.config.additional_params["max_notes_in_onset"] - 1
        elif token_type == "RelOnsetDev":
            return values
        elif token_type == "PerfDuration":
            return super()._denormalize_values(values, "Duration")
        elif token_type == "RelPerfDuration":
            values = np.exp(-values * np.log(self.rel_performed_durations[0]))
        elif token_type in (
            "TimeShift",
            "TimeDuration",
            "TimePosition",
            "TimeDurationSustain",
            "Sustained",
        ):
            return values
        else:
            return super()._denormalize_values(values, token_type)

        values[is_special] = special_values
        return values

    @property
    def score_sizes(self):
        """The vocabulary sizes for token types specific to score encoding.

        Filters the main vocabulary sizes to include only keys defined in
        `symupe.data.tokenizers.constants.SCORE_KEYS`.

        Returns:
            Dictionary mapping score-side token names to their vocabulary sizes.
        """
        return {key: value for key, value in self.sizes.items() if key in SCORE_KEYS}

    @property
    def performance_sizes(self):
        """The complete dictionary of vocabulary sizes for all supported token types."""
        return self.sizes

    @property
    def time_performance_sizes(self) -> dict[str, int]:
        """The vocabulary sizes for token types used in time-only performance encoding.

        Includes keys defined in `symupe.data.tokenizers.constants.TIME_PERFORMANCE_KEYS`,
        such as `Pitch`, `Velocity`, and absolute time tokens.

        Returns:
            Dictionary mapping time-performance token names to their vocabulary sizes.
        """
        return {key: value for key, value in self.sizes.items() if key in TIME_PERFORMANCE_KEYS}

    @property
    def onset_deviation_token(self) -> str:
        """The string name of the active onset deviation token.

        Determines whether the tokenizer is configured for relative (`RelOnsetDev`)
        or absolute (`OnsetDev`) timing deviations.
        """
        return "RelOnsetDev" if self.config.additional_params["rel_onset_dev"] else "OnsetDev"

    @property
    def performed_duration_token(self) -> str:
        """The string name of the active performance duration token.

        Determines whether the tokenizer is configured for relative (`RelPerfDuration`)
        or absolute (`PerfDuration`) articulation/duration.
        """
        return (
            "RelPerfDuration"
            if self.config.additional_params["rel_perf_duration"]
            else "PerfDuration"
        )

    @property
    def score_only_tokens(self) -> list[str]:
        """A list of token types that are exclusive to score-metrical representations.

        This includes metrical indices like `Bar` and `Position`, as well as
        score-side auxiliary tokens like `PositionShift` and `NotesInOnset`.
        """
        token_types = ["Bar", "Position", "Duration"] + self.time_signature_tokens
        if self.config.additional_params["use_position_shifts"]:
            token_types.append("PositionShift")
        if self.config.additional_params["use_onset_indices"]:
            token_types.extend(["NotesInOnset", "PositionInOnset"])
        return token_types

    @property
    def bar_line_id(self) -> int | None:
        """The vocabulary ID associated with the `BAR_LINE` special token.

        Returns:
            Integer ID if the token exists in the vocabulary, otherwise ``None``.
        """
        if BAR_LINE_TOKEN in self.special_tokens:
            return self[0, BAR_LINE_TOKEN]
        return None

    @property
    def pedal_ids(self) -> tuple[int | None, int | None]:
        """The vocabulary IDs for sustain pedal events.

        Returns:
            Tuple containing the IDs for (`PEDAL_ON`, `PEDAL_OFF`).
            Elements are ``None`` if the respective tokens are not in the vocabulary.
        """
        if PEDAL_ON_TOKEN in self.special_tokens:
            return self[0, PEDAL_ON_TOKEN], self[0, PEDAL_OFF_TOKEN]
        return None, None
