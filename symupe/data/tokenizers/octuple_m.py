"""
OctupleM encoding method, a modified Octuple encoding,
introduced in MusicBERT https://arxiv.org/abs/2106.05630

Reimagines the Octuple tokenizer in MidiTok package (https://github.com/Natooz/MidiTok)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import partial
from math import ceil

import numpy as np
import torch
from miditok import Event
from miditok.constants import TIME_SIGNATURE, TEMPO
from miditok.utils.utils import tempo_qpm_to_mspq
from symusic import Score, Note, Tempo, TimeSignature

from symupe.utils import find_closest, forward_fill, backward_fill
from .classes import TokSequence, SequenceType, EncodingType, TokSequenceContext, backend
from .constants import (
    TICKS_PER_QUARTER,
    SPECIAL_TOKENS_VALUE,
    MASK_TOKEN,
    SOS_TOKEN,
    EOS_TOKEN,
    IGNORE_TOKEN,
    BAR_LINE_TOKEN,
)
from .midi_tokenizer import MusicTokenizer
from ..midi.beats import get_bar_beat_ticks
from ..midi.utils import sort_notes, UNPERFORMED_TRACK_NAME, create_unperformed_notes_track


class OctupleM(MusicTokenizer):
    r"""OctupleM: A modified and fast Octuple encoding method based on vectorized array operations.

    Introduced as a variation of the Octuple encoding from MusicBERT (https://arxiv.org/abs/2106.05630),
    this class reimplements the logic using NumPy/vectorized operations for significant
    performance gains over standard iterative tokenizers.

    **Dimensions:**
    * 0: `Bar`
    * 1: `Position`
    * 2: `Pitch`
    * 3: `Duration`
    * 4: `Velocity`
    * (+ Optional) `Tempo`
    * (+ Optional) `TimeSignature` or (`BeatDuration`, `BeatsInBar` / `MaxBarPosition`)
    * (+ Optional) `Program`
    """

    def _tweak_config_before_creating_voc(self):
        """Tweaks default :class:`TokenizerConfig`."""
        self.config.use_chords = False
        self.config.use_rests = False
        self.config.use_sustain_pedals = False
        self.config.use_pitch_bends = False
        self.config.use_pitch_intervals = False
        self.config.use_pitchdrum_tokens = False
        self.config.delete_equal_successive_tempo_changes = True
        self.config.delete_equal_successive_time_sig_changes = True

        # override miditok's configuration
        self.one_token_stream = self.config.one_token_stream_for_programs
        self._disable_attribute_controls()

        additional_params = self.config.additional_params

        # used in place of positional encoding
        # max embedding as seen outside the tokenizer and used by the model
        additional_params["max_bar_embedding"] = additional_params.get("max_bar_embedding", 64)

        # max embedding used by tokenizer, might increase over tokenizations, if the tokenizer encounters longer MIDIs
        additional_params["real_max_bar_embedding"] = additional_params.get(
            "real_max_bar_embedding", additional_params["max_bar_embedding"]
        )

        # time signature configuration
        additional_params["compound_time_signature"] = additional_params.get(
            "compound_time_signature", additional_params.get("compound_time_signatures", False)
        )
        additional_params["time_signature_max_position"] = additional_params.get(
            "time_signature_max_position", False
        )

        # data preprocessing
        additional_params["fill_unperformed_notes"] = True
        self.config.remove_duplicated_notes = False

        self._duration_values = None

    def fill_unperformed_notes(self, score: Score) -> Score:
        """Adds unperformed notes encoded as markers on a separate track
        unless these notes are already added to MIDI object.

        Args:
            score: :class:`symusic.Score` object to process.

        Returns:
            Modified :class:`symusic.Score` object with unperformed notes track added.
        """
        if (
            self.config.additional_params["fill_unperformed_notes"]
            and score.tracks[-1].name != UNPERFORMED_TRACK_NAME
        ):
            notes = []
            for m in score.markers:
                if m.text.startswith("NoteS"):
                    pitch, start, duration = map(int, m.text.split("_")[1:])
                    notes.append(Note(time=start, duration=duration, pitch=pitch, velocity=0))
            if notes:
                score.tracks.append(create_unperformed_notes_track())
                score.tracks[-1].notes = notes

        return score

    def preprocess_score(
        self,
        score: Score,
        quantize_times: bool = True,
        quantize_velocities: bool = True,
        quantize_time_signatures: bool = True,
        quantize_tempos: bool = True,
    ) -> Score:
        """Preprocesses a score :class:`symusic.Score` object for the OctupleM encoding.

        Args:
            score: :class:`symusic.Score` object to process.
            quantize_times: Resample and quantize note times.
            quantize_velocities: Quantize velocity of each note.
            quantize_time_signatures: Resample and quantize time signature times.
            quantize_tempos: Quantize tempo values.

        Returns:
            Preprocessed :class:`symusic.Score` object.
        """
        # Insert unperformed notes on a new track
        score = self.fill_unperformed_notes(score)

        # Do base preprocessing
        return super().preprocess_score(
            score,
            quantize_times=quantize_times,
            quantize_velocities=quantize_velocities,
            quantize_time_signatures=quantize_time_signatures,
            quantize_tempos=quantize_tempos,
        )

    def _add_time_events(self, events: list[Event], time_division: int) -> list[list[Event]]:
        """Creates the time events from a list of global and track events.

        Left as a plug in since we override `_score_to_tokens` directly.

        Args:
            events: Sequence of global and track events to create tokens time from.
            time_division: MIDI time division / resolution, in ticks/beat.

        Returns:
            Sequence of events with time events inserted.
        """
        ...

    def encode(self, score: Score, **kwargs) -> TokSequence | list[TokSequence]:
        """Converts a MIDI into a sequence of OctupleM tokens.

        Args:
            score: :class:`symusic.Score` object to convert.

        Returns:
            :class:`TokSequence` object.
        """
        # Preprocess the MIDI file
        score = self.preprocess_score(
            score,
            quantize_times=False,
            quantize_velocities=False,
            quantize_tempos=False,
        )

        # Sort notes and compute note order change
        token_to_note_alignments = []
        for track in score.tracks:
            track.notes, track_token_to_note = sort_notes(track.notes)
            token_to_note_alignments.append(track_token_to_note)
        token_to_note = np.concatenate(token_to_note_alignments)

        # Tokenize it
        tokens = self._score_to_tokens(score)

        # Add alignment between notes and tokens
        tokens = vars(tokens)
        tokens.update(token_to_note=token_to_note)
        tokens = TokSequence(**tokens)

        return tokens

    def _score_to_tokens(
        self,
        score: Score,
        attribute_controls_indexes: Mapping[int, Mapping[int, Sequence[int] | bool]] | None = None,
    ) -> TokSequence | list[TokSequence]:
        """Converts a **preprocessed** file object to a sequence of tokens.

        Args:
            score: {The p}reprocessed :class:`symusic.Score` object.
            attribute_controls_indexes: Indices of the attribute controls to compute
                and associated tracks and bars.

        Returns:
            :class:`TokSequence` representing the score.
        """
        ticks_per_sample = score.ticks_per_quarter / self.config.max_num_pos_per_beat
        bar_ticks, _ = get_bar_beat_ticks(score)

        # Check bar embedding limit, update if needed
        num_bars = len(bar_ticks)
        if self.config.additional_params["real_max_bar_embedding"] < num_bars:
            for i in range(self.config.additional_params["real_max_bar_embedding"], num_bars):
                self.add_to_vocab(f"Bar_{i}", vocab_idx=self.vocab_types_idx["Bar"])
            self.config.additional_params["real_max_bar_embedding"] = num_bars

        values = []
        for track in score.tracks:
            note_soa = track.notes.numpy()
            ticks = note_soa["time"]

            bar_values = np.searchsorted(bar_ticks, ticks, side="right") - 1
            pos_values = (ticks - bar_ticks[bar_values]) / ticks_per_sample

            pitch_values = note_soa["pitch"]
            velocity_values = note_soa["velocity"]
            duration_values = note_soa["duration"] / ticks_per_sample

            track_values = np.stack(
                [
                    bar_values,
                    pos_values,
                    pitch_values,
                    duration_values,
                    velocity_values,
                ],
                axis=1,
            )

            if self.config.use_tempos:
                tempo_soa = score.tempos.numpy()
                tempo_ids = np.minimum(
                    np.searchsorted(tempo_soa["time"], ticks, side="right") - 1,
                    tempo_soa["time"].shape[0] - 1,
                )
                tempo_values = tempo_qpm_to_mspq(tempo_soa["mspq"])
                track_values = np.concatenate(
                    [track_values, tempo_values[tempo_ids][:, None]], axis=1
                )

            if self.config.use_time_signatures:
                ts_soa = score.time_signatures.numpy()
                ts_ids = np.searchsorted(ts_soa["time"], ticks, side="right") - 1

                if self.config.additional_params["compound_time_signature"]:
                    beat_values = np.array([1 / denom for denom in ts_soa["denominator"]])
                    track_values = np.concatenate(
                        [track_values, beat_values[ts_ids][:, None]], axis=1
                    )

                    if self.config.additional_params["time_signature_max_position"]:
                        num_quarters = np.array(
                            [
                                4 * num / denom
                                for num, denom in zip(ts_soa["numerator"], ts_soa["denominator"])
                            ]
                        )
                        max_position_values = num_quarters * self.config.max_num_pos_per_beat
                        track_values = np.concatenate(
                            [track_values, max_position_values[ts_ids][:, None]], axis=1
                        )
                    else:
                        num_beats_values = np.array([num for num in ts_soa["numerator"]])
                        track_values = np.concatenate(
                            [track_values, num_beats_values[ts_ids][:, None]], axis=1
                        )
                else:
                    ts_values = np.array(
                        [
                            num + 1 / denom
                            for num, denom in zip(ts_soa["numerator"], ts_soa["denominator"])
                        ]
                    )
                    track_values = np.concatenate(
                        [track_values, ts_values[ts_ids][:, None]], axis=1
                    )

            if self.config.use_programs:
                program_values = np.full_like(track_values[0], fill_value=track.program)
                track_values = np.concatenate([track_values, program_values], axis=1)

            values.append(track_values)

        if len(values) > 1:
            values = np.concatenate(values, axis=0)
            tokens = self.encode_tokens(values)

            if self.config.use_programs:
                sort_ids = np.lexsort(
                    [
                        tokens[:, 2],
                        tokens[:, self.vocab_types_idx["Program"], tokens[:, 1], tokens[:, 0]],
                    ]
                )
            else:
                sort_ids = np.lexsort([tokens[:, 2], tokens[:, 1], tokens[:, 0]])
            values, tokens = values[sort_ids], tokens[sort_ids]
        else:
            values = values[0]
            tokens = self.encode_tokens(values)

        tokens = tokens.astype(int)

        tok_sequence = TokSequence(
            ids=tokens,
            values=values,
            type=SequenceType.SCORE,
            encoding=EncodingType.SCORE,
            vocab={
                ttype: idx for ttype, idx in self.vocab_types_idx.items() if idx < len(tokens[0])
            },
            meta={
                "time_division": score.ticks_per_quarter,
                "bars": int(tokens[-1, self.vocab_types_idx["Bar"]] - self.zero_token + 1),
            },
        )

        return tok_sequence

    def decode_note_positions(
        self,
        tokens: TokSequence,
        context: TokSequenceContext | None = None,
        time_division: int | None = TICKS_PER_QUARTER,
    ) -> tuple[dict[str, any], TokSequenceContext]:
        """Decodes temporal metadata from a token sequence to determine note onsets/offsets.

        Extracts `Bar`, `Position`, and `TimeShift` information to calculate absolute
        ticks and seconds.

        Args:
            tokens: :class:`TokSequence` object to decode.
            context: Optional :class:`TokSequenceContext` for incremental decoding.
            time_division: MIDI resolution (Ticks Per Quarter).

        Returns:
            Tuple containing a dictionary of position data (ticks, times, tempos)
            and the updated :class:`TokSequenceContext` object.
        """
        time_division = time_division or self.time_division

        context = context or TokSequenceContext()
        prev_tempos, prev_tempo_ticks, prev_tempo_times = context.tempos or (None, None, None)

        # Compute NoteON, Time Signature, Bar and Beat ticks
        ticks_data = self.compute_ticks(tokens, context=context, time_division=time_division)

        note_on_ticks = ticks_data["note_on"].round().astype(int)
        durations = ticks_data["duration"].round().astype(int)
        note_off_ticks = note_on_ticks + durations

        # Process Tempo changes
        tempos, tempo_ticks, tempo_times = self._decode_tempos(
            tempo_values=self.get_values(tokens, "Tempo"),
            prev_tempos=prev_tempos,
            prev_tempo_ticks=prev_tempo_ticks,
            prev_tempo_times=prev_tempo_times,
            beat_ticks=ticks_data["beat"],
            score_ticks=note_on_ticks,
            time_division=time_division,
        )

        note_on_times, note_off_times = self._decode_note_times(
            note_on_ticks,
            note_off_ticks,
            tempos,
            tempo_ticks,
            tempo_times,
            time_division=time_division,
        )

        position_data = {
            "ticks_data": ticks_data,
            "note_on_ticks": note_on_ticks,
            "note_off_ticks": note_off_ticks,
            "note_on_times": note_on_times,
            "note_off_times": note_off_times,
            "tempos": (tempos, tempo_ticks, tempo_times),
        }

        # Build new context
        if prev_tempos is not None:
            tempos = np.concatenate([prev_tempos, tempos[1:]], axis=0)
            tempo_ticks = np.concatenate([prev_tempo_ticks, tempo_ticks[1:]], axis=0)
            tempo_times = np.concatenate([prev_tempo_times, tempo_times[1:]], axis=0)

        # remove duplicates by ticks and tempos
        tempos, tempo_ticks, tempo_times = self._filter_equal_tempos(
            tempos, tempo_ticks, tempo_times
        )

        def extend_context(prev_data, new_data):
            return np.concatenate([prev_data, new_data]) if prev_data is not None else new_data

        new_context = TokSequenceContext(
            time_signatures=ticks_data["time_sig"],
            tempos=(tempos, tempo_ticks, tempo_times),
            score_ticks=extend_context(context.score_ticks, note_on_ticks),
            note_on_ticks=extend_context(context.note_on_ticks, note_on_ticks),
            note_on_times=extend_context(context.note_on_times, note_on_times),
        )

        return position_data, new_context

    def _decode_tempos(
        self,
        tempo_values: np.ndarray,
        prev_tempos: np.ndarray,
        prev_tempo_ticks: np.ndarray,
        prev_tempo_times: np.ndarray,
        beat_ticks: np.ndarray,
        score_ticks: np.ndarray,
        time_division: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Internal logic to decode tempo changes into absolute ticks and seconds.

        Args:
            tempo_values: Array of tempo values from tokens.
            prev_tempos: Tempos from previous context.
            prev_tempo_ticks: Ticks from previous context.
            prev_tempo_times: Times from previous context.
            beat_ticks: Array of beat positions in ticks.
            score_ticks: Array of note onsets in ticks.
            time_division: MIDI time division / resolution, in ticks/beat.

        Returns:
            Tuple of (tempos, tempo_ticks, tempo_times).
        """
        time_division = time_division or self.time_division

        # Process Tempo changes
        tempo_indices = np.concatenate([[0], np.where(np.diff(tempo_values))[0] + 1])
        tempos = tempo_values[tempo_indices]

        start_tempo_change = prev_tempos is not None and prev_tempos[-1] != tempos[0]
        if start_tempo_change:
            tempos = np.concatenate([[prev_tempos[-1]], tempos])

        prev_tempo_tick = 0 if prev_tempo_ticks is None else prev_tempo_ticks[-1]
        prev_tempo_time = 0.0 if prev_tempo_times is None else prev_tempo_times[-1]

        # Tempo ticks and Tempo changes
        tempo_ticks = score_ticks[tempo_indices]  # Note: position at the start of the beat
        tempo_ticks = beat_ticks[
            np.minimum(
                np.searchsorted(beat_ticks, tempo_ticks, side="right") - 1, beat_ticks.shape[0] - 1
            )
        ]
        tempo_ticks[0] = prev_tempo_tick

        if start_tempo_change:
            new_tempo_tick = beat_ticks[
                np.minimum(
                    np.searchsorted(beat_ticks, score_ticks[0], side="right") - 1,
                    beat_ticks.shape[0] - 1,
                )
            ]
            tempo_ticks = np.concatenate([[tempo_ticks[0]], [new_tempo_tick], tempo_ticks[1:]])

        tempo_times = np.cumsum(
            np.concatenate(
                [[prev_tempo_time], np.diff(tempo_ticks) / time_division * 60 / tempos[:-1]]
            )
        )

        return tempos, tempo_ticks, tempo_times

    def _decode_note_times(
        self,
        note_on_ticks,
        note_off_ticks,
        tempos: np.ndarray,
        tempo_ticks: np.ndarray,
        tempo_times: np.ndarray,
        time_division: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Converts absolute tick positions to performance times in seconds.

        Args:
            note_on_ticks: Onsets in ticks.
            note_off_ticks: Offsets in ticks.
            tempos: Decoded tempo values.
            tempo_ticks: Decoded tempo change ticks.
            tempo_times: Decoded tempo change times.
            time_division: MIDI time division / resolution, in ticks/beat.

        Returns:
            Tuple of (note_on_times, note_off_times) in seconds.
        """
        time_division = time_division or self.time_division

        note_ticks = np.concatenate([note_on_ticks, note_off_ticks])

        tempo_ids = np.searchsorted(tempo_ticks, note_ticks, side="right") - 1
        _tempos, _tempo_ticks, _tempo_times = map(
            lambda t: t[tempo_ids], (tempos, tempo_ticks, tempo_times)
        )
        note_times = _tempo_times + (note_ticks - _tempo_ticks) / time_division * 60 / _tempos

        note_on_times, note_off_times = np.split(note_times, 2)

        return note_on_times, note_off_times

    @staticmethod
    def _filter_equal_tempos(
        tempos: np.ndarray,
        tempo_ticks: np.ndarray,
        tempo_times: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Removes redundant successive tempo changes at the same tick or with the same value.

        Args:
            tempos: Array of tempo values.
            tempo_ticks: Array of ticks.
            tempo_times: Array of times.

        Returns:
            Tuple of cleaned arrays of tempos, ticks, and times.
        """
        _tempo_ticks = np.concatenate([tempo_ticks, [-1]])
        mask = (_tempo_ticks[1:] - _tempo_ticks[:-1]) != 0
        tempos, tempo_ticks, tempo_times = map(
            lambda t: t[mask], (tempos, tempo_ticks, tempo_times)
        )

        _tempos = np.concatenate([[-1], tempos])
        mask = (_tempos[1:] - _tempos[:-1]) != 0
        tempos, tempo_ticks, tempo_times = map(
            lambda t: t[mask], (tempos, tempo_ticks, tempo_times)
        )

        return tempos, tempo_ticks, tempo_times

    def _tokens_to_score(
        self,
        tokens: TokSequence | list[TokSequence],
        programs: list[tuple[int, bool]] | None = None,
    ) -> Score:
        """Internal logic to convert :class:`TokSequence` object back into a :class:`symusic.Score`.

        Args:
            tokens: Token sequence to convert.
            programs: Optional list of (program, is_drum) for tracks.

        Returns:
            Reconstructed :class:`symusic.Score` object.
        """
        tokens.meta = tokens.meta or {}
        time_division = tokens.meta.get("time_division", self.time_division)

        # Compute NoteON, Duration, Time Signature, Bar and Beat ticks
        ticks_data = self.compute_ticks(tokens, time_division=time_division)

        # Note attributes
        note_on_ticks = ticks_data["note_on"].round().astype(int)
        durations = ticks_data["duration"].round().astype(int)
        pitches = self.get_values(tokens, "Pitch", from_ids=True)
        velocities = self.get_values(tokens, "Velocity")

        # Build Time Signature changes
        time_sigs, time_sig_ticks = ticks_data["time_sig"]
        time_signatures = TimeSignature.from_numpy(
            time=time_sig_ticks,
            numerator=time_sigs[:, 0],
            denominator=time_sigs[:, 1],
        )

        # Process Tempo changes
        if self.config.use_tempos and self.has_token_types(tokens, "Tempo"):
            tempos = self.get_values(tokens, "Tempo")
            tempo_indices = np.concatenate([[0], np.where(np.diff(tempos))[0] + 1])
            tempos = tempo_qpm_to_mspq(tempos[tempo_indices])

            if len(tempos) > 0:
                # Get beat ticks to tie Tempo change to them
                beat_ticks = ticks_data["beat"]
                # Note: position at the start of the beat
                tempo_ticks = note_on_ticks[tempo_indices]
                tempo_ticks = beat_ticks[
                    np.minimum(np.searchsorted(beat_ticks, tempo_ticks), beat_ticks.shape[0] - 1)
                ]
                tempo_ticks[0] = 0
            else:
                tempo_ticks = [0]

            tempos = Tempo.from_numpy(time=tempo_ticks, mspq=tempos)
        else:
            tempos = [Tempo(time=0, qpm=TEMPO)]

        # Process Programs
        programs = (
            self.get_values(tokens, "Program", from_ids=True) if self.config.use_programs else None
        )

        score = self._build_score(
            times=note_on_ticks,
            durations=durations,
            pitches=pitches,
            velocities=velocities,
            programs=programs,
            time_signatures=time_signatures,
            tempos=tempos,
            time_division=time_division,
            ttype="tick",
        )
        return score

    def _tokens_to_midi_messages(
        self,
        tokens: TokSequence,
        context: TokSequenceContext | None = None,
        note_attributes: bool = True,
        note_on_events: bool = True,
        note_off_events: bool = True,
        sort: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, TokSequenceContext]:
        """Decodes :class:`TokSequence` into raw MIDI event attributes and temporal metadata.

        This internal method decomposes the compound tokens into their constituent
        performance data, facilitating the reconstruction of MIDI tracks.

        Args:
            tokens: :class:`TokSequence` to convert.
            context: Optional :class:`TokSequenceContext` for incremental decoding.
            note_attributes: If ``True``, extracts pitch and velocity values.
            note_on_events: If ``True``, extracts note-on timing events.
            note_off_events: If ``True``, extracts note-off timing events.
            sort: If ``True``, ensures the resulting events are chronologically ordered.

        Returns:
            Tuple containing (note_on_times, note_off_times, pitches, velocities, new_context).
        """
        position_data, new_context = self.decode_note_positions(tokens=tokens, context=context)

        note_on_times = position_data["note_on_times"]
        note_off_times = position_data["note_off_times"]

        pitches, velocities = None, None
        if note_attributes:
            pitches = self.get_values(tokens, "Pitch", from_ids=True)
            velocities = self.get_values(tokens, "Velocity")

        return note_on_times, note_off_times, pitches, velocities, new_context

    def sort_tokens(self, tokens: TokSequence) -> TokSequence:
        """Sorts :class:`TokSequence` object by `Bar`, then `Position`, then `Pitch`.

        Args:
            tokens: :class:`TokSequence` object to sort.

        Returns:
            Sorted :class:`TokSequence` object.
        """
        vocab = tokens.vocab or self.vocab_types_idx

        sort_ids = np.lexsort(
            (
                tokens.ids[:, vocab["Pitch"]],
                tokens.ids[:, vocab["Position"]],
                tokens.ids[:, vocab["Bar"]],
            )
        )

        tokens.ids = tokens.ids[sort_ids]
        if tokens.values is not None:
            tokens.values = tokens.values[sort_ids]
        if tokens.interpolated is not None:
            tokens.interpolated = tokens.interpolated[sort_ids]

        return tokens

    def _create_base_vocabulary(self) -> list[list[str]]:
        """Creates vocabulary, as a list of lists of string token names.

        Returns:
            Stacked vocabulary of token names.
        """
        vocab = []

        # BAR
        vocab.append(
            [f"Bar_{i}" for i in range(self.config.additional_params["real_max_bar_embedding"])]
        )

        # POSITION
        self._max_num_quarters = max(map(lambda ts: ceil(4 * ts[0] / ts[1]), self.time_signatures))
        num_positions = self._max_num_quarters * self.config.max_num_pos_per_beat
        vocab.append([f"Position_{i}" for i in range(num_positions + 1)])

        # PITCH
        pitch_range = range(self.config.pitch_range[0], self.config.pitch_range[1] + 1)
        vocab.append([f"Pitch_{i}" for i in pitch_range])

        # DURATION
        self.durations = [(0, 0, self.durations[0][-1])] + self.durations  # allow 0 duration
        vocab.append([f"Duration_{'.'.join(map(str, duration))}" for duration in self.durations])
        self._tpb_to_time_array = self._MusicTokenizer__create_tpb_to_ticks_array()
        self._tpb_tokens_to_ticks = self._MusicTokenizer__create_tpb_tokens_to_ticks()
        self._tpb_ticks_to_tokens = self._MusicTokenizer__create_tpb_ticks_to_tokens()

        # VELOCITY
        velocity_step = round(128 / self.config.num_velocities)
        self.velocities = np.concatenate(
            (np.arange(0, 127, velocity_step), [127])
        )  # allow 0 (unperformed note)
        vocab.append([f"Velocity_{i}" for i in self.velocities])

        # TEMPO
        if self.config.use_tempos:
            vocab.append([f"Tempo_{i}" for i in self.tempos])

        # TIME_SIGNATURE
        if self.config.use_time_signatures:
            if self.config.additional_params["compound_time_signature"]:
                denominators = sorted(set(ts[1] for ts in self.time_signatures))
                beat_durations = [
                    self.durations[np.where(self.duration_values == 4 / d)[0][0]]
                    for d in denominators
                ]
                vocab.append(
                    [f"Duration_{'.'.join(map(str, duration))}" for duration in beat_durations]
                )
                self._beat_durations = 1 / np.array(denominators)

                if self.config.additional_params["time_signature_max_position"]:
                    max_bar_positions = sorted(
                        set(
                            self.config.max_num_pos_per_beat * 4 * num / denom
                            for num, denom in self.time_signatures
                        )
                    )
                    vocab.append([f"Position_{i}" for i in max_bar_positions])
                    self._max_bar_positions = np.array(max_bar_positions)
                else:
                    max_num_beats = max(map(lambda ts: ts[0], self.time_signatures))
                    vocab.append([f"BeatsInBar_{i}" for i in range(1, max_num_beats + 1)])
            else:
                vocab.append([f"TimeSig_{ts[0]}/{ts[1]}" for ts in self.time_signatures])

        # PROGRAM
        if self.config.use_programs:
            vocab.append([f"Program_{i}" for i in self.config.programs])

        token_types = self._get_token_types()
        self.vocab_types_idx = {ttype: idx for idx, ttype in enumerate(token_types)}

        return vocab

    def _get_token_types(self) -> list[str]:
        """Creates an ordered list of available token types."""
        token_types = ["Bar", "Position", "Pitch", "Duration", "Velocity"]

        if self.config.use_tempos:
            token_types.append("Tempo")

        if self.config.use_time_signatures:
            token_types.extend(self.time_signature_tokens)

        if self.config.use_programs:
            token_types.append("Program")

        return token_types

    def _create_token_types_graph(self) -> dict[str, list[str]]:
        """Returns a graph (as a dictionary) of the possible token types successions.

        Not relevant for Octuple.

        Returns:
            Empty dictionary.
        """
        return {}

    def has_token_types(
        self,
        tokens: TokSequence,
        token_types: str | list[str],
        check_values: bool = True,
    ) -> bool:
        """Checks if :class:`TokSequence` object contains specific token dimensions.

        Args:
            tokens: Sequence to check.
            token_types: Single name or list of dimension names.
            check_values: If ``True``, also checks if values are valid (not ignored).

        Returns:
            ``True`` if all types are present and valid, else ``False``.
        """
        token_types = [token_types] if isinstance(token_types, str) else token_types
        vocab = tokens.vocab or self.vocab_types_idx
        if any(token_type not in vocab for token_type in token_types):
            return False
        if not check_values:
            return True

        _backend = backend(tokens)
        values = self.get_values(tokens, token_types)
        return _backend.all(
            _backend.logical_or(values > SPECIAL_TOKENS_VALUE, values < self.ignore_value)
        )

    def get_values(
        self,
        tokens: TokSequence,
        token_type: str | list[str] | None,
        from_ids: bool = False,
    ):
        """Extracts real-valued features from :class:`TokSequence` object.

        Args:
            tokens: :class:`TokSequence` to extract from.
            token_type: Token name (dimension) to retrieve.
            from_ids: If ``True``, decodes values from IDs instead of using precomputed values.

        Returns:
            NumPy array of values for the requested dimension(s).
        """
        assert tokens.ids is not None or tokens.values is not None
        from_ids = (from_ids and tokens.ids is not None) or tokens.values is None
        vocab = tokens.vocab or self.vocab_types_idx

        if token_type is None:
            if from_ids:
                return self.decode_values(tokens)
            else:
                return tokens.values
        elif isinstance(token_type, list):
            if from_ids:
                return backend(tokens).stack(
                    [self.decode_values(tokens.ids, _token_type) for _token_type in token_type], -1
                )
            else:
                return tokens.values[:, [vocab[_token_type] for _token_type in token_type]]
        else:
            if from_ids:
                return self.decode_values(tokens, token_type)
            else:
                return tokens.values[:, vocab[token_type]]

    def _transform_tokens_or_values(
        self,
        tokens_or_values: np.ndarray | torch.Tensor | TokSequence | int | float,
        transform_func: callable,
        token_type: str | list[str] | None = None,
        vocab: dict[str, int] | None = None,
    ) -> np.ndarray | torch.Tensor | TokSequence:
        """Internal method for universal conversion between tokens and values

        Args:
            tokens_or_values: Tokens or values or :class:`TokSequence` to convert.
            transform_func: Function to apply to tokens or values.
            token_type: Optional token name or list of dimension names.
            vocab: Optional token vocabulary for token type indexing.

        Returns:
            Array or :class:`TokSequence` with clamped values.
        """
        is_array = isinstance(tokens_or_values, (np.ndarray, torch.Tensor))
        is_torch = isinstance(tokens_or_values, torch.Tensor)

        device = None
        if is_torch:
            device = tokens_or_values.device
            tokens_or_values = tokens_or_values.detach().cpu().numpy()

        vocab = vocab or self.vocab_types_idx
        token_type = list(vocab.keys()) if token_type is None else token_type

        if isinstance(token_type, list):
            assert is_array

            new_tokens_or_values = []
            for idx, key in enumerate(token_type):
                if idx == tokens_or_values.shape[-1]:
                    break
                new_tokens_or_values.append(transform_func(tokens_or_values[..., idx], key))

            outputs = np.stack(new_tokens_or_values, axis=-1)
        elif is_array and tokens_or_values.ndim >= 2:
            outputs = transform_func(
                tokens_or_values[..., vocab[token_type]], token_type=token_type
            )
        else:
            outputs = transform_func(tokens_or_values, token_type=token_type)

        if is_torch:
            outputs = torch.from_numpy(outputs).to(device=device)
        return outputs

    def encode_tokens(
        self,
        values: np.ndarray | torch.Tensor | TokSequence,
        token_type: str | list[str] | None = None,
        vocab: dict[str, int] | None = None,
        denormalize: bool = False,
        clip: bool = False,
    ) -> np.ndarray | torch.Tensor:
        """Encodes tokens from values for all or a specific `token_type`.

        Args:
            values: Array of values to encode.
            token_type: Optional type of tokens (name) to encode.
            vocab: Optional token vocabulary for token type indexing.
            denormalize: Denormalize values before encoding.
            clip: Clip values to their quantized boundaries.

        Returns:
            Array of encoded tokens.
        """
        is_tok_seq = isinstance(values, TokSequence)
        tokens = self._transform_tokens_or_values(
            values.values if is_tok_seq else values,
            transform_func=partial(self._values_to_tokens, denormalize=denormalize, clip=clip),
            token_type=token_type,
            vocab=values.vocab if is_tok_seq else vocab,
        )
        return tokens.long() if isinstance(tokens, torch.Tensor) else tokens.astype(int)

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
        if clip:
            values = self._clip_values(values, token_type)

        is_special = values <= SPECIAL_TOKENS_VALUE
        special_values = values[is_special]
        tokens = np.zeros_like(values)

        if token_type == "Position":
            max_position = self._max_num_quarters * self.config.max_num_pos_per_beat
            tokens = np.minimum(values.round(), max_position - 1)
        elif token_type == "Pitch":
            tokens = values - self.config.pitch_range[0]
        elif token_type == "Velocity":
            tokens = find_closest(self.velocities, values)
        elif token_type == "Duration":
            tokens = (
                find_closest(self.duration_values[1:] * self.config.max_num_pos_per_beat, values)
                + 1
            )
            tokens[values == self.duration_values[0] * self.config.max_num_pos_per_beat] = 0
        elif token_type == "Tempo":
            tokens = find_closest(self.tempos, values)
        elif token_type == "TimeSig":
            time_sigs = np.stack(
                [np.floor(values[~is_special]), (1 / (values[~is_special] % 1.0)).round()], axis=1
            )
            tokens[~is_special] = np.where(
                np.all(time_sigs[..., None] == np.array(self.time_signatures), axis=-1)
            )[1]
        elif token_type == "BeatDuration":
            tokens = np.argmin(np.fabs(values[..., None] - self._beat_durations), axis=-1)
        elif token_type == "MaxPosition":
            tokens = np.argmin(np.fabs(values[..., None] - self._max_bar_positions), axis=-1)
        elif token_type == "BeatsInBar":
            tokens = values - 1
        else:
            tokens = values.astype(int)

        tokens[is_special] = SPECIAL_TOKENS_VALUE - special_values  # special tokens
        tokens[~is_special] = tokens[~is_special] + self.zero_token

        return tokens

    def decode_values(
        self,
        tokens: np.ndarray | torch.Tensor | TokSequence,
        token_type: str | list[str] | None = None,
        vocab: dict[str, int] | None = None,
        clip: bool = False,
        normalize: bool = False,
    ) -> np.ndarray | torch.Tensor:
        """Decodes values from tokens for all or a specific `token_type`.

        Args:
            tokens: Array of tokens to decode.
            token_type: Optional type of tokens (name) to decode.
            vocab: Optional token vocabulary for token type indexing.
            clip: Clip values to their quantized boundaries.
            normalize: normalize values after decoding.

        Returns:
            Array of decoded values.
        """
        is_tok_seq = isinstance(tokens, TokSequence)
        values = self._transform_tokens_or_values(
            tokens.ids if is_tok_seq else tokens,
            transform_func=partial(self._tokens_to_values, clip=clip, normalize=normalize),
            token_type=token_type,
            vocab=tokens.vocab if is_tok_seq else vocab,
        )
        return values

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
        if token_type in ("Bar", "Position"):
            values = tokens
        elif token_type == "Pitch":
            values = tokens + self.config.pitch_range[0]
        elif token_type == "Velocity":
            values = self.velocities[tokens]
        elif token_type == "Duration":
            values = self.duration_values[tokens] * self.config.max_num_pos_per_beat
        elif token_type == "Tempo":
            values = self.tempos[tokens]
        elif token_type == "TimeSig":
            time_sigs = np.array(self.time_signatures)[tokens]
            values = time_sigs[:, 0] + 1 / time_sigs[:, 1]
        elif token_type == "BeatDuration":
            values = self._beat_durations[np.maximum(0, tokens)]
        elif token_type == "MaxPosition":
            values = self._max_bar_positions[np.maximum(0, tokens)]
        elif token_type == "BeatsInBar":
            values = tokens + 1
        else:
            values = tokens

        values[is_special] = SPECIAL_TOKENS_VALUE - special_tokens  # special tokens

        if clip:
            values = self._clip_values(values, token_type)
        if normalize:
            values = self._normalize_values(values, token_type)
        return values

    def clip_values(
        self,
        values: np.ndarray | torch.Tensor | TokSequence,
        token_type: str | list[str] | None = None,
        vocab: dict[str, int] | None = None,
    ) -> np.ndarray | torch.Tensor | TokSequence:
        """Clamps values to the valid range for all or a specific `token_type`.

        Args:
            values: Array of values or :class:`TokSequence` with values to clamp.
            token_type: Type of tokens (name) to clamp.
            vocab: Optional token vocabulary for token type indexing.

        Returns:
            Array or :class:`TokSequence` with clamped values.
        """
        if isinstance(values, TokSequence):
            if token_type is not None:
                token_type = [token_type] if isinstance(token_type, str) else token_type
                for key in token_type:
                    idx = values.vocab[key]
                    values.values[..., idx] = self._transform_tokens_or_values(
                        values.values[..., idx], transform_func=self._clip_values, token_type=key
                    )
            else:
                values.values = self._transform_tokens_or_values(
                    values.values, transform_func=self._clip_values, vocab=values.vocab
                )
            return values
        else:
            return self._transform_tokens_or_values(
                values, transform_func=self._clip_values, token_type=token_type, vocab=vocab
            )

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

        if token_type == "Bar":
            values = np.clip(values, 0, None)
        elif token_type == "Position":
            values = np.clip(values, 0, self._max_num_quarters * self.config.max_num_pos_per_beat)
        elif token_type in ("Pitch", "Velocity"):
            values = np.clip(values, 0, 127)
        elif token_type == "Duration":
            values = np.clip(
                values, 0.0, self.duration_values[-1] * self.config.max_num_pos_per_beat
            )
        elif token_type == "Tempo":
            values = np.clip(values, self.tempos[0], self.tempos[-1])

        values[is_special] = special_values
        return values

    def normalize_values(
        self,
        values: np.ndarray | torch.Tensor | TokSequence,
        token_type: str | list[str] | None = None,
        vocab: dict[str, int] | None = None,
    ) -> np.ndarray | torch.Tensor | TokSequence:
        """Scales values to a defined range for all or a specific `token_type`.

        Args:
            values: Array of values or :class:`TokSequence` with values to normalize.
            token_type: Type of tokens (name) to normalize.
            vocab: Optional token vocabulary for token type indexing.

        Returns:
            Array or :class:`TokSequence` with normalized values.
        """
        if isinstance(values, TokSequence):
            values.values = self._transform_tokens_or_values(
                values.values, transform_func=self._normalize_values, vocab=values.vocab
            )
            return values
        else:
            return self._transform_tokens_or_values(
                values, transform_func=self._normalize_values, token_type=token_type, vocab=vocab
            )

    def _normalize_values(self, values: np.ndarray, token_type: str) -> np.ndarray:
        """Scales values to a defined range for a specific `token_type`.

        Args:
            values: Array of values to clamp.
            token_type: Type of tokens (name) to clamp.

        Returns:
            Array of normalized values for provided `token_type`.
        """
        is_special = values <= SPECIAL_TOKENS_VALUE
        special_values = values[is_special]

        if token_type == "Bar":
            values = (values + 1) / self.config.additional_params["max_bar_embedding"]
        elif token_type in ("Position", "Duration", "MaxPosition"):
            values = values / self.config.max_num_pos_per_beat / 4  # max(self._tpb_per_ts.keys())
        elif token_type in ("Pitch", "Velocity"):
            values = values / 127
        elif token_type == "Tempo":
            values = values.copy()
            non_zero = values > 0.0
            values[non_zero] = np.log2(values[non_zero] / TEMPO)
        elif token_type == "TimeSig":
            values = values.copy()
            time_sigs = np.stack(
                [np.floor(values[~is_special]), (1 / (values[~is_special] % 1.0)).round()], axis=1
            )
            values[~is_special] = time_sigs[:, 0] / time_sigs[:, 1]
        elif token_type == "BeatDuration":
            return values
        elif token_type == "BeatsInBar":
            values = values / 4

        values[is_special] = special_values
        return values

    def denormalize_values(
        self,
        values: np.ndarray | torch.Tensor | TokSequence,
        token_type: str | list[str] | None = None,
        vocab: dict[str, int] | None = None,
    ) -> np.ndarray | torch.Tensor | TokSequence:
        """Rescales values back to their original units for all or a specific `token_type`.

        Args:
            values: Array of values or :class:`TokSequence` with values to denormalize.
            token_type: Type of tokens (name) to denormalize.
            vocab: Optional token vocabulary for token type indexing.

        Returns:
            Array or :class:`TokSequence` with denormalized values.
        """
        if isinstance(values, TokSequence):
            values.values = self._transform_tokens_or_values(
                values.values, transform_func=self._denormalize_values, vocab=values.vocab
            )
            return values
        else:
            return self._transform_tokens_or_values(
                values, transform_func=self._denormalize_values, token_type=token_type, vocab=vocab
            )

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

        if token_type == "Bar":
            values = values * self.config.additional_params["max_bar_embedding"] - 1
        elif token_type in ("Position", "Duration", "MaxPosition"):
            values = values * self.config.max_num_pos_per_beat * 4
        elif token_type in ("Pitch", "Velocity"):
            values = np.round(values * 127)
        elif token_type == "Tempo":
            values = np.round(np.exp2(values) * TEMPO, 2)
        elif token_type == "TimeSig":
            values = values * 4 + 0.25  # not exactly correct
        elif token_type == "BeatDuration":
            return values
        elif token_type == "BeatsInBar":
            values = values * 4

        values[is_special] = special_values
        return values

    def token_values(self, normalize: bool | list[str] = False) -> dict[str, np.ndarray]:
        """Returns the real values associated with every possible token ID for each type.

        Args:
            normalize: If ``True``, returns normalized values for all types.
                If a list, normalizes only specified types.

        Returns:
            Dictionary mapping dimension names to arrays of all possible values.
        """
        if isinstance(normalize, bool):
            normalize = list(self.vocab_types_idx.keys()) if normalize else []

        token_values = {}
        for key in self.vocab_types_idx.keys():
            token_values[key] = self.decode_values(
                tokens=np.arange(self.sizes[key]), token_type=key, normalize=key in normalize
            )

        return token_values

    def compute_ticks(
        self,
        tokens: TokSequence,
        context: TokSequenceContext | None = None,
        time_division: int | None = None,
    ) -> dict[str, np.ndarray | tuple[np.ndarray, np.ndarray]]:
        """Calculates absolute tick positions for notes, bars, and beats from tokens.

        Args:
            tokens: :class:`TokSequence` to analyze.
            context: Optional preceding context for relative timing.
            time_division: MIDI time division / resolution, in ticks/beat.

        Returns:
            Dictionary containing 'note_on', 'duration', 'bar', 'beat', and 'time_sig' ticks.
        """
        time_division = time_division or self.time_division
        ticks_per_sample = time_division / self.config.max_num_pos_per_beat
        additional_params = self.config.additional_params

        # Incorporate context
        context = context or TokSequenceContext()
        prev_time_sigs, prev_time_sig_ticks = context.time_signatures or (None, np.zeros(1))
        prev_score_ticks = context.score_ticks if context.score_ticks is not None else None

        # Compute tick positions considering the previous context
        if self.config.use_time_signatures and self.has_token_types(
            tokens, self.time_signature_tokens
        ):
            # Compute ticks per position for each note according to time signatures
            if additional_params["compound_time_signature"]:
                all_beat_values = self.get_values(tokens, "BeatDuration", from_ids=True)
                all_beat_values = (1 / all_beat_values).round()

                if additional_params["time_signature_max_position"]:
                    all_max_positions = self.get_values(tokens, "MaxPosition", from_ids=True)
                    all_num_beats = (
                        all_max_positions / self.config.max_num_pos_per_beat / 4 * all_beat_values
                    )
                else:
                    all_num_beats = self.get_values(tokens, "BeatsInBar", from_ids=True)
            else:
                all_time_sig_values = self.get_values(tokens, "TimeSig", from_ids=True)

                all_num_beats = np.floor(all_time_sig_values)
                all_beat_values = (1 / (all_time_sig_values % 1.0)).round()

            all_time_sigs = np.stack([all_num_beats, all_beat_values], axis=1).astype(int)
            ticks_per_pos = ticks_per_sample * 4 / all_time_sigs[:, 1]

            # Compute Time Signature change positions
            time_sig_indices = np.where(np.any(np.diff(all_time_sigs, axis=0), axis=1))[0] + 1
            time_sig_indices = np.concatenate([[0], time_sig_indices])

            # Get time signatures
            time_sigs = all_time_sigs[time_sig_indices]
        else:
            ticks_per_pos = ticks_per_sample

            time_sig_indices = np.zeros((1,), dtype=np.int32)
            time_sigs = np.array([[*TIME_SIGNATURE]])

        # Incorporate previous time signatures and bar lengths
        repeated_time_sig = False
        if prev_time_sigs is not None:
            repeated_time_sig = np.all(time_sigs[0] == prev_time_sigs[-1])
            if repeated_time_sig:
                time_sigs = time_sigs[1:]
            time_sigs = (
                np.concatenate([prev_time_sigs, time_sigs])
                if len(time_sigs) > 0
                else prev_time_sigs
            )

        ticks_per_bar = time_division * 4 * time_sigs[:, 0] / time_sigs[:, 1]
        ticks_per_beat = ticks_per_bar // time_sigs[:, 0]

        bars, note_on_ticks = None, None

        has_score = self.has_token_types(tokens, ["Bar", "Position"])
        if has_score:
            bars = self.get_values(tokens, "Bar").astype(int)

            # Compute time signature ticks
            time_sig_bars = bars[time_sig_indices[int(repeated_time_sig) :]]
            if prev_time_sigs is not None:
                prev_time_sig_bars = np.sum(
                    np.diff(prev_time_sig_ticks) / ticks_per_bar[: len(prev_time_sig_ticks) - 1]
                )
                time_sig_bars = np.concatenate([[prev_time_sig_bars], time_sig_bars])

            time_sig_ticks = np.concatenate(
                [
                    prev_time_sig_ticks,
                    prev_time_sig_ticks[-1]
                    + np.cumsum(
                        ticks_per_bar[len(prev_time_sig_ticks) - 1 : -1] * np.diff(time_sig_bars)
                    ),
                ]
            )
        else:
            assert additional_params["use_position_shifts"] and self.has_token_types(
                tokens, "PositionShift"
            )
            note_on_ticks = np.cumsum(self.get_values(tokens, "PositionShift")) * ticks_per_sample

            # Incorporate previous note on ticks
            if prev_score_ticks is not None:
                note_on_ticks = note_on_ticks + prev_score_ticks[-1]

            # Compute time signature ticks
            time_sig_ticks = np.concatenate(
                [
                    prev_time_sig_ticks if prev_time_sigs is not None else [],
                    note_on_ticks[time_sig_indices][int(repeated_time_sig) :],
                ]
            )
            if time_sig_ticks[0] != 0.0:
                time_sig_ticks[0] = 0.0

        # Compute ticks for each bar
        bar_ticks, beat_ticks = [], []
        for i, time_sig in enumerate(time_sigs[:-1]):
            # move `time_sig_tick` to the start of the bar
            time_sig_ticks[i + 1] -= (time_sig_ticks[i + 1] - time_sig_ticks[i]) % ticks_per_bar[i]

            bar_ticks.append(np.arange(time_sig_ticks[i], time_sig_ticks[i + 1], ticks_per_bar[i]))
            beat_ticks.append(
                np.arange(time_sig_ticks[i], time_sig_ticks[i + 1], ticks_per_beat[i])
            )

        bar_ticks = np.concatenate(bar_ticks) if len(bar_ticks) else []
        if has_score:
            last_tick = time_sig_ticks[-1] + (bars[-1] - len(bar_ticks) + 1) * ticks_per_bar[-1]
        else:
            last_tick = note_on_ticks.max() + ticks_per_bar[-1]
        bar_ticks = np.concatenate(
            [bar_ticks, np.arange(time_sig_ticks[-1], last_tick + 1, ticks_per_bar[-1])]
        )

        beat_ticks.append(np.arange(time_sig_ticks[-1], last_tick + 1, ticks_per_beat[-1]))
        beat_ticks = np.concatenate(beat_ticks)

        if has_score:
            # Compute note on ticks
            positions = self.get_values(tokens, "Position")
            note_on_ticks = bar_ticks[bars] + positions * ticks_per_sample

        # Compute durations
        durations = self.get_values(tokens, "Duration") * ticks_per_sample

        # Combine ticks data
        ticks_data = {
            "time_division": time_division,
            "note_on": note_on_ticks,
            "duration": durations,
            "time_sig": (time_sigs, time_sig_ticks),
            "ticks_per_pos": ticks_per_pos,
            "bar": bar_ticks,
            "beat": beat_ticks,
        }

        return ticks_data

    def compute_bar_beat_onset_indices(
        self,
        tokens: TokSequence | None,
        ticks_data: dict[str, np.ndarray | tuple[np.ndarray, np.ndarray]] | None = None,
        shift_to_zero: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimated bar, beat and onset indices for each note in the token sequence.

        Args:
            tokens: :class:`TokSequence` object to process.
            ticks_data: Dictionary returned by `self.compute_ticks`.
            shift_to_zero: If ``True``, the indices start from 0.

        Returns:
            Tuple of (bar, beat, onset) indices.
        """
        if ticks_data is None:
            assert tokens is not None
            ticks_data = self.compute_ticks(tokens)

        if tokens is not None:
            bars = self.get_values(tokens, "Bar").astype(int)
        else:
            bars = np.searchsorted(ticks_data["bar"], ticks_data["note_on"], side="right") - 1

        beats = np.searchsorted(ticks_data["beat"], ticks_data["note_on"], side="right") - 1

        unique_onsets, onset_notes = np.unique(ticks_data["note_on"], return_counts=True)
        onsets = np.arange(len(unique_onsets)).repeat(onset_notes)

        if shift_to_zero:
            bars, beats, onsets = map(lambda s: s - s.min(), (bars, beats, onsets))
        return bars, beats, onsets

    def compute_position_shifts(
        self, positions: np.ndarray, onset_shift: bool = False
    ) -> np.ndarray:
        """Computes absolute position shifts between onsets from positions.

        Args:
            positions: Array of positions (ticks/beats).
            onset_shift: If ``True``, overwrites tokenizer setting for onset_shift position shift.

        Returns:
            Array of position shits.
        """
        if onset_shift:
            unique_score_pos, score_pos_counts = np.unique(positions, return_counts=True)
            score_pos_ids = np.arange(len(unique_score_pos)).repeat(score_pos_counts)
            pos_shifts = unique_score_pos[score_pos_ids] - unique_score_pos[score_pos_ids - 1]
            pos_shifts[pos_shifts < 0] = positions[pos_shifts < 0]
        else:
            pos_shifts = np.concatenate([positions[:1], np.diff(positions)])
        return pos_shifts

    def shift_positions(
        self,
        tokens: TokSequence,
        shifts: dict[str, int | float] | None = None,
        inverse_shifts: bool = False,
        normalized_values: bool = False,
        shift_to_zero: bool = False,
    ) -> tuple[TokSequence, dict[str, int | float]]:
        """Applies a global metrical shift (Bar offset) to the entire sequence.

        Args:
            tokens: :class:`TokSequence` to shift.
            shifts: Dictionary containing 'Bar' offset.
            inverse_shifts: If ``True``, subtracts the shift.
            normalized_values: Whether values are currently normalized.
            shift_to_zero: If ``True``, shifts the sequence so the first bar is 0.

        Returns:
            Tuple of (shifted_sequence, applied_shifts).
        """
        assert not shift_to_zero or shifts is None

        vocab = tokens.vocab or self.vocab_types_idx

        has_bars = (
            tokens.encoding != EncodingType.TIME_PERFORMANCE
            and "Bar" in vocab
            and "Position" in vocab
            and self.has_token_types(tokens, ["Bar", "Position"])
        )

        shifts = shifts or {"Bar": 0}

        if inverse_shifts:
            shifts = {key: -value for key, value in shifts.items()}

        if shift_to_zero and has_bars:  # move the first note to zero time
            shifts["Bar"] = -int(self.get_values(tokens, "Bar", from_ids=True).min())

        bar_shift = shifts["Bar"]
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

        return tokens, shifts

    def add_sos_token(self, tokens: TokSequence) -> TokSequence:
        """Prepends a Start-Of-Sequence (SOS) token to the sequence.

        Args:
            tokens: :class:`TokSequence` object to update.

        Returns:
            Updated :class:`TokSequence` object.
        """
        assert SOS_TOKEN in self.special_tokens

        _backend = backend(tokens)
        sos_token_id = self[0, SOS_TOKEN]

        tokens.ids = _backend.concatenate(
            (_backend.full_like(tokens.ids[:1], sos_token_id), tokens.ids), 0
        )
        if tokens.values is not None:
            values = (
                _backend.full_like(tokens.values[:1], SPECIAL_TOKENS_VALUE - sos_token_id),
                tokens.values,
            )
            tokens.values = _backend.concatenate(values, 0)
        if tokens.interpolated is not None:
            tokens.interpolated = _backend.concatenate((_backend.zeros(1), tokens.interpolated), 0)

        return tokens

    def add_eos_token(self, tokens: TokSequence, token_name: str | None = None) -> TokSequence:
        """Appends an End-Of-Sequence (EOS) token to the sequence.

        Args:
            tokens: :class:`TokSequence` object to update.

        Returns:
            Updated :class:`TokSequence` object.
        """
        token_name = token_name or EOS_TOKEN
        assert token_name in self.special_tokens

        _backend = backend(tokens)
        eos_token_id = self[0, token_name]

        tokens.ids = _backend.concatenate(
            (tokens.ids, _backend.full_like(tokens.ids[:1], eos_token_id)), 0
        )
        if tokens.values is not None:
            values = (
                tokens.values,
                _backend.full_like(tokens.values[:1], SPECIAL_TOKENS_VALUE - eos_token_id),
            )
            tokens.values = _backend.concatenate(values, 0)
        if tokens.interpolated is not None:
            tokens.interpolated = _backend.concatenate((tokens.interpolated, _backend.zeros(1)), 0)

        return tokens

    def add_bar_line_tokens(self, tokens: TokSequence, start: bool = True) -> TokSequence:
        """Injects `BAR_LINE` special tokens at the boundaries of each bar.

        Args:
            tokens: :class:`TokSequence` object to process.
            start: If ``True``, `BAR_LINE` tokens are inserted at the beginning of each bar.
                If ``False``, `BAR_LINE` tokens are inserted at the end of each bar.

        Returns:
            :class:`TokSequence` with injected `BAR_LINE` tokens.
        """
        if BAR_LINE_TOKEN not in self.special_tokens:
            return tokens

        vocab = tokens.vocab or self.vocab_types_idx
        assert "Bar" in vocab

        _backend = backend(tokens)
        token_id = self[0, BAR_LINE_TOKEN]

        bars = self.get_values(tokens, "Bar")
        min_bar, max_bar = map(int, (bars.min(), bars.max()))
        num_bars = max_bar - min_bar + 1

        new_values = _backend.full((num_bars, tokens.ids.shape[1]), fill_value=self.ignore_value)
        new_values[:, vocab["Pitch"]] = SPECIAL_TOKENS_VALUE - token_id
        new_values[:, vocab["Bar"]] = _backend.arange(min_bar, max_bar + 1)
        if start:
            new_values[:, vocab["Position"]] = 0.0
        else:
            ticks_data = self.compute_ticks(tokens, time_division=self.config.max_num_pos_per_beat)
            new_values[:, vocab["Position"]] = np.diff(ticks_data["bar"][min_bar:])

        new_tokens = replace(
            tokens,
            ids=self.encode_tokens(new_values),
            values=new_values,
            interpolated=(
                np.zeros_like(new_values[:, 0]) if tokens.interpolated is not None else None
            ),
        )
        new_tokens = tokens + new_tokens

        new_tokens = self.sort_tokens(new_tokens)

        for token_type in self.time_signature_tokens:
            if token_type not in vocab:
                continue
            type_idx = vocab[token_type]
            fill_func = backward_fill if start else forward_fill
            new_tokens.values[:, type_idx] = fill_func(
                new_tokens.values[:, type_idx], self.ignore_value
            )
            new_tokens.ids[:, type_idx] = self.encode_tokens(
                new_tokens.values[:, type_idx], token_type=token_type
            )
        tokens.ids, tokens.values = new_tokens.ids, new_tokens.values
        tokens.interpolated = new_tokens.interpolated

        return tokens

    def remove_bar_line_tokens(self, tokens: TokSequence) -> TokSequence:
        """Removes `BAR_LINE` special tokens and recomputes surrounding position shifts.

        Args:
            tokens: :class:`TokSequence` object to process.

        Returns:
            :class:`TokSequence` with deleted `BAR_LINE` tokens.
        """
        if BAR_LINE_TOKEN not in self.special_tokens:
            return tokens

        vocab = tokens.vocab or self.vocab_types_idx
        mask = tokens.ids[:, vocab["Pitch"]] != self[0, BAR_LINE_TOKEN]

        tokens.ids = tokens.ids[mask]
        tokens.values = tokens.values[mask]
        if tokens.interpolated is not None:
            tokens.interpolated = tokens.interpolated[mask]

        return tokens

    @property
    def sizes(self) -> dict[str, int]:
        """The complete dictionary of vocabulary sizes for all supported token types."""
        sizes = {k: len(v) for k, v in zip(self.vocab_types_idx, self.vocab)}
        sizes["Bar"] -= (
            self.config.additional_params["real_max_bar_embedding"]
            - self.config.additional_params["max_bar_embedding"]
        )
        return sizes

    @property
    def zero_token(self) -> int:
        """The vocabulary ID offset where the non-special vocabulary begins."""
        return len(self.special_tokens)

    @property
    def ignore_token(self) -> int:
        """The vocabulary ID used for ignoring a dimension."""
        return self.vocab[0].get(IGNORE_TOKEN, self.vocab[0].get(MASK_TOKEN, 0))

    @property
    def ignore_value(self) -> float:
        """The real-value equivalent of the ignore token."""
        return SPECIAL_TOKENS_VALUE - self.ignore_token

    @property
    def duration_values(self) -> np.ndarray:
        """An array of all possible metrical duration values in quarter notes."""
        if self._duration_values is None:
            self._duration_values = np.array(
                [(beat * res + pos) / res if res > 0 else 0 for beat, pos, res in self.durations]
            )
        return self._duration_values

    @property
    def time_signature_tokens(self) -> list[str]:
        """A list of dimension names used to represent time signatures in the tokenizer."""
        if self.config.use_time_signatures:
            if self.config.additional_params["compound_time_signature"]:
                return [
                    "BeatDuration",
                    "MaxPosition"
                    if self.config.additional_params["time_signature_max_position"]
                    else "BeatsInBar",
                ]
            else:
                return ["TimeSig"]
        return []
