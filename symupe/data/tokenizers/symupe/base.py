"""SyMuPe (Symbolic Music Performance) encoding for score and performance music sequences."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from miditok.constants import TIME_SIGNATURE
from symusic import Score, TimeSignature, Tempo, ControlChange

from ..classes import TokSequence
from ..common import OctupleM
from ..constants import TICKS_PER_QUARTER, SCORE_KEYS
from ...midi.sync import sync_performance_midi, GridLevel
from ...midi.utils import cut_overlapping_notes, sort_notes


@dataclass
class SyMuPeTokSequence(TokSequence):
    score_to_perf_token: np.ndarray | None = None


class SyMuPeBase(OctupleM):
    r"""
    SyMuPeBase: a base class for a family of SyMuPe encodings.

    An extended OctupleM encoding with performance-specific tokens for performance MIDIs.
    """

    def _tweak_config_before_creating_voc(self):
        super()._tweak_config_before_creating_voc()

        # midi postprocessing
        self.config.additional_params["cut_overlapping_notes"] = True

    def preprocess_score(
        self,
        midi: Score,
        quantize_times: bool = True,
        quantize_velocities: bool = False,
        quantize_time_signatures: bool = True,
        quantize_tempos: bool = False,
    ) -> Score:
        r"""
        Preprocess a score ``symusic.Score`` to be used by SPMuple encoding.

        :param midi: `symusic.Score`` object to preprocess.
        :param quantize_times: resample and quantize note times
        :param quantize_velocities: quantize velocity of each note
        :param quantize_time_signatures: resample and quantize time signature times
        :param quantize_tempos: quantize tempo values of each tempo change
        """
        return super().preprocess_score(
            midi,
            quantize_times=quantize_times,
            quantize_velocities=quantize_velocities,
            quantize_time_signatures=quantize_time_signatures,
            quantize_tempos=quantize_tempos,
        )

    def preprocess_performance(self, midi: Score) -> Score:
        r"""
        Preprocess a performance ``symusic.Score`` to be used by SPMuple encoding.

        :param midi: `symusic.Score`` object to preprocess
        """
        return self.preprocess_score(
            midi,
            quantize_times=False,
            quantize_velocities=False,
            quantize_time_signatures=False,
            quantize_tempos=False,
        )

    def encode_score(self, midi: Score) -> TokSequence:
        r"""
        Tokenize a score MIDI file into :class:`miditok.TokSequence` using OctupleM encoding.

        The resulting `miditok.TokSequence` provides the alignment between the token and note
        sequences (`score_token_to_score_note`) resulting after the note quantization and sorting.
        This alignment should be used to compute the token-level alignment between the score
        and performance token sequences.

        :param midi: the MIDI objet to convert
        :return: a :class:`miditok.TokSequence`.
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
        tokens = vars(tokens)
        tokens.update(token_to_note=token_to_note)
        tokens = TokSequence(**tokens)

        return tokens

    def encode_performance(
        self,
        midi: Score,
        score_tokens: TokSequence | None,
        note_alignment: np.ndarray | None = None,
    ) -> SyMuPeTokSequence:
        r"""
        Tokenize a performance MIDI file into :class:`miditok.TokSequence`.

        Use `alignment` to provide the MIDI-level mapping between the score and performance notes.
        The alignment on the token level is computed inside using `score_tokens.token_to_note`
        (alignment between score tokens and notes) and is returned as a token sequence metadata.

        :param midi: the MIDI object to convert.
        :param score_tokens: corresponding score tokens :class:`miditok.TokSequence`.
        :param note_alignment: optional alignment between score and performance notes (`score_note_to_perf_note`).
        :return: a :class:`miditok.TokSequence`.
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
            token_to_note=perf_token_to_perf_note, score_to_perf_token=score_token_to_perf_token
        )
        tokens["meta"] = tokens.get("meta", {})
        tokens["meta"].update(time_division=midi.ticks_per_quarter)
        tokens = SyMuPeTokSequence(**tokens)

        return tokens

    @abstractmethod
    def _encode_performance(
        self,
        midi: Score,
        score_tokens: TokSequence,
        note_alignment: np.ndarray | None = None,
    ) -> TokSequence:
        r"""
        Convert a MIDI file to a performance tokens representation, a sequence of "time steps"
        of score tokens stacked with performance specific features (e.g., OnsetDeviation).

        :param midi: the MIDI object to convert.
        :param score_tokens: corresponding score tokens :class:`miditok.TokSequence`.
        :param note_alignment: optional alignment between performance and score notes.
        :return: the performance token representation, i.e. tracks converted into sequences of tokens
        """
        raise NotImplementedError

    def decode_score(
        self,
        tokens: TokSequence | list[list[int]] | np.ndarray,
        programs: list[tuple[int, bool]] | None = None,
        output_path: str | None = None,
    ) -> Score:
        r"""
        Detokenize a sequence of score tokens into a ``symusic.Score``.

        :param tokens: tokens to convert. Can be a list :class:`miditok.TokSequence`,
            a numpy array or a Python list of ints.
        :param programs: programs of the tracks. If none is given, will default to
            piano, program 0. (default: ``None``)
        :param output_path: path to save the file. (default: ``None``)
        :return: the ``symusic.Score`` object.
        """
        return self.decode(tokens, programs=programs, output_path=output_path)

    def decode_performance(
        self,
        tokens: TokSequence | list[list[int]] | np.ndarray,
        programs: list[tuple[int, bool]] | None = None,
        time_division: int = TICKS_PER_QUARTER,
        output_path: str | None = None,
        **kwargs,
    ) -> Score:
        r"""
        Detokenize a sequences of performance tokens into a ``symusic.Score``.

        :param tokens: tokens to convert. Can be a list :class:`miditok.TokSequence`,
            a numpy array or a Python list of ints.
        :param programs: programs of the tracks. If none is given, will default to
            piano, program 0. (default: ``None``)
        :param time_division: MIDI time division / resolution, in ticks/beat
        :param output_path: path to save the file. (default: ``None``)
        :return: the ``symusic.Score`` object.
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

    @abstractmethod
    def _decode_performance(
        self,
        tokens: TokSequence,
        programs: list[tuple[int, bool]] | None = None,
        time_division: int = TICKS_PER_QUARTER,
        **kwargs,
    ) -> Score:
        r"""
        Convert performance tokens (:class:`miditok.TokSequence`) into a ``symusic.Score``.

        This is an internal method called by ``self.decode_performance``, intended to be
        implemented by classes inheriting :class:`miditok.MusicTokenizer`.

        :param tokens: tokens to convert. Can be either a list of
            :class:`miditok.TokSequence` or a list of :class:`miditok.TokSequence`s.
        :param programs: programs of the tracks. If none is given, will default to
            piano, program 0. (default: ``None``)
        :param time_division: MIDI time division / resolution, in ticks/beat
        :return: the ``symusic.Score`` object.
        """
        raise NotImplementedError

    def synchronize_performance_midi(
        self,
        perf_midi: Score,
        score_midi: Score,
        note_alignment: np.ndarray,
    ) -> Score:
        r"""
        Synchronize a performance MIDI file with a score MIDI file bar/beat grid,
        compute bar/beat tempos and change ticks of all notes according to these tempos.

        Should be used for tokenizers with beat-/bar- performance tempo tokens.

        **NOTE**: not an inplace operation.

        :param perf_midi: the performance MIDI object to convert
        :param score_midi: the reference score MIDI object to convert
        :param note_alignment: alignment between performance and score notes
        :return: the bar-/beat-synchronized performance MIDI
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

    @abstractmethod
    def score_tokens_as_performance(
        self,
        score_tokens: TokSequence | list[list[int]] | np.ndarray,
    ) -> TokSequence:
        r"""
        Convert a sequence of score tokens into a sequence of performance tokens,
        the tokens corresponding to a deadpan performance with no variation from score notes.
        """
        raise NotImplementedError

    @abstractmethod
    def _create_base_vocabulary(self) -> list[list[str]]:
        r"""
        Create the vocabulary, as a list of string tokens.

        :return: the vocabulary as a list of string.
        """
        return super()._create_base_vocabulary()

    @abstractmethod
    def _get_token_types(self) -> list[str]:
        r"""Create an ordered list of available token types."""
        return super()._get_token_types()

    @property
    def score_sizes(self):
        return {key: value for key, value in self.sizes.items() if key in SCORE_KEYS}

    @property
    def performance_sizes(self):
        return self.sizes
