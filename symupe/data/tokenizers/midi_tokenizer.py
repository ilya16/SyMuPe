"""Base tokenizer class, extending miditok.MusicTokenizer."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from miditok import MusicTokenizer as _MusicTokenizer, Event
from miditok.constants import TIME_SIGNATURE, MIDI_INSTRUMENTS
from miditok.utils import (
    is_track_empty, merge_same_program_tracks,
    remove_duplicated_notes,
    get_score_ticks_per_beat
)
from symusic import Score, Track, Note, TimeSignature, Tempo
from symusic.core import NoteTickList, TempoTickList, TimeSignatureTickList

from symupe.utils import find_closest
from .classes import TokSequence, TokSequenceContext
from .constants import NOTE_ON_MIDI_EVENT


class MusicTokenizer(_MusicTokenizer, ABC):
    r"""
    Base music tokenizer class, acting as a common framework.
    See :class:`miditok.MusicTokenizer` for a detailed documentation.
    """

    def preprocess_score(
            self,
            score: Score,
            quantize_times: bool = True,
            quantize_velocities: bool = True,
            quantize_time_signatures: bool = True,
            quantize_tempos: bool = True
    ) -> Score:
        r"""
        Pre-process a ``symusic.Score`` object to resample its time and events values.

        This method is called before parsing a Score's contents for tokenization.
        Its notes attributes (times, pitches, velocities) will be downsampled and
        sorted, duplicated notes removed, as well as tempos. Empty tracks (with no
        note) will be removed from the ``symusic.Score`` object. Notes with pitches
        outside ``self.config.pitch_range`` will be deleted. Tracks with programs not
        supported by the tokenizer will be deleted.

        This method is **not inplace** and does not alter the provided ``score`` object.

        :param score: ``symusic.Score`` object to preprocess.
        :param quantize_times: resample and quantize note times.
        :param quantize_velocities: quantize velocity of each note.
        :param quantize_time_signatures: resample and quantize time signature times.
        :param quantize_tempos: quantize tempo values of each tempo change.
        :return: the preprocessed ``score``.
        """
        # Filter time signatures.
        # We need to do this first to determine the Score's new time division.
        # A copy of the time signatures is made here to make inplace operations without
        # modifying the provided Score object. This copy will be set to the copy of the
        # score after resampling it.
        time_signatures_copy = score.time_signatures.copy()
        if self.config.use_time_signatures:
            self._filter_unsupported_time_signatures(time_signatures_copy)
            # We mock the first with 0, even if there are already time signatures. This
            # is required as if the Score only had */2 time signatures, we must make
            # sure the resampling tpq is calculated according to a maximum denom of 4
            # if the beginning of the Score is mocked at 4/4.
            if len(time_signatures_copy) == 0 or time_signatures_copy[0].time != 0:
                time_signatures_copy.insert(0, TimeSignature(0, *TIME_SIGNATURE))
            # The new time division is chosen depending on its highest time signature
            # denominator, and is equivalent to the highest possible tick/beat ratio.
            max_ts_denom = max(ts.denominator for ts in time_signatures_copy)
            new_tpq = int(self.config.max_num_pos_per_beat * max_ts_denom / 4)
        else:
            time_signatures_copy = TimeSignatureTickList(
                [TimeSignature(0, *TIME_SIGNATURE)]
            )
            new_tpq = self.config.max_num_pos_per_beat

        if quantize_times:
            # Resample time if needed (not inplace) and attribute preprocessed time sig.
            score = self._resample_score(score, new_tpq, time_signatures_copy)

        # Merge instruments of the same program / inst before preprocessing them.
        # This allows to avoid potential duplicated notes in some multitrack settings
        # This can however mess up chord detections.
        if self.config.use_programs and self.config.one_token_stream_for_programs:
            merge_same_program_tracks(score.tracks)

        # Process time signature changes
        # We need to do it before computing the ticks_per_beat sections
        if quantize_time_signatures and self.config.use_time_signatures and len(score.time_signatures) > 0:
            self._preprocess_time_signatures(
                score.time_signatures, score.ticks_per_quarter
            )

        # Compute resampling ratios to update times of events when several time sig,
        # and ticks per beat ratios.
        # Resampling factors are used to resample times of events when the Score has
        # several different time signature denominators.
        # ticks_per_beat ratios are used to adjust durations values according to the
        # tokenizer's vocabulary, i.e. *Duration* tokens.
        if not self._note_on_off or (
                self.config.use_sustain_pedals and self.config.sustain_pedal_duration
        ):
            if self.config.use_time_signatures and len(score.time_signatures) > 0:
                ticks_per_beat = get_score_ticks_per_beat(score)
            else:
                ticks_per_beat = np.array([[score.end(), score.ticks_per_quarter]])
        else:
            ticks_per_beat = None

        if (
                self.config.use_time_signatures
                and len({ts.denominator for ts in score.time_signatures}) > 1
        ):
            tpq_resampling_factors = self._get_score_resampling_factor(score)
        else:
            tpq_resampling_factors = None

        # Preprocess track events
        for t in range(len(score.tracks) - 1, -1, -1):
            # Delete track only there is nothing inside being used
            program = -1 if score.tracks[t].is_drum else score.tracks[t].program
            if is_track_empty(
                    score.tracks[t],
                    check_pedals=self.config.use_sustain_pedals,
                    check_pitch_bend=self.config.use_pitch_bends,
            ) or (self.config.use_programs and program not in self.config.programs):
                del score.tracks[t]
                continue

            # Preprocesses notes
            if len(score.tracks[t].notes) > 0:
                self._preprocess_notes(
                    score.tracks[t],
                    tpq_resampling_factors,
                    ticks_per_beat,
                    quantize_times=quantize_times,
                    quantize_velocities=quantize_velocities
                )

            if quantize_times:
                # Resample pitch bend values
                if self.config.use_pitch_bends and len(score.tracks[t].pitch_bends) > 0:
                    score.tracks[t].pitch_bends = self._preprocess_pitch_bends(
                        score.tracks[t].pitch_bends, tpq_resampling_factors
                    )

                # Resample pedals durations
                if self.config.use_sustain_pedals and len(score.tracks[t].pedals) > 0:
                    score.tracks[t].pedals = self._preprocess_pedals(
                        score.tracks[t].pedals, tpq_resampling_factors, ticks_per_beat
                    )

            # Delete track only there is nothing inside being used
            if is_track_empty(
                    score.tracks[t],
                    check_pedals=self.config.use_sustain_pedals,
                    check_pitch_bend=self.config.use_pitch_bends,
            ):
                del score.tracks[t]
                continue

        # Process tempo changes
        if self.config.use_tempos:
            score.tempos = self._preprocess_tempos(
                score.tempos,
                tpq_resampling_factors,
                quantize_tempos=quantize_tempos
            )

        # We do not change key signature changes, markers and lyrics here as they are
        # not used by MidiTok (yet)

        return score

    def _preprocess_notes(
        self,
        track: Track,
        resampling_factors: np.ndarray = None,
        ticks_per_beat: np.ndarray = None,
        min_duration: int = 1,
        quantize_times: bool = True,
        quantize_velocities: bool = True
    ) -> None:
        r"""
        Resample inplace the note velocities, remove notes outside of pitch range.

        Note durations will be clipped to the maximum duration that can be handled by
        the tokenizer. This is done to prevent having incorrect offset values when
        computing rests. Notes with pitches outside of self.pitch_range will be
        deleted.

        :param track: track containing the notes to resample.
        :param resampling_factors: sections of resampling factors, when we need to
            adjust the times of events to a specific ticks/beat value. This is required
            when the file has time signatures with different denominators. The factors
            are given as a numpy array of shape ``(N,2)``, for ``N`` changes of ticks
            per beat, and the second dimension representing the end tick of each
            section and the number of ticks per beat respectively. (default: ``None``)
        :param ticks_per_beat: array indicating the number of ticks per beat per time
            signature denominator section. The numbers of ticks per beat depend on the
            time signatures of the file being parsed. The array has a shape ``(N,2)``,
            for ``N`` changes of ticks per beat, and the second dimension representing
            the end tick of each section and the number of ticks per beat respectively.
            This argument is not required if
            ``tokenizer.config.sustain_pedal_duration`` is disabled.
            (default: ``None``)
        :param min_duration: minimum duration (in tick) to set to notes that have
            durations of 0 ticks after resampling. (default: ``1``)
        :param quantize_times: resample and quantize note times
        :param quantize_velocities: quantize velocity of each note
        """
        note_soa = track.notes.numpy()

        # Delete notes outside of pitch range
        pitch_range = (
            self.config.drums_pitch_range
            if track.is_drum and self.config.use_pitchdrum_tokens
            else self.config.pitch_range
        )
        idx_out_of_pitch_range = np.where(
            np.logical_or(
                note_soa["pitch"] < pitch_range[0], note_soa["pitch"] > pitch_range[1]
            )
        )[0]
        if len(idx_out_of_pitch_range) > 0:
            mask = np.ones(len(note_soa["time"]), dtype=bool)
            mask[idx_out_of_pitch_range] = False
            for key in note_soa:
                note_soa[key] = note_soa[key][mask]
        if len(note_soa["time"]) == 0:
            track.notes = NoteTickList()
            return

        # Compute new velocities
        if self.config.use_velocities and quantize_velocities:
            note_soa["velocity"] = find_closest(
                self.velocities, np.array(note_soa["velocity"])
            )

        # Adjust times if needed
        if quantize_times:
            if resampling_factors is not None:
                # First get the idx of the notes covered per section
                resampling_factors = self._MusicTokenizer__convert_resampling_ratios_ticks_to_idx(
                    resampling_factors, note_soa["time"]
                )
                note_soa["time"] = self._adjust_time_to_tpb(
                    note_soa["time"], resampling_factors
                )

            # Resample duration values if NoteOff, otherwise adjust to the vocab
            program = -1 if track.is_drum else track.program
            if program in self.config.use_note_duration_programs:
                if not self._note_on_off and ticks_per_beat is not None:
                    self._adjust_durations(note_soa, ticks_per_beat)
                elif resampling_factors is not None:
                    note_soa["duration"] = self._adjust_time_to_tpb(
                        note_soa["duration"], resampling_factors, min_duration
                    )
                    self._adjust_offset_spanning_across_time_sig(
                        note_soa, resampling_factors
                    )

        # Symusic automatically sorts the notes by (time, duration, pitch) keys when
        # reading a music file. We hence don't need to sort the notes.
        # However, when using `NoteOn`/`NoteOff`, we can encounter note order
        # alterations with the velocity values as they are not sorted on velocities and
        # that the tokens are decoded following a FIFO logic.
        # To alleviate this, a user can sort them before calling the tokenizer.
        # We do not do it here as it is not considered a disturbing issue, and that it
        # would add a significant overhead preprocessing time. This is however done in
        # the tokenization tests of MidiTok for concerned tokenizers in order to keep
        # 100% of the data integrity, so that the tests pass.

        notes_new = Note.from_numpy(**note_soa)

        if self.config.remove_duplicated_notes:
            # we need to resort here, as symusic does it by (time, duration, pitch).
            notes_new.sort(key=lambda n: (n.time, n.pitch, n.duration, n.velocity))
            remove_duplicated_notes(notes_new)

        track.notes = notes_new

    def _preprocess_tempos(
            self,
            tempos: TempoTickList,
            resampling_factors: np.ndarray = None,
            quantize_tempos: bool = True
    ) -> TempoTickList:
        r"""
        Resample the tempo values of tempo change events.

        For tempo changes occurring at the same tick/time, we only keep the last one.
        Consecutive identical tempo changes will be removed if
        ``self.config.delete_equal_successive_tempo_changes`` is True.

        :param tempos: tempo changes to resample.
        :param resampling_factors: sections of resampling factors, when we need to
            adjust the times of events to a specific ticks/beat value. This is required
            when the file has time signatures with different denominators. The factors
            are given as a numpy array of shape ``(N,2)``, for ``N`` changes of ticks
            per beat, and the second dimension representing the end tick of each
            section and the number of ticks per beat respectively. (default: ``None``)
        :param quantize_tempos: quantize tempo values of each tempo change
        """
        # If we delete the successive equal tempo changes, we need to sort them by time
        # Fortunately, sorting is already performed by symusic when loading the file.

        # Use the default tempo if there is None (shouldn't happen)
        if len(tempos) == 0:
            tempos.insert(0, Tempo(0, self.default_tempo))
            return tempos

        tempos_soa = tempos.numpy()

        # Find the closest tempos
        if quantize_tempos:
            tempos_soa["mspq"] = find_closest(self._tempos_mspq, tempos_soa["mspq"], return_values=True)

        # Adjust times if needed
        if resampling_factors is not None:
            tempos_soa["time"] = self._adjust_time_to_tpb(
                tempos_soa["time"], resampling_factors
            )

        # Find groups of tempos at the same onset ticks, equal consecutive ones
        # Keep only last tempo change for groups with same tick
        idx_groups = np.split(
            np.arange(len(tempos_soa["time"])),
            np.where(np.diff(tempos_soa["time"]) != 0)[0] + 1,
        )
        for idx_group in reversed(idx_groups):
            if len(idx_group) > 1:
                for key in tempos_soa:
                    # We don't use a mask here as the number of idx to delete is
                    # likely to be small.
                    for idx_to_del in reversed(idx_group[:-1]):
                        tempos_soa[key] = np.delete(tempos_soa[key], idx_to_del)
        # Deduplicate successive tempo changes with same tempo value
        if self.config.delete_equal_successive_tempo_changes:
            idx_groups = np.split(
                np.arange(len(tempos_soa["time"])),
                np.where(np.diff(tempos_soa["mspq"]) != 0)[0] + 1,
            )
            for idx_group in reversed(idx_groups):
                if len(idx_group) > 1:
                    for key in tempos_soa:
                        for idx_to_del in reversed(idx_group[1:]):
                            tempos_soa[key] = np.delete(tempos_soa[key], idx_to_del)

        tempos = Tempo.from_numpy(**tempos_soa)

        # Make sure there is at least one tempo at tick 0
        if len(tempos) > 0:
            if (
                    self.config.delete_equal_successive_tempo_changes
                    and tempos[0].tempo == self.default_tempo
            ):
                tempos[0].time = 0
            elif tempos[0].time != 0:
                tempos.insert(0, Tempo(0, self.default_tempo))
        else:
            tempos.insert(0, Tempo(0, self.default_tempo))

        return tempos

    def _build_score(
            self,
            times: np.ndarray,
            durations: np.ndarray,
            pitches: np.ndarray,
            velocities: np.array,
            programs: np.ndarray | None,
            time_signatures: list[TimeSignature] | None,
            tempos: list[Tempo] | None,
            time_division: int | None = None,
            ttype: str = "tick"
    ) -> Score:
        r"""
        Build symusic.Score MIDI from the provided data.
        """
        score = Score(time_division or self.time_division, ttype=ttype)

        score.time_signatures = time_signatures or [TimeSignature(0, *TIME_SIGNATURE, ttype=ttype)]
        score.tempos = tempos or [Tempo(0, self.default_tempo, ttype=ttype)]

        tracks: dict[int, Track] = {}
        programs = np.zeros_like(pitches) if programs is None else programs

        for program in np.unique(programs):
            program = int(program)
            tracks[program] = Track(
                program=0 if program == -1 else program,
                is_drum=program == -1,
                name="Drums" if program == -1 else MIDI_INSTRUMENTS[program]["name"],
                ttype=ttype
            )

            program_ids = np.where(programs == program)[0]
            tracks[program].notes = Note.from_numpy(
                time=times[program_ids],
                duration=durations[program_ids],
                pitch=pitches[program_ids],
                velocity=velocities[program_ids],
                ttype=ttype
            )

        score.tracks = list(tracks.values())

        return score

    def _ids_to_tokens(
            self, ids: list[int | list[int]], as_str: bool = True
    ) -> list[str | Event | list[str | Event]]:
        r"""
        Convert a sequence of ids (int) to their tokens format (str or Event).

        **This method will not work with ids encoded with the tokenizer's model. You
        will need to decode them first (
        :py:meth:`miditok.MusicTokenizer.decode_token_ids`)**.

        :param ids: sequence of ids (int) to convert.
        :param as_str: return the tokens as string objects, otherwise Event objects
            (default: True)
        :return: the sequence of corresponding tokens (str or Event).
        """
        tokens = []
        if len(ids) == 0:
            return tokens

        if isinstance(ids[0], list) or isinstance(ids[0], np.ndarray):  # multiple vocabularies
            ids = np.array(ids) if isinstance(ids, list) else ids
            tokens = np.stack([
                np.array(list(self.vocab[i].keys()))[ids[:, i]]
                for i in range(ids.shape[1])
            ], axis=1)
            return tokens.tolist()

        for id_ in ids:
            event_str = self[id_]
            tokens.append(event_str if as_str else Event(*event_str.split("_")))
        return tokens

    @property
    def special_tokens_dict(self) -> dict[str, int]:
        r"""
        Return the map of the special tokens to their ids in the vocabulary.

        :return: dictionary of special tokens and their ids
        """
        return {token: self[token] for token in self.special_tokens}

    def tokens_to_midi_messages(
            self,
            tokens: TokSequence,
            context: TokSequenceContext | None = None,
            note_attributes: bool = True,
            note_on_events: bool = True,
            note_off_events: bool = True,
            sort: bool = True
    ):
        assert note_on_events or note_off_events
        tokens = tokens.numpy()

        note_on_times, note_off_times, pitches, velocities, new_context = self._tokens_to_midi_messages(
            tokens=tokens, context=context, note_attributes=note_attributes
        )

        messages = []
        if note_attributes:
            assert pitches is not None and velocities is not None
            midi_msgs = np.full_like(pitches, NOTE_ON_MIDI_EVENT)
            if note_on_events:
                messages.append(np.stack([note_on_times, midi_msgs, pitches, velocities], axis=-1))
            if note_off_events:
                messages.append(np.stack([note_off_times, midi_msgs, pitches, np.zeros(velocities.shape[0])], axis=-1))
        else:
            if note_on_events:
                messages.append(note_on_times)
            if note_off_events:
                messages.append(note_off_times)
        messages = np.concatenate(messages, axis=0)

        if sort:
            messages = self.sort_messages(messages)

        return messages, new_context

    @abstractmethod
    def _tokens_to_midi_messages(
            self,
            tokens: TokSequence,
            context: TokSequenceContext | None = None,
            note_attributes: bool = True
    ):
        raise NotImplementedError

    @staticmethod
    def sort_messages(messages: np.ndarray) -> np.ndarray:
        if len(messages.shape) == 2:
            return messages[np.lexsort((-messages[:, 3], messages[:, 2], messages[:, 0]))]
        else:
            return messages[np.lexsort((messages,))]
