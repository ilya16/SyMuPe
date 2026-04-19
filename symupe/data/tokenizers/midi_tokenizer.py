"""Base tokenizer class, extending miditok.MusicTokenizer."""

from __future__ import annotations

import json
import sys
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from miditok import MusicTokenizer as _MusicTokenizer, Event
from miditok.constants import (
    TIME_SIGNATURE,
    MIDI_INSTRUMENTS,
    DEFAULT_TOKENIZER_FILE_NAME,
    CURRENT_MIDITOK_VERSION,
)
from miditok.utils import (
    is_track_empty,
    merge_same_program_tracks,
    remove_duplicated_notes,
    get_score_ticks_per_beat,
)
from symusic import Score, Track, Note, TimeSignature, Tempo
from symusic.core import NoteTickList, TempoTickList, TimeSignatureTickList

from symupe.utils import find_closest
from .classes import TokSequence, TokSequenceContext
from .constants import NOTE_ON_MIDI_EVENT


class MusicTokenizer(_MusicTokenizer, ABC):
    """Base music tokenizer class extending :class:`miditok.MusicTokenizer`.

    Acts as a common framework for SyMuPe-based tokenizers, providing
    vectorized preprocessing, score building, and MIDI message generation.

    See :class:`miditok.MusicTokenizer` for a detailed documentation.
    """

    def preprocess_score(
        self,
        score: Score,
        quantize_times: bool = True,
        quantize_velocities: bool = True,
        quantize_time_signatures: bool = True,
        quantize_tempos: bool = True,
    ) -> Score:
        """Preprocesses a score :class:`symusic.Score` object for the SyMuPe encoding.

        Filters unsupported time signatures, resamples time based on maximum
        denominator, merges tracks of same program if configured, and quantizes
        notes, tempos, and control changes.

        This method is called before parsing a Score's contents for tokenization.
        Its notes attributes (times, pitches, velocities) will be downsampled and
        sorted, duplicated notes removed, as well as tempos. Empty tracks (with no
        note) will be removed from the :class:`symusic.Score` object. Notes with pitches
        outside `self.config.pitch_range` will be deleted. Tracks with programs not
        supported by the tokenizer will be deleted.

        Args:
            score: :class:`symusic.Score` object to process.
            quantize_times: Resample and quantize note times.
            quantize_velocities: Quantize velocity of each note.
            quantize_time_signatures: Resample and quantize time signature times.
            quantize_tempos: Quantize tempo values.

        Returns:
            Preprocessed :class:`symusic.Score` object.
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
            time_signatures_copy = TimeSignatureTickList([TimeSignature(0, *TIME_SIGNATURE)])
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
        if (
            quantize_time_signatures
            and self.config.use_time_signatures
            and len(score.time_signatures) > 0
        ):
            self._preprocess_time_signatures(score.time_signatures, score.ticks_per_quarter)

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
                    quantize_velocities=quantize_velocities,
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
                score.tempos, tpq_resampling_factors, quantize_tempos=quantize_tempos
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
        quantize_velocities: bool = True,
    ) -> None:
        """Resamples note velocities and durations inplace; removes notes out of range.

        Clips note durations to maximum handled by tokenizer and adjusts
        timing based on resampling factors for varying time signatures.

        Args:
            track: :class:`symusic.Track` containing notes to resample.
            resampling_factors: Sections of resampling factors, when we need to
                adjust the times of events to a specific ticks/beat value. This is required
                when the file has time signatures with different denominators. The factors
                are given as a numpy array of shape ``(N,2)``, for ``N`` changes of ticks
                per beat, and the second dimension representing the end tick of each
                section and the number of ticks per beat respectively.
            ticks_per_beat: Array indicating the number of ticks per beat per time
                signature denominator section. The numbers of ticks per beat depend on the
                time signatures of the file being parsed. The array has a shape ``(N,2)``,
                for ``N`` changes of ticks per beat, and the second dimension representing
                the end tick of each section and the number of ticks per beat respectively.
            min_duration: Minimum duration (in tick) to set to notes that have
                durations of 0 ticks after resampling.
            quantize_times: Resample and quantize note times.
            quantize_velocities: Quantize velocity of each note.
        """
        note_soa = track.notes.numpy()

        # Delete notes outside of pitch range
        pitch_range = (
            self.config.drums_pitch_range
            if track.is_drum and self.config.use_pitchdrum_tokens
            else self.config.pitch_range
        )
        idx_out_of_pitch_range = np.where(
            np.logical_or(note_soa["pitch"] < pitch_range[0], note_soa["pitch"] > pitch_range[1])
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
            note_soa["velocity"] = find_closest(self.velocities, np.array(note_soa["velocity"]))

        # Adjust times if needed
        if quantize_times:
            if resampling_factors is not None:
                # First get the idx of the notes covered per section
                resampling_factors = self._MusicTokenizer__convert_resampling_ratios_ticks_to_idx(
                    resampling_factors, note_soa["time"]
                )
                note_soa["time"] = self._adjust_time_to_tpb(note_soa["time"], resampling_factors)

            # Resample duration values if NoteOff, otherwise adjust to the vocab
            program = -1 if track.is_drum else track.program
            if program in self.config.use_note_duration_programs:
                if not self._note_on_off and ticks_per_beat is not None:
                    self._adjust_durations(note_soa, ticks_per_beat)
                elif resampling_factors is not None:
                    note_soa["duration"] = self._adjust_time_to_tpb(
                        note_soa["duration"], resampling_factors, min_duration
                    )
                    self._adjust_offset_spanning_across_time_sig(note_soa, resampling_factors)

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
        quantize_tempos: bool = True,
    ) -> TempoTickList:
        """Resamples tempo change events and removes redundant successive changes.

        Ensures at least one tempo event exists at tick 0 and applies
        quantization to tempo values based on tokenizer vocabulary.

        Args:
            tempos: Tempo changes to resample.
            resampling_factors: Sections of resampling factors, when we need to
                adjust the times of events to a specific ticks/beat value. This is required
                when the file has time signatures with different denominators. The factors
                are given as a numpy array of shape ``(N,2)``, for ``N`` changes of ticks
                per beat, and the second dimension representing the end tick of each
                section and the number of ticks per beat respectively.
            quantize_tempos: Whether to quantize tempo values.

        Returns:
            Processed :class:`symusic.TempoTickList`.
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
            tempos_soa["mspq"] = find_closest(
                self._tempos_mspq, tempos_soa["mspq"], return_values=True
            )

        # Adjust times if needed
        if resampling_factors is not None:
            tempos_soa["time"] = self._adjust_time_to_tpb(tempos_soa["time"], resampling_factors)

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
        ttype: str = "tick",
    ) -> Score:
        """Constructs :class:`symusic.Score` from raw attribute arrays.

        Groups notes into tracks based on program IDs and sets global
        metrical metadata.

        Args:
            times: Array of onset times.
            durations: Array of durations.
            pitches: Array of MIDI pitches.
            velocities: Array of MIDI velocities.
            programs: Array of track programs.
            time_signatures: List of :class:`symusic.TimeSignature` objects.
            tempos: List of :class:`symusic.Tempo` objects.
            time_division: MIDI resolution.
            ttype: Time type, either 'tick' or 'second'.

        Returns:
            Reconstructed :class:`symusic.Score` object.
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
                ttype=ttype,
            )

            program_ids = np.where(programs == program)[0]
            tracks[program].notes = Note.from_numpy(
                time=times[program_ids],
                duration=durations[program_ids],
                pitch=pitches[program_ids],
                velocity=velocities[program_ids],
                ttype=ttype,
            )

        score.tracks = list(tracks.values())

        return score

    def _ids_to_tokens(
        self, ids: list[int | list[int]], as_str: bool = True
    ) -> list[str | Event | list[str | Event]]:
        """Converts sequence of IDs to their tokens format (str or Event).

        Args:
            ids: Sequence of IDs to convert.
            as_str: If ``True``, returns tokens as strings, otherwise as Events.

        Returns:
            Sequence of corresponding tokens.
        """
        tokens = []
        if len(ids) == 0:
            return tokens

        if isinstance(ids[0], list) or isinstance(ids[0], np.ndarray):  # multiple vocabularies
            ids = np.array(ids) if isinstance(ids, list) else ids
            tokens = np.stack(
                [np.array(list(self.vocab[i].keys()))[ids[:, i]] for i in range(ids.shape[1])],
                axis=1,
            )
            return tokens.tolist()

        for id_ in ids:
            event_str = self[id_]
            tokens.append(event_str if as_str else Event(*event_str.split("_")))
        return tokens

    @property
    def special_tokens_dict(self) -> dict[str, int]:
        """Mapping of special token names to their respective vocabulary IDs."""
        return {token: self[token] for token in self.special_tokens}

    def tokens_to_midi_messages(
        self,
        tokens: TokSequence,
        context: TokSequenceContext | None = None,
        note_attributes: bool = True,
        note_on_events: bool = True,
        note_off_events: bool = True,
        sort: bool = True,
    ):
        """Converts :class:`TokSequence` into raw MIDI message arrays.

        Facilitates low-level MIDI event processing by returning flat arrays
        of times, event types, pitches, and velocities.

        Args:
            tokens: :class:`TokSequence` to convert.
            context: Optional :class:`TokSequenceContext` for incremental decoding.
            note_attributes: If ``True``, extracts pitch and velocity values.
            note_on_events: If ``True``, extracts note-on timing events.
            note_off_events: If ``True``, extracts note-off timing events.
            sort: If ``True``, ensures the resulting events are chronologically ordered.

        Returns:
            Tuple containing array of MIDI messages and updated context.
        """
        assert note_on_events or note_off_events
        tokens = tokens.numpy()

        note_on_times, note_off_times, pitches, velocities, new_context = (
            self._tokens_to_midi_messages(
                tokens=tokens, context=context, note_attributes=note_attributes
            )
        )

        messages = []
        if note_attributes:
            assert pitches is not None and velocities is not None
            midi_msgs = np.full_like(pitches, NOTE_ON_MIDI_EVENT)
            if note_on_events:
                messages.append(np.stack([note_on_times, midi_msgs, pitches, velocities], axis=-1))
            if note_off_events:
                messages.append(
                    np.stack(
                        [note_off_times, midi_msgs, pitches, np.zeros(velocities.shape[0])], axis=-1
                    )
                )
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
        note_attributes: bool = True,
    ):
        """Internal abstract method for decoding tokens into event components."""
        raise NotImplementedError

    @staticmethod
    def sort_messages(messages: np.ndarray) -> np.ndarray:
        """Sorts MIDI message arrays by time, then pitch, then velocity.

        Args:
            messages: Array of MIDI messages.

        Returns:
            Sorted NumPy array.
        """
        if len(messages.shape) == 2:
            return messages[np.lexsort((-messages[:, 3], messages[:, 2], messages[:, 0]))]
        else:
            return messages[np.lexsort((messages,))]

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str | None,
        cache_dir: str | Path | None,
        force_download: bool,
        proxies: dict | None = None,
        resume_download: bool = False,
        local_files_only: bool,
        token: str | bool | None,
        **kwargs,
    ) -> MusicTokenizer:
        """Loads tokenizer from a pretrained configuration on HuggingFace Hub.

        Args:
            model_id: Repository ID or local path.
            revision: Specific model version.
            cache_dir: Path to cache directory.
            force_download: Whether to force re-download.
            proxies: Dictionary of proxy servers.
            resume_download: Whether to resume interrupted download.
            local_files_only: Whether to skip network calls.
            token: Authentication token.
            **kwargs: Additional parameters for loading.

        Returns:
            Instance of the appropriate :class:`MusicTokenizer` subclass.
        """
        # Called by `ModelHubMixin.from_pretrained`
        pretrained_path = Path(model_id)
        if pretrained_path.is_file():
            params_path = pretrained_path
        else:
            filename = kwargs.get("filename", DEFAULT_TOKENIZER_FILE_NAME)
            if (pretrained_path / filename).is_file():
                params_path = pretrained_path / filename
            else:
                hf_hub_kwargs = dict(
                    repo_id=model_id,
                    filename=filename,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                    token=token,
                    library_name="MidiTok",
                    library_version=CURRENT_MIDITOK_VERSION,
                )

                import inspect

                hf_download_params = inspect.signature(hf_hub_download).parameters
                if "proxies" in hf_download_params:
                    hf_hub_kwargs["proxies"] = proxies
                if "resume_download" in hf_download_params:
                    hf_hub_kwargs["resume_download"] = resume_download

                params_path = hf_hub_download(**hf_hub_kwargs)

        # Checking config file tokenization
        with Path(params_path).open() as file:
            tokenization = json.load(file)["tokenization"]
        cls_name = cls.__name__
        if cls_name not in ["MusicTokenizer", tokenization]:
            warnings.warn(
                ".from_pretrained called with an invalid class name. The current class"
                f"is {cls_name} whereas the config file comes from a {tokenization} "
                f"tokenizer. Returning an instance of {tokenization}.",
                stacklevel=2,
            )

        if cls_name == tokenization:
            return cls(params=params_path)

        miditok_module = sys.modules[".".join(__name__.split(".")[:-1])]
        return getattr(miditok_module, tokenization)(params=params_path)
