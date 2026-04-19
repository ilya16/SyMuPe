"""
SyMuPeLocal encoding.

Adapts the ideas of the SPMuple encoding from ScorePerformer with local window tempos to SyMuPe encoding.
"""

from __future__ import annotations

import numpy as np
from miditok.constants import TEMPO
from symusic import Score

from symupe.utils import find_closest
from .classes import TokSequence, SequenceType, EncodingType, TokSequenceContext
from .constants import TICKS_PER_QUARTER
from .symupe import SyMuPe
from ..midi.timing import MIDITimeMapper
from ..midi.utils import sort_notes


class SyMuPeLocal(SyMuPe):
    r"""
    SyMuPeLocal: a Symbolic Music Performance encoding with local window tempos.

    A mix of SyMuPe encoding [1] and local window tempos from the SPMuple encoding [2].

    Each compound token is a tuple of the form (index: Token type):
    * 0: Bar
    * 1: Position
    * 2: Pitch
    * 3: Duration
    * 4: Velocity
    * (+ Optional) Tempo
    * (+ Optional) TimeSignature / (BeatDuration, BeatsInBar / MaxBarPosition)
    * (+ Optional) Program
    * (+ Optional, score) PositionShift
    * (+ Optional, score) NotesInOnset
    * (+ Optional, score) PositionInOnset
    * (+ performance) RelativeOnsetDeviation
    * (+ performance) RelativePerformedDuration
    * (+ Optional, performance) TimeShift
    * (+ Optional, performance) TimeDuration
    * (+ Optional, performance) TimePosition
    * (+ Optional, performance) TimeDurationSustain
    * (+ Optional, performance) Sustained

    References:
        [1]: Borovik, I., Gavrilev, D., and Viro, V. (2025). "SyMuPe: Affective and
        Controllable Symbolic Music Performance." In Proceedings of the 33rd ACM International
        Conference on Multimedia (ACM MM).
        [2]: Borovik, I., & Viro, V. (2023). "ScorePerformer: Expressive Piano Performance
        Rendering with Fine-Grained Control." In Proceedings of  the 24th International Society
        for Music Information Retrieval Conference (ISMIR).
    """

    def _tweak_config_before_creating_voc(self):
        additional_params = self.config.additional_params

        # default parameters
        additional_params["use_onset_tokens"] = True

        additional_params["rel_onset_dev"] = True
        additional_params["num_onset_devs"] = additional_params.get("num_onset_devs", 161)

        additional_params["rel_perf_duration"] = True
        additional_params["num_perf_durations"] = additional_params.get("num_perf_durations", 81)

        super()._tweak_config_before_creating_voc()

        # tempo encoding/decoding parameters
        additional_params["onset_tempos"] = additional_params.get("onset_tempos", False)
        additional_params["tempo_window"] = additional_params.get("tempo_window", 8.0)
        additional_params["tempo_min_onset_dist"] = additional_params.get(
            "tempo_min_onset_dist", 0.5
        )
        additional_params["tempo_min_onsets"] = additional_params.get("tempo_min_onsets", 8)

        additional_params["use_quantized_tempos"] = additional_params.get(
            "use_quantized_tempos", False
        )
        additional_params["decode_recompute_tempos"] = additional_params.get(
            "decode_recompute_tempos", False
        )

        # outlier detection and processing
        additional_params["limit_onset_devs"] = additional_params.get("limit_onset_devs", True)

    def _encode_performance(
        self,
        midi: Score,
        score_tokens: TokSequence,
        note_alignment: np.ndarray | None = None,
    ) -> TokSequence:
        r"""
        Tokenize a performance MIDI file into :class:`miditok.TokSequence`
        with score tokens plus RelativeOnsetDeviation and RelativePerformedDuration tokens.
        Converts a MIDI file to a performance tokens representation, a sequence of "time steps"
        of score tokens stacked with performance specific features (e.g., OnsetDeviation).

        :param midi: the MIDI object to convert.
        :param score_tokens: corresponding score tokens :class:`miditok.TokSequence`.
        :param note_alignment: optional alignment between performance and score tokens.
        :return: the performance token representation, i.e. tracks converted into sequences of tokens
        """
        if score_tokens is None:
            return self._performance_midi_to_time_only_tokens(midi)

        additional_params = self.config.additional_params

        # Prepare constants used for calculations
        time_division = midi.ticks_per_quarter
        ticks_per_sample = time_division / self.config.max_num_pos_per_beat
        tempo_scale = 60 / time_division

        self._current_midi_metadata = {
            "time_division": midi.ticks_per_quarter,
            "tempos": midi.tempos,
        }

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

        # Construct an array of values
        values = np.zeros((len(perf_positions), len(self.score_sizes)))

        # Fill in pitch and velocity values
        values[:, self.vocab_types_idx["Pitch"]] = note_soa["pitch"]
        values[:, self.vocab_types_idx["Velocity"]] = note_soa["velocity"]

        # Apply alignment
        if note_alignment is not None:
            values, perf_positions, perf_durations = map(
                lambda x: x[note_alignment], (values, perf_positions, perf_durations)
            )

        # Copy score values to performance values
        for token_type in self.score_only_tokens:
            idx = self.vocab_types_idx[token_type]
            values[:, idx] = score_tokens.values[:, idx]

        # Compute NoteON, Time Signature and Bar ticks
        ticks_data = self.compute_ticks(score_tokens, time_division=time_division)

        # Get score position and duration ticks
        score_ticks = ticks_data["note_on"]
        duration_ticks = ticks_data["duration"]

        # Compute performance note positions
        time_mapper = MIDITimeMapper(midi)
        perf_times = time_mapper.t2s(perf_positions * ticks_per_sample)
        perf_offset_times = time_mapper.t2s((perf_positions + perf_durations) * ticks_per_sample)

        # Record performed notes
        is_performed = values[:, self.vocab_types_idx["Velocity"]] > 0.0

        # Mask out score ticks for unperformed notes
        performed_ticks = score_ticks.copy()
        performed_ticks[~is_performed] = -1

        # Build onset pairs: a list of tuples (onset_score_tick, onset_perf_time)
        onset_pairs = self.compute_onset_pairs(score_ticks=performed_ticks, perf_times=perf_times)

        # Compute initial tempo using a subset of onset pairs
        start_pairs = onset_pairs[onset_pairs[:, 1] <= 2 * additional_params["tempo_window"]]
        if len(start_pairs) < additional_params["tempo_min_onsets"]:
            start_pairs = onset_pairs[: additional_params["tempo_min_onsets"]]

        # Compute weighted initial tempo
        initial_tempo = self.compute_local_tempo(
            distances=start_pairs[start_pairs[:, 1] > 0.0] - start_pairs[0], tempo_scale=tempo_scale
        )
        self._current_midi_metadata["initial_tempo"] = initial_tempo

        # Process zero first onset
        if onset_pairs[1, 0] == 0:
            onset_pairs[0] = [-time_division, -time_division / initial_tempo * tempo_scale]

        if additional_params["onset_tempos"]:
            initial_tempo = self.compute_onset_tempo(
                onset_pairs[1], prev_onset_pair=onset_pairs[0], tempo_scale=tempo_scale
            )

        # Iteratively compute weighted local tempos, assign them to notes
        _offset, num_tokens = 0, len(values)
        tempos = [initial_tempo]
        note_tempos = np.ones(num_tokens)
        note_onsets, note_prev_onsets = np.zeros((num_tokens, 2)), np.zeros((num_tokens, 2))
        for i, onset_pair in enumerate(onset_pairs[1:]):
            onset_tick, onset_time = onset_pair
            prev_onset_tick, prev_onset_time = onset_pairs[i]

            # Compute onset deviations for current notes
            onset_mask = score_ticks == onset_tick
            onset_time_shift = (onset_tick - prev_onset_tick) / tempos[-1] * tempo_scale
            note_perf_times = perf_times[onset_mask][is_performed[onset_mask]]
            note_onset_devs = note_perf_times - (prev_onset_time + onset_time_shift)
            note_rel_onset_devs = note_onset_devs / onset_time_shift
            note_abs_rel_onset_devs = np.abs(note_rel_onset_devs)

            # Limit relative onset deviations to max relative onset deviation if required
            # Compute the required onset shift (such that all deviations fall under
            # the maximum relative onset deviation) and shift the deviating notes
            if (
                additional_params["limit_onset_devs"]
                and note_abs_rel_onset_devs.max() > self.rel_onset_deviations[-1]
            ):
                # Compute time shift for the deviating notes
                shift_ids = np.where(note_abs_rel_onset_devs > self.rel_onset_deviations[-1])
                note_shifts = np.zeros_like(note_onset_devs)
                note_shifts[shift_ids] = (
                    self.rel_onset_deviations[-1]
                    * onset_time_shift
                    * np.sign(note_rel_onset_devs[shift_ids])
                    - note_onset_devs[shift_ids]
                )

                perf_times[onset_mask] += note_shifts
                perf_offset_times[onset_mask] += note_shifts
                onset_pairs[i + 1, 1] = perf_times[onset_mask].mean()
                onset_time = onset_pairs[i + 1, 1]

            if additional_params["onset_tempos"]:
                tempo = self.compute_onset_tempo(
                    onset_pairs[i + 1], prev_onset_pair=onset_pairs[i], tempo_scale=tempo_scale
                )
            else:
                if onset_time < 2 * additional_params["tempo_min_onset_dist"]:
                    tempo = initial_tempo  # not enough history, use initial tempo
                else:
                    # Cut onsets in a local window
                    pairs_in_window = self.filter_onsets_in_window(onset_time, onset_pairs[: i + 1])

                    # Compute local tempo
                    tempo = self.compute_local_tempo(
                        distances=onset_pair - pairs_in_window, tempo_scale=tempo_scale
                    )

            tempos.append(tempo)

            note_tempos[onset_mask] = tempos[i]
            note_prev_onsets[onset_mask] = onset_pairs[i]
            note_onsets[onset_mask] = onset_pairs[i + 1]

        # Save MIDI data for external use
        self._current_midi_metadata.update(
            **{
                "onset_pairs": onset_pairs,
                "tempos": np.array(tempos),
                "note_tempos": note_tempos,
            }
        )

        # Assign neighbouring tempos for not performed notes
        is_empty = note_tempos == 0.0
        if np.any(is_empty):
            fill_ids = np.where(~is_empty, np.arange(is_empty.shape[0]), 0)
            np.maximum.accumulate(fill_ids, axis=0, out=fill_ids)
            note_tempos = note_tempos[fill_ids]

        # Add tempo values if they are present in the encoding
        if self.config.use_tempos:
            values[:, self.vocab_types_idx["Tempo"]] = note_tempos

        # Compute onset deviations and RelativeOnsetDeviation values
        note_time_shifts = (note_onsets[:, 0] - note_prev_onsets[:, 0]) / note_tempos * tempo_scale
        note_onset_devs = perf_times - (note_prev_onsets[:, 1] + note_time_shifts)
        note_onset_devs[~is_performed] = 0  # zero out onsets for not performed notes

        note_rel_onset_devs = np.zeros_like(note_onset_devs)
        note_rel_onset_devs[is_performed] = (
            note_onset_devs[is_performed] / note_time_shifts[is_performed]
        )

        # Compute performed durations RelativePerformedDuration values
        perf_time_durations = perf_offset_times - perf_times
        score_time_durations = duration_ticks / note_tempos * tempo_scale

        note_rel_perf_durations = perf_time_durations / score_time_durations
        note_rel_perf_durations[~is_performed] = 1  # "zero out" durations for not performed notes

        self._current_midi_metadata.update(
            **{
                "perf_times": perf_times,
                "note_time_shifts": note_time_shifts,
                "note_onset_devs": note_onset_devs,
                "score_time_durations": score_time_durations,
                "perf_time_durations": perf_time_durations,
            }
        )

        # Append RelOnsetDev and RelPerfDuration values
        values = np.concatenate(
            [
                values,
                note_rel_onset_devs[:, None],
                note_rel_perf_durations[:, None],
            ],
            axis=1,
        )

        # Append TimeShift/TimeDuration values
        if additional_params["use_time_tokens"]:
            sort_ids = np.argsort(perf_times)
            perf_time_shifts = np.diff(np.concatenate([[0.0], perf_times[sort_ids]]))[
                np.argsort(sort_ids)
            ]
            values = np.concatenate(
                [
                    values,
                    perf_time_shifts[:, None] * 1000.0,
                    perf_time_durations[:, None] * 1000.0,
                ],
                axis=1,
            )

            # Append TimePosition values
            if additional_params["use_time_positions"]:
                values = np.concatenate(
                    [
                        values,
                        (perf_times[:, None] // 1.0),
                        np.round((perf_times[:, None] % 1.0) * 1000.0, 3),
                    ],
                    axis=1,
                )

        # Convert values to tokens and build final TokSequence
        tokens = self.encode_tokens(values, clip=True)

        return TokSequence(
            ids=tokens,
            values=values,
            type=SequenceType.PERFORMANCE,
            encoding=EncodingType.PERFORMANCE,
            vocab=self.vocab_types_idx,
            meta={
                "time_division": midi.ticks_per_quarter,
                "bars": int(tokens[-1, self.vocab_types_idx["Bar"]] - self.zero_token + 1),
                "initial_tempo": initial_tempo,
            },
        )

    def decode_note_positions(
        self,
        tokens: TokSequence,
        context: TokSequenceContext | None = None,
        time_division: int = TICKS_PER_QUARTER,
    ) -> tuple[dict[str, any], TokSequenceContext]:
        additional_params = self.config.additional_params
        time_division = time_division or self.time_division

        context = context or TokSequenceContext()
        prev_note_on_times = (
            context.note_on_times if context.note_on_times is not None else np.zeros(1)
        )

        ticks_data, score_ticks = None, None
        note_on_times, note_off_times = None, None
        tempos, tempo_ticks, tempo_times = None, None, None
        onset_pairs = None

        has_time_tokens = additional_params["use_time_tokens"] and self.has_token_types(
            tokens, ["TimeShift", "TimeDuration"]
        )
        if has_time_tokens:
            # Note times
            time_shifts = self.get_values(tokens, "TimeShift")
            note_on_times = np.cumsum(time_shifts) + prev_note_on_times.max()

            perf_time_durations = self.get_values(tokens, "TimeDuration")
            note_off_times = note_on_times + perf_time_durations

        has_score = self.has_token_types(tokens, ["Bar", "Position"]) or (
            additional_params["use_position_shifts"]
            and self.has_token_types(tokens, ["PositionShift"])
        )
        if has_score and tokens.type not in (
            SequenceType.TIME_PERFORMANCE,
            SequenceType.TIME_PERFORMANCE_SUSTAIN,
        ):
            tempo_scale = 60 / time_division

            meta = tokens.meta or {}
            context.initial_tempo = context.initial_tempo or meta.get("initial_tempo", TEMPO)

            # Compute NoteON, Time Signature, Bar and Beat ticks
            ticks_data = self.compute_ticks(tokens, context=context, time_division=time_division)

            # Get score positions and durations
            score_ticks = ticks_data["note_on"]
            score_durations = ticks_data["duration"]

            # Record performed notes
            is_performed = self.get_values(tokens, "Velocity") != 0.0

            # Get unique performed score onsets
            score_onsets = np.unique(score_ticks[is_performed])

            # Get token tempos
            tempo_values = self.get_values(tokens, "Tempo")

            # Create a list of tempos
            tempos, tempo_ticks, tempo_times = context.tempos or (None, None, None)
            if tempos is None:
                if (
                    not additional_params["decode_recompute_tempos"]
                    or additional_params["onset_tempos"]
                ):
                    tempos = np.array([tempo_values[score_ticks == score_onsets[0]].mean()])
                else:
                    tempos = np.array([context.initial_tempo or self.default_tempo])
                tempo_ticks, tempo_times = np.zeros(1), np.zeros(1)
            tempo = tempos[-1]

            # Decode RelativeOnsetDeviation and RelativePerformedDuration values/tokens
            note_rel_onset_devs = self.get_values(tokens, "RelOnsetDev")
            note_rel_perf_durations = self.get_values(tokens, "RelPerfDuration")

            # Build onset pairs, compute performance notes start and end times
            onset_pairs = context.onset_pairs
            if onset_pairs is None:
                if score_ticks[0] > 0:
                    onset_pairs = np.array([(0, 0, 1)])
                else:
                    onset_pairs = np.array(
                        [(-time_division, -time_division / tempo * tempo_scale, 1)]
                    )
            prev_onset_tick, prev_onset_time, prev_num = onset_pairs[-1]

            _offset, num_tokens = 0, len(score_ticks)
            perf_times, perf_durations = np.zeros(num_tokens), np.zeros(num_tokens)

            for i, onset_tick in enumerate(score_onsets):
                repeated_onset = onset_tick == tempo_ticks[-1] and onset_tick > 0
                if repeated_onset:
                    prev_onset_tick, prev_onset_time = onset_pairs[-2, :2]

                onset_mask = score_ticks[_offset:] == onset_tick
                num = onset_mask.sum()

                is_performed_onset = is_performed[_offset:][onset_mask]
                num_perf = is_performed_onset.sum()

                _tempo = tempo
                if (
                    not additional_params["decode_recompute_tempos"]
                    or additional_params["onset_tempos"]
                ):
                    if repeated_onset:
                        tempo = (tempo * prev_num + tempo_values[_offset:][onset_mask].sum()) / (
                            prev_num + num
                        )
                    else:
                        tempo = tempo_values[_offset:][onset_mask].mean()

                score_shift = onset_tick - prev_onset_tick

                # Compute time shift using tempo
                time_shift = score_shift / tempo * tempo_scale
                onset_time = prev_onset_time + time_shift

                # Compute onset deviations for each note
                onset_devs = note_rel_onset_devs[_offset:][onset_mask] * time_shift
                onset_perf_times = onset_time + onset_devs

                # Average across performed notes
                if repeated_onset:
                    # Data computed for the preceding notes in the onset using previously averaged tempo
                    _time_shift = score_shift / _tempo * tempo_scale
                    _onset_time = prev_onset_time + _time_shift

                    _onset_perf_times_mean = onset_pairs[-1, 1]
                    _onset_devs_mean = _onset_perf_times_mean - _onset_time
                    _rel_onset_devs_mean = _onset_devs_mean / _time_shift

                    _onset_perf_times_mean = onset_time + _rel_onset_devs_mean * time_shift

                    onset_time = (
                        _onset_perf_times_mean * prev_num
                        + onset_perf_times[is_performed_onset].sum()
                    )
                    onset_time /= prev_num + num_perf
                else:
                    onset_time = onset_perf_times[is_performed_onset].mean()

                # Add new onset pair
                if repeated_onset:
                    onset_pairs[-1] = np.array([onset_tick, onset_time, prev_num + num])
                else:
                    onset_pairs = np.concatenate([onset_pairs, [(onset_tick, onset_time, num)]])
                onset_pair = onset_pairs[-1]

                # Process performed durations to compute note offset time
                onset_score_time_durations = (
                    score_durations[_offset:][onset_mask] / tempo * tempo_scale
                )
                onset_perf_time_durations = (
                    note_rel_perf_durations[_offset:][onset_mask] * onset_score_time_durations
                )

                # Save note attributes
                perf_times[_offset:][onset_mask] = onset_perf_times
                perf_durations[_offset:][onset_mask] = onset_perf_time_durations

                # Compute next tempo
                if (
                    additional_params["decode_recompute_tempos"]
                    and not additional_params["onset_tempos"]
                ):
                    if onset_time < 2 * additional_params["tempo_min_onset_dist"]:
                        tempo = context.initial_tempo  # not enough history, use initial tempo
                    else:
                        # Cut onsets in a local window
                        pairs_in_window = self.filter_onsets_in_window(
                            onset_time, onset_pairs[:-1, :2]
                        )

                        # Compute local tempo
                        tempo = self.compute_local_tempo(
                            distances=onset_pair[:2] - pairs_in_window, tempo_scale=tempo_scale
                        )

                if repeated_onset:
                    tempos[-1], tempo_times[-1] = tempo, onset_time
                else:
                    tempos = np.concatenate([tempos, [tempo]])
                    tempo_ticks = np.concatenate([tempo_ticks, [onset_tick]])
                    tempo_times = np.concatenate([tempo_times, [onset_time]])

                prev_onset_tick, prev_onset_time, prev_num = onset_pair
                _offset += num

            note_on_times = perf_times
            note_off_times = perf_times + perf_durations

        position_data = {
            "ticks_data": ticks_data,
            "note_on_ticks": None,
            "note_off_ticks": None,
            "note_on_times": note_on_times,
            "note_off_times": note_off_times,
            "tempos": (tempos, tempo_ticks, tempo_times),
        }

        def extend_context(prev_data, new_data):
            return np.concatenate([prev_data, new_data]) if prev_data is not None else new_data

        new_context = TokSequenceContext(
            time_signatures=ticks_data["time_sig"],
            tempos=(tempos, tempo_ticks, tempo_times),
            score_ticks=extend_context(context.score_ticks, score_ticks),
            note_on_ticks=None,
            note_on_times=extend_context(context.note_on_times, note_on_times),
            initial_tempo=context.initial_tempo,
            onset_pairs=onset_pairs,
        )

        return position_data, new_context

    def _create_relative_onset_deviations(self) -> np.ndarray:
        r"""
        Create the relative onset deviation bins.
        The larger the factor, the smaller the resolution.

        :return: the relative onset deviation bins
        """
        onset_dev_quant = (self.config.additional_params["num_onset_devs"] - 1) // 10

        def exp_segment(a, b, scale):
            steps = np.arange(onset_dev_quant // scale + 1) / onset_dev_quant * scale
            return (2 ** (np.log(b) / np.log(2) * steps) * a)[1:]

        rel_onset_devs = np.concatenate(
            [
                # 20% from 0 to 1/20
                np.linspace(0, 1 / 20, onset_dev_quant + 1),
                # 20% from 1/20 to 1/10
                np.linspace(1 / 20, 1 / 10, onset_dev_quant + 1)[1:],
                # 20% from 1/10 to 1/6
                np.linspace(1 / 10, 1 / 6, onset_dev_quant + 1)[1:],
                # 20% from 1/6 to 1/3
                exp_segment(1 / 6, 2, 1),
                # 10% from 1/3 to 1/2
                exp_segment(1 / 3, 3 / 2, 2),
                # 5% from 1/2 to 3/4
                exp_segment(1 / 2, 3 / 2, 4),
                # 2.5% from 3/4 to 1
                exp_segment(3 / 4, 4 / 3, 8),
                # 2.5% from 1 to 2
                exp_segment(1, 2, 8),
            ]
        )
        rel_onset_devs = np.round(rel_onset_devs, 4)
        rel_onset_devs = np.sort(
            np.concatenate([-rel_onset_devs[1:], rel_onset_devs])
        )  # add negative deviations

        return rel_onset_devs

    def _create_relative_performed_durations(self) -> np.ndarray:
        r"""
        Create the relative performed duration bins based on some heuristics.
        The larger the factor, the smaller the resolution.

        :return: the relative onset deviation bins
        """
        perf_dur_quant = (self.config.additional_params["num_perf_durations"] - 1) // 5

        rel_performed_durations = np.concatenate(
            [
                # 20% from 1/16 to 1/4
                np.log2(np.linspace(2, 16, perf_dur_quant)) / 16,
                # 60% from 1/4 to 1
                np.linspace(1 / 4, 1.0, 3 * perf_dur_quant + 1)[1:],
                # 10% from 1 to 2^(1/2)
                (2 ** (np.arange(perf_dur_quant // 2 + 1) / perf_dur_quant))[1:],
                # 5% from 2^(1/2) to 2
                (2 ** (2 * np.arange(perf_dur_quant // 4 + 1) / perf_dur_quant) * np.sqrt(2))[1:],
                # 5% from 2 to 4
                (2 ** (4 * np.arange(perf_dur_quant // 4 + 1) / perf_dur_quant) * 2)[1:],
            ]
        )
        rel_performed_durations = np.round(rel_performed_durations, 4)

        return rel_performed_durations

    def filter_onsets_in_window(self, onset_time: float, onset_pairs: np.ndarray) -> np.ndarray:
        r"""
        Select onsets in the local window for the specified onset.

        :param onset_time: current onset time
        :param onset_pairs: all onset (tick, time) pairs
        :return: the subset of onset pairs in the local window
        """
        additional_params = self.config.additional_params

        time_diffs = onset_time - onset_pairs[:, 1]
        candidate_pairs = onset_pairs[time_diffs >= additional_params["tempo_min_onset_dist"]]
        if len(candidate_pairs) == 0:
            candidate_pairs = onset_pairs[time_diffs >= 0]

        pairs_in_window = candidate_pairs[
            onset_time - candidate_pairs[:, 1] <= additional_params["tempo_window"]
        ]

        if (
            len(pairs_in_window) < additional_params["tempo_min_onsets"]
        ):  # collect minimum required number of onsets
            pairs_in_window = candidate_pairs[-additional_params["tempo_min_onsets"] :]
            pairs_in_window = pairs_in_window[
                onset_time - pairs_in_window[:, 1] <= 4 * additional_params["tempo_window"]
            ]

        if (
            len(pairs_in_window) == 0
        ):  # if suddenly no pairs found, take all candidates and hope for the best
            pairs_in_window = candidate_pairs

        return pairs_in_window

    def compute_local_tempo(
        self,
        distances: np.ndarray,
        eps: float = 1e-2,
        tempo_scale: float | None = None,
    ) -> float:
        r"""
        Compute weighted local tempo from the tick and time distances.

        :param distances: all onset (tick, time) distance pairs
        :param eps: small added value for the weighting function
        :param tempo_scale: tempo scaling factor
        :return: the computed local tempo
        """
        tempo_scale = tempo_scale or 60 / self.time_division

        local_tempos = distances[:, 0] / distances[:, 1] * tempo_scale
        weights = 1 - distances[:, 1] / (distances[:, 1].max() + eps)
        weights /= weights.sum()

        tempo = np.dot(weights, local_tempos)

        if self.config.use_tempos:
            tempo = min(self.tempos[-1], max(self.tempos[0], tempo))

            if self.config.additional_params["use_quantized_tempos"]:
                tempo = find_closest(self.tempos, tempo, return_values=True)

        return tempo

    def compute_onset_tempo(
        self,
        onset_pair: np.ndarray,
        prev_onset_pair: np.ndarray,
        tempo_scale: float | None = None,
    ) -> float:
        r"""
        Compute onset tempo from the tick and time distance for current and previous onsets.

        :param onset_pair: current pair onset (tick, time)
        :param prev_onset_pair: previous onset pair (tick, time)
        :param tempo_scale: tempo scaling factor
        :return: the computed local tempo
        """
        tempo_scale = tempo_scale or 60 / self.time_division

        if onset_pair[1] <= prev_onset_pair[1]:
            tempo = self.tempos[-1]
        else:
            tempo = (
                (onset_pair[0] - prev_onset_pair[0])
                / (onset_pair[1] - prev_onset_pair[1])
                * tempo_scale
            )

        if self.config.use_tempos:
            tempo = min(self.tempos[-1], max(self.tempos[0], tempo))

            if self.config.additional_params["use_quantized_tempos"]:
                tempo = find_closest(self.tempos, tempo, return_values=True)

        return tempo
