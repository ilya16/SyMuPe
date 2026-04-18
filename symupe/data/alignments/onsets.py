"""RAScoP pipeline, Stage (O): Onset Cleaning and Temporal Refinement."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from symusic import Score, TextMeta
from symusic.core import Tick

from .alignment import PositionPair, Alignment
from ..midi.timing import MIDITimeMapper
from ..midi.utils import shift_midi_events, clean_controls_in_interval


@dataclass
class OnsetCleanerConfig:
    """Configuration for Stage (O): Onset Cleaning and Temporal Refinement.

    Attributes:
        max_onset_delay: Multiplier for the minimum allowed time shift when the
            interval between notes is less than one quarter note. A negative value allows
            for higher flexibility/jitter in fast passages or dense chords ('lagging' notes).
        max_break: The maximum allowed silence (in seconds) between
            consecutive onsets before it is flagged as an 'alignment jump' or 'break'.
        min_time_diff: The absolute minimum time difference (seconds) required
            between two distinct onsets to avoid them being treated as simultaneous.

        min_tempo: The lower bound of the 'plausible tempo range' (BPM).
            Used to filter out unrealistically slow performance segments.
        max_tempo: The upper bound of the 'plausible tempo range' (BPM).
            Used to filter out unrealistically fast performance segments.

        min_deviation: The minimum threshold (seconds) for chord jitter.
            Deviations below this are never flagged as outliers.
        max_deviation: The maximum allowable deviation (seconds) within
            a chord. Any note deviating from the chord mean by more than this is flagged
            as an outlier, regardless of sigma.
        num_deviations: The 'n' multiplier for standard deviation (sigma).
            A note is an outlier if its deviation > n * sigma.

        tempo_time_window (tuple[float, float]): The sliding window (min, max) in seconds
            used to select previous notes for local tempo estimation.
        tempo_quarter_window (tuple[float, float]): The sliding window (min, max) in quarter notes
            used to bound the tempo estimation candidates.
        min_window_onsets: The minimum number of previous onset pairs
            required within the window to compute a valid local tempo.

        shift_note_breaks: If ``True``, algorithm will explicitly shift the global timing
            of all subsequent performance events to close detected alignment jumps.
        ticks_per_quarter: The MIDI resolution (TPQ) used for internal beat-to-second conversions.
    """

    # inter-onset management
    max_onset_delay: float = -1.0
    max_break: float = 10.0
    min_time_diff: float = 1e-2

    # tempo range
    min_tempo: int = 15
    max_tempo: int = 480

    # intra-onset deviations
    min_deviation: float = 0.02
    max_deviation: float = 0.25
    num_deviations: int = 2

    # local tempo computation
    tempo_time_window: tuple[float, float] = (0.5, 8.0)
    tempo_quarter_window: tuple[float, float] = (0.5, 16.0)
    min_window_onsets: int = 8

    # processing of jumps/breaks
    shift_note_breaks: bool = False
    ticks_per_quarter: int = 480


def compute_onset_position_pairs(
    alignment: Alignment,
    score_midi: Score,
    perf_midi: Score,
    config: OnsetCleanerConfig = OnsetCleanerConfig(),
    update_alignment_and_midi: bool = False,
) -> tuple[list[PositionPair], list[tuple[int, float, float]]]:
    """Computes a sequence of aligned score-performance onset pairs.

    If `update_alignment_and_midi` is True, this function filters the
    ``Alignment`` object to remove outliers and (if configured) shifts
    the performance MIDI events to correct temporal discontinuities.

    Args:
        alignment: :class:`symupe.data.alignments.Alignment` object containing note matches.
        score_midi: :class:`symusic.Score` object representing the score MIDI.
        perf_midi: :class:`symusic.Score` object representing the performance MIDI.
        config: :class:`OnsetCleanerConfig` configuration for cleaning and shifting logic.
        update_alignment_and_midi: If ``True``, modifies the `alignment` and `perf_midi` objects in-place.
    """
    assert not config.shift_note_breaks or update_alignment_and_midi, (
        "Note breaks can be shifted only together with notes in Alignment and performance MIDI"
    )

    position_pairs = alignment.build_position_pairs(score_midi)
    if len(position_pairs) == 0:
        return [], []

    onset_cleaner = OnsetCleaner(
        config=replace(config, ticks_per_quarter=position_pairs[0].ticks_per_quarter),
    )
    onset_pairs, mismatched_ids, shifts = onset_cleaner.clean_onset_pairs(
        position_pairs=position_pairs,
    )

    if update_alignment_and_midi:
        _update_alignment_and_midi(
            alignment=alignment,
            perf_midi=perf_midi,
            mismatched_ids=mismatched_ids,
            shifts=shifts,
        )

    return onset_pairs, shifts


class OnsetCleaner:
    """Implementation of Stage (O): Onset Cleaning and Temporal Refinement.

    Sequentially processes chords to:
    1. Clean intra-onset jitter (chords).
    2. Detect and optionally shift alignment jumps (breaks).
    3. Clean inter-onset outliers based on local tempo projections.
    """

    def __init__(
        self,
        config: OnsetCleanerConfig = OnsetCleanerConfig(),
    ):
        self.config = config or OnsetCleanerConfig()

        self.min_tick_time = 60 / self.config.max_tempo / self.config.ticks_per_quarter
        self.max_tick_time = 60 / self.config.min_tempo / self.config.ticks_per_quarter

        # state management
        self.position_pairs = []
        self.cleaned_pairs: list[PositionPair] = []
        self.mismatched_ids: set[int] = set()
        self.shifts: list[tuple[int, float, float]] = []

        # buffers for tempo calculation
        self.count = 0
        self.cleaned_asc_ticks = None
        self.cleaned_asc_times = None

        self.prev_tick = None
        self.prev_time = None

        self.clear()

    def clear(self):
        # state management
        self.cleaned_pairs: list[PositionPair] = []
        self.mismatched_ids: set[int] = set()
        self.shifts: list[tuple[int, float, float]] = []

        # buffers for tempo calculation
        self.count = 0
        self.cleaned_asc_ticks = np.zeros(len(self.position_pairs))
        self.cleaned_asc_times = np.zeros(len(self.position_pairs))

        self.prev_tick = None
        self.prev_time = None

    def clean_onset_pairs(
        self,
        position_pairs: list[PositionPair],
    ) -> tuple[list[PositionPair], set[int], list[tuple[int, float, float]]]:
        """The main loop for the Onset Cleaning stage.

        Iterates through position pairs, groups them into chords, and triggers chord-level cleaning.
        """
        self.position_pairs = position_pairs
        self.clear()

        self.prev_tick = prev_pair_tick = min(map(lambda p: p.score_tick, position_pairs[:20]))
        self.prev_time = min(map(lambda p: p.perf_time, position_pairs[:20]))

        chord_idx = 0
        chord_notes = []
        for i, pos_pair in enumerate(position_pairs):
            tick = pos_pair.score_tick

            if tick == prev_pair_tick:
                chord_notes.append(pos_pair)
            elif tick > prev_pair_tick:
                old_time = self.cleaned_asc_times[-1] if len(self.cleaned_asc_times) > 0 else 0

                # process collected notes
                self._process_chord_notes(chord_notes=chord_notes, chord_idx=chord_idx)
                new_time = self.cleaned_pairs[-1].perf_time if len(self.cleaned_pairs) > 0 else 0

                prev_pair_tick = tick
                if new_time > old_time:  # do not rely on delayed chords
                    self.prev_tick = self.cleaned_pairs[-1].score_tick
                    self.prev_time = self.cleaned_pairs[-1].perf_time
                chord_notes = [pos_pair]  # add the first chord note
                chord_idx = i
            else:
                self.mismatched_ids.add(pos_pair.idx)

        # process last position pair notes
        self._process_chord_notes(chord_notes=chord_notes, chord_idx=chord_idx)

        return self.cleaned_pairs, self.mismatched_ids, self.shifts

    def _process_chord_notes(self, chord_notes: list[PositionPair], chord_idx: int):
        """Core logic for a single beat (chord).

        Performs three sequential steps:
        1. Intra-onset cleaning (removes jitter within the chord).
        2. Break detection (identifies alignment jumps).
        3. Inter-onset cleaning (removes tempo outliers relative to tau_local).
        """
        if len(chord_notes) == 0:
            return

        onset_pair = PositionPair(**chord_notes[0].__dict__)

        onset_tick = onset_pair.score_tick
        if len(self.cleaned_asc_ticks) > 0:
            prev_onset_tick = self.cleaned_asc_ticks[self.count - 1]
            prev_onset_time = self.cleaned_asc_times[self.count - 1]
            tick_shift = onset_tick - prev_onset_tick
            min_time_shift = tick_shift * self.min_tick_time
            if onset_tick - prev_onset_tick <= self.config.ticks_per_quarter:
                min_time_shift *= self.config.max_onset_delay
            max_time_shift = tick_shift * self.max_tick_time
        else:
            prev_onset_tick = prev_onset_time = min_time_shift = 0
            max_time_shift = float("Inf")

        # clean intra-onset deviations (high deviations inside a chord)
        chord_notes_cleaned, onset_pair.perf_time = clean_onset_deviations(chord_notes)

        if len(chord_notes_cleaned) == 0:
            for note in chord_notes:
                self.mismatched_ids.add(note.idx)
            return

        onset_time = onset_pair.perf_time

        # check potential breaks and jumps in alignment
        onset_time_shift = onset_time - prev_onset_time
        if (
            self.config.shift_note_breaks
            and self.count > 0
            and (
                0 < onset_time_shift < min_time_shift
                or onset_time_shift > min(self.config.max_break, max_time_shift)
            )
        ):
            self._apply_global_shift(
                onset_tick=onset_tick,
                onset_time=onset_time,
                onset_pair=onset_pair,
                chord_idx=chord_idx,
                chord_notes_cleaned=chord_notes_cleaned,
            )

        # local tempo in seconds per tick
        tempo_spt = self._compute_local_tempo(
            onset_tick=onset_tick,
            onset_time=onset_time,
            clean_pair_idx=self.count,
        )

        if tempo_spt is not None:
            # projected time of the pair
            proj_time = prev_onset_time + tempo_spt * (onset_tick - self.prev_tick)

            # clean inter-onset deviations (onset is not in the expected local tempo)
            chord_notes_cleaned, onset_pair.perf_time = clean_onset_deviations(
                chord_notes_cleaned,
                prev_onset_time=prev_onset_time,
                proj_time=proj_time,
                min_time_shift=min_time_shift,
                max_time_shift=max_time_shift,
            )

            if len(chord_notes_cleaned) == 0:
                for note in chord_notes:
                    self.mismatched_ids.add(note.idx)
                return

        onset_time = onset_pair.perf_time
        onset_time_shift = onset_time - prev_onset_time

        tick_shift = onset_tick - prev_onset_tick
        _min_time_diff = min(self.config.min_time_diff, tick_shift * self.min_tick_time)

        if len(self.cleaned_pairs) == 0 or (
            min_time_shift < onset_time_shift < max_time_shift
            and abs(onset_time_shift) > _min_time_diff
        ):
            if self.count == 0 or onset_time > self.cleaned_asc_times[self.count - 1]:
                self.cleaned_asc_ticks[self.count] = onset_pair.score_tick
                self.cleaned_asc_times[self.count] = onset_pair.perf_time
                self.count += 1

            self.cleaned_pairs.append(onset_pair)

            for note in chord_notes:
                if note not in chord_notes_cleaned:
                    # note is far from other notes in the chord
                    self.mismatched_ids.add(note.idx)
        else:
            # onset is too close or too far from the previous onset
            for note in chord_notes:
                self.mismatched_ids.add(note.idx)

        return

    def _compute_local_tempo(
        self,
        onset_tick: int,
        onset_time: float,
        clean_pair_idx: int,
        max_prev_notes: int = 100,
    ):
        """Computes local performance tempo (tau_local) using the history of previously cleaned onsets."""
        return compute_local_tempo(
            onset_tick,
            onset_time,
            candidate_ticks=self.cleaned_asc_ticks[
                max(0, clean_pair_idx - max_prev_notes) : clean_pair_idx
            ],
            candidate_times=self.cleaned_asc_times[
                max(0, clean_pair_idx - max_prev_notes) : clean_pair_idx
            ],
            tempo_time_window=self.config.tempo_time_window,
            tempo_quarter_window=self.config.tempo_quarter_window,
            min_window_onsets=self.config.min_window_onsets,
            min_tempo=self.config.min_tempo,
            max_tempo=self.config.max_tempo,
            ticks_per_quarter=self.config.ticks_per_quarter,
        )

    def _apply_global_shift(
        self,
        onset_tick: int,
        onset_time: float,
        onset_pair: PositionPair,
        chord_idx: int,
        chord_notes_cleaned: list[PositionPair],
    ):
        """Handles 'Alignment Jumps'.

        Calculates the required time shift to align a jumpy performance onset with
        its projected time based on local tempo, and shifts all subsequent performance events.
        """
        j, tempo_spt = self.count - 1, None
        _prev_tick, _prev_time = self.cleaned_asc_ticks[j], self.cleaned_asc_times[j]

        for j in range(self.count - 1, max(0, self.count - 10), -1):
            tempo_spt = self._compute_local_tempo(
                onset_tick=int(self.cleaned_asc_ticks[j]),
                onset_time=float(self.cleaned_asc_times[j]),
                clean_pair_idx=j,
            )
            if tempo_spt is not None:
                break

        tempo_spt = 1 / (2 * self.config.ticks_per_quarter) if tempo_spt is None else tempo_spt

        # projected time of the pair
        proj_time = _prev_time + tempo_spt * (onset_tick - _prev_tick)
        shift_time = proj_time - onset_time

        # shift current and all succeeding pairs in time
        offset = min(break_pos_pair.perf_time for break_pos_pair in chord_notes_cleaned)
        for j, break_pos_pair in enumerate(self.position_pairs[chord_idx:]):
            if break_pos_pair.perf_time < offset:
                offset = break_pos_pair.perf_time
            break_pos_pair.perf_time += shift_time

        onset_pair.perf_time += shift_time
        onset_time = onset_pair.perf_time

        # save shift to update alignment later
        self.shifts.append((self.position_pairs[chord_idx].idx, offset - 1e-3, shift_time))

        return onset_time


def clean_onset_deviations(
    chord_notes: list[PositionPair],
    prev_onset_time: float | None = None,
    proj_time: float | None = None,
    min_deviation: float = 0.02,
    max_deviation: float = 0.25,
    num_deviations: int = 2,
    max_tempo_change_ratio: float = 0.5,
    min_time_shift: float | None = None,
    max_time_shift: float | None = None,
) -> tuple[list[PositionPair], float]:
    """Iteratively removes note outliers from a chord based on intra- and inter-onset deviations.

    Intra-onset Cleaning:
        Calculates the mean onset and standard deviation (sigma) of notes in a chord.
        Notes deviating by more than `num_deviations` * sigma
        (clamped by `min_deviation`/`max_deviation`) are removed.

    Inter-onset Cleaning:
        Compares the chord's mean onset to `proj_time` (calculated via local tempo).
        If the timing implies a tempo change exceeding `max_tempo_change_ratio`,
        the chord is flagged as a tempo outlier.

    Returns:
        A tuple containing the list of cleaned ``PositionPair`` objects and the final mean onset time.
    """
    time_diff = None
    if prev_onset_time is not None and proj_time is not None:
        time_diff = proj_time - prev_onset_time
        min_time_shift = (
            time_diff / (1 + max_tempo_change_ratio) if min_time_shift is None else min_time_shift
        )
        max_time_shift = (
            time_diff * (1 + max_tempo_change_ratio) if max_time_shift is None else max_time_shift
        )

    if len(chord_notes) == 1 and time_diff is None:
        return chord_notes, chord_notes[0].perf_time

    original_notes = chord_notes
    original_perf_times = np.array([note.perf_time for note in original_notes])
    included_indices = list(range(len(original_notes)))

    while len(included_indices) > 0:
        current_perf_times = original_perf_times[included_indices]
        mean_onset_time = (
            current_perf_times.mean() if len(current_perf_times) > 1 else current_perf_times[0]
        )

        # check for onset outliers
        onset_outlier_index = None
        if len(included_indices) > 1:
            abs_deviations = np.abs(current_perf_times - mean_onset_time)
            sigma = np.sqrt((abs_deviations**2).mean())
            threshold_onset = min(max_deviation, max(min_deviation, num_deviations * sigma))
            is_onset_outlier = abs_deviations.max() > threshold_onset
            if is_onset_outlier:
                onset_outlier_index = np.argmax(abs_deviations)

        # check for tempo outliers
        tempo_outlier_index = None
        if time_diff is not None:
            min_proj_time = prev_onset_time + min_time_shift
            max_proj_shift = prev_onset_time + max_time_shift
            is_tempo_outlier = (current_perf_times < min_proj_time) | (
                current_perf_times > max_proj_shift
            )
            tempo_outlier_indices = np.where(is_tempo_outlier)[0]
            if len(tempo_outlier_indices) > 0:
                tempo_outlier_index = tempo_outlier_indices[0]

        if onset_outlier_index is None and tempo_outlier_index is None:
            break

        if tempo_outlier_index is not None:
            del included_indices[tempo_outlier_index]
        elif onset_outlier_index is not None:
            if len(included_indices) == 2:
                included_indices = []
                break
            else:
                del included_indices[onset_outlier_index]
    else:
        return [], np.nan

    final_chord_notes = [original_notes[i] for i in included_indices]
    return final_chord_notes, mean_onset_time


def compute_local_tempo(
    end_tick: int,
    end_time: float,
    candidate_ticks: np.ndarray,
    candidate_times: np.ndarray,
    tempo_time_window: tuple[float, float] = (0.5, 8.0),
    tempo_quarter_window: tuple[float, float] = (0.5, 16.0),
    min_window_onsets: int = 8,
    min_tempo: int = 15,
    max_tempo: int = 480,
    ticks_per_quarter: int = 480,
) -> float | None:
    """Estimates the local tempo (tau_local) using a sliding w-second window.

    Estimates the maximum and minimum plausible time shifts between score onsets
    and uses a weighted average of previous onsets where closer notes contribute a higher weight.

    Returns:
        Local tempo in seconds-per-tick.
    """
    if len(candidate_ticks) == 0:
        return None

    tempo_tick_window = (
        ticks_per_quarter * tempo_quarter_window[0],
        ticks_per_quarter * tempo_quarter_window[1],
    )

    # extract candidate ticks and times
    candidate_ticks, candidate_times = np.array(candidate_ticks), np.array(candidate_times)
    tick_diff = end_tick - candidate_ticks
    time_diff = end_time - candidate_times

    # build masks according to the criteria
    cond1 = (
        (tick_diff >= tempo_tick_window[0])
        & (tick_diff <= tempo_tick_window[1])
        & (time_diff >= tempo_time_window[0])
        & (time_diff <= tempo_time_window[1])
    )
    mask = cond1

    # check extended criteria for extra window
    if mask.sum() < min_window_onsets:
        cond2 = (time_diff >= tempo_time_window[0]) & (time_diff <= 2 * tempo_time_window[1])
        mask = cond1 | cond2
        k = min(len(mask), min_window_onsets)
        mask[np.argpartition(tick_diff, -k)[:-k]] = False

    # if we have too few valid candidates, return None
    if mask.sum() == 0:
        return None

    valid_tick_diff = tick_diff[mask]
    valid_time_diff = time_diff[mask]

    # compute local tempos in beats per minute
    # (tick_diff / time_diff) gives ticks/second, converting to bpm
    local_tempos = valid_tick_diff / valid_time_diff * 60 / ticks_per_quarter

    # use weights that favor smaller time differences
    weights = 1 - valid_time_diff / (valid_time_diff.max() + 0.01)
    weights /= weights.sum()
    tempo = (weights * local_tempos).sum()

    # clamp tempo between min_tempo and max_tempo
    tempo = min(max_tempo, max(min_tempo, tempo))

    # return the seconds-per-tick value
    return 60 / (tempo * ticks_per_quarter)


def _update_alignment_and_midi(alignment, perf_midi, mismatched_ids, shifts):
    """Helper to finalize changes into the MIDI and alignment objects."""
    pair_to_perf, perf_to_pair = alignment.match_with_midi(perf_midi, is_score_midi=False)
    perf_notes = perf_midi.tracks[0].notes
    time_mapper = MIDITimeMapper(perf_midi) if isinstance(perf_midi, Tick) else None

    for idx in mismatched_ids:
        alignment.pairs[idx].perf_note = None

    ttype = "tick" if isinstance(perf_midi.ttype, Tick) else "second"
    all_shifted_indices = []
    for pos_idx, offset, time_shift in shifts:
        if time_shift < 0.0:
            perf_midi = clean_controls_in_interval(perf_midi, start=offset + time_shift, end=offset)

        perf_note_ids = pair_to_perf[int(pos_idx) :]
        perf_note_ids = perf_note_ids[perf_note_ids != -1]

        _, shifted_indices = shift_midi_events(
            perf_midi,
            time_shift=time_shift,
            offset=offset,
            note_indices=perf_note_ids,
            inplace=True,
            return_shifted_indices=True,
        )
        perf_midi.markers.append(
            TextMeta(
                time=0,
                text=f"ShiftOnset_{-time_shift:.6f}_{offset + time_shift:.6f}",
                ttype=ttype,
            )
        )

        all_shifted_indices.append(shifted_indices["note"][0][1])

    if all_shifted_indices:
        for idx in np.unique(np.concatenate(all_shifted_indices).astype(int)):
            pair_idx = perf_to_pair[idx]
            pair_note = alignment.pairs[pair_idx].perf_note
            if pair_idx != -1 and pair_note is not None:
                note = perf_notes[idx]
                pair_note.start = (
                    time_mapper.t2s(note.start) if time_mapper is not None else note.start
                )
                pair_note.end = time_mapper.t2s(note.end) if time_mapper is not None else note.end

    alignment.clear_cache()
