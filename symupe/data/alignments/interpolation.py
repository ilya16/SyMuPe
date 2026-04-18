"""RAScoP pipeline, Stage (I): Note Interpolation."""

import bisect
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from symusic import Score, Note, TextMeta
from symusic.core import Tick

from .alignment import Alignment, AlignmentNote
from ..midi.timing import MIDITimeMapper
from ..midi.utils import shift_midi_events, sort_notes, cut_overlapping_notes
from symupe.utils import find_closest


@dataclass
class NoteInterpolationConfig:
    """Configuration for Stage (I): Note Interpolation.

    Attributes:
        reuse_new_notes: If ``True``, allows newly interpolated notes to serve
            as anchors for subsequent missing notes in the same pass.
        save_markers: If ``True``, appends MIDI TextMeta markers (e.g., 'NoteI_...')
            to identify synthesized notes in the performance track.

        min_inter_interval: The (quarters, seconds) minimum spacing required between
            anchor notes for standard linear interpolation.
        min_extra_interval: The (quarters, seconds) minimum spacing used when searching
            for anchors outside the immediate neighborhood.
        local_window: The range in quarter notes (left, right) around the target note
            to search for reference velocity and articulation.
        min_local_pairs: The minimum number of nearby performed notes required
             to compute a reliable weighted average for dynamics.

        min_duration: The absolute minimum duration (seconds) for a synthesized note.
        min_time_shift: Minimum time (seconds) to separate onsets of the same pitch,
            preventing unrealistic note collisions.
    """

    reuse_new_notes: bool = False
    save_markers: bool = True

    min_inter_interval: tuple[float, float] = (0.5, 0.1)
    min_extra_interval: tuple[float, float] = (1, 0.5)
    local_window: tuple[int, int] = (-3, 3)
    min_local_pairs: int = 8

    min_duration: float = 0.02
    min_time_shift: float = 0.02


def interpolate_missing_notes(
    score_midi: Score,
    perf_midi: Score,
    alignment: Alignment,
    config: NoteInterpolationConfig = NoteInterpolationConfig(),
) -> tuple[Score, Alignment, int]:
    """Implementation of Stage (I): Note Interpolation.

    Synthesizes missing performance notes to create a complete parallel context.

    This function:
    1. Linearly interpolates onset times $t(n_i)$ from neighboring anchor notes.
    2. Estimates velocity and articulation via a weighted average of nearby
        performed notes, where weights are inversely proportional to beat distance.
    3. Adds 'NoteI' markers to the performance MIDI to distinguish synthetic events.
        Markers follow the format: `NoteI_{pitch}_{start}_{duration}`.

    Args:
        score_midi: The score reference providing the target context.
        perf_midi: The performance MIDI where notes will be synthesized.
        alignment: Current alignment object to be filled with synthetic pairs.
        config: Configuration for interpolation logic and dynamics estimation.

    Returns:
        A tuple of (Updated perf_midi, Updated alignment, num_added).
    """

    pair_to_score, score_to_pair = alignment.match_with_midi(score_midi, is_score_midi=True)
    pair_to_perf, perf_to_pair = alignment.match_with_midi(perf_midi, is_score_midi=False)

    is_performed = np.array([pair.perf_note is not None for pair in alignment.pairs])
    if np.all(is_performed):
        return perf_midi, alignment, 0

    performed_indices = np.where(is_performed)[0]
    is_performed_score = is_performed[score_to_pair]
    new_pair_to_perf = pair_to_perf.copy()

    score_notes = score_midi.tracks[0].notes
    score_note_soa = score_notes.numpy()
    score_ticks, score_pitches = score_note_soa["time"], score_note_soa["pitch"]
    score_durations = score_note_soa["duration"]
    performed_score_ticks = score_ticks[is_performed_score]
    closest_performed_ticks = performed_score_ticks[
        find_closest(performed_score_ticks, score_ticks)
    ]
    score_end_tick = score_ticks.max()

    perf_notes = perf_midi.tracks[0].notes
    perf_note_soa = perf_notes.numpy()
    perf_pitches, perf_velocities = perf_note_soa["pitch"], perf_note_soa["velocity"]
    perf_note_soa_s = perf_midi.to("second").tracks[0].notes.numpy()
    perf_times, perf_time_durations = perf_note_soa_s["time"], perf_note_soa_s["duration"]

    pair_pitches = perf_pitches[pair_to_perf]

    ticks_per_quarter = score_midi.ticks_per_quarter
    time_mapper = MIDITimeMapper(perf_midi) if isinstance(perf_midi.ttype, Tick) else None
    ttype = "tick" if isinstance(perf_midi.ttype, Tick) else "second"

    # precompute score tick to indices mapping
    score_tick_to_indices = defaultdict(list)
    for idx, tick in enumerate(score_ticks):
        score_tick_to_indices[tick].append(idx)
    score_tick_to_indices = {tick: np.array(ids) for tick, ids in score_tick_to_indices.items()}

    pitch_times = defaultdict(list)
    for pitch in np.unique(pair_pitches):
        pitch_times[pitch] = np.sort(perf_times[perf_pitches == pitch]).tolist()

    new_notes, perf_shift = 0, 0.0
    for i, pair in enumerate(alignment.pairs):
        if pair.perf_note is not None:
            continue

        new_notes += 1

        pitch = pair.score_note.pitch
        score_note = score_notes[pair_to_score[i]]

        # find all performed notes in the same chord
        chord_ids = score_to_pair[score_tick_to_indices[score_note.start]]
        chord_ids = chord_ids[is_performed[chord_ids]]

        note_off, weights = None, None
        if len(chord_ids) > 0:
            # use average onset time
            note_on = perf_times[pair_to_perf[chord_ids]].mean()
        else:
            # find known left and right notes to interpolate position
            left = -1
            if performed_indices[0] < i:
                left = performed_indices[np.searchsorted(performed_indices, i, side="left") - 1]

            if left < 0:
                left = len(pair_to_perf)
                if performed_indices[-1] > i:
                    left = performed_indices[np.searchsorted(performed_indices, i, side="right")]

            tick_interval, time_interval = (
                config.min_inter_interval if left < i else config.min_extra_interval
            )
            tick_interval = tick_interval * ticks_per_quarter
            left_perf_note = alignment[left].perf_note
            left_score_note = score_notes[pair_to_score[left]]

            right = max(i + 1, left + 1)
            right_performed = np.searchsorted(performed_indices, right, side="left")
            for idx in range(right_performed, len(performed_indices)):
                right = int(performed_indices[idx])
                right_perf_note = alignment[right].perf_note
                right_score_note = score_notes[pair_to_score[right]]

                if not (
                    left_perf_note.start > right_perf_note.start - time_interval
                    or left_score_note.start > right_score_note.start - tick_interval
                    or left_score_note.start == right_score_note.start
                ):
                    break
            else:
                tick_interval, time_interval = config.min_extra_interval
                tick_interval = tick_interval * ticks_per_quarter
                left, right = left - 1, left
                right_perf_note = alignment[right].perf_note
                right_score_note = score_notes[pair_to_score[right]]

                while left > 0 and (
                    not is_performed[left]
                    or left_perf_note.start > right_perf_note.start - time_interval
                    or left_score_note.start > right_score_note.start - tick_interval
                    or left_score_note.start == right_score_note.start
                ):
                    left -= 1
                    left_perf_note = alignment[left].perf_note
                    left_score_note = score_notes[pair_to_score[left]]

            left_tick, right_tick = (
                score_notes[pair_to_score[left]].start,
                score_notes[pair_to_score[right]].start,
            )

            left_chord_ids = score_to_pair[score_tick_to_indices[left_tick]]
            left_chord_ids = left_chord_ids[is_performed[left_chord_ids]]

            right_chord_ids = score_to_pair[score_tick_to_indices[right_tick]]
            right_chord_ids = right_chord_ids[is_performed[right_chord_ids]]

            left_time = perf_times[pair_to_perf[left_chord_ids]].mean()
            right_time = perf_times[pair_to_perf[right_chord_ids]].mean()

            tick_shift, time_shift = right_tick - left_tick, right_time - left_time

            note_on = left_time + time_shift * (score_note.start - left_tick) / tick_shift

        note_on = round(note_on, 6)

        if config.min_time_shift > 0.0 and pitch in pitch_times:
            while True:
                pts = pitch_times[pitch]
                idx = bisect.bisect_left(pts, note_on)
                if idx > 0:
                    left_neighbor = pts[idx - 1]
                    if note_on - left_neighbor < config.min_time_shift - 1e-4:
                        note_on = left_neighbor + config.min_time_shift
                        continue

                if idx < len(pts):
                    right_neighbor = pts[idx]
                    if right_neighbor - note_on < config.min_time_shift - 1e-4:
                        note_on = right_neighbor + config.min_time_shift
                        continue

                break

            bisect.insort(pitch_times[pitch], note_on)
        else:
            pitch_times[pitch] = [note_on]

        # collect performed notes in the local window to compute velocity and articulation
        local_ids = []
        closest_score_tick = closest_performed_ticks[pair_to_score[i]]
        extra_windows = abs(closest_score_tick - score_note.start) // ticks_per_quarter
        extra_windows = max(
            0, extra_windows - min(abs(config.local_window[0]), abs(config.local_window[1]))
        )
        left_offset = (config.local_window[0] - extra_windows) * ticks_per_quarter
        right_offset = (config.local_window[1] + extra_windows) * ticks_per_quarter
        while len(local_ids) < config.min_local_pairs and (
            len(local_ids) == 0 or (left_offset >= 0 or right_offset <= score_end_tick)
        ):
            left_idx = np.searchsorted(score_ticks, score_note.start + left_offset, side="left")
            right_idx = np.searchsorted(score_ticks, score_note.start + right_offset, side="right")
            local_ids = np.arange(left_idx, right_idx)[is_performed_score[left_idx:right_idx]]

            left_offset -= ticks_per_quarter
            right_offset += ticks_per_quarter

        distances = np.abs(score_note.start - score_ticks[local_ids]) / ticks_per_quarter

        # duplicate (increase weight of) notes with the same pitch
        same_pitch = score_pitches[local_ids] == pitch
        local_same_pitch_ids = local_ids[same_pitch]
        if len(local_same_pitch_ids) > 0:
            distances = np.concatenate([distances, distances[same_pitch]])
            local_ids = np.concatenate([local_ids, local_same_pitch_ids])

        weights = 1 - distances / (distances.max() + 1)
        weights /= weights.sum()

        local_pair_ids = score_to_pair[local_ids]

        if note_on < 0.0:  # slow but faster than to think
            # shift all notes
            perf_shift -= note_on
            shift_midi_events(perf_midi, time_shift=-note_on, inplace=True)
            alignment.shift_notes(time_shift=-note_on, score_notes=False)
            perf_times -= note_on
            pitch_times = {
                _pitch: (np.array(pts) - note_on).tolist() for _pitch, pts in pitch_times.items()
            }
            if note_off is not None:
                note_off += -note_on
            note_on = 0.0

            time_mapper = MIDITimeMapper(perf_midi) if isinstance(perf_midi.ttype, Tick) else None

        weights = 1 / len(local_ids) if weights is None else weights

        # use weighted velocity and articulation
        velocities = perf_velocities[pair_to_perf[local_pair_ids]]
        velocity = int((velocities * weights).sum())

        if note_off is None:
            articulations = (
                perf_time_durations[pair_to_perf[local_pair_ids]] / score_durations[local_ids]
            )

            # add more weight to the notes having a similar score duration
            duration_ratios = score_durations[local_ids] / score_note.duration
            duration_ratios[duration_ratios > 1] **= -1

            weights *= duration_ratios
            weights /= weights.sum()

            articulation = (articulations * weights).sum()

            note_off = note_on + articulation * score_note.duration

        note_off = float(max(note_off, note_on + config.min_duration))

        # map times to performance ticks
        if time_mapper is None:
            start, end = note_on, note_off
        else:
            start = time_mapper.s2t(note_on)
            end = max(start + 1, time_mapper.s2t(note_off))

        # add new note and update alignment pair
        perf_notes.append(
            Note(
                time=start,
                duration=end - start,
                pitch=pitch,
                velocity=velocity,
                ttype=ttype,
            )
        )
        alignment.pairs[i].perf_note = AlignmentNote(
            idx=-1,
            pitch=pitch,
            start=max(0.0, round(note_on, 6)),
            end=max(0.0, round(note_off, 6)),
        )
        if config.reuse_new_notes:
            pair_to_perf[i] = len(perf_notes) - 1
            is_performed[i] = True

        new_pair_to_perf[i] = len(perf_notes) - 1

    if new_notes == 0:
        return perf_midi, alignment, new_notes

    perf_notes, sort_ids = sort_notes(perf_notes, order="time")
    perf_midi.tracks[0].notes = perf_notes = cut_overlapping_notes(perf_notes)

    unsort_ids = np.argsort(sort_ids)
    for idx in np.arange(len(perf_notes)):
        note = perf_notes[unsort_ids[new_pair_to_perf[idx]]]
        alignment.pairs[idx].perf_note.start = (
            time_mapper.t2s(note.start) if time_mapper is not None else note.start
        )
        alignment.pairs[idx].perf_note.end = (
            time_mapper.t2s(note.end) if time_mapper is not None else note.end
        )

    alignment.preprocess_pairs(sort=True, clean_duplicates=False)

    if config.save_markers:
        markers = []
        for idx in np.argsort(sort_ids)[-new_notes:]:
            note = perf_notes[idx]
            start = time_mapper.t2s(note.start) if time_mapper is not None else note.start
            duration = (
                time_mapper.t2s(note.end) - start if time_mapper is not None else note.duration
            )
            markers.append(
                TextMeta(
                    time=note.start,
                    text=f"NoteI_{note.pitch}_{start:.6f}_{duration:.6f}",
                    ttype=ttype,
                )
            )

        perf_midi.markers.extend(markers)

    perf_midi.markers.insert(0, TextMeta(time=0, text=f"ShiftInterp_{perf_shift:.6f}", ttype=ttype))

    return perf_midi, alignment, new_notes


def process_unperformed_notes(
    score_midi: Score,
    perf_midi: Score,
    alignment: Alignment,
) -> int:
    """Identifies score notes that have no performance alignment and appends
    them as TextMeta markers to the performance MIDI.

    Markers follow the format: `NoteS_{pitch}_{start}_{duration}`.

    Args:
        score_midi: The score reference providing the target context.
        perf_midi: The performance MIDI where notes will be synthesized.
        alignment: Current alignment object to be filled with synthetic pairs.

    Returns:
        The number of added unperformed note markers.
    """
    ttype = "tick" if isinstance(perf_midi.ttype, Tick) else "second"
    time_mapper = MIDITimeMapper(perf_midi)
    perf_score_tick_ratio = perf_midi.ticks_per_quarter / score_midi.ticks_per_quarter

    pair_to_score, _ = alignment.match_with_midi(score_midi)
    note_soa = score_midi.tracks[0].notes.numpy()

    score_start_ticks = (note_soa["time"][pair_to_score] * perf_score_tick_ratio).astype(int)
    score_end_ticks = (
        score_start_ticks + note_soa["duration"][pair_to_score] * perf_score_tick_ratio
    ).astype(int)

    new_notes = 0
    for i, pair in enumerate(alignment.pairs):
        if pair.perf_note is None:
            pitch = pair.score_note.pitch
            start, end = map(
                lambda t: float(round(time_mapper.t2s(t), 6)),
                (score_start_ticks[i], score_end_ticks[i]),
            )
            perf_midi.markers.append(
                TextMeta(
                    time=int(score_start_ticks[i]) if ttype == "tick" else start,
                    text=f"NoteS_{pitch}_{start:.6f}_{end - start:.6f}",
                    ttype=ttype,
                )
            )
            pair.perf_note = AlignmentNote(idx=-1, pitch=pitch, start=start, end=end)
            new_notes += 1

    if new_notes > 0:
        alignment.clear_cache()

    return new_notes
