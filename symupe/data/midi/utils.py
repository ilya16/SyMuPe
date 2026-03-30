from __future__ import annotations

from collections import defaultdict

import numpy as np
from symusic import Score, Note, Track
from symusic.core import NoteTickList, Tick

from .timing import MIDITimeMapper


def sort_notes(notes: NoteTickList, order: str = "time") -> tuple[NoteTickList, np.ndarray | None]:
    if len(notes) == 0:
        return notes, None

    assert order in ("time", "pitch")

    note_soa = notes.numpy()

    sort_ids = None
    if order == "time":
        sort_ids = np.lexsort(
            [note_soa["velocity"], note_soa["duration"], note_soa["pitch"], note_soa["time"]]
        )
        notes = Note.from_numpy(
            **{key: values[sort_ids] for key, values in note_soa.items()}, ttype=notes[0].ttype
        )
    elif order == "pitch":
        sort_ids = np.lexsort(
            [note_soa["velocity"], note_soa["duration"], note_soa["time"], note_soa["pitch"]]
        )
        notes = Note.from_numpy(
            **{key: values[sort_ids] for key, values in note_soa.items()}, ttype=notes[0].ttype
        )

    return notes, sort_ids


def cut_overlapping_notes(
    notes: NoteTickList,
    duplicate_max_duration: bool = True,
    min_shift: float | None = None,
    sort: bool = False,
) -> NoteTickList:
    r"""Find and cut the first of the two overlapping notes, i.e. with the same pitch,
    and the second note starting before the ending of the first note.

    :param notes: notes to analyse
    :param duplicate_max_duration: make notes with the same start point have the same maximized duration
    :param min_shift: minimal tick/time shift between two overlapping notes with the same pitch
    :param sort: whether to sort notes before computation
    """
    if sort:
        notes, sort_ids = sort_notes(notes, order="time")

    prev_pitch_notes = {}
    for note in notes:
        prev_notes = prev_pitch_notes.get(note.pitch, None)
        if prev_notes is None:
            prev_pitch_notes[note.pitch] = [note]
            continue

        prev_note = prev_notes[-1]

        if prev_note.time == note.time:  # `note` and `prev_note` start at the same time
            if duplicate_max_duration:
                duration = max(prev_note.duration, note.duration)
            else:
                duration = min(prev_note.duration, note.duration)

            note.duration = duration
            for _prev_note in prev_notes:
                _prev_note.duration = duration
            prev_pitch_notes[note.pitch].append(note)
            continue

        elif prev_note.end > note.time:  # `note` starts before `prev_note` ended
            if (
                min_shift is not None and note.time - prev_note.time < min_shift
            ):  # previous note will be too short
                note.duration = max(prev_note.end, note.end) - prev_note.time  # duplicate notes
                note.time = prev_note.time

                for _prev_note in prev_notes:
                    _prev_note.duration = note.duration
                prev_pitch_notes[note.pitch].append(note)
                continue

            if prev_note.end > note.end:  # `note` is inside `prev_note`, do not cut total duration
                note.duration = prev_note.end - note.time

            for _prev_note in prev_notes:  # cut `prev_note` until `note` start
                _prev_note.duration = note.time - prev_note.start

        prev_pitch_notes[note.pitch] = [note]

    return notes


def remove_duplicated_notes(notes: NoteTickList):
    r"""Find and remove exactly similar notes, i.e. with the same pitch, start and end.

    :param notes: notes to analyse
    """
    i, prev_pitch_note = 0, {}
    while i < len(notes):
        note = notes[i]
        next_note = prev_pitch_note.get(note.pitch, None)

        if next_note is not None:
            if note.time == next_note.time and note.duration == next_note.duration:
                del notes[i]
                continue

        prev_pitch_note[note.pitch] = note
        i += 1

    return notes


def remove_duplicated_midi_changes(midi: Score) -> Score:
    r"""Find and remove exactly similar change events in MIDIs, i.e. with the same values or same start time.

    :param midi: MIDI to analyse
    """
    # Process tempos
    if len(midi.tempos) > 0:
        i, prev_tempo = 1, midi.tempos[0]
        while i < len(midi.tempos):
            if midi.tempos[i].qpm == prev_tempo.qpm:
                del midi.tempos[i]
                continue
            elif midi.tempos[i].time == prev_tempo.time:
                del midi.tempos[i - 1]
                prev_tempo = midi.tempos[i - 1]
                continue
            prev_tempo = midi.tempos[i]
            i += 1

    # Process time signatures
    if len(midi.time_signatures) > 0:
        i, prev_time_sig = 1, midi.time_signatures[0]
        while i < len(midi.time_signatures):
            time_sig = midi.time_signatures[i]

            if (
                time_sig.numerator == prev_time_sig.numerator
                and time_sig.denominator == prev_time_sig.denominator
            ):
                del midi.time_signatures[i]
                continue
            elif time_sig.time == prev_time_sig.time:
                del midi.time_signatures[i - 1]
                prev_time_sig = midi.time_signatures[i - 1]
                continue
            prev_time_sig = time_sig
            i += 1

    # Process key signatures
    if len(midi.key_signatures) > 0:
        i, prev_key_sig = 1, midi.key_signatures[0]
        while i < len(midi.key_signatures):
            key_sig = midi.key_signatures[i]

            if (key_sig.key, key_sig.tonality) == (prev_key_sig.key, prev_key_sig.tonality):
                del midi.key_signatures[i]
                continue
            elif key_sig.time == prev_key_sig.time:
                del midi.key_signatures[i - 1]
                prev_key_sig = midi.key_signatures[i - 1]
                continue
            prev_key_sig = key_sig
            i += 1

    return midi


def remove_short_notes(
    notes: NoteTickList,
    time_division: int,
    max_beat_res: int = 48,
    min_duration: float | int | None = None,
):
    r"""Find and remove short notes.

    :param notes: notes to analyse
    :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI being parsed)
    :param max_beat_res: maximum beat resolution for one sample
    :param min_duration: minimum duration for a note
    """
    if min_duration is not None:
        for i in range(len(notes) - 1, 0, -1):
            if notes[i].duration < min_duration:
                del notes[i]
    else:
        ticks_per_sample = int(time_division / max_beat_res)
        for i in range(len(notes) - 1, 0, -1):
            if notes[i].duration < ticks_per_sample // 2:
                del notes[i]

    return notes


def filter_notes_by_pitch_range(notes: NoteTickList, pitch_range: tuple[int, int] = (21, 108)):
    i = 0
    while i < len(notes):
        note = notes[i]
        if note.pitch < pitch_range[0] or note.pitch > pitch_range[1]:
            del notes[i]
            continue
        i += 1

    return notes


def filter_extra_midi_events(
    midi: Score,
    min_tick: int | None = None,
    max_tick: int | None = None,
    sort: bool = False,
    use_sustain_boundaries: bool = False,
):
    if use_sustain_boundaries:
        min_tick, max_tick = compute_global_sustain_control_boundaries(midi)

    min_tick = (
        min_tick
        if min_tick is not None
        else min(track.notes.numpy()["time"].min() for track in midi.tracks)
    )
    max_tick = (
        max_tick
        if max_tick is not None
        else max(
            (track.notes.numpy()["time"] + track.notes.numpy()["duration"]).max()
            for track in midi.tracks
        )
    )

    for track in midi.tracks:
        if sort:
            track.controls.sort(key=lambda c: c.time)
            track.pedals.sort(key=lambda p: p.time)
            track.pitch_bends.sort(key=lambda p: p.time)

        track.controls = list(filter(lambda c: min_tick <= c.time <= max_tick, track.controls))
        track.pedals = list(filter(lambda p: min_tick <= p.time <= max_tick, track.pedals))
        track.pitch_bends = list(
            filter(lambda p: min_tick <= p.time <= max_tick, track.pitch_bends)
        )

    return midi


def shift_midi_events(
    midi: Score,
    time_shift: float = 0.0,
    offset: float = 0.0,
    note_offset: int = 0,
    note_indices: np.ndarray | None = None,
    inplace: bool = True,
    return_shifted_indices: bool = False,
):
    midi = midi if inplace else midi.copy()

    time_mapper = MIDITimeMapper(midi) if isinstance(midi.ttype, Tick) else None

    def process_continuous_events(
        elements, offset_index: int = 0, indices: np.ndarray | None = None
    ):
        el_soa = elements.numpy()
        start_ticks = el_soa["time"]
        end_ticks = start_ticks + el_soa["duration"]

        if time_mapper is None:
            start_times = start_ticks
            new_start_ticks = start_times + time_shift
            new_end_ticks = end_ticks + time_shift
        else:
            start_times, end_times = time_mapper.t2s(start_ticks), time_mapper.t2s(end_ticks)
            new_start_ticks = time_mapper.s2t(start_times + time_shift)
            new_end_ticks = time_mapper.s2t(end_times + time_shift)

            same_mask = new_end_ticks == new_start_ticks
            new_end_ticks[same_mask] = new_end_ticks[same_mask] + 1

        if indices is None:
            mask = np.ones(len(elements), dtype=bool)
        else:
            mask = np.zeros(len(elements), dtype=bool)
            mask[indices] = True

        shift_ids = []
        for idx, (el, time, start_t, end_t) in enumerate(
            zip(elements, start_times, new_start_ticks, new_end_ticks)
        ):
            if idx >= offset_index and time >= offset and mask[idx]:
                shift_ids.append(idx)
                el.time = start_t
                if time_mapper is not None:
                    el.duration = end_t - start_t

        return np.array(shift_ids)

    def process_instant_events(elements):
        ticks = elements.numpy()["time"]

        if time_mapper is None:
            times = ticks
            new_ticks = times + time_shift
        else:
            times = time_mapper.t2s(ticks)
            new_ticks = time_mapper.s2t(times + time_shift)

        for el, time, tick in zip(elements, times, new_ticks):
            if time >= offset:
                el.time = tick

        return np.where(times >= offset)[0]

    # shift relevant notes in MIDI
    shifted_indices = defaultdict(list)
    for track_idx, track in enumerate(midi.tracks):
        shifted_indices["note"].append(
            (
                track_idx,
                process_continuous_events(
                    track.notes, offset_index=note_offset, indices=note_indices
                ),
            )
        )
        if track.pedals:
            shifted_indices["pedal"].append((track_idx, process_continuous_events(track.pedals)))
        if track.controls:
            shifted_indices["control_change"].append(
                (track_idx, process_instant_events(track.controls))
            )
        if track.pitch_bends:
            shifted_indices["pitch_bend"].append(
                (track_idx, process_instant_events(track.pitch_bends))
            )

    if return_shifted_indices:
        return midi, shifted_indices
    return midi


def clip_silence(midi: Score, max_silence: float = 5.0) -> Score:
    for track in midi.tracks:
        note_soa = track.notes.numpy()
        note_on = note_soa["time"]
        note_off = note_on + note_soa["duration"]

        max_note_off = np.maximum.accumulate(np.concatenate([[0.0], note_off[:-1]], axis=0))

        is_silence = note_on > max_note_off
        silences = np.stack([max_note_off[is_silence], note_on[is_silence]], axis=1)

        silences = silences[np.diff(silences).flatten() > max_silence]

        for i, (left, right) in enumerate(silences):
            left, right = float(left), float(right)
            new_right = float(left + max_silence)
            shift = new_right - right

            midi = clean_controls_in_interval(midi, start=new_right, end=right)

            midi = shift_midi_events(midi, time_shift=shift, offset=new_right - 1e-3)
            silences[i + 1 :] += shift

    return midi


def resample_midi(midi: Score, ticks_per_quarter: int, min_duration: int | None = 1):
    if midi.ticks_per_quarter == ticks_per_quarter:
        return midi
    return midi.resample(ticks_per_quarter, min_dur=min_duration)


def convert_note_markers(midi: Score) -> Score:
    ttype = "tick" if isinstance(midi.ttype, Tick) else "second"
    time_mapper = MIDITimeMapper(midi)

    for marker in midi.markers:
        if marker.text.startswith("Note") and marker.text[5] == "_":
            note_marker, pitch, start, duration = marker.text.split("_")
            start, duration = map(float if ttype == "tick" else int, (start, duration))

            if ttype == "tick":
                duration = time_mapper.s2t(start + duration) - marker.time
            else:
                duration = time_mapper.t2s(start + duration) - marker.time

            marker.text = f"{note_marker}_{pitch}_{marker.time}_{duration}"

    return midi


def extract_track_pedals(track: Track):
    ctrl_soa = track.controls.numpy()

    is_sustain = ctrl_soa["number"] == 64
    sustain_values = ctrl_soa["value"][is_sustain]
    if len(sustain_values) == 0:
        return [], []

    sustain_changes = np.diff(np.concatenate([[0], (sustain_values >= 64).astype(int)]))

    sustain_ons = ctrl_soa["time"][is_sustain][sustain_changes > 0]
    sustain_offs = ctrl_soa["time"][is_sustain][sustain_changes < 0]

    if len(sustain_ons) > len(sustain_offs):
        sustain_offs = np.concatenate([sustain_offs, [track.end()]])

    return sustain_ons, sustain_offs


def apply_sustain_control_changes(
    midi: Score, inplace: bool = True, max_duration: int | float | None = None
):
    midi = midi if inplace else midi.copy()

    for track in midi.tracks:
        sustain_ons, sustain_offs = extract_track_pedals(track)
        if len(sustain_ons) == 0:
            continue

        note_soa = track.notes.numpy()
        note_offs = note_soa["time"] + note_soa["duration"]

        # a note off during a pedal
        start_search = np.searchsorted(sustain_ons, note_offs, side="right") - 1
        end_search = np.searchsorted(sustain_offs, note_offs, side="left")
        note_off_sustain = (start_search >= 0) & (end_search == start_search)

        if np.any(note_off_sustain):
            note_ids, sustain_ids = np.where(note_off_sustain)[0], start_search[note_off_sustain]
            note_soa["duration"][note_ids] = sustain_offs[sustain_ids] - note_soa["time"][note_ids]

            if max_duration is not None:
                note_soa["duration"][note_ids] = np.minimum(
                    max_duration, note_soa["duration"][note_ids]
                )

            track.notes = Note.from_numpy(**note_soa)
            track.notes = cut_overlapping_notes(track.notes)

    return midi


def compute_global_sustain_control_boundaries(midi: Score):
    start, end = midi.end(), midi.start()
    has_pedals = False

    for track in midi.tracks:
        if len(track.controls) == 0:
            continue

        sustain_ons, sustain_offs = extract_track_pedals(track)
        if len(sustain_ons) == 0:
            continue

        note_soa = track.notes.numpy()
        note_ons = note_soa["time"]
        note_offs = note_ons + note_soa["duration"]

        used_ids = np.where(
            np.any(
                np.logical_or(
                    # a note off during a pedal
                    (note_offs[:, None] >= sustain_ons[None])
                    & (note_offs[:, None] <= sustain_offs[None]),
                    # a pedal (on+off) during a note
                    (note_ons[:, None] <= sustain_ons[None])
                    & (note_offs[:, None] >= sustain_offs[None]),
                ),
                axis=0,
            )
        )[0]

        if len(used_ids) == 0:
            continue

        has_pedals = True
        start = min(start, sustain_ons[used_ids].min())
        end = max(end, sustain_offs[used_ids].max())

    start, end = (start, end) if has_pedals else (None, None)
    return start, end


def clean_controls_in_interval(midi: Score, start: float, end: float, eps: float = 1e-3):
    for track in midi.tracks:
        pedal_start, pedal_end = start, end

        sustain_ons, sustain_offs = extract_track_pedals(track)

        if len(sustain_ons) > 0:
            mask_off = (start <= sustain_offs) & (sustain_offs <= end)
            if np.any(mask_off):
                pedal_start = max(start, sustain_offs[mask_off].min())

            mask_on = (start <= sustain_ons) & (sustain_ons <= end)
            if np.any(mask_on):
                pedal_end = min(end, sustain_ons[mask_on].max())

        if pedal_start != start:
            for c in track.controls:
                if abs(c.time - pedal_start) < eps:
                    c.time = start - eps
                    break

        if pedal_end != end:
            for c in track.controls:
                if abs(c.time - pedal_end) < eps:
                    c.time = end + eps
                    break

        if pedal_start < pedal_end:
            track.controls = list(
                filter(lambda c: c.time < pedal_start or c.time > pedal_end, track.controls)
            )
            track.pedals = list(
                filter(lambda p: p.time < pedal_start or p.time > pedal_end, track.pedals)
            )

    return midi


def fix_incorrect_durations(notes: NoteTickList, sort: bool = True):
    if sort:
        notes, _ = sort_notes(notes, order="time")

    prev_pitch_notes = {}
    update_notes = []
    for note in notes:
        prev_note = prev_pitch_notes.get(note.pitch, None)
        if prev_note is not None:
            if note.time == prev_note.time:
                # `note` and `prev_note` start at the same time, leave only prev
                note.velocity = 0
                continue
            elif note.end == prev_note.end:
                # `note` and `prev_note` end at the same time, remove prev
                prev_note.velocity = 0

            elif note.time < prev_note.end:
                # `note` starts before `prev_note` ended, update durations to cut notes
                update_notes.append((note, prev_note.end - note.time))

        prev_pitch_notes[note.pitch] = note

    for note, new_duration in update_notes:
        note.duration = new_duration

    notes = [n for n in notes if n.velocity > 0]

    return notes


UNPERFORMED_TRACK_NAME = "Unperformed Notes"


def create_unperformed_notes_track(program: int = 0):
    return Track(program=program, name=UNPERFORMED_TRACK_NAME, is_drum=False)
