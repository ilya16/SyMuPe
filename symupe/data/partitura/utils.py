from __future__ import annotations

import numpy as np
import partitura as pt
from symusic import Score

from symupe.data.midi.preprocess import preprocess_midi


def cut_overlapping_partitura_score_notes(
        score: pt.score.Score,
        duplicate_max_duration: bool = True,
        min_shift: float | None = 1 / 32,  # in quarter notes
        trills: bool = True
):
    for part in score.parts:
        tpq = part.first_point.quarter
        min_tick_shift = int(tpq * min_shift) if min_shift is not None else None

        prev_pitch_notes = {}

        for note in part.notes[::-1]:
            if note.tie_next is not None and note.tie_next.duration_tied is None:
                note.tie_next = None
            note.end = note.start + (note.duration if note.tie_next is None else note.duration_tied)
            note.tie_next = None

        for note in part.notes_tied:
            if note.tie_prev is not None:
                continue

            prev_notes = prev_pitch_notes.get(note.midi_pitch, None)
            if prev_notes is not None:
                prev_note = prev_notes[-1]

                prev_duration = prev_note.duration if prev_note.tie_next is None else prev_note.duration_tied
                duration = note.duration if note.tie_next is None else note.duration_tied

                if prev_note.start == note.start:  # `note` and `prev_note` start at the same time
                    duration = max(prev_duration, duration) if duplicate_max_duration else min(prev_duration, duration)
                    # note.tie_next = None

                    note.end = note.start + duration
                    for _prev_note in prev_notes:
                        _prev_note.end = note.end
                    prev_pitch_notes[note.midi_pitch].append(note)
                    continue

                elif prev_note.end > note.start and trills and "trill-mark" in prev_note.ornaments:
                    prev_note.end = prev_note.start + note.duration

                if prev_note.end > note.start:  # `note` starts before `prev_note` ended
                    if prev_duration == duration:  # a close note with the same duration, make them duplicated
                        note.start = prev_note.start
                        prev_pitch_notes[note.midi_pitch].append(note)
                        continue

                    if min_shift is not None and note.start.t - prev_note.start.t < min_tick_shift:  # previous note will be too short
                        note.end = max(prev_note.end, note.end)  # duplicate notes
                        note.start = prev_note.start

                        for _prev_note in prev_notes:
                            _prev_note.end = note.end
                        prev_pitch_notes[note.midi_pitch].append(note)
                        continue

                    elif prev_note.end > note.end:  # `note` is inside `prev_note`, do not cut total duration
                        note.end = prev_note.end
                    for _prev_note in prev_notes:  # cut `prev_note` until `note` start
                        _prev_note.end = note.start

            # note.tie_next = None  # duration might be altered, and we do not control duration of the tied note

            prev_pitch_notes[note.midi_pitch] = [note]

    return score


def remove_duplicated_partitura_score_notes(score: pt.score.Score):
    for part in score.parts:
        notes = part.notes_tied
        i, prev_pitch_note = 0, {}
        while i < len(notes):
            note = notes[i]
            next_note = prev_pitch_note.get(note.midi_pitch, None)

            if next_note is not None:
                if note.start == next_note.start and note.duration_tied == next_note.duration_tied:
                    part.remove(next_note)
                    continue

            prev_pitch_note[note.midi_pitch] = note
            i += 1

    return score


def preprocess_partitura_score(
        score: pt.score.Score,
        unfold_repeats: bool = True,
        remove_grace_notes: bool = True,
        clean_duplicates: bool = True,
        cut_overlapped_notes: bool = True,
        voice_is_staff: bool = True
):
    if unfold_repeats and len(score.parts[0].repeats) > 0:
        score = pt.score.unfold_part_maximal(score)

    if remove_grace_notes:
        for part in score.parts:
            pt.score.remove_grace_notes(part)

    if clean_duplicates:
        score = remove_duplicated_partitura_score_notes(score)

    if cut_overlapped_notes:
        score = cut_overlapping_partitura_score_notes(score, trills=True)
        if clean_duplicates:
            score = remove_duplicated_partitura_score_notes(score)

    for part in score.parts:
        for note in part.notes:
            if voice_is_staff:
                note.voice = note.staff

            if note.duration <= 0:
                part.remove(note)

    return score


def partitura_score_to_midi(
        score: str | pt.score.Score,
        midi_path: str,
        unfold_repeats: bool = True,
        clean_duplicates: bool = True,
        cut_overlapped_notes: bool = True,
        downsample_ticks_per_quarter: int | None = None,
        ticks_per_quarter: int | None = 480,
        min_shift: float | None = 1 / 24,
        min_duration: float | None = 1 / 24
) -> Score:
    if not isinstance(score, pt.score.Score):
        score = pt.load_score(score)

    score = preprocess_partitura_score(
        score,
        unfold_repeats=unfold_repeats,
        clean_duplicates=clean_duplicates,
        cut_overlapped_notes=cut_overlapped_notes,
    )

    pt.save_score_midi(score, midi_path, part_voice_assign_mode=5, velocity=80, anacrusis_behavior="pad_bar")
    midi = Score(midi_path)

    midi = preprocess_midi(
        midi,
        to_single_track=False,
        clean_duplicates=True,
        cut_overlapped_notes=True,
        clean_short_notes=True,
        min_tick_shift=int(ticks_per_quarter * min_shift) if ticks_per_quarter is not None else None,
        min_tick_duration=int(ticks_per_quarter * min_duration) if ticks_per_quarter is not None else None,
        downsample_ticks_per_quarter=downsample_ticks_per_quarter,
        target_ticks_per_quarter=ticks_per_quarter
    )

    midi.dump_midi(midi_path)
    return midi


def load_performance_note_array(midi_path):
    midi_t = Score(midi_path)
    midi_s = midi_t.to("second")

    ppart_fields = [
        ("onset_sec", "f4"),
        ("duration_sec", "f4"),
        ("onset_tick", "i4"),
        ("duration_tick", "i4"),
        ("pitch", "i4"),
        ("velocity", "i4"),
        ("track", "i4"),
        ("channel", "i4"),
        ("id", "U256"),
    ]

    note_arr = np.zeros(midi_t.note_num(), dtype=ppart_fields)

    cur = 0
    for i, (track_t, track_s) in enumerate(zip(midi_t.tracks, midi_s.tracks)):
        note_soa_t = track_t.notes.numpy()
        note_soa_s = track_s.notes.numpy()
        num = len(track_t.notes)

        note_arr["onset_sec"][cur:cur + num] = note_soa_s["time"]
        note_arr["duration_sec"][cur:cur + num] = note_soa_s["duration"]
        note_arr["onset_tick"][cur:cur + num] = note_soa_t["time"]
        note_arr["duration_tick"][cur:cur + num] = note_soa_t["duration"]
        note_arr["pitch"][cur:cur + num] = note_soa_s["pitch"]
        note_arr["velocity"][cur:cur + num] = note_soa_s["velocity"]
        note_arr["track"][cur:cur + num] = np.full_like(note_soa_t["time"], fill_value=track_t.program)

        cur += num

    sort_ids = np.lexsort([note_arr["track"], note_arr["duration_sec"], note_arr["pitch"], note_arr["onset_sec"]])
    note_arr = note_arr[sort_ids]

    note_arr["id"] = [f"n{k}" for k in range(len(note_arr))]

    return note_arr
