from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np
import partitura as pt
from partitura.utils.music import symbolic_to_numeric_duration
from symusic import Score

from symupe.data.midi.preprocess import preprocess_midi


def cut_overlapping_partitura_score_notes(
    score: pt.score.Score,
    duplicate_max_duration: bool = True,
    min_shift: float | None = 1 / 32,  # in quarter notes
):
    for part in score.parts:
        tpq = part.first_point.quarter
        min_tick_shift = max(1, round(tpq * min_shift)) if min_shift is not None else None

        prev_pitch_notes = {}

        for note in part.notes_tied:
            note.end = note.start + (note.duration if note.tie_next is None else note.duration_tied)
            note.tie_next = None

            prev_notes = prev_pitch_notes.get(note.midi_pitch, None)
            if prev_notes is None:
                prev_pitch_notes[note.midi_pitch] = [note]
                continue

            prev_note = prev_notes[-1]

            prev_duration = prev_note.duration
            duration = note.duration

            if prev_note.start == note.start:
                # `note` and `prev_note` start at the same time
                duration = (
                    max(prev_duration, duration)
                    if duplicate_max_duration
                    else min(prev_duration, duration)
                )

                note.end = note.start + duration
                for _prev_note in prev_notes:
                    _prev_note.end = note.end
                prev_pitch_notes[note.midi_pitch].append(note)
                continue

            if prev_note.end > note.start:
                # `note` starts before `prev_note` ended

                if min_shift is not None and note.start.t - prev_note.start.t < min_tick_shift:
                    # previous note will be too short
                    note.end = max(prev_note.end, note.end)  # duplicate notes
                    note.start = prev_note.start

                    for _prev_note in prev_notes:
                        _prev_note.end = note.end
                    prev_pitch_notes[note.midi_pitch].append(note)
                    continue

                if prev_note.end > note.end:
                    # `note` is inside `prev_note`, do not cut total duration
                    note.end = prev_note.end

                for _prev_note in prev_notes:  # cut `prev_note` until `note` start
                    _prev_note.end = note.start

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


def expand_partitura_score_grace_notes(
    score: pt.score.Score,
    default_duration_ratio=0.125,
    max_steal_ratio=0.75,
):
    for part in score.parts:
        tpq = part.first_point.quarter

        # group all grace notes by their main note
        all_grace_notes = [note for note in part.notes_tied if isinstance(note, pt.score.GraceNote)]
        groups = defaultdict(list)
        for grace_note in all_grace_notes:
            grace_note.tie_next = None
            if grace_note.main_note and grace_note.main_note.start:
                groups[grace_note.main_note].append(grace_note)

        for main_note, grace_notes in groups.items():
            # store the original boundaries of the main note
            main_start = main_note.start.t
            main_end = main_note.end.t
            main_duration = main_end - main_start

            # separate by type
            acciaccaturas, appoggiaturas = [], []
            for grace_note in grace_notes:
                if getattr(grace_note, "grace_type", "acciaccatura") == "acciaccatura":
                    acciaccaturas.append(grace_note)
                else:
                    appoggiaturas.append(grace_note)

            # acciaccaturas (lead to the main note)
            next_start = main_start
            for grace_note in reversed(acciaccaturas):
                duration = max(1, round(tpq * default_duration_ratio))
                new_start = max(0, next_start - duration)
                part.remove(grace_note)
                part.add(grace_note, new_start, next_start)
                next_start = new_start

            # appoggiaturas (on the beat)
            if appoggiaturas:
                # calculate total time to steal
                total_req_duration = sum(
                    max(1, round(symbolic_to_numeric_duration(g.symbolic_duration, tpq)))
                    for g in appoggiaturas
                )

                # don't let the main note disappear
                allowed_steal = int(main_duration * max_steal_ratio)
                actual_steal = min(total_req_duration, allowed_steal)

                # compute scale factor
                scale_factor = actual_steal / total_req_duration if total_req_duration > 0 else 1.0

                # distribute the stolen time among the grace notes
                current_time = main_start
                for grace_note in appoggiaturas:
                    grace_duration = max(
                        1,
                        int(
                            round(symbolic_to_numeric_duration(grace_note.symbolic_duration, tpq))
                            * scale_factor
                        ),
                    )
                    new_end = current_time + grace_duration

                    part.remove(grace_note)
                    part.add(grace_note, current_time, new_end)
                    current_time = new_end

                # ensure final_start is at least 1 tick before the `main_end`
                final_main_start = min(current_time, main_end - 1)
                # ensure it doesn't start before the `main_start`
                final_main_start = max(final_main_start, main_start)

                part.remove(main_note)
                part.add(main_note, final_main_start, main_end)

    return score


def process_partitura_score_ornaments(
    score: pt.score.Score,
    min_notes_in_cluster: int = 3,
    max_pitch_difference: int = 2,
):
    for part in score.parts:
        all_notes = [n for n in part.notes_tied if not isinstance(n, pt.score.GraceNote)]
        hidden = sorted(
            [n for n in all_notes if not getattr(n, "is_printed", True)], key=lambda x: x.start
        )

        if not hidden:
            continue

        # group hidden notes into clusters
        active_clusters = []

        for h_note in hidden:
            assigned = False
            for cluster in active_clusters:
                last_note = cluster[-1]

                # check time contiguity and pitch proximity
                time_match = abs(h_note.start.t - last_note.end.t) < 1e-4
                pitch_match = abs(h_note.midi_pitch - last_note.midi_pitch) <= max_pitch_difference

                if time_match and pitch_match:
                    cluster.append(h_note)
                    assigned = True
                    break

            if not assigned:
                active_clusters.append([h_note])

        visible = [
            n
            for n in all_notes
            if not getattr(n, "is_cue", False) and getattr(n, "is_printed", True)
        ]

        # process each cluster
        removed_visible = set()
        for cluster in active_clusters:
            if len(cluster) < min_notes_in_cluster:
                continue  # too short to be a trill/ornament

            # get the set of pitches used in this realization (usually 2 pitches)
            realization_pitches = set(n.midi_pitch for n in cluster)
            cluster_start = cluster[0].start
            cluster_end = cluster[-1].end

            # target visible note placeholders: only if they share a pitch with the cluster
            placeholders = [
                v
                for v in visible
                if v not in removed_visible
                and v.midi_pitch in realization_pitches
                and (v.start < cluster_end and v.end > cluster_start)
            ]

            if placeholders:  # this cluster is officially a trill/mordent/ornament realization
                for h_note in cluster[:2]:
                    h_note.ornaments.append("_ornament_start")

                for h_note in cluster[-2:]:
                    h_note.ornaments.append("_ornament_end")

                for v_note in placeholders:
                    cluster[0].ornaments.extend(
                        [text for text in v_note.ornaments if text not in cluster[0].ornaments]
                    )
                    part.remove(v_note)
                    removed_visible.add(v_note)

    return score


def preprocess_partitura_score(
    score: pt.score.Score,
    unfold_repeats: bool = True,
    unfold_minimal: bool = False,
    remove_grace_notes: bool = False,
    expand_grace_notes: bool = True,
    process_ornaments: bool = True,
    clean_duplicates: bool = True,
    cut_overlapped_notes: bool = True,
    voice_is_staff: bool = True,
    recursion_depth: int = 10000,
):
    if unfold_repeats and len(score.parts[0].repeats) > 0:
        sys.setrecursionlimit(recursion_depth)

        if unfold_minimal:
            score = pt.score.unfold_part_minimal(score)
        else:
            score = pt.score.unfold_part_maximal(score)

    assert not remove_grace_notes or not expand_grace_notes, (
        "Only one of `remove_grace_notes` or `expand_grace_notes` must be True"
    )

    if remove_grace_notes:
        for part in score.parts:
            pt.score.remove_grace_notes(part)
    elif expand_grace_notes:
        score = expand_partitura_score_grace_notes(score)

    if process_ornaments:
        score = process_partitura_score_ornaments(score)

    if clean_duplicates:
        score = remove_duplicated_partitura_score_notes(score)

    if cut_overlapped_notes:
        score = cut_overlapping_partitura_score_notes(score)
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
    unfold_minimal: bool = False,
    remove_grace_notes: bool = False,
    expand_grace_notes: bool = True,
    process_ornaments: bool = True,
    clean_duplicates: bool = True,
    cut_overlapped_notes: bool = True,
    downsample_ticks_per_quarter: int | None = None,
    ticks_per_quarter: int | None = 480,
    min_shift: float | None = 1 / 24,
    min_duration: float | None = 1 / 24,
) -> Score:
    if not isinstance(score, pt.score.Score):
        score = pt.load_score(score)

    score = preprocess_partitura_score(
        score,
        unfold_repeats=unfold_repeats,
        unfold_minimal=unfold_minimal,
        remove_grace_notes=remove_grace_notes,
        expand_grace_notes=expand_grace_notes,
        process_ornaments=process_ornaments,
        clean_duplicates=clean_duplicates,
        cut_overlapped_notes=cut_overlapped_notes,
    )

    pt.save_score_midi(
        score, midi_path, part_voice_assign_mode=5, velocity=80, anacrusis_behavior="pad_bar"
    )
    midi = Score(midi_path)

    midi = preprocess_midi(
        midi,
        to_single_track=False,
        clean_duplicates=True,
        cut_overlapped_notes=True,
        clean_short_notes=True,
        min_tick_shift=(
            int(ticks_per_quarter * min_shift) if ticks_per_quarter is not None else None
        ),
        min_tick_duration=(
            int(ticks_per_quarter * min_duration) if ticks_per_quarter is not None else None
        ),
        downsample_ticks_per_quarter=downsample_ticks_per_quarter,
        target_ticks_per_quarter=ticks_per_quarter,
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

        note_arr["onset_sec"][cur : cur + num] = note_soa_s["time"]
        note_arr["duration_sec"][cur : cur + num] = note_soa_s["duration"]
        note_arr["onset_tick"][cur : cur + num] = note_soa_t["time"]
        note_arr["duration_tick"][cur : cur + num] = note_soa_t["duration"]
        note_arr["pitch"][cur : cur + num] = note_soa_s["pitch"]
        note_arr["velocity"][cur : cur + num] = note_soa_s["velocity"]
        note_arr["track"][cur : cur + num] = np.full_like(
            note_soa_t["time"], fill_value=track_t.program
        )

        cur += num

    sort_ids = np.lexsort(
        [note_arr["track"], note_arr["duration_sec"], note_arr["pitch"], note_arr["onset_sec"]]
    )
    note_arr = note_arr[sort_ids]

    note_arr["id"] = [f"n{k}" for k in range(len(note_arr))]

    return note_arr
