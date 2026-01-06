""" MusicXML performance direction data parsing. """
from __future__ import annotations

import partitura as pt

from .articulation import ARTICULATION_PREFIX, ArticulationClasses, ARTICULATION_DIRECTIONS_MAPS
from .dynamic import DYNAMIC_PREFIX, DYNAMIC_DIRECTIONS_MAPS
from .tempo import TEMPO_PREFIX, TEMPO_DIRECTIONS_MAPS
from .words import extract_direction_with_class, word_regularization


def parse_directions(
        score: pt.score.Score,
        delete_duplicates: bool = False,
        ticks_scale: float = 1.
):
    # anacrusis
    xml_shift = 0
    if len(score.parts[0].measures) > 1:
        xml_shift = score.parts[0].measures[1].duration - score.parts[0].measures[0].duration

    score_directions = []
    for part in score.parts:
        part_directions = []

        # dynamic markings
        for d in part.dynamics:
            dkey = DYNAMIC_PREFIX
            dtype = extract_direction_with_class(d.text, DYNAMIC_DIRECTIONS_MAPS)
            if dtype is None:
                dtype = extract_direction_with_class(d.text, TEMPO_DIRECTIONS_MAPS)
                if dtype is not None:
                    dkey = TEMPO_PREFIX

            if dtype is not None:
                d_dict = {
                    "type": f"{dkey}/{dtype}",
                    "start": d.start.t,
                    "end": d.end.t if d.end else d.start.t,
                    "raw_text": d.raw_text
                }
                part_directions.append(d_dict)

        # tempo markings
        for d in part.tempo_directions:
            dtype = extract_direction_with_class(d.text, TEMPO_DIRECTIONS_MAPS)
            if "più mosso" in d.raw_text.lower():
                dtype = "increase/più mosso"
            elif "meno mosso" in d.raw_text.lower():
                dtype = "decrease/meno mosso"
            if dtype is not None:
                d_dict = {
                    "type": f"{TEMPO_PREFIX}/{dtype}",
                    "start": d.start.t,
                    "end": d.end.t if d.end else None,
                    "raw_text": d.raw_text
                }
                part_directions.append(d_dict)

        # words that might be tempos
        for d in part.iter_all(pt.score.Words):
            word = word_regularization(d.text)
            dtype = extract_direction_with_class(word, TEMPO_DIRECTIONS_MAPS)
            if dtype is not None:
                d_dict = {
                    "type": f"{TEMPO_PREFIX}/{dtype}",
                    "start": d.start.t,
                    "end": d.end.t if d.end else None,
                    "raw_text": d.text
                }
                part_directions.append(d_dict)

        # process start/end of tempo directions without an `end`
        part_directions = list(sorted(part_directions, key=lambda d: (d["start"], d["type"], d["end"])))

        endless_tempo = None
        prev_tempo = None
        for d_dict in part_directions:
            if "tempo" in d_dict["type"]:
                if prev_tempo is not None and prev_tempo["end"] is not None and prev_tempo["end"] > d_dict["start"]:
                    prev_tempo["end"] = d_dict["start"]
                prev_tempo = d_dict

                if endless_tempo is not None:
                    endless_tempo["end"] = d_dict["start"]
                    endless_tempo = None
                if d_dict["end"] is None:
                    endless_tempo = d_dict
        if endless_tempo is not None:
            endless_tempo["end"] = part.last_point.t

        # text articulations
        for d in part.articulations:
            dtype = extract_direction_with_class(
                d.text, ARTICULATION_DIRECTIONS_MAPS, classes=[ArticulationClasses.INTERVAL]
            )
            if dtype is not None:
                d_dict = {
                    "type": f"{ARTICULATION_PREFIX}/{dtype}",
                    "start": d.start.t,
                    "end": d.end.t if d.end else None,
                    "raw_text": d.raw_text
                }
                part_directions.append(d_dict)

        # note articulations
        for note in part.notes:
            for art in note.articulations:
                d_dict = {
                    "type": f"{ARTICULATION_PREFIX}/note/{art}",
                    "start": note.start.t,
                    "end": note.end.t,
                    "pitch": note.midi_pitch,
                }
                part_directions.append(d_dict)

            if note.fermata:
                d_dict = {
                    "type": f"{ARTICULATION_PREFIX}/note/fermata",
                    "start": note.start.t,
                    "end": note.end.t,
                    "pitch": note.midi_pitch,
                }
                part_directions.append(d_dict)

        # note ornaments
        for note in part.notes:
            for orn in note.ornaments:
                d_dict = {
                    "type": f"{ARTICULATION_PREFIX}/ornament/{orn}",
                    "start": note.start.t,
                    "end": note.end.t,
                    "pitch": note.midi_pitch,
                }
                part_directions.append(d_dict)

        # slurs
        for note in part.notes:
            if note.slur_starts:
                main_note = note.main_note if isinstance(note, pt.score.GraceNote) else None
                end = main_note.end if main_note is not None else note.end
                d_dict = {
                    "type": f"{ARTICULATION_PREFIX}/slur/slur-start",
                    "start": note.start.t,
                    "end": end.t if end is not None else note.start.t,
                    "pitch": note.midi_pitch,
                    "pitch_main": main_note.midi_pitch if main_note is not None else None
                }
                part_directions.append(d_dict)
            if note.slur_stops:
                d_dict = {
                    "type": f"{ARTICULATION_PREFIX}/slur/slur-stop",
                    "start": note.start.t,
                    "end": note.end.t,
                    "pitch": note.midi_pitch,
                    "pitch_main": None
                }
                part_directions.append(d_dict)

        # scale xml positions if needed
        if xml_shift != 0 or ticks_scale != 1.:
            for d_dict in part_directions:
                d_dict["start"] = int(ticks_scale * (d_dict["start"] + xml_shift))
                if d_dict["end"]:
                    d_dict["end"] = int(ticks_scale * (d_dict["end"] + xml_shift))

        # sort directions
        part_directions = list(sorted(part_directions, key=lambda d: (d["start"], d["type"], d["end"])))

        if delete_duplicates:
            i = 0
            while i < len(part_directions) - 1:
                d_dict, next_d_dict = part_directions[i], part_directions[i + 1]
                if d_dict["type"] == next_d_dict["type"] and d_dict["start"] == next_d_dict["start"]:
                    del part_directions[i + 1]
                    continue
                i += 1

        score_directions.append(part_directions)

    return score_directions


def get_part_directions(part):
    directions = []
    for measure_idx, measure in enumerate(part.measures):
        for direction in measure.directions:
            direction.type["measure"] = measure_idx
            directions.append(direction)

    directions.sort(key=lambda x: x.xml_position)
    cleaned_direction = []
    for i, d in enumerate(directions):
        if d.type is None:
            continue

        if d.type["type"] == "none":
            for j in range(i):
                prev_dir = directions[i - j - 1]
                if "number" in prev_dir.type.keys():
                    prev_key = prev_dir.type["type"]
                    prev_num = prev_dir.type["number"]
                else:
                    continue
                if prev_num == d.type["number"]:
                    if prev_key == "crescendo":
                        d.type["type"] = "crescendo"
                        break
                    elif prev_key == "diminuendo":
                        d.type["type"] = "diminuendo"
                        break
        cleaned_direction.append(d)

    return cleaned_direction
