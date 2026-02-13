from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from symusic import Score
from symusic.core import Tick

from symupe.utils import find_closest, ExplicitEnum
from ..midi.timing import MIDITimeMapper
from ..music_constants import sitch2pitch, pitch2sitch

TIME_TOLERANCE = 1e-3
TIME_PRECISION = 6
INVALID_TIME_PITCH_TUPLE = 2 * (10000,)


@dataclass
class AlignmentNote:
    idx: int
    pitch: int
    start: float
    end: float | None = None
    velocity: float | None = None

    @property
    def duration(self):
        return self.end - self.start

    def __eq__(self, other):
        return other is not None \
            and self.pitch == other.pitch \
            and math.fabs(self.start - other.start) < TIME_TOLERANCE \
            and (self.end is None or other.end is None or math.fabs(self.end - other.end) < TIME_TOLERANCE)

    def __hash__(self):
        note_hash = hash(self.pitch) + hash(round(self.start, TIME_PRECISION // 2))
        note_hash += hash(round(self.end, TIME_PRECISION // 2)) if self.end is not None else 0
        return note_hash

    def to_print_str(self):
        return (
            f"{self.idx}"
            f"\t{round(self.start, TIME_PRECISION)}"
            f"\t{round(self.end or -1, TIME_PRECISION)}"
            f"\t{pitch2sitch(self.pitch)}"
        )


@dataclass
class AlignmentPair:
    score_note: AlignmentNote | None = None
    perf_note: AlignmentNote | None = None

    def __eq__(self, other):
        return self.score_note == other.score_note and self.perf_note == other.perf_note

    def __hash__(self):
        return hash(self.score_note) + hash(self.perf_note)


@dataclass
class PositionPair:
    idx: int
    score_tick: int
    perf_time: float
    pitch: int
    ticks_per_quarter: int


class AlignmentFileType(ExplicitEnum):
    ALIGN = "align"
    CORRESP = "corresp"


class Alignment:
    _tpmult = 100000

    def __init__(
            self,
            path: str | None,
            pairs: list[AlignmentPair] | None = None,
            score_name: str | None = None,
            perf_name: str | None = None,
            score_first: bool = True,
            clean_duplicates: bool = True,
            clean_mismatched_pitches: bool = True,
            score_index_range: tuple[int, int] | None = None,
            filetype: str = "align"
    ):
        assert path is not None or pairs is not None

        if pairs is None:
            (self.score_name, self.perf_name), self.pairs = self.load(
                filepath=path,
                clean_duplicates=clean_duplicates,
                clean_mismatched_pitches=clean_mismatched_pitches,
                filetype=filetype
            )
        else:
            (self.score_name, self.perf_name), self.pairs = (score_name, perf_name), pairs

        self.preprocess_pairs(score_first=score_first, clean_duplicates=False)

        self.score_index_range = score_index_range

    @staticmethod
    def load(
            filepath: str,
            clean_duplicates: bool = True,
            clean_mismatched_pitches: bool = True,
            filetype: str = "align"
    ):
        with open(filepath, "r") as f:
            # load score/performance name metadata
            if filetype == AlignmentFileType.ALIGN:
                meta = f.readline().strip().split("\t")
                if len(meta) == 2:  # backward compatibility
                    meta = ["P-S"] + meta
                score_first = meta[0] == "S-P"
                score_name, perf_name = (meta[1], meta[2]) if score_first else (meta[2], meta[1])
            else:  # no name meta
                score_first = False
                score_name, perf_name = None, None

            _perf_ids, _score_ids = set(), set()

            alignment = []
            for line in f:
                items = line.strip().split("\t")

                if filetype == AlignmentFileType.CORRESP:
                    if line.startswith("//") or len(items) != 10:
                        continue

                if filetype == AlignmentFileType.ALIGN:
                    note_1, note_2 = map(
                        lambda ni: AlignmentNote(
                            idx=int(ni[0]),
                            start=float(ni[1]),
                            end=float(ni[2]) if ni[2] != -1 else None,
                            pitch=sitch2pitch(ni[3])
                        ) if ni[3] != "*" else None,
                        (items[:4], items[4:8])
                    )
                elif filetype == AlignmentFileType.CORRESP:
                    note_1, note_2 = map(
                        lambda ni: AlignmentNote(
                            idx=int(ni[0]),
                            start=float(ni[1]),
                            pitch=int(ni[3])
                        ) if ni[0] != "*" else None,
                        (items[:5], items[5:10])
                    )

                s_note, p_note = (note_1, note_2) if score_first else (note_2, note_1)

                if clean_duplicates:
                    if (p_note and p_note.idx in _perf_ids) or (s_note and s_note.idx in _score_ids):
                        continue  # duplicate

                    if p_note:
                        _perf_ids.add(p_note.idx)

                    if s_note:
                        _score_ids.add(s_note.idx)

                if clean_mismatched_pitches and s_note and p_note and s_note.pitch != p_note.pitch:
                    p_note = None

                alignment.append(AlignmentPair(s_note, p_note))

            del _perf_ids, _score_ids

        return (score_name, perf_name), alignment

    @staticmethod
    def from_midis(
            score_midi: Score,
            perf_midi: Score,
            alignment: np.ndarray | None = None
    ):
        def build_alignment_notes(midi):
            time_mapper = MIDITimeMapper(midi)

            note_soa = midi.tracks[0].notes.numpy()
            start_times = time_mapper.t2s(note_soa["time"]).astype(float)
            end_times = time_mapper.t2s(note_soa["time"] + note_soa["duration"]).astype(float)
            pitches = note_soa["pitch"]

            return [
                AlignmentNote(
                    idx=i,
                    start=start_times[i],
                    end=end_times[i],
                    pitch=pitches[i]
                )
                for i in range(len(midi.tracks[0].notes))
            ]

        score_notes = build_alignment_notes(score_midi)
        perf_notes = build_alignment_notes(perf_midi)

        pairs = []
        alignment = np.arange(len(score_notes)) if alignment is None else alignment
        for score_idx, perf_idx in enumerate(alignment):
            pairs.append(AlignmentPair(score_notes[score_idx], perf_notes[perf_idx]))

        return Alignment(path=None, pairs=pairs)

    def write(
            self,
            filepath: str,
            score_first: bool = False
    ):
        _empty = "-1\t-1\t-1\t*"
        self.preprocess_pairs(sort=True, score_first=score_first, clean_duplicates=False)

        with open(filepath, "w") as f:
            if score_first:
                meta = f"S-P\t{self.score_name}\t{self.perf_name}\n"
            else:
                meta = f"P-S\t{self.perf_name}\t{self.score_name}\n"
            f.write(meta)

            for pair in self.pairs:
                s_note, p_note = pair.score_note, pair.perf_note
                s_str = s_note.to_print_str() if s_note is not None else _empty
                p_str = p_note.to_print_str() if p_note is not None else _empty
                f.write(f"{s_str}\t{p_str}\n" if score_first else f"{p_str}\t{s_str}\n")

    def preprocess_pairs(self, sort: bool = True, score_first: bool = True, clean_duplicates: bool = True):
        if clean_duplicates:
            self.pairs = list(set(self.pairs))

        if sort:
            if score_first:
                self.pairs.sort(key=lambda p: (
                    (p.score_note.start, p.score_note.pitch) if p.score_note else INVALID_TIME_PITCH_TUPLE,
                    (p.perf_note.start, p.perf_note.pitch) if p.perf_note else INVALID_TIME_PITCH_TUPLE
                ))
            else:
                self.pairs.sort(key=lambda p: (
                    (p.perf_note.start, p.perf_note.pitch) if p.perf_note else INVALID_TIME_PITCH_TUPLE,
                    (p.score_note.start, p.score_note.pitch) if p.score_note else INVALID_TIME_PITCH_TUPLE
                ))

        self._score_data = self._get_pitch_time_data(
            pitches=np.array(list(map(lambda x: x.score_note.pitch if x.score_note else -1, self.pairs))),
            times=np.array(list(map(lambda x: x.score_note.start if x.score_note else -1, self.pairs)))
        )
        self._perf_data = self._get_pitch_time_data(
            pitches=np.array(list(map(lambda x: x.perf_note.pitch if x.perf_note else -1, self.pairs))),
            times=np.array(list(map(lambda x: x.perf_note.start if x.perf_note else -1, self.pairs)))
        )

        return self

    def compare_notes_with_midi(
            self,
            score_midi: Score | None = None,
            perf_midi: Score | None = None,
            clean_unmatched: bool = True,
            fill_note_attributes: bool = False
    ):
        def _fill_attributes(midi: Score, indices: np.ndarray, is_score_midi: bool = False):
            notes = midi.tracks[0].notes
            time_mapper = MIDITimeMapper(midi) if isinstance(midi.ttype, Tick) else None
            for i, pair in enumerate(self.pairs):
                pair_note = pair.score_note if is_score_midi else pair.perf_note
                if pair_note is None:
                    continue
                note = notes[indices[i]]
                if isinstance(midi.ttype, Tick):
                    pair_note.start = round(time_mapper.t2s(note.start), TIME_PRECISION)
                    pair_note.end = round(time_mapper.t2s(note.end), TIME_PRECISION)
                else:
                    pair_note.start = round(note.start, TIME_PRECISION)
                    pair_note.end = round(note.end, TIME_PRECISION)
                pair_note.velocity = note.velocity

        if score_midi is not None:
            pair_to_score, _ = self.match_with_midi(midi=score_midi, is_score_midi=True)

            if clean_unmatched:  # remove score notes not found in score midi
                for idx in np.where(pair_to_score == -1)[0]:
                    self.pairs[idx].score_note = None

            if fill_note_attributes:
                _fill_attributes(score_midi, pair_to_score, is_score_midi=True)

        if perf_midi is not None:
            (pair_to_note, _), (pair_data, midi_data) = self.match_with_midi(
                midi=perf_midi, is_score_midi=False, return_midi_data=True
            )

            if clean_unmatched:  # remove performance notes not found in performance midi
                diff = np.abs(midi_data[pair_to_note] - pair_data)
                for idx in np.where(diff > TIME_TOLERANCE)[0]:
                    self.pairs[idx].perf_note = None

            if fill_note_attributes:
                _fill_attributes(perf_midi, pair_to_note, is_score_midi=False)

        self.preprocess_pairs(sort=False, clean_duplicates=False)

        return self

    def create_pair_from_midi_notes(
            self,
            index: int,
            score_midi: Score,
            perf_midi: Score,
            score_note_idx: int = 0,
            perf_note_idx: int = 0,
            replace: bool = False
    ):
        score_note = score_midi.tracks[0].notes[score_note_idx]
        perf_note = perf_midi.tracks[0].notes[perf_note_idx]

        if score_note.pitch == perf_note.pitch:
            score_time_mapper = MIDITimeMapper(score_midi)
            perf_time_mapper = MIDITimeMapper(perf_midi)
            pair = AlignmentPair(
                score_note=AlignmentNote(
                    idx=score_note_idx,
                    start=round(score_time_mapper.t2s(score_note.start), TIME_PRECISION),
                    end=round(score_time_mapper.t2s(score_note.end), TIME_PRECISION),
                    pitch=score_note.pitch,
                    velocity=score_note.velocity
                ),
                perf_note=AlignmentNote(
                    idx=perf_note_idx,
                    start=round(perf_time_mapper.t2s(perf_note.start), TIME_PRECISION),
                    end=round(perf_time_mapper.t2s(perf_note.start), TIME_PRECISION),
                    pitch=perf_note.pitch,
                    velocity=perf_note.velocity
                )
            )

            if not replace:
                self.pairs.insert(index, AlignmentPair())
            self.pairs[index] = pair

            self.preprocess_pairs(sort=index > 0, clean_duplicates=False)
            return pair

        return None

    def _get_pitch_time_data(self, pitches: np.ndarray, times: np.ndarray, precision: int = 4):
        return self._tpmult * pitches + np.round(times, precision)

    def _get_midi_data(self, midi: Score):
        note_soa = midi.tracks[0].notes.numpy()
        midi_pitches = note_soa["pitch"]
        midi_times = note_soa["time"]

        if isinstance(midi.ttype, Tick):
            time_mapper = MIDITimeMapper(midi)
            midi_times = time_mapper.t2s(midi_times)

        return self._get_pitch_time_data(midi_pitches, midi_times)

    def match_with_midi(
            self,
            midi: Score,
            is_score_midi: bool = True,
            fix_non_unique: bool = True,
            return_midi_data: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        midi_data = self._get_midi_data(midi)
        pair_data = self._score_data if is_score_midi else self._perf_data

        indices = np.argsort(midi_data)
        if len(midi_data) == 0:
            pair_to_note = np.full(len(pair_data), fill_value=-1)
        else:
            pair_to_note = indices[find_closest(midi_data[indices], pair_data)]

        pair_to_note[pair_data < 0] = -1

        if fix_non_unique:
            ids, counts = np.unique(pair_to_note, return_counts=True)
            nonunique_ids = ids[counts > 1]

            unused_ids = np.arange(midi_data.shape[0])
            unused_ids = unused_ids[~np.isin(unused_ids, ids)]

            for idx in nonunique_ids:
                if idx == -1:
                    continue
                repeat_pair_ids = np.where(pair_to_note == idx)[0]
                for rep_idx in repeat_pair_ids[1:]:
                    for i, un_idx in enumerate(unused_ids):
                        pair_pitch, pair_time = pair_data[rep_idx] // self._tpmult, pair_data[rep_idx] % self._tpmult
                        midi_pitch, midi_time = midi_data[un_idx] // self._tpmult, midi_data[un_idx] % self._tpmult
                        if abs(pair_time - midi_time) < 1 and pair_pitch == midi_pitch:
                            pair_to_note[rep_idx] = un_idx
                            unused_ids = np.delete(unused_ids, i)
                            break
                        else:
                            pair_to_note[rep_idx] = -1  # not found, set to unmatched

        note_to_pair = np.full(len(midi_data), fill_value=-1)
        if len(note_to_pair) > 0:
            note_to_pair[pair_to_note] = np.arange(len(pair_to_note))

        if return_midi_data:
            return (pair_to_note, note_to_pair), (pair_data, midi_data)
        else:
            return (pair_to_note, note_to_pair)

    def build_position_pairs(self, score_midi: Score):
        pair_to_score, _ = self.match_with_midi(score_midi)

        if not isinstance(score_midi.ttype, Tick):
            score_midi = score_midi.to("tick")

        note_soa = score_midi.tracks[0].notes.numpy()
        score_ticks = note_soa["time"][pair_to_score]

        position_pairs = []
        for i, pair in enumerate(self.pairs):
            if pair.score_note and pair.perf_note:
                position_pairs.append(
                    PositionPair(
                        idx=i,
                        score_tick=score_ticks[i],
                        perf_time=pair.perf_note.start,
                        pitch=pair.score_note.pitch,
                        ticks_per_quarter=score_midi.ticks_per_quarter
                    )
                )

        return position_pairs

    def build_score_to_perf_note_alignment(
            self,
            score_midi: Score,
            perf_midi: Score
    ) -> np.ndarray:
        _, score_note_to_pair = self.match_with_midi(score_midi, is_score_midi=True)
        pair_to_perf_note, _ = self.match_with_midi(perf_midi, is_score_midi=False)
        score_to_perf_note = pair_to_perf_note[score_note_to_pair]
        return score_to_perf_note

    def clean_midi(self, midi: Score, is_score_midi: bool = True):
        pair_to_midi, _ = self.match_with_midi(midi, is_score_midi=is_score_midi)

        notes = midi.tracks[0].notes
        remove_ids = np.where(~np.isin(np.arange(len(notes)), pair_to_midi))[0].tolist()

        if is_score_midi:
            for p, idx in zip(self.pairs, pair_to_midi):
                if p.score_note is not None and p.perf_note is None:
                    remove_ids.append(idx)
        else:
            for p, idx in zip(self.pairs, pair_to_midi):
                if p.score_note is None and p.perf_note is not None:
                    remove_ids.append(idx)

        remove_ids = sorted(set(remove_ids))
        for i, idx in enumerate(remove_ids):
            del notes[idx - i]

        return midi

    def delete_empty_pairs(self, no_score_note: bool = True, no_perf_note: bool = False):
        if no_score_note:
            self.pairs = list(filter(lambda p: p.score_note is not None, self.pairs))

        if no_perf_note:
            self.pairs = list(filter(lambda p: p.perf_note is not None, self.pairs))

        self.preprocess_pairs(sort=False, clean_duplicates=False)

        return self

    def fill_missing_score_notes(
            self,
            score_midi: Score,
            start_idx: int | None = None,
            end_idx: int | None = None
    ):
        pair_to_score, _ = self.match_with_midi(score_midi, is_score_midi=True)

        notes = score_midi.tracks[0].notes
        fill_ids = np.where(~np.isin(np.arange(len(notes)), pair_to_score))[0]

        if start_idx is not None:
            fill_ids = fill_ids[fill_ids >= start_idx]
        if end_idx is not None:
            fill_ids = fill_ids[fill_ids <= end_idx]

        if len(fill_ids) > 0:
            time_mapper = MIDITimeMapper(score_midi)

            for idx in fill_ids:
                note = notes[idx]
                s_note = AlignmentNote(
                    idx=-idx,
                    start=time_mapper.t2s(note.start),
                    end=time_mapper.t2s(note.end),
                    pitch=note.pitch,
                    velocity=note.velocity
                )
                self.pairs.append(AlignmentPair(s_note, perf_note=None))

            self.preprocess_pairs(clean_duplicates=False)

        return self

    def update_pair_note_ids(self, midi: Score, is_score_midi: bool = True):
        pair_to_note, _ = self.match_with_midi(midi, is_score_midi=is_score_midi)
        for pair, idx in zip(self.pairs, pair_to_note):
            pair_note = pair.score_note if is_score_midi else pair.perf_note
            if pair_note is None:
                continue

            pair_note.idx = idx if pair_note.idx >= 0 else -idx

    def shift_notes(
            self,
            time_shift: float = 0.,
            offset: float = 0.,
            score_notes: bool = False,
            shift_indices: np.ndarray | None = None
    ):
        ids = []
        if shift_indices is None:
            for i, pair in enumerate(self.pairs):
                note = pair.score_note if score_notes else pair.perf_note
                if note is not None and note.start >= offset:
                    note.start += time_shift
                    note.end += time_shift
                    ids.append(i)
        else:
            for idx in shift_indices:
                note = self.pairs[idx].score_note if score_notes else self.pairs[idx].perf_note
                if note is not None:
                    note.start += time_shift
                    note.end += time_shift

        self.preprocess_pairs(sort=False, clean_duplicates=False)
        return np.array(ids)

    def join_with(self, other):
        _score_notes = set(map(lambda p: p.score_note, self.pairs))
        _perf_notes = set(map(lambda p: p.perf_note, self.pairs))

        new_pairs = False
        for pair in other.pairs:
            if pair.score_note not in _score_notes and pair.perf_note not in _perf_notes:
                self.pairs.append(pair)
                new_pairs = True

        if new_pairs:
            self.preprocess_pairs()

        return self

    def __getitem__(self, idx: int):
        return self.pairs[idx] if idx < len(self) else None

    def __len__(self):
        return len(self.pairs)

    @property
    def num_full_pairs(self) -> int:
        return sum(map(lambda p: int(p.score_note is not None and p.perf_note is not None), self.pairs))

    @property
    def start_index(self):
        if self.score_index_range is None:
            start_idx = 1e10
            for pair in self.pairs:
                if pair.score_note is not None and pair.perf_note is not None:
                    start_idx = min(start_idx, pair.score_note.idx)
        else:
            start_idx = self.score_index_range[0]
        return start_idx

    @property
    def end_index(self):
        if self.score_index_range is None:
            end_idx = 0
            for pair in self.pairs:
                if pair.score_note is not None and pair.perf_note is not None:
                    end_idx = max(end_idx, pair.score_note.idx)
        else:
            end_idx = self.score_index_range[1]
        return end_idx
