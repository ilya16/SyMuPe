"""Alignment storing and processing logic."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path

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
    """Symbolic representation of a MIDI note $n = (p, o, d, v)$ used for alignment.

    Attributes:
        idx: The unique index of the note within its parent MIDI track.
        pitch: MIDI pitch $p$.
        start: Onset time $o$ (seconds).
        end: Offset time $o + d$ (seconds).
        velocity: MIDI velocity $v$.
    """

    idx: int
    pitch: int
    start: float
    end: float | None = None
    velocity: float | None = None

    @property
    def duration(self) -> float | None:
        return self.end - self.start if self.end is not None else None

    def __eq__(self, other) -> bool:
        if other is None:
            return False
        return (
            self.pitch == other.pitch
            and math.fabs(self.start - other.start) < TIME_TOLERANCE
            and (
                self.end is None
                or other.end is None
                or math.fabs(self.end - other.end) < TIME_TOLERANCE
            )
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.pitch,
                round(self.start, TIME_PRECISION // 2),
                round(self.end, TIME_PRECISION // 2) if self.end is not None else None,
            )
        )

    def to_print_str(self) -> str:
        """Formats note data for standard `.txt` alignment files."""
        return (
            f"{self.idx}"
            f"\t{round(self.start, TIME_PRECISION)}"
            f"\t{round(self.end or -1, TIME_PRECISION)}"
            f"\t{pitch2sitch(self.pitch)}"
        )

    def to_array(self) -> list[int | float | None]:
        """Converts note to a list for numpy/compressed storage."""
        return [
            self.idx,
            self.pitch,
            self.start,
            self.end if self.end is not None else -1.0,
        ]

    @classmethod
    def from_array(cls, row) -> AlignmentNote | None:
        """Reconstructs a note from a numpy array row."""
        if row[0] == -1:
            return None
        return cls(
            idx=int(row[0]),
            pitch=int(row[1]),
            start=float(row[2]),
            end=float(row[3]) if row[3] != -1 else None,
        )


@dataclass
class AlignmentPair:
    """Represents a single alignment link $a_{ij} = (y_i, x_j)$.

    A pair connects a score note to a performance note.
    If either is ``None``, it represents an insertion or deletion.
    """

    score_note: AlignmentNote | None = None
    perf_note: AlignmentNote | None = None

    def __eq__(self, other) -> bool:
        return self.score_note == other.score_note and self.perf_note == other.perf_note

    def __hash__(self) -> int:
        return hash((hash(self.score_note), hash(self.perf_note)))


@dataclass
class PositionPair:
    """A simplified note representation for temporal alignment between score and performance.

    Contains the mapping between a score beat (ticks) and a performance time (seconds).
    """

    idx: int
    score_tick: int
    perf_time: float
    pitch: int
    ticks_per_quarter: int


class AlignmentFileType(ExplicitEnum):
    """Supported alignment file standards."""

    ALIGN = "align"
    COMPRESSED = "compressed"
    CORRESP = "corresp"


class AlignmentFormat(ExplicitEnum):
    """File extensions associated with alignment types."""

    ALIGN = ".txt"
    COMPRESSED = ".npz"
    CORRESP = "_corresp.txt"


class Alignment:
    """A collection of matched score and performance notes $A = \{a_{ij}\}$.

    This class provides the core interface for loading, saving, and processing note-level alignments.
    It supports multiple formats, including the standardized `.npz` compressed format.
    """

    def __init__(
        self,
        pairs: list[AlignmentPair],
        score_name: str = None,
        perf_name: str = None,
    ):
        """Initializes an Alignment object.

        Args:
            pairs: List of :class:`AlignmentPair` objects.
            score_name: Metadata identifier for the score.
            perf_name: Metadata identifier for the performance.
        """
        assert isinstance(pairs, list)
        self.pairs = pairs
        self.score_name = score_name
        self.perf_name = perf_name

        self._num_full_pairs = None
        self._score_data = None
        self._perf_data = None

    # properties

    def __getitem__(self, idx: int) -> AlignmentPair | None:
        return self.pairs[idx] if idx < len(self) else None

    def __len__(self) -> int:
        return len(self.pairs)

    @property
    def num_full_pairs(self) -> int:
        """Returns the number of matched notes $N_m$ (where both score and performance notes exist)."""
        if self._num_full_pairs is None:
            self._num_full_pairs = sum(
                1 for pair in self.pairs if pair.score_note and pair.perf_note
            )
        return self._num_full_pairs

    @property
    def score_note_array(self) -> np.ndarray:
        """Vectorized representation of score-side notes in the alignment."""
        if self._score_data is None:
            self._score_data = np.array(
                [
                    pair.score_note.to_array() if pair.score_note else [-1, -1, -1, -1]
                    for pair in self.pairs
                ]
            )
        return self._score_data

    @property
    def perf_note_array(self) -> np.ndarray:
        """Vectorized representation of performance-side notes in the alignment."""
        if self._perf_data is None:
            self._perf_data = np.array(
                [
                    pair.perf_note.to_array() if pair.perf_note else [-1, -1, -1, -1]
                    for pair in self.pairs
                ]
            )
        return self._perf_data

    def clear_cache(self) -> None:
        """Invalidates internal cached arrays after alignment modifications."""
        self._num_full_pairs = None
        self._score_data = None
        self._perf_data = None

    @property
    def start_index(self) -> int | None:
        """The index of the first matched score note."""
        if self.num_full_pairs == 0:
            return None
        return min(pair.score_note.idx for pair in self.pairs if pair.score_note and pair.perf_note)

    @property
    def end_index(self) -> int | None:
        """The index of the last matched score note."""
        if self.num_full_pairs == 0:
            return None
        return max(pair.score_note.idx for pair in self.pairs if pair.score_note and pair.perf_note)

    # loading/saving methods

    @classmethod
    def from_file(cls, path: str | Path) -> Alignment:
        """Loads an alignment from disk.

        Supports `.npz` (PianoCoRe), `.txt` (standard align), and `_corresp.txt` formats.
        """
        path = Path(path)
        if path.suffix == AlignmentFormat.COMPRESSED:
            return cls._load_compressed(path)
        elif path.name.endswith(AlignmentFormat.CORRESP.value):
            return cls._load_text(path, filetype=AlignmentFileType.CORRESP)
        else:
            return cls._load_text(path, filetype=AlignmentFileType.ALIGN)

    @classmethod
    def from_bytes(cls, data: bytes) -> Alignment:
        """Loads a compressed alignment from memory."""
        return cls._load_compressed(data)

    @classmethod
    def _load_text(
        cls,
        path: str | Path,
        filetype: str | AlignmentFileType = AlignmentFileType.ALIGN,
    ) -> Alignment:
        """Internal parser for text-based alignment formats."""
        is_align = filetype == AlignmentFileType.ALIGN

        with open(path, "r") as f:
            if is_align:
                header = f.readline().strip().split("\t")
                if len(header) == 2:
                    header = ["P-S"] + header
                score_first = header[0] == "S-P"
                score_name, perf_name = (
                    (header[1], header[2]) if score_first else (header[2], header[1])
                )
            else:  # no name meta
                score_first, score_name, perf_name = False, None, None

            alignment = []
            for line in f:
                items = line.strip().split("\t")
                if filetype == AlignmentFileType.CORRESP and (
                    line.startswith("//") or len(items) != 10
                ):
                    continue

                def parse_note(note_data):
                    if (is_align and note_data[3] == "*") or (not is_align and note_data[0] == "*"):
                        return None

                    return AlignmentNote(
                        idx=int(note_data[0]),
                        start=float(note_data[1]),
                        end=float(note_data[2]) if is_align and note_data[2] != "-1" else None,
                        pitch=sitch2pitch(note_data[3]) if is_align else int(note_data[3]),
                    )

                note_1 = parse_note(items[:4] if is_align else items[:5])
                note_2 = parse_note(items[4:8] if is_align else items[5:10])

                s_note, p_note = (note_1, note_2) if score_first else (note_2, note_1)
                alignment.append(AlignmentPair(s_note, p_note))

        return cls(alignment, score_name, perf_name)

    @classmethod
    def _load_compressed(cls, path: str | Path | bytes) -> Alignment:
        """Internal parser for the `.npz` format."""
        with np.load(io.BytesIO(path) if isinstance(path, bytes) else path) as data:
            score_idx, score_pitch, score_times = (
                data["score_idx"],
                data["score_pitch"],
                data["score_times"],
            )
            perf_idx, perf_pitch, perf_times = (
                data["perf_idx"],
                data["perf_pitch"],
                data["perf_times"],
            )

            pairs = []
            for i in range(len(score_idx)):
                s_note = (
                    AlignmentNote(
                        idx=int(score_idx[i]),
                        pitch=int(score_pitch[i]),
                        start=float(score_times[i][0]),
                        end=float(score_times[i][1]) if score_times[i][1] != -1 else None,
                    )
                    if score_idx[i] != -1
                    else None
                )

                p_note = (
                    AlignmentNote(
                        idx=int(perf_idx[i]),
                        pitch=int(perf_pitch[i]),
                        start=float(perf_times[i][0]),
                        end=float(perf_times[i][1]) if perf_times[i][1] != -1 else None,
                    )
                    if perf_idx[i] != -1
                    else None
                )

                pairs.append(AlignmentPair(s_note, p_note))

            return cls(
                pairs=pairs,
                score_name=str(data["score_name"]),
                perf_name=str(data["perf_name"]),
            )

    def save(self, path: str | Path, **kwargs) -> None:
        """Saves alignment to disk, automatically detecting format by suffix."""
        path = Path(path)
        if path.suffix == ".npz":
            self._save_compressed(path)
        else:
            self._save_text(path, **kwargs)

    def _save_text(self, path: Path, score_first: bool = True) -> None:
        """Internal saver for human-readable text formats."""
        _empty = "-1\t-1\t-1\t*"
        self.preprocess_pairs(sort=True, score_first=score_first, clean_duplicates=False)

        with open(path, "w") as f:
            if score_first:
                header = f"S-P\t{self.score_name}\t{self.perf_name}\n"
            else:
                header = f"P-S\t{self.perf_name}\t{self.score_name}\n"
            f.write(header)

            for pair in self.pairs:
                s_str = pair.score_note.to_print_str() if pair.score_note else _empty
                p_str = pair.perf_note.to_print_str() if pair.perf_note else _empty
                f.write(f"{s_str}\t{p_str}\n" if score_first else f"{p_str}\t{s_str}\n")

    def _save_compressed(self, path: Path) -> None:
        """Internal saver for standardized `.npz` format."""
        score_data = np.array(
            [
                pair.score_note.to_array() if pair.score_note else [-1, -1, -1, -1]
                for pair in self.pairs
            ]
        )
        perf_data = np.array(
            [
                pair.perf_note.to_array() if pair.perf_note else [-1, -1, -1, -1]
                for pair in self.pairs
            ]
        )

        np.savez_compressed(
            path,
            score_name=str(self.score_name),
            score_idx=score_data[:, 0].astype(np.int32),
            score_pitch=score_data[:, 1].astype(np.int8),
            score_times=score_data[:, 2:].astype(np.float64),
            perf_name=str(self.perf_name),
            perf_idx=perf_data[:, 0].astype(np.int32),
            perf_pitch=perf_data[:, 1].astype(np.int8),
            perf_times=perf_data[:, 2:].astype(np.float64),
        )

    @classmethod
    def from_midi(
        cls,
        score_midi: Score,
        perf_midi: Score,
        alignment: np.ndarray | None = None,
    ) -> Alignment:
        """Generates an :class:`Alignment` object from two :class:`Score` objects and an index map."""

        def build_alignment_notes(midi):
            time_mapper = MIDITimeMapper(midi)

            note_soa = midi.tracks[0].notes.numpy()
            start_times = time_mapper.t2s(note_soa["time"]).astype(float)
            end_times = time_mapper.t2s(note_soa["time"] + note_soa["duration"]).astype(float)
            pitches = note_soa["pitch"]

            return [
                AlignmentNote(
                    idx=i,
                    start=float(start_times[i]),
                    end=float(end_times[i]),
                    pitch=pitches[i],
                )
                for i in range(len(midi.tracks[0].notes))
            ]

        score_notes = build_alignment_notes(score_midi)
        perf_notes = build_alignment_notes(perf_midi)

        pairs = []
        alignment = np.arange(len(score_notes)) if alignment is None else alignment
        for score_idx, perf_idx in enumerate(alignment):
            pairs.append(AlignmentPair(score_notes[score_idx], perf_notes[perf_idx]))

        return Alignment(pairs=pairs)

    # processing methods

    def preprocess_pairs(
        self,
        sort: bool = True,
        score_first: bool = True,
        clean_duplicates: bool = False,
        clean_mismatched_pitches: bool = False,
        clear_cache: bool = True,
    ) -> Alignment:
        """Standardizes the alignment state by sorting and cleaning invalid links."""
        if clean_duplicates:
            self.clean_duplicates()

        if clean_mismatched_pitches:
            self.clean_mismatched_pitches()

        if sort:
            self.sort(score_first=score_first)

        if clear_cache:
            self.clear_cache()

        return self

    def sort(self, score_first: bool = True) -> Alignment:
        """Sorts alignment pairs chronologically based on note onsets."""
        if score_first:
            self.pairs.sort(
                key=lambda pair: (
                    (
                        (pair.score_note.start, pair.score_note.pitch)
                        if pair.score_note
                        else INVALID_TIME_PITCH_TUPLE
                    ),
                    (
                        (pair.perf_note.start, pair.perf_note.pitch)
                        if pair.perf_note
                        else INVALID_TIME_PITCH_TUPLE
                    ),
                )
            )
        else:
            self.pairs.sort(
                key=lambda pair: (
                    (
                        (pair.perf_note.start, pair.perf_note.pitch)
                        if pair.perf_note
                        else INVALID_TIME_PITCH_TUPLE
                    ),
                    (
                        (pair.score_note.start, pair.score_note.pitch)
                        if pair.score_note
                        else INVALID_TIME_PITCH_TUPLE
                    ),
                )
            )

        self.clear_cache()
        return self

    def clean_duplicates(self) -> Alignment:
        """Removes duplicate :class:`AlignmentPair` objects."""
        self.pairs = list(set(self.pairs))
        self.clear_cache()
        return self

    def clean_mismatched_pitches(self) -> Alignment:
        """Filters out links where score and performance pitches do not match."""
        for pair in self.pairs:
            if (
                pair.score_note is not None
                and pair.perf_note is not None
                and pair.score_note.pitch != pair.perf_note.pitch
            ):
                pair.perf_note = None
        self.clear_cache()
        return self

    # utility methods used during the alignment cleaning

    def compare_notes_with_midi(
        self,
        score_midi: Score | None = None,
        perf_midi: Score | None = None,
        clean_unmatched: bool = True,
        fill_note_attributes: bool = False,
    ) -> Alignment:
        """Validates alignment notes against actual MIDI file contents."""
        if score_midi is not None:
            pair_to_score, _ = self.match_with_midi(midi=score_midi, is_score_midi=True)

            if clean_unmatched:  # remove score notes not found in score midi
                for idx in np.where(pair_to_score == -1)[0]:
                    self.pairs[idx].score_note = None

            if fill_note_attributes:
                self.fill_note_attributes(score_midi, pair_to_score, is_score_midi=True)

        if perf_midi is not None:
            (pair_to_note, _), (pair_data, midi_data) = self.match_with_midi(
                midi=perf_midi, is_score_midi=False, return_midi_data=True
            )

            if clean_unmatched:  # remove performance notes not found in performance midi
                diff = np.abs(midi_data[pair_to_note] - pair_data)
                for idx in np.where(diff > TIME_TOLERANCE)[0]:
                    self.pairs[idx].perf_note = None

            if fill_note_attributes:
                self.fill_note_attributes(perf_midi, pair_to_note, is_score_midi=False)

        self.clear_cache()

        return self

    def fill_note_attributes(
        self,
        midi: Score,
        pair_to_note: np.ndarray,
        is_score_midi: bool = False,
    ):
        """Populates :class:`AlignmentNote` objects with velocity and timing from a MIDI Score."""
        notes = midi.tracks[0].notes
        note_soa = midi.tracks[0].notes.numpy()
        start_times = note_soa["time"].astype(float)
        end_times = (note_soa["time"] + note_soa["duration"]).astype(float)

        if isinstance(midi.ttype, Tick):
            time_mapper = MIDITimeMapper(midi)
            start_times, end_times = map(time_mapper.t2s, (start_times, end_times))

        for i, pair in enumerate(self.pairs):
            pair_note = pair.score_note if is_score_midi else pair.perf_note
            if pair_note is None:
                continue
            note_idx = pair_to_note[i]
            pair_note.start = round(float(start_times[note_idx]), TIME_PRECISION)
            pair_note.end = round(float(end_times[note_idx]), TIME_PRECISION)
            pair_note.velocity = notes[note_idx].velocity

    def create_pair_from_midi_notes(
        self,
        index: int,
        score_midi: Score,
        perf_midi: Score,
        score_note_idx: int = 0,
        perf_note_idx: int = 0,
        replace: bool = False,
    ) -> AlignmentPair | None:
        """Utility for manually creating a link from two MIDI note indices."""
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
                    velocity=score_note.velocity,
                ),
                perf_note=AlignmentNote(
                    idx=perf_note_idx,
                    start=round(perf_time_mapper.t2s(perf_note.start), TIME_PRECISION),
                    end=round(perf_time_mapper.t2s(perf_note.end), TIME_PRECISION),
                    pitch=perf_note.pitch,
                    velocity=perf_note.velocity,
                ),
            )

            if not replace:
                self.pairs.insert(index, AlignmentPair())
            self.pairs[index] = pair

            self.preprocess_pairs(sort=index > 0, clean_duplicates=False)
            return pair

        return None

    def match_with_midi(
        self,
        midi: Score,
        is_score_midi: bool = True,
        fix_non_unique: bool = True,
        return_midi_data: bool = False,
    ) -> (
        tuple[np.ndarray, np.ndarray]
        | tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
    ):
        """Resolves the mapping between symbolic alignment pairs and a :class:`symusic.Score`.

        Performs a pitch-and-time lookup to find the actual indices of the notes in the provided Score object.

        Returns:
            A mapping array where ``result[pair_index] = midi_note_index``.
        """

        def _get_pitch_time_array(
            pitches: np.ndarray, times: np.ndarray, precision: int = 4
        ) -> np.ndarray:
            return 1_000_000 * pitches.astype(int) + np.round(times.astype(float), precision)

        # get MIDI data
        note_soa = midi.tracks[0].notes.numpy()
        midi_pitches, midi_times = note_soa["pitch"], note_soa["time"]

        if isinstance(midi.ttype, Tick):
            time_mapper = MIDITimeMapper(midi)
            midi_times = time_mapper.t2s(midi_times)

        midi_data = _get_pitch_time_array(pitches=midi_pitches, times=midi_times)

        # get pair data (score/perf)
        pair_array = self.score_note_array if is_score_midi else self.perf_note_array
        pair_pitches, pair_times = pair_array[:, 1], pair_array[:, 2]
        pair_data = _get_pitch_time_array(pitches=pair_pitches, times=pair_times)

        if len(midi_data) == 0:
            pair_to_note = np.full(len(pair_data), fill_value=-1)
        else:
            # sort midi_data for efficient searching
            sorted_indices = np.argsort(midi_data)
            sorted_midi_data = midi_data[sorted_indices]
            pair_to_note = sorted_indices[find_closest(sorted_midi_data, pair_data)]

            # set invalid pairs to -1
            pair_to_note[pair_pitches < 0] = -1

        # fix non-unique mappings
        if fix_non_unique and len(midi_data) > 0:
            ids, counts = np.unique(pair_to_note, return_counts=True)
            nonunique_ids = ids[counts > 1]
            nonunique_ids = nonunique_ids[nonunique_ids != -1]

            # create mask for unused midi indices
            unused_mask = np.ones(len(midi_data), dtype=bool)
            unused_mask[ids[ids != -1]] = False
            unused_ids = np.where(unused_mask)[0]

            for idx in nonunique_ids:
                repeat_pair_ids = np.where(pair_to_note == idx)[0]
                for rep_idx in repeat_pair_ids[1:]:
                    pair_pitch, pair_time = pair_pitches[rep_idx], pair_times[rep_idx]

                    candidate_ids = np.where(
                        np.logical_and(
                            pair_pitch == midi_pitches[unused_ids],
                            np.abs(pair_time - midi_times[unused_ids]) < 1,
                        )
                    )[0]
                    if len(candidate_ids):
                        un_idx = unused_ids[candidate_ids[0]]

                        pair_to_note[rep_idx] = un_idx
                        unused_mask[un_idx] = False
                        unused_ids = np.where(unused_mask)[0]
                    else:
                        pair_to_note[rep_idx] = -1  # not found, set to unmatched

        # build note_to_pair mapping
        note_to_pair = np.full(len(midi_data), fill_value=-1)
        valid_pair_indices = pair_to_note[pair_to_note >= 0]
        if len(valid_pair_indices) > 0:
            note_to_pair[valid_pair_indices] = np.where(pair_to_note >= 0)[0]

        if return_midi_data:
            return (pair_to_note, note_to_pair), (pair_data, midi_data)
        else:
            return pair_to_note, note_to_pair

    def build_position_pairs(self, score_midi: Score) -> list[PositionPair]:
        """Converts the alignment into a list of matched :class:`PositionPair` objects."""
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
                        ticks_per_quarter=score_midi.ticks_per_quarter,
                    )
                )

        return position_pairs

    def build_score_to_perf_note_alignment(
        self,
        score_midi: Score,
        perf_midi: Score,
    ) -> np.ndarray:
        """Returns a direct index mapping from score notes to performance notes."""
        _, score_note_to_pair = self.match_with_midi(score_midi, is_score_midi=True)
        pair_to_perf_note, _ = self.match_with_midi(perf_midi, is_score_midi=False)
        score_to_perf_note = pair_to_perf_note[score_note_to_pair]
        return score_to_perf_note

    def clean_midi(self, midi: Score, is_score_midi: bool = True) -> Score:
        """Removes notes from a Score that are not linked in this alignment."""
        pair_to_midi, _ = self.match_with_midi(midi, is_score_midi=is_score_midi)

        notes = midi.tracks[0].notes
        remove_ids = np.where(~np.isin(np.arange(len(notes)), pair_to_midi))[0].tolist()

        if is_score_midi:
            for pair, idx in zip(self.pairs, pair_to_midi):
                if pair.score_note is not None and pair.perf_note is None:
                    remove_ids.append(idx)
        else:
            for pair, idx in zip(self.pairs, pair_to_midi):
                if pair.score_note is None and pair.perf_note is not None:
                    remove_ids.append(idx)

        remove_ids = sorted(set(remove_ids))
        for i, idx in enumerate(remove_ids):
            del notes[idx - i]

        return midi

    def delete_empty_pairs(
        self,
        no_score_note: bool = True,
        no_perf_note: bool = False,
    ) -> Alignment:
        """Filters out :class:`AlignmentPairs` that lack a score or performance note."""
        if no_score_note:
            self.pairs = list(filter(lambda pair: pair.score_note is not None, self.pairs))

        if no_perf_note:
            self.pairs = list(filter(lambda pair: pair.perf_note is not None, self.pairs))

        if no_score_note or no_perf_note:
            self.clear_cache()

        return self

    def fill_missing_score_notes(
        self,
        score_midi: Score,
        start_idx: int | None = None,
        end_idx: int | None = None,
    ) -> Alignment:
        """Populates the alignment with unaligned score notes as deletions."""
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
                    velocity=note.velocity,
                )
                self.pairs.append(AlignmentPair(s_note, perf_note=None))

            self.preprocess_pairs(sort=True, clean_duplicates=False)

        return self

    def update_pair_note_ids(self, midi: Score, is_score_midi: bool = True) -> Alignment:
        """Refreshes the internal `idx` of :class:`AlignmentNote` objects to match a MIDI."""
        pair_to_note, _ = self.match_with_midi(midi, is_score_midi=is_score_midi)
        for pair, idx in zip(self.pairs, pair_to_note):
            pair_note = pair.score_note if is_score_midi else pair.perf_note
            if pair_note is None:
                continue

            pair_note.idx = idx if pair_note.idx >= 0 else -idx

        return self

    def shift_notes(
        self,
        time_shift: float = 0.0,
        offset: float = 0.0,
        score_notes: bool = False,
        shift_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Applies a temporal shift to notes in the alignment."""
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

        self.clear_cache()

        return np.array(ids)
