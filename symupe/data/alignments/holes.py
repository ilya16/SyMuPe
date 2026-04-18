"""RAScoP pipeline, Stage (H): Alignment Hole Processing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from symusic import Score

from .alignment import Alignment
from ..midi.utils import clean_controls_in_interval


@dataclass
class AlignmentHoleProcessorConfig:
    """Configuration for Stage (H): Alignment Hole Processing.

    Attributes:
        score_threshold: The $H_r$ threshold for flagging holes in the score.
            Represents the ratio of unaligned notes within the window.
        performance_threshold: The $H_r$ threshold for flagging holes in the performance.
        window: The $H_w$ sliding window size (number of notes) used to calculate alignment density.
    """

    score_threshold: float = 0.75
    performance_threshold: float = 0.75
    window: int = 31


class AlignmentHoleProcessor:
    """Implementation of Stage (H): Alignment Hole Processing.

    This processor identifies 'alignment holes', continuous regions where matches
    are sparse or nonsensical (e.g., skipped repeats or transcription artifacts).

    It can detect holes from two perspectives:
    1. Score Holes: Unperformed score segments.
    2. Performance Holes: Extra or noisy performed segments.
    """

    def __init__(
        self,
        score_midi: Score,
        perf_midi: Score,
        alignment: Alignment,
        config: AlignmentHoleProcessorConfig = AlignmentHoleProcessorConfig(),
    ):
        self.score_midi = score_midi
        self.perf_midi = perf_midi
        self.alignment = alignment

        self.config = config

    def __call__(
        self,
        detect_score_holes: bool = True,
        detect_performance_holes: bool = False,
        delete_notes_in_holes: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Executes the hole detection and optional deletion pipeline.

        Args:
            detect_score_holes: Whether to detect unperformed score segments.
            detect_performance_holes: Whether to detect extra performance segments.
            delete_notes_in_holes: If ``True``, removes the matching links in the
                alignment and filters the notes from the MIDI objects.

        Returns:
            A tuple containing (hole_ticks, hole_times, note_missing_ratios).
        """

        hole_ticks, hole_times = np.empty((0, 2)), np.empty((0, 2))
        note_missing_ratios = np.empty(0)

        if detect_score_holes:
            score_hole_ticks, score_hole_times, score_note_missing_ratios, _ = (
                detect_alignment_holes(
                    score_midi=self.score_midi,
                    perf_midi=self.perf_midi,
                    alignment=self.alignment,
                    score_holes=True,
                    threshold=self.config.score_threshold,
                    window=self.config.window,
                )
            )

            hole_ticks = np.concatenate([hole_ticks, score_hole_ticks]).astype(int)
            hole_times = np.concatenate([hole_times, score_hole_times])
            note_missing_ratios = np.concatenate([note_missing_ratios, score_note_missing_ratios])

        if detect_performance_holes:
            perf_hole_ticks, perf_hole_times, perf_note_missing_ratios, _ = detect_alignment_holes(
                score_midi=self.score_midi,
                perf_midi=self.perf_midi,
                alignment=self.alignment,
                score_holes=False,
                threshold=self.config.performance_threshold,
                window=self.config.window,
            )
            if len(perf_hole_ticks) > 0:
                tempos = (
                    (np.diff(perf_hole_ticks) / np.diff(perf_hole_times)).reshape(-1)
                    / self.perf_midi.ticks_per_quarter
                    * 60
                )
                mask = (tempos < 15) | (tempos > 480)
                perf_hole_ticks, perf_hole_times = perf_hole_ticks[mask], perf_hole_times[mask]

            hole_ticks = np.concatenate([hole_ticks, perf_hole_ticks]).astype(int)
            hole_times = np.concatenate([hole_times, perf_hole_times])
            note_missing_ratios = np.concatenate([note_missing_ratios, perf_note_missing_ratios])

        if len(hole_ticks) > 0:
            if delete_notes_in_holes:
                self.delete_notes_in_tick_holes(hole_ticks=hole_ticks)
                self.delete_notes_in_time_holes(hole_times=hole_times)

        return hole_ticks, hole_times, note_missing_ratios

    def delete_notes_in_tick_holes(self, hole_ticks: np.ndarray) -> tuple[Score, Alignment]:
        """Removes alignment pairs and performance notes located within score-side tick holes."""
        score_midi, perf_midi, alignment = self.score_midi, self.perf_midi, self.alignment

        score_notes = score_midi.tracks[0].notes
        _, score_to_pair = alignment.match_with_midi(score_midi, is_score_midi=True)

        perf_notes = perf_midi.tracks[0].notes
        pair_to_perf, _ = alignment.match_with_midi(perf_midi, is_score_midi=False)

        for hole_start, hole_end in hole_ticks:
            hole_start_time, hole_end_time = perf_midi.end(), perf_midi.start()
            for i, note in enumerate(score_notes):
                if hole_start <= note.start <= hole_end:
                    pair = alignment.pairs[score_to_pair[i]]

                    if pair.perf_note is not None:
                        perf_note = perf_notes[pair_to_perf[score_to_pair[i]]]
                        hole_start_time = min(perf_note.time, hole_start_time)
                        hole_end_time = max(perf_note.time, hole_end_time)
                        perf_note.time = -1

                        pair.perf_note = None
                elif note.start > hole_end:
                    break

            if hole_start_time > hole_end_time:
                continue

            perf_midi = clean_controls_in_interval(
                perf_midi, start=hole_start_time, end=hole_end_time
            )

        perf_midi.tracks[0].notes.filter(lambda n: n.time != -1, inplace=True)
        alignment.preprocess_pairs(sort=True, clean_duplicates=False)

        return perf_midi, alignment

    def delete_notes_in_time_holes(self, hole_times: np.ndarray) -> tuple[Score, Alignment]:
        """Removes alignment pairs and performance notes located within performance-side time holes."""
        perf_midi, alignment = self.perf_midi, self.alignment

        perf_notes = perf_midi.tracks[0].notes
        pair_to_perf, _ = alignment.match_with_midi(perf_midi, is_score_midi=False)

        for hole_start_time, hole_end_time in hole_times:
            for i, pair in enumerate(alignment.pairs):
                if pair.perf_note is not None:
                    if hole_start_time <= pair.perf_note.start <= hole_end_time:
                        perf_notes[pair_to_perf[i]].time = -1
                        pair.perf_note = None

            perf_midi = clean_controls_in_interval(
                perf_midi, start=hole_start_time, end=hole_end_time
            )

        perf_midi.tracks[0].notes.filter(lambda n: n.time != -1, inplace=True)
        alignment.preprocess_pairs(sort=True, clean_duplicates=False)

        return perf_midi, alignment


def detect_alignment_holes(
    score_midi: Score,
    perf_midi: Score,
    alignment: Alignment,
    score_holes: bool = True,
    threshold: float = 0.75,
    window: int = 31,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Low-level function to detect structural alignment gaps using a sliding window.

    Calculates the ratio of unaligned notes ($H_a$) within a window ($H_w$).
    If $H_a > H_r$ (threshold), the region is designated as a hole.

    Args:
        score_midi: The score reference.
        perf_midi: The performance sequence.
        alignment: Current note-level alignment.
        score_holes: If ``True``, detects holes in score ticks. Otherwise, detects holes in performance time.
        threshold: The ratio threshold for hole designation.
        window: The sliding window size in notes.

    Returns:
        A tuple of (hole_ticks, hole_times, missing_ratios, inside_hole_mask).
    """
    score_notes = score_midi.tracks[0].notes
    score_ticks = score_notes.numpy()["time"]
    pair_to_score, score_to_pair = alignment.match_with_midi(score_midi, is_score_midi=True)

    perf_notes = perf_midi.to("second").tracks[0].notes
    perf_times = perf_notes.numpy()["time"]
    pair_to_perf, perf_to_pair = alignment.match_with_midi(perf_midi, is_score_midi=False)

    if score_holes:
        is_missing = (pair_to_perf == -1)[score_to_pair]
    else:
        is_missing = (pair_to_score == -1)[perf_to_pair]

    note_missing_ratios = np.convolve(is_missing, np.ones(window), mode="same") / window

    start_ids = np.where(note_missing_ratios >= threshold)[0]
    end_ids = []
    for idx in start_ids:
        normal_ids = np.where(note_missing_ratios[idx:] < threshold * 0.8)[0]
        end_ids.append(normal_ids[0] + idx if len(normal_ids) > 0 else len(pair_to_score))
    end_ids = np.array(end_ids)

    hole_note_ranges = np.array(
        [(start_ids[end_ids == end].min(), end) for end in np.unique(end_ids)]
    )

    if score_holes:
        data_times, other_times = score_ticks, perf_times
        data_to_pair = score_to_pair
        pair_to_other = pair_to_perf
    else:
        data_times, other_times = perf_times, score_ticks
        data_to_pair = perf_to_pair
        pair_to_other = pair_to_score

    hole_data_times = np.array(
        [(data_times[start], data_times[end]) for start, end in hole_note_ranges]
    )

    inside_hole = np.zeros_like(data_times, dtype=bool)
    for start, end in hole_data_times:
        inside_hole = inside_hole | (data_times >= start) & (data_times <= end)

    # compute hole performance times
    if len(hole_data_times) > 0:
        hole_other_times = []
        data_to_other = pair_to_other[data_to_pair]

        for hole_start, hole_end in hole_data_times:
            left = np.where(
                (data_times < hole_start) & (data_to_pair != -1) & (data_to_other != -1)
            )[0]
            left = left[-1] if len(left) else None

            right = np.where(
                (data_times > hole_end) & (data_to_pair != -1) & (data_to_other != -1)
            )[0]
            right = right[0] if len(right) else None

            hole_other_times.append(
                (
                    other_times[pair_to_other[data_to_pair[left]]] if left is not None else 0.0,
                    other_times[pair_to_other[data_to_pair[right]]]
                    if right is not None
                    else other_times.max(),
                )
            )

        hole_other_times = np.array(hole_other_times)
    else:
        hole_data_times, hole_other_times = np.empty((0, 2)), np.empty((0, 2))

    if score_holes:
        hole_ticks, hole_times = hole_data_times, hole_other_times
    else:
        hole_ticks, hole_times = hole_other_times, hole_data_times

    return hole_ticks, hole_times, note_missing_ratios, inside_hole
