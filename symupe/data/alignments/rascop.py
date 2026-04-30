"""RAScoP (Refined Alignment for Scores and Performances) pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace, field
from functools import partial

import numpy as np
from symusic import Score

from . import PositionPair
from .alignment import Alignment
from .holes import AlignmentHoleProcessorConfig, AlignmentHoleProcessor
from .interpolation import (
    NoteInterpolationConfig,
    interpolate_missing_notes,
    process_unperformed_notes,
)
from .onsets import OnsetCleanerConfig, compute_onset_position_pairs
from .sync import PerformanceSyncConfig, synchronize_performance
from ..midi.preprocess import preprocess_midi
from ..midi.sync import SyncMetadata
from ..midi.timing import MIDITimeMapper


@dataclass
class RefinementStages:
    """Statistics for recall or precision at different pipeline stages."""

    aligner: float | None = None
    hole_pre: float | None = None
    onset_cleaner: float | None = None
    hole_post: float | None = None
    cleaner: float | None = None


@dataclass
class AlignmentMetrics:
    """Container for tracking Recall and Precision through the refinement process."""

    recall: RefinementStages
    precision: RefinementStages


@dataclass
class RefinementTimes:
    """Computational time (seconds) for each pipeline stage."""

    init: float | None = None
    check: float | None = None
    holes_pre: float | None = None
    onsets: float | None = None
    clean: float | None = None
    holes_post: float | None = None
    interpolation: float | None = None
    sync: float | None = None
    total: float | None = None


@dataclass
class RAScoPConfig:
    """Configuration for the RAScoP refinement pipeline.

    Attributes:
        check_notes: Validate alignment against current MIDI note attributes.
        match_note_ids: Re-index alignment pairs indices to match current MIDI indices.

        score_holes: Enable Stage (H) for score-side holes (unperformed segments).
        performance_holes: Enable Stage (H) for performance-side holes (extra notes).
        hole_processor: Configuration for :class:`symupe.data.alignments.holes.AlignmentHoleProcessor`.

        clean_onsets: Enable Stage (O) to remove timing outliers and jitter.
        clean_score: Remove unaligned notes from the score MIDI.
        clean_performance: Remove unaligned notes from the performance MIDI.
        onset_cleaner: Configuration for :class:`symupe.data.alignments.onsets.OnsetCleaner`.

        interpolate_notes: Enable Stage (I) to synthesize missing notes.
        unperformed_note_markers: Add metadata markers for notes that remain unperformed.
        note_interpolation: Configuration for :method:`symupe.data.alignments.interpolation.interpolate_missing_notes`.

        synchronize_performance: Enable Stage (S) to align performance to the score beat grid.
        synchronization: Configuration for :method:`symupe.data.alignments.sync.synchronize_performance`.

        min_recall: Halt pipeline if recall drops below this threshold.
        num_runs: Number of iterative refinement passes.
    """

    # Stage 0. Initial Check
    check_notes: bool = True
    match_note_ids: bool = True

    # Stage 1. (H): Alignment Hole Processing
    score_holes: bool = False
    performance_holes: bool = False
    hole_processor: AlignmentHoleProcessorConfig = field(
        default_factory=lambda: AlignmentHoleProcessorConfig(),
    )

    # Stage 2. (O): Onset Cleaning and Temporal Refinement
    clean_onsets: bool = True
    clean_score: bool = False
    clean_performance: bool = True
    onset_cleaner: OnsetCleanerConfig = field(
        default_factory=lambda: OnsetCleanerConfig(
            shift_note_breaks=True,
        ),
    )

    # Stage 3. (I): Note Interpolation
    interpolate_notes: bool = False
    unperformed_note_markers: bool = False
    note_interpolation: NoteInterpolationConfig = field(
        default_factory=lambda: NoteInterpolationConfig(),
    )

    # Stage 4. (S): Performance-to-Score Synchronization
    synchronize_performance: bool = False
    synchronization: PerformanceSyncConfig = field(
        default_factory=lambda: PerformanceSyncConfig(),
    )

    # Global settings
    min_recall: float = 0.7
    num_runs: int = 1

    @classmethod
    def alignment_metrics(cls):
        """Preset for computing metrics after basic O+H refinement."""
        return cls(
            score_holes=True,
            performance_holes=True,
            min_recall=0.0,
        )

    @classmethod
    def pianocore_a(cls):
        """Preset to reproduce the PianoCoRe-A subset (RAScoP Recall >= 0.7)."""
        return cls(
            score_holes=True,
            performance_holes=True,
            interpolate_notes=True,
            min_recall=0.7,
        )

    @classmethod
    def pianocore_a_star(cls):
        """Preset to reproduce the PianoCoRe-A* high-quality subset (RAScoP Recall >= 0.85)."""
        return cls(
            score_holes=True,
            performance_holes=True,
            interpolate_notes=True,
            min_recall=0.85,
        )

    @classmethod
    def matched_only(cls):
        """Preset to filter out only initially matched score and performance MIDI notes."""
        return cls(
            clean_onsets=False,
            clean_score=True,
            clean_performance=True,
        )


class RAScoP:
    """RAScoP (Refined Alignment for Scores and Performances).

    This class implements the alignment refinement pipeline introduced in
    "PianoCoRe: Combined and Refined Piano MIDI Dataset" (Borovik, 2026).
    It transforms raw, noisy note-level alignments into clean, temporally
    coherent parallel data pairs by executing four sequential stages:

    1. (H) Hole Processing: detects and removes continuous, structurally incorrect alignment sections.
    2. (O) Onset Cleaning and Temporal Refinement: corrects temporal outliers and intra-onset jitter.
    3. (I) Note Interpolation: synthesizes unperformed notes to parallel note-aligned score-performance pairs.
    4. (S) Score-to-Performance Synchronization: aligns the beat structure of the refined performance with the score.
    """

    def __init__(
        self,
        *,
        score_midi: str | Score,
        perf_midi: str | Score,
        alignment: str | Alignment,
        config: RAScoPConfig | None = None,
        verbose: int = 0,
    ):
        """Initializes the refiner with score, performance, and initial alignment.

        Note: alignment refinement works with single-track MIDI files.
            If the original score and performance MIDI files had multiple tracks,
            the all tracks will be merged into one before the refinement.

        Args:
            score_midi: Path to MIDI or :class:`symusic.Score` object representing the score.
            perf_midi: Path to MIDI or :class:`symusic.Score` object representing the performance.
            alignment: Path to .npz or :class:`Alignment` object containing initial matches.
            verbose: Logging verbosity level (0: silent, 1: stages, 2: detailed stats).
        """
        self.score_midi = Score(score_midi) if isinstance(score_midi, str) else score_midi
        self.perf_midi = Score(perf_midi) if isinstance(perf_midi, str) else perf_midi
        self.alignment = Alignment.from_file(alignment) if isinstance(alignment, str) else alignment

        prepare_midi = partial(preprocess_midi, to_single_track=True, clean_duplicates=True)
        self.score_midi = prepare_midi(self.score_midi)
        self.perf_midi = prepare_midi(self.perf_midi)

        self.config = config or RAScoPConfig()
        self.verbose = verbose

        self._onset_pairs = None

    @staticmethod
    def empty_metrics_dict() -> AlignmentMetrics:
        """Creates a fresh AlignmentMetrics container."""
        return AlignmentMetrics(
            recall=RefinementStages(),
            precision=RefinementStages(),
        )

    def compute_metrics(self) -> tuple[float, float]:
        """Computes current Recall and Precision for the internal alignment state.

        Returns:
            A tuple of (recall, precision).
        """
        num_matched = self.alignment.num_full_pairs
        num_score, num_perf = self.score_midi.note_num(), self.perf_midi.note_num()
        recall = num_matched / num_score if num_score > 0 else 0.0
        precision = num_matched / num_perf if num_perf > 0 else 0.0
        return recall, precision

    @property
    def onset_pairs(self):
        if self._onset_pairs is None:
            self._onset_pairs, _ = compute_onset_position_pairs(
                self.alignment,
                self.score_midi,
                self.perf_midi,
                config=replace(
                    self.config.onset_cleaner,
                    shift_note_breaks=False,
                ),
                update_alignment_and_midi=False,
            )
        return self._onset_pairs

    def _log(
        self,
        stage: str,
        note_num: bool = False,
        recall: float | None = None,
        precision: float | None = None,
        onset_pairs: bool = False,
        skipped: bool = False,
        msg: str = "",
        stage_time: float | None = None,
        verbose: int = 0,
    ):
        """Internal logging helper for pipeline progress."""
        if not verbose:
            return

        stage_time = stage_time if stage_time is not None else 0.0
        log_str = f"  [{stage:<8} | {stage_time:.3f}s]"
        log_str += f" matched/all: {self.alignment.num_full_pairs}/{len(self.alignment)}"
        msg = "skipped" if skipped else msg
        if note_num:
            log_str += f" | score: {self.score_midi.note_num()}, perf: {self.perf_midi.note_num()}"
        if recall is not None:
            log_str += f" | R: {recall:.3f}"
        if precision is not None:
            log_str += f" | P: {precision:.3f}"
        if onset_pairs and self.onset_pairs is not None:
            log_str += f" | onset_pairs: {len(self.onset_pairs)}"
        if msg:
            log_str += f" | ({msg})"
        print(log_str)

    @staticmethod
    def _log_detail(label: str | None, data: str | list | np.ndarray) -> None:
        """Internal logging helper for detailed stage data."""
        if label is not None:
            print(f"            ↳ {label}: {data}")
        else:
            print(f"            ↳ {data}")

    def __call__(self, **kwargs) -> tuple[Alignment, AlignmentMetrics, RefinementTimes]:
        """Alias for :meth:`refine_alignment`."""
        return self.refine_alignment(**kwargs)

    def refine_alignment(
        self,
        *,
        verbose: int | None = None,
        num_runs: int | None = None,
    ) -> tuple[Alignment, AlignmentMetrics, RefinementTimes]:
        """Executes the RAScoP pipeline.

        This method coordinates the configurable sequential stages of the note alignment refinement.

        All parameters are configured using :class:`RAScoPConfig`

        Args:
            verbose: Override default logging verbosity.
            num_runs: Override default number of runs.

        Returns:
            A tuple of (Refined :class:`symupe.data.alignments.Alignment` object, Tracked metrics through stages).
        """
        cfg = self.config
        verbose = verbose if verbose is not None else self.verbose
        num_runs = num_runs if num_runs is not None else cfg.num_runs

        # stage times
        pipeline_start = stage_start = time.perf_counter()
        stage_times = RefinementTimes()

        # make sure alignment is in the right format
        self.alignment.preprocess_pairs(sort=True, score_first=True, clean_duplicates=True)

        # RAScoP performs temporal calculations in seconds
        self.perf_midi = self.perf_midi.to("second")

        # initial metrics
        metrics = self.empty_metrics_dict()
        metrics.recall.aligner, metrics.precision.aligner = self.compute_metrics()

        if verbose:
            print("RAScoP Pipeline:")
        stage_times.init = time.perf_counter() - stage_start
        self._log(
            stage="0:INIT",
            note_num=True,
            recall=metrics.recall.aligner,
            precision=metrics.precision.aligner,
            stage_time=stage_times.init,
            verbose=verbose,
        )

        def _plural(number):
            return "" if number == 1 else "s"

        def _halt_computation(recall: float, precision: float, stage: int):
            if (
                recall == 0.0
                or recall < cfg.min_recall
                or self.perf_midi.note_num() < 2
                or self.alignment.num_full_pairs == 0
            ):
                if verbose:
                    self._log(
                        stage="HALT",
                        msg=f"recall {recall:.3f} < threshold {cfg.min_recall}, pipeline terminated.",
                        verbose=verbose,
                    )
                    self._log_detail(
                        label=None,
                        data=f"Stages {', '.join(map(str, range(stage + 1, 5)))} cancelled.",
                    )
                self.perf_midi = self.perf_midi.to("tick")
                metrics.recall.cleaner, metrics.precision.cleaner = recall, precision
                stage_times.total = time.perf_counter() - pipeline_start
                self._log(
                    stage="TOTAL",
                    note_num=True,
                    recall=metrics.recall.cleaner,
                    precision=metrics.precision.cleaner,
                    stage_time=stage_times.total,
                    verbose=verbose,
                )
                return True
            return False

        if _halt_computation(metrics.recall.aligner, metrics.precision.aligner, stage=0):
            return self.alignment, metrics, stage_times

        if cfg.check_notes:
            stage_start = time.perf_counter()
            self.check_alignment_notes(match_ids=cfg.match_note_ids)
            stage_times.check = time.perf_counter() - stage_start
            self._log(stage="0:CHECK", stage_time=stage_times.check, verbose=verbose)

        for run in range(num_runs):
            if verbose and num_runs >= 2:
                print(f"RUN-{run + 1}")

            # STAGE 1: Hole Processing
            stage_start = time.perf_counter()
            if cfg.score_holes or cfg.performance_holes:
                hole_beat_ticks, hole_times = self.process_holes(
                    score_holes=cfg.score_holes,
                    performance_holes=cfg.performance_holes,
                )
                if run == 0:
                    metrics.recall.hole_pre, metrics.precision.hole_pre = self.compute_metrics()

                self.alignment.delete_empty_pairs(
                    no_score_note=cfg.clean_performance,
                    no_perf_note=cfg.clean_score,
                )

                stage_time = time.perf_counter() - stage_start
                stage_times.holes_pre = (stage_times.holes_pre or 0.0) + stage_time
                self._log(
                    stage="1:HOLES",
                    note_num=True,
                    recall=metrics.recall.hole_pre,
                    precision=metrics.precision.hole_pre,
                    msg=f"detected {len(hole_beat_ticks)} hole{_plural(len(hole_beat_ticks))}",
                    stage_time=time.perf_counter() - stage_start,
                    verbose=verbose,
                )
                if len(hole_beat_ticks) > 0 and verbose >= 2:
                    _data = list(
                        zip(hole_beat_ticks.round(6).tolist(), hole_times.round(6).tolist())
                    )
                    self._log_detail(label="holes", data=_data)

                if _halt_computation(metrics.recall.hole_pre, metrics.precision.hole_pre, stage=1):
                    return self.alignment, metrics, stage_times
            else:
                self._log("1:HOLES", skipped=True, verbose=verbose)

            # STAGE 2: Onset Cleaning
            stage_start = time.perf_counter()
            if cfg.clean_onsets:
                _, shifts = self.clean_onsets()
                if run == 0:
                    metrics.recall.onset_cleaner, metrics.precision.onset_cleaner = (
                        self.compute_metrics()
                    )
                if verbose:
                    stage_time = time.perf_counter() - stage_start
                    stage_times.onsets = (stage_times.onsets or 0.0) + stage_time
                    self._log(
                        stage="2:ONSET",
                        note_num=True,
                        onset_pairs=True,
                        recall=metrics.recall.onset_cleaner,
                        precision=metrics.precision.onset_cleaner,
                        msg=f"applied {len(shifts)} shift{_plural(len(shifts))}",
                        stage_time=stage_time,
                        verbose=verbose,
                    )
                if len(shifts) > 0 and verbose >= 2:
                    _data = [
                        (pos_idx, round(offset, 6), round(shift, 6))
                        for pos_idx, offset, shift in shifts
                    ]
                    self._log_detail(label="shifts", data=_data)
            else:
                self._log("2:ONSET", skipped=True, verbose=verbose)

            if cfg.score_holes or cfg.performance_holes or cfg.clean_onsets:
                self.check_alignment_notes(fill_score_notes=False)

            # clean MIDI data
            stage_start = time.perf_counter()

            if cfg.clean_score:
                self.alignment.clean_midi(self.score_midi, is_score_midi=True)
                self.score_midi = preprocess_midi(self.score_midi, sort_events=False)

            if cfg.clean_performance:
                self.alignment.clean_midi(self.perf_midi, is_score_midi=False)
                self.perf_midi = preprocess_midi(self.perf_midi, sort_events=False)

            self.alignment.delete_empty_pairs(
                no_score_note=cfg.clean_performance,
                no_perf_note=cfg.clean_score,
            )

            if cfg.clean_score or cfg.clean_performance:
                stage_time = time.perf_counter() - stage_start
                stage_times.clean = (stage_times.clean or 0.0) + stage_time
                self._log(stage="2:CLEAN", note_num=True, stage_time=stage_time, verbose=verbose)
            else:
                self._log("2:CLEAN", skipped=True, verbose=verbose)

            if cfg.clean_onsets:
                if _halt_computation(
                    metrics.recall.onset_cleaner, metrics.precision.onset_cleaner, stage=2
                ):
                    return self.alignment, metrics, stage_times

            # process holes again after onset cleaning to catch errors exposed by temporal shifts
            stage_start = time.perf_counter()
            if cfg.clean_onsets and cfg.score_holes:
                hole_beat_ticks, hole_times = self.process_holes()
                if run == 0:
                    metrics.recall.hole_post, metrics.precision.hole_post = self.compute_metrics()
                stage_time = time.perf_counter() - stage_start
                stage_times.holes_post = (stage_times.holes_post or 0.0) + stage_time
                self._log(
                    stage="2:HOLES",
                    note_num=True,
                    recall=metrics.recall.hole_post,
                    precision=metrics.precision.hole_post,
                    msg=f"detected {len(hole_beat_ticks)} hole{_plural(len(hole_beat_ticks))}",
                    stage_time=time.perf_counter() - stage_start,
                    verbose=verbose,
                )
                if len(hole_beat_ticks) > 0 and verbose >= 2:
                    _data = list(
                        zip(hole_beat_ticks.round(6).tolist(), hole_times.round(6).tolist())
                    )
                    self._log_detail(label="holes", data=_data)
            else:
                self._log("2:HOLES", skipped=True, verbose=verbose)

            # update final "Cleaner" stage metrics for this run
            if run == 0:
                metrics.recall.cleaner, metrics.precision.cleaner = self.compute_metrics()

            if _halt_computation(metrics.recall.cleaner, metrics.precision.cleaner, stage=2):
                return self.alignment, metrics, stage_times

            # STAGE 3: Interpolation
            stage_start = time.perf_counter()
            if cfg.interpolate_notes and self.perf_midi.note_num() < self.score_midi.note_num():
                new_notes = self.interpolate_performance_notes()
                stage_time = time.perf_counter() - stage_start
                stage_times.interpolation = (stage_times.interpolation or 0.0) + stage_time
                self._log(
                    stage="3:INTERP",
                    note_num=True,
                    onset_pairs=True,
                    recall=metrics.recall.cleaner,
                    precision=metrics.precision.cleaner,
                    msg=f"interpolated {new_notes} note{_plural(new_notes)}",
                    stage_time=time.perf_counter() - stage_start,
                    verbose=verbose,
                )
            else:
                self._log("3:INTERP", skipped=True, verbose=verbose)

                if cfg.unperformed_note_markers:
                    new_notes = self.process_unperformed_notes()
                    stage_time = time.perf_counter() - stage_start
                    stage_times.interpolation = (stage_times.interpolation or 0.0) + stage_time
                    self._log(
                        stage="3:UNPERF",
                        note_num=True,
                        recall=metrics.recall.cleaner,
                        precision=metrics.precision.cleaner,
                        msg=f"added {new_notes} unperformed note marker{_plural(new_notes)}",
                        stage_time=stage_time,
                        verbose=verbose,
                    )

        # STAGE 4: Performance-Score Synchronization
        stage_start = time.perf_counter()
        if cfg.synchronize_performance:
            _, meta = self.synchronize_performance()
            msg = "shifted performance onsets"
            if cfg.synchronization.synchronize_grid:
                msg += f", synchronized {cfg.synchronization.grid_level} grid"
            stage_time = time.perf_counter() - stage_start
            stage_times.sync = (stage_times.sync or 0.0) + stage_time
            self._log(
                stage="4:SYNC",
                note_num=True,
                recall=metrics.recall.cleaner,
                precision=metrics.precision.cleaner,
                msg=msg,
                stage_time=time.perf_counter() - stage_start,
                verbose=verbose,
            )

            if meta is None and verbose:
                _data = "Non-monotonic performance onsets detected. Sync failed."
                self._log_detail(label="ERROR", data=_data)
            elif meta is not None and verbose >= 2:
                if cfg.synchronization.synchronize_grid:
                    ts_str = ", ".join(
                        [
                            f"{ts.numerator}/{ts.denominator}@{ts.time}"
                            for ts in meta.time_signatures
                        ]
                    )
                    self._log_detail(
                        label="score",
                        data=f"tpq={meta.ticks_per_quarter} | time_sigs={ts_str}",
                    )
                    if meta.ticks_per_bar is not None:
                        self._log_detail(
                            label="grid",
                            data=(
                                f"tpb={meta.ticks_per_bar.tolist()}"
                                f" | ibi_q={meta.ibi_in_quarters.round(3).tolist()}"
                                f" | points={meta.num_points}"
                            ),
                        )
                    if meta.tempos is not None:
                        self._log_detail(
                            label="tempos",
                            data=(
                                f"min={meta.tempos.min():.1f} | max={meta.tempos.max():.1f}"
                                f" | median={np.median(meta.tempos):.1f} (n={len(meta.tempos)})"
                            ),
                        )
                self._log_detail(
                    label="perf",
                    data=f"initial_shift={meta.initial_shift:.4f}s",
                )

        else:
            stage_time = time.perf_counter() - stage_start
            self._log("4:SYNC", skipped=True, stage_time=stage_time, verbose=verbose)

        # get back to ticks
        self.perf_midi = self.perf_midi.to("tick")

        # sort after quantization
        self.score_midi = preprocess_midi(self.score_midi, sort_events=True)
        self.perf_midi = preprocess_midi(self.perf_midi, sort_events=True)

        if cfg.match_note_ids:
            self._update_note_ids()

        stage_times.total = time.perf_counter() - pipeline_start
        self._log(
            stage="TOTAL",
            note_num=True,
            recall=metrics.recall.cleaner,
            precision=metrics.precision.cleaner,
            stage_time=stage_times.total,
            verbose=verbose,
        )

        return self.alignment, metrics, stage_times

    def _update_note_ids(self) -> None:
        """Synchronizes the internal alignment indices with current MIDI note indices."""
        self.alignment.update_pair_note_ids(midi=self.score_midi, is_score_midi=True)
        self.alignment.update_pair_note_ids(midi=self.perf_midi, is_score_midi=False)

    def check_alignment_notes(
        self,
        match_ids: bool = True,
        fill_score_notes: bool = True,
    ) -> None:
        """Validates the alignment against provided MIDI :class:`symusic.Score` objects.

        Ensures that every note referenced in the alignment exists in the MIDI scores,
        filling in missing attributes (pitch, timing) and matching indices if requested.

        Args:
            match_ids: If ``True``, the alignment indices are matched against MIDI note indices.
            fill_score_notes: If ``True``, the alignment is filled with pairs for missing score notes.
        """
        if match_ids:
            self._update_note_ids()

        self.alignment.compare_notes_with_midi(
            score_midi=self.score_midi,
            perf_midi=self.perf_midi,
            clean_unmatched=True,
            fill_note_attributes=True,
        )

        if fill_score_notes:
            self.alignment.fill_missing_score_notes(self.score_midi)

    def process_holes(
        self,
        score_holes: bool = True,
        performance_holes: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stage (H): Detects and removes structurally incorrect alignment sections.

        Identifies 'alignment holes', continuous regions where matches are sparse or
        nonsensical (e.g., skipped repeats). Uses a sliding window approach to
        calculate alignment density.

        Args:
            score_holes: Detect holes in the score (unperformed segments).
            performance_holes: Detect holes in performance (extra/wrong notes).

        Returns:
            The arrays of score beat and performance time holes.
        """
        hole_processor = AlignmentHoleProcessor(
            score_midi=self.score_midi,
            perf_midi=self.perf_midi,
            alignment=self.alignment,
            config=self.config.hole_processor,
        )
        hole_beat_ticks, hole_times, _ = hole_processor(
            detect_score_holes=score_holes,
            detect_performance_holes=performance_holes,
            delete_notes_in_holes=True,
        )

        return hole_beat_ticks, hole_times

    def clean_onsets(self) -> tuple[list[PositionPair], list[tuple[int, float, float]]]:
        """Stage (O): Corrects timing outliers and intra-onset jitter.

        Refines the alignment by:
        1. Removing notes with high intra-onset deviation (outliers within chords).
        2. Correcting inter-onset intervals that imply unrealistic local tempi.
        3. Shifting subsequent notes to maintain temporal coherence.

        Returns:
            The number of temporal shifts/corrections applied.
        """
        self._onset_pairs, shifts = compute_onset_position_pairs(
            self.alignment,
            self.score_midi,
            self.perf_midi,
            config=self.config.onset_cleaner,
            update_alignment_and_midi=True,
        )
        return self.onset_pairs, shifts

    def interpolate_performance_notes(self):
        """Stage (I): Synthesizes performance notes for unperformed score segments.

        Uses linear interpolation between neighboring 'anchor' notes in the alignment
        to estimate onsets, durations, and velocities.

        Returns:
            The number of added synthesized notes.
        """
        self.alignment.preprocess_pairs(sort=True, clean_duplicates=False)

        _, _, new_notes = interpolate_missing_notes(
            score_midi=self.score_midi,
            perf_midi=self.perf_midi,
            alignment=self.alignment,
            config=self.config.note_interpolation,
        )

        if new_notes > 0:
            self._onset_pairs = None  # drop onset pair cache

        return new_notes

    def process_unperformed_notes(self) -> int:
        """Identifies score notes that have no performance alignment and appends
        them as TextMeta markers to the performance MIDI.

        Markers follow the format: `NoteS_{pitch}_{start}_{duration}`.

        Returns:
            The number of added unperformed note markers.
        """
        new_notes = process_unperformed_notes(
            score_midi=self.score_midi,
            perf_midi=self.perf_midi,
            alignment=self.alignment,
        )

        return new_notes

    def synchronize_performance(self, inplace: bool = True) -> tuple[Score, SyncMetadata | None]:
        """Stage (S): Synchronizes performance beat-structure with the score.

        Calculates a beat-to-time mapping based on the alignment and updates the
        performance MIDI's tempo structure or onset timing.

        Args:
            inplace: If ``True``, updates ``self.perf_midi`` directly.

        Returns:
            The synchronized :class:`symusic.Score` object.
        """
        onset_pairs = np.array([(p.score_tick, p.perf_time) for p in self.onset_pairs])

        pair_to_perf = None
        if inplace:
            pair_to_perf, _ = self.alignment.match_with_midi(self.perf_midi, is_score_midi=False)

        midi, meta = synchronize_performance(
            score_midi=self.score_midi,
            perf_midi=self.perf_midi,
            onset_pairs=onset_pairs,
            config=self.config.synchronization,
            inplace=inplace,
        )

        if meta is None:
            return midi, None

        if inplace:
            self.perf_midi = midi
            time_mapper = MIDITimeMapper(midi)
            perf_notes = midi.tracks[0].notes

            for idx in pair_to_perf:
                if idx == -1 or self.alignment.pairs[idx].perf_note is None:
                    continue
                note = perf_notes[pair_to_perf[idx]]
                self.alignment.pairs[idx].perf_note.start = time_mapper.t2s(note.start)
                self.alignment.pairs[idx].perf_note.end = time_mapper.t2s(note.end)

            self.alignment.preprocess_pairs(sort=True, clean_duplicates=False)

        return midi, meta


# backward compatibility with the original name
AlignmentCleaner = RAScoP
