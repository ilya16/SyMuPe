"""RAScoP pipeline, Stage (S): Performance-to-Score Synchronization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from symusic import Score

from ..midi.sync import sync_performance_midi, GridLevel, SyncMetadata


@dataclass
class PerformanceSyncConfig:
    """Configuration for Stage (S): Performance-to-Score Synchronization.

    Attributes:
        synchronize_grid: If ``True``, aligns performance beats to the exact
            score grid by inserting tempo changes.
        grid_level: The granularity of the synchronization grid.
            ``GridLevel.BAR`` (or "bar") for measure-level synchronization
            ``GridLevel.BEAT`` (or "beat") for beat-level synchronization.
        ticks_per_quarter: The target MIDI resolution (TPQ) for the synchronized
            performance. Defaults to the global project constant.
    """

    synchronize_grid: bool = True
    grid_level: str | GridLevel = GridLevel.BEAT
    ticks_per_quarter: int = 480


def synchronize_performance(
    score_midi: Score,
    perf_midi: Score,
    onset_pairs: np.ndarray,
    config: PerformanceSyncConfig = PerformanceSyncConfig(),
    inplace: bool = True,
) -> tuple[Score, SyncMetadata | None]:
    """Implementation of Stage (S): Performance-to-Score Synchronization.

    Synchronizes the performance beat structure with the score grid.

    This stage:
    1. Calculates a beat-to-time mapping based on aligned onset pairs.
    2. Optional: Inserts inter-beat tempo changes to align with the score's
       symbolic grid (bars or beats).
    3. Shifts the performance so the first note aligns with the score start.

    Args:
        score_midi: The score providing the symbolic beat grid.
        perf_midi: The performance to be synchronized.
        onset_pairs: A 2D array of matched (score_tick, perf_time) pairs.
        config: Configuration for synchronization levels and resolution.
        inplace: If ``True``, modifies the perf_midi object directly.

    Returns:
        The synchronized performance Score.
    """
    return sync_performance_midi(
        score_midi=score_midi,
        perf_midi=perf_midi,
        onset_pairs=onset_pairs,
        ticks_per_quarter=config.ticks_per_quarter,
        synchronize_grid=config.synchronize_grid,
        grid_level=config.grid_level,
        inplace=inplace,
    )
