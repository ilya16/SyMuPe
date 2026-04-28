from __future__ import annotations

import numpy as np
from symusic import Score


class MIDITimeMapper:
    """Converter for translating between MIDI ticks and absolute seconds.

    Maintains internal tempo map to handle multiple tempo changes in score.
    """

    def __init__(self, midi: Score):
        """Initializes mapper with score tempo data.

        Args:
            midi: :class:`symusic.Score` to analyze.
        """
        self.midi = midi
        self.tempos = self.compute_tempo_times(midi)

    @staticmethod
    def compute_tempo_times(midi: Score) -> np.ndarray:
        """Generates lookup table mapping MIDI ticks to cumulative performance time.

        Iterates through tempo changes to calculate absolute time in seconds at each transition
        and determines the tick-to-second conversion factor for the subsequent interval.

        Args:
            midi: :class:`symusic.Score` object to analyze.

        Returns:
            NumPy array of shape (4, N) containing BPM, tick position,
            absolute time, and seconds-per-tick factor for each tempo section.
        """
        if len(midi.tempos) > 0:
            midi.tempos[0].time = 0

        prev_tempo_tick, prev_tempo_time = 0, 0
        scale_factor = 60 / float(midi.ticks_per_quarter)
        seconds_per_tick = scale_factor / 120.0

        tempo_data = []
        for tempo in midi.tempos:
            tempo_time = prev_tempo_time + seconds_per_tick * (tempo.time - prev_tempo_tick)

            seconds_per_tick = scale_factor / tempo.qpm
            tempo_data.append([tempo.qpm, tempo.time, tempo_time, seconds_per_tick])
            prev_tempo_tick, prev_tempo_time = tempo.time, tempo_time

        tempo_data = np.stack(tempo_data, axis=1)
        return tempo_data

    def ticks_to_seconds(self, ticks: int | np.ndarray) -> float | np.ndarray:
        """Converts tick positions to absolute timestamps in seconds.

        Args:
            ticks: Tick value(s) to convert.

        Returns:
            Timestamp(s) in seconds.
        """
        ids = np.searchsorted(self.tempos[1], ticks, side="right") - 1
        tempo_ticks, tempo_times, seconds_per_tick = self.tempos[1:4, ids]
        return tempo_times + seconds_per_tick * (ticks - tempo_ticks)

    def seconds_to_ticks(self, seconds: float | np.ndarray) -> int | np.ndarray:
        """Converts absolute timestamps in seconds to MIDI tick positions.

        Args:
            seconds: Timestamp(s) in seconds.

        Returns:
            Integer tick position(s).
        """
        ids = np.searchsorted(self.tempos[2], seconds, side="right") - 1
        tempo_ticks, tempo_times, seconds_per_tick = self.tempos[1:4, ids]
        return (tempo_ticks + (seconds - tempo_times) / seconds_per_tick).astype(int)

    def t2s(self, ticks: int | np.ndarray) -> float | np.ndarray:
        """Alias for :meth:`ticks_to_seconds`."""
        return self.ticks_to_seconds(ticks)

    def s2t(self, seconds: float | np.ndarray) -> int | np.ndarray:
        """Alias for :meth:`seconds_to_ticks`."""
        return self.seconds_to_ticks(seconds)
