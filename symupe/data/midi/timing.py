from __future__ import annotations

import numpy as np
from symusic import Score


class MIDITimeMapper:
    def __init__(self, midi: Score):
        self.midi = midi
        self.tempos = self.compute_tempo_times(midi)

    @staticmethod
    def compute_tempo_times(midi: Score) -> np.ndarray:
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
        ids = np.searchsorted(self.tempos[1], ticks, side="right") - 1
        tempo_ticks, tempo_times, seconds_per_tick = self.tempos[1:4, ids]
        return tempo_times + seconds_per_tick * (ticks - tempo_ticks)

    def seconds_to_ticks(self, seconds: float | np.ndarray) -> int | np.ndarray:
        ids = np.searchsorted(self.tempos[2], seconds, side="right") - 1
        tempo_ticks, tempo_times, seconds_per_tick = self.tempos[1:4, ids]
        return (tempo_ticks + (seconds - tempo_times) / seconds_per_tick).astype(int)

    def t2s(self, ticks: int | np.ndarray) -> float | np.ndarray:
        return self.ticks_to_seconds(ticks)

    def s2t(self, seconds: float | np.ndarray) -> int | np.ndarray:
        return self.seconds_to_ticks(seconds)
