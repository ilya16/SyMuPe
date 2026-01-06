from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class Aligner(ABC):
    """ Base Aligner class. """

    @abstractmethod
    def align(
            self,
            score_midi: str | Path,
            perf_midi: str | Path,
            timeout: float | None = 1000.,
            memory_limit: int | None = None
    ):
        raise NotImplementedError
