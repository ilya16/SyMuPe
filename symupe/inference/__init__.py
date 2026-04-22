from .auto import AutoGenerator, AutoClassifier
from .base import Generator, GeneratorData, Classifier
from .classification import MusicClassificationResult, MusicClassifier
from .performance import (
    PerformanceGenerator,
    perform_score,
    save_performances,
    PerformanceRenderingResult,
)

__all__ = [
    "AutoGenerator",
    "AutoClassifier",
    "Generator",
    "GeneratorData",
    "Classifier",
    "PerformanceGenerator",
    "perform_score",
    "save_performances",
    "PerformanceRenderingResult",
    "MusicClassifier",
    "MusicClassificationResult",
]
