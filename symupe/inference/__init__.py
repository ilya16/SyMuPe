from .auto import AutoGenerator, AutoClassifier, AutoEmbedder
from .auto import AutoGenerator, AutoClassifier, AutoEmbedder
from .base import Generator, GeneratorData, Classifier, Embedder
from .classification import MusicClassificationResult, MusicClassifier
from .embeddings import MusicEmbeddingResult, MusicEmbedder
from .performance import (
    PerformanceGenerator,
    perform_score,
    save_performances,
    PerformanceRenderingResult,
)

__all__ = [
    # Auto Classes
    "AutoGenerator",
    "AutoClassifier",
    "AutoEmbedder",
    # Generators
    "Generator",
    "GeneratorData",
    "PerformanceGenerator",
    "PerformanceRenderingResult",
    "perform_score",
    "save_performances",
    # Classifiers
    "Classifier",
    "MusicClassifier",
    "MusicClassificationResult",
    # Embedders
    "Embedder",
    "MusicEmbedder",
    "MusicEmbeddingResult",
]
