MAJOR = 1
MINOR = 0
PATCH = 0

VERSION = (MAJOR, MINOR, PATCH)
__version__ = f"{MAJOR}.{MINOR}.{PATCH}"

from .data.alignments import Alignment, RAScoP
from .data.tokenizers import AutoTokenizer, SyMuPeTokenizer
from .inference import (
    AutoGenerator,
    PerformanceGenerator,
    AutoClassifier,
    MusicClassifier,
    AutoEmbedder,
    MusicEmbedder,
)
from .models import AutoModel

__all__ = [
    "__version__",
    "AutoModel",
    "AutoGenerator",
    "PerformanceGenerator",
    "AutoClassifier",
    "MusicClassifier",
    "AutoEmbedder",
    "MusicEmbedder",
    "AutoTokenizer",
    "SyMuPeTokenizer",
    "Alignment",
    "RAScoP",
]
