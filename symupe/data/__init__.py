from torch.utils.data import Dataset

from .datasets import (
    LocalSequenceDataset,
    LocalScorePerformanceDataset
)
from .collators import (
    DataCollator,
    SequenceCollator,
    LMSequenceCollator,
    LMMultiSequenceCollator,
    MixedLMSequenceCollator,
    LMScorePerformanceCollator,
    MixedLMScorePerformanceCollator,
    LMSeq2SeqCollator
)

DATASETS = {
    name: cls for name, cls in locals().items()
    if isinstance(cls, type) and issubclass(cls, Dataset) and cls is not Dataset
}
COLLATORS = {
    name: cls for name, cls in locals().items()
    if isinstance(cls, type) and issubclass(cls, DataCollator) and cls is not DataCollator
}
