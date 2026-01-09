from .collators import (
    SequenceCollator,
    LMSequenceCollator,
    LMMultiSequenceCollator,
    MixedLMSequenceCollator,
    LMScorePerformanceCollator,
    MixedLMScorePerformanceCollator,
    LMSeq2SeqCollator
)
from .datasets import (
    LocalSequenceDataset,
    LocalScorePerformanceDataset
)

DATASETS = {name: cls for name, cls in globals().items() if ".datasets." in str(cls)}
COLLATORS = {name: cls for name, cls in globals().items() if ".collators." in str(cls)}
