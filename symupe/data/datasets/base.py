from dataclasses import dataclass

import numpy as np

from symupe.utils import ExplicitEnum
from ..tokenizers import EncodingType, SortingType

DATA_SPLITS = ["all", "train", "eval", "val", "test"]


class SequenceTask(ExplicitEnum):
    SCORE = EncodingType.SCORE.value  # S
    PLAIN_SCORE = EncodingType.PLAIN_SCORE.value  # S
    PERFORMANCE = EncodingType.PERFORMANCE.value  # SRT
    REL_PERFORMANCE = EncodingType.REL_PERFORMANCE.value  # SR
    TIME_PERFORMANCE = EncodingType.TIME_PERFORMANCE.value  # T

    SCORE_TO_TIME_PERFORMANCE = "score_to_time_performance"  # ST
    TIME_PERFORMANCE_TO_SCORE = "time_performance_to_score"  # ST

    SCORE_TO_TIME_PERFORMANCE_DECOUPLED = "score_to_time_performance_decoupled"  # ST
    TIME_PERFORMANCE_TO_SCORE_DECOUPLED = "time_performance_to_score_decoupled"  # ST


TASK_TO_ENCODINGS = {
    SequenceTask.SCORE: (EncodingType.SCORE, None),
    SequenceTask.PLAIN_SCORE: (EncodingType.PLAIN_SCORE, None),
    SequenceTask.PERFORMANCE: (EncodingType.PERFORMANCE, None),
    SequenceTask.REL_PERFORMANCE: (EncodingType.REL_PERFORMANCE, None),
    SequenceTask.TIME_PERFORMANCE: (EncodingType.TIME_PERFORMANCE, None),

    SequenceTask.SCORE_TO_TIME_PERFORMANCE: (EncodingType.SCORE_TIME_PERFORMANCE, None),
    SequenceTask.TIME_PERFORMANCE_TO_SCORE: (EncodingType.SCORE_TIME_PERFORMANCE, None),

    SequenceTask.SCORE_TO_TIME_PERFORMANCE_DECOUPLED: (EncodingType.PLAIN_SCORE, EncodingType.TIME_PERFORMANCE),
    SequenceTask.TIME_PERFORMANCE_TO_SCORE_DECOUPLED: (EncodingType.TIME_PERFORMANCE, EncodingType.PLAIN_SCORE)
}


TASK_SEQUENCE_SORTING = {
    SortingType.SCORE: {
        SequenceTask.SCORE,
        SequenceTask.PLAIN_SCORE,
        SequenceTask.REL_PERFORMANCE,
        SequenceTask.SCORE_TO_TIME_PERFORMANCE
    },
    SortingType.TIME: {
        SequenceTask.TIME_PERFORMANCE,
        SequenceTask.TIME_PERFORMANCE_TO_SCORE
    },
    SortingType.ANY: {
        SequenceTask.PERFORMANCE
    }
}


@dataclass
class NoteSegments:
    bar: np.ndarray
    beat: np.ndarray
    onset: np.ndarray
