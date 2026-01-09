""" Score-Performance data collators. """
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from symupe.utils import asdict
from .common import SeqInputs, SeqSegments
from .sequence import (
    SequenceCollator,
    LMSequenceCollator,
    MixedLMSequenceCollator,
    MaskLevel
)
from ..datasets.score_performance import ScorePerformanceSample


@dataclass
class ScorePerformanceInputs:
    scores: SeqInputs
    performances: SeqInputs
    noisy_performances: SeqInputs | None = None
    segments: SeqSegments | None = None
    directions: dict[str, torch.Tensor] | torch.Tensor | None = None
    task_ids: torch.Tensor | None = None
    deadpan_mask: torch.Tensor | None = None
    random_performances: SeqInputs | None = None


class ScorePerformanceCollator(SequenceCollator):
    def __init__(
            self,
            pad_token_id: int = 0,
            pad_to_multiple_of: int = 1
    ):
        super().__init__(pad_token_id, pad_to_multiple_of)

    def get_max_lengths(self, batch: Sequence[ScorePerformanceSample], inference: bool = False):
        lens_perf = np.array(list(map(lambda sample: len(sample.perf), batch))).T
        lens_score = np.array(list(map(lambda sample: len(sample.score), batch))).T
        max_lens = {
            "performance": self.pad_len(np.max(lens_perf)),
            "score": self.pad_len(np.max(lens_score))
        }

        if all((sample.noisy_perf is not None for sample in batch)):
            lens_noisy_perf = np.array(list(map(lambda sample: len(sample.noisy_perf), batch))).T
            max_lens["noisy_perf"] = self.pad_len(np.max(lens_noisy_perf))

        return max_lens

    def init_data(self, batch: Sequence[ScorePerformanceSample], inference: bool = False):
        max_lens = self.get_max_lengths(batch, inference=inference)

        sample, b = batch[0], len(batch)
        return ScorePerformanceInputs(
            scores=self._init_seq_data(
                b, max_lens["score"],
                compound_factor=sample.score.ids.shape[-1]
            ),
            performances=self._init_seq_data(
                b, max_lens["performance"],
                compound_factor=sample.perf.ids.shape[-1]
            ),
            noisy_performances=self._init_seq_data(
                b, max_lens["noisy_perf"],
                compound_factor=sample.noisy_perf.ids.shape[-1]
            ) if "noisy_perf" in max_lens else None,
            segments=SeqSegments(
                bar=torch.zeros(b, max_lens["score"], dtype=torch.long),
                beat=torch.zeros(b, max_lens["score"], dtype=torch.long),
                onset=torch.zeros(b, max_lens["score"], dtype=torch.long)
            ) if sample.segments is not None else None,
            directions=torch.zeros(
                b, max_lens["score"], len(sample.directions), dtype=torch.long
            ) if sample.directions is not None else None,
            task_ids=torch.zeros(b, dtype=torch.long),
            deadpan_mask=torch.zeros(b, dtype=torch.bool),
            random_performances=self._init_seq_data(
                b, max_lens["performance"],
                compound_factor=sample.perf.ids.shape[-1]
            ) if sample.random_perf is not None else None
        )

    def process_sample(
            self, i: int, sample: ScorePerformanceSample, data: ScorePerformanceInputs, inference: bool = False
    ):
        # process score
        self._process_sequence(i, seq=sample.score, seq_data=data.scores)

        # process performance
        self._process_sequence(i, seq=sample.perf, seq_data=data.performances)

        # process note segments if present
        self._process_segments(i, sample=sample, seg_data=data.segments, seq_len=len(sample.perf))

        # process task id
        data.task_ids[i] = sample.task_idx

        # process random performance is present
        if sample.random_perf is not None:
            self._process_sequence(i, seq=sample.random_perf, seq_data=data.random_performances)

        # process noisy performance is present
        if sample.noisy_perf is not None:
            self._process_sequence(i, seq=sample.noisy_perf, seq_data=data.noisy_performances)

        # process directions if present
        if sample.directions is not None:
            seq_len = len(sample.score)
            for j, (group_name, group_directions) in enumerate(sample.directions.items()):
                for (label, key), direction_map in group_directions.items():
                    mask = direction_map != 0.
                    if np.any(mask):
                        data.directions[i, :seq_len, j][mask] = label * torch.from_numpy(direction_map[mask])

        data.deadpan_mask[i] = sample.is_deadpan

    def __call__(self, batch: Sequence[ScorePerformanceSample], inference: bool = False, return_dict: bool = True):
        data = self.init_data(batch, inference=inference)
        for i, sample in enumerate(batch):
            self.process_sample(i, sample, data)

        return asdict(data) if return_dict else data


# FOR LANGUAGE MODELING
@dataclass
class LMScorePerformanceInputs(ScorePerformanceInputs):
    labels: SeqInputs | None = None


class LMScorePerformanceCollator(ScorePerformanceCollator, LMSequenceCollator):
    def __init__(
            self,
            pad_token_id: int = 0,
            pad_to_multiple_of: int = 1,

            mlm: bool = False,
            mask_level: str | MaskLevel | dict[str | MaskLevel, float] = MaskLevel.NOTE,
            mask_compound: bool = True,
            mask_prob: float = 0.15,
            replace_prob: float = 0.9,
            random_token_prob: float = 0.,
            copy_sequence_prob: float = 0.,
            mask_token_id: int = 1,
            mask_ignore_token_ids: list[int] | None = None,
            mask_token_dims: list[list[int]] | list[int] | None = None,
            label_pad_token_id: int = -100
    ):
        LMSequenceCollator.__init__(
            self,
            pad_token_id=pad_token_id,
            pad_to_multiple_of=pad_to_multiple_of,
            mlm=mlm,
            mask_level=mask_level,
            mask_compound=mask_compound,
            mask_prob=mask_prob,
            replace_prob=replace_prob,
            random_token_prob=random_token_prob,
            copy_sequence_prob=copy_sequence_prob,
            mask_token_id=mask_token_id,
            mask_ignore_token_ids=mask_ignore_token_ids,
            mask_token_dims=mask_token_dims,
            label_pad_token_id=label_pad_token_id
        )

    def __call__(self, batch: Sequence[ScorePerformanceSample], inference: bool = False, return_dict: bool = True):
        data = super().__call__(batch, inference=inference, return_dict=False)

        labels = self.mask_and_compute_labels(
            sequences=data.performances,
            random_sequences=data.random_performances,
            segments=data.segments,
            task_ids=data.task_ids,
            num_tokens=batch[0].meta.token_sizes
        )

        data = LMScorePerformanceInputs(
            scores=data.scores,
            performances=data.performances,
            noisy_performances=data.noisy_performances,
            segments=data.segments,
            directions=data.directions,
            deadpan_mask=data.deadpan_mask,
            labels=labels
        )

        return asdict(data) if return_dict else data


@dataclass
class MixedLMScorePerformanceInputs(LMScorePerformanceInputs):
    masked_performances: SeqInputs | None = None


class MixedLMScorePerformanceCollator(ScorePerformanceCollator, MixedLMSequenceCollator):
    def __init__(
            self,
            pad_token_id: int = 0,
            pad_to_multiple_of: int = 1,

            mask_token_id: int = 1,
            mask_ignore_token_ids: list[int] | None = None,
            mask_token_dims: list[int] | None = None,
            label_pad_token_id: int = -100
    ):
        MixedLMSequenceCollator.__init__(
            self,
            pad_token_id=pad_token_id,
            pad_to_multiple_of=pad_to_multiple_of,
            mask_token_id=mask_token_id,
            mask_ignore_token_ids=mask_ignore_token_ids,
            mask_token_dims=mask_token_dims,
            label_pad_token_id=label_pad_token_id
        )

    def __call__(self, batch: Sequence[ScorePerformanceSample], inference: bool = False, return_dict: bool = True):
        data = super().__call__(batch, inference=inference, return_dict=False)

        masked_performances, labels = self.mask_and_compute_labels(sequences=data.performances)

        data = MixedLMScorePerformanceInputs(
            scores=data.scores,
            performances=data.performances,
            noisy_performances=data.noisy_performances,
            segments=data.segments,
            directions=data.directions,
            deadpan_mask=data.deadpan_mask,
            masked_performances=masked_performances,
            labels=labels
        )

        return asdict(data) if return_dict else data
