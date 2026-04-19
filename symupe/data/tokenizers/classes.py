"""Extended miditok classes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import torch
from miditok.classes import (
    TokSequence as MidiTokTokSequence,
    TokenizerConfig as MidiTokTokenizerConfig,
    Event,
)

from symupe.utils import ExplicitEnum
from .constants import SPECIAL_TOKENS


class SequenceType(ExplicitEnum):
    SCORE = "score"
    PERFORMANCE = "performance"
    SYNC_PERFORMANCE = "sync_performance"
    TIME_PERFORMANCE = "time_performance"
    PERFORMANCE_SUSTAIN = "performance_sustain"
    SYNC_PERFORMANCE_SUSTAIN = "sync_performance_sustain"
    TIME_PERFORMANCE_SUSTAIN = "time_performance_sustain"


class EncodingType(ExplicitEnum):
    SCORE = "score"  # S - s
    PLAIN_SCORE = "plain_score"  # S - s
    PERFORMANCE = "performance"  # SRT - st
    REL_PERFORMANCE = "rel_performance"  # SR - s
    TIME_PERFORMANCE = "time_performance"  # T - t
    SCORE_TIME_PERFORMANCE = "score_time_performance"  # ST - st


class SortingType(ExplicitEnum):
    SCORE = "score"
    TIME = "time"
    ANY = "any"


ENCODING_SORTING = {
    SortingType.SCORE: {
        EncodingType.SCORE,
        EncodingType.PLAIN_SCORE,
        EncodingType.REL_PERFORMANCE,
    },
    SortingType.TIME: {
        EncodingType.TIME_PERFORMANCE,
    },
    SortingType.ANY: {
        EncodingType.PERFORMANCE,
        EncodingType.SCORE_TIME_PERFORMANCE,
    },
}

SEQUENCE_DEFAULT_ENCODING = {
    SequenceType.SCORE: EncodingType.SCORE,
    SequenceType.PERFORMANCE: EncodingType.PERFORMANCE,
    SequenceType.SYNC_PERFORMANCE: EncodingType.SCORE,
    SequenceType.TIME_PERFORMANCE: EncodingType.TIME_PERFORMANCE,
    SequenceType.PERFORMANCE_SUSTAIN: EncodingType.PERFORMANCE,
    SequenceType.TIME_PERFORMANCE_SUSTAIN: EncodingType.TIME_PERFORMANCE,
}


@dataclass
class TokSequence(MidiTokTokSequence):
    ids: np.ndarray | torch.Tensor | list[int | list[int]] | None = None
    values: np.ndarray | torch.Tensor | None = None
    interpolated: np.ndarray | torch.Tensor | None = None
    pedals: np.ndarray | torch.Tensor | None = None
    type: str | SequenceType | None = SequenceType.SCORE
    encoding: str | EncodingType | None = EncodingType.SCORE
    vocab: dict[str, int] | None = None
    meta: dict[str, ...] | None = None
    token_to_note: np.ndarray | None = None
    score_to_perf_token: np.ndarray | None = None

    def __getitem__(self, val: int | slice) -> int | str | Event | TokSequence:
        """
        Return the ``idx``th element or slice of the sequence.

        If an integer is providing, it checks by order: ids, tokens, values, events, bytes, interpolated.

        :param val: index of the element to retrieve.
        :return: ``idx``th element.
        """
        if isinstance(val, slice):
            return self.__slice(val)

        attributes = ["ids", "values", "tokens", "events", "bytes", "interpolated"]
        for attr in attributes:
            data = getattr(self, attr)
            if data is not None and len(data) > 0:
                return data[val]

        msg = "This TokSequence seems to not be initialized, all its attributes are None."
        raise ValueError(msg)

    def __slice(self, sli: slice) -> TokSequence:
        """
        Slice the ``TokSequence``.

        :param sli: slice object.
        :return: the slice of the self ``TokSequence``.
        """
        seq = replace(self)
        attributes = ["tokens", "ids", "values", "bytes", "events", "interpolated"]
        for attr in attributes:
            data = getattr(self, attr)
            if data is not None and len(data) > 0:
                setattr(seq, attr, data[sli] if data.ndim <= 2 else data[:, sli])
        return seq

    def __iadd__(self, other: TokSequence) -> TokSequence:
        """
        Concatenate the self ``TokSequence`` to another one.

        The `ìds``, ``tokens``, ``values``, ``interpolated``, ``events`` and ``bytes`` will be concatenated.

        :param other: other ``TokSequence``.
        :return: the two sequences concatenated.
        """
        if not isinstance(other, TokSequence):
            msg = (
                "Addition to a `TokSequence` object can only be performed with other"
                f"`TokSequence` objects. Received: {other.__class__.__name__}"
            )
            raise ValueError(msg)

        attributes = ["tokens", "ids", "values", "bytes", "events", "interpolated", "pedals"]
        for attr in attributes:
            self_attr, other_attr = getattr(self, attr), getattr(other, attr)
            if self_attr is not None and other_attr is not None:
                if isinstance(self_attr, (np.ndarray, torch.Tensor)):
                    _backend = torch if isinstance(self_attr, torch.Tensor) else np
                    axis = 0 if self_attr.ndim <= 2 else 1
                    new_attr = _backend.concatenate([self_attr, other_attr], axis)
                else:
                    new_attr = self_attr + other_attr
                setattr(self, attr, new_attr)

        return self

    def numpy(self) -> TokSequence:
        """
        Convert ``TokSequence`` `torch.Tensor` attributes into `np.ndarray`.

        The `ìds``, ``values`` and ``interpolated`` will be transformed.

        :return: the sequence with converted attributes.
        """
        seq = replace(self)
        attributes = ["ids", "values", "interpolated"]
        for attr in attributes:
            data = getattr(self, attr)
            if data is not None and isinstance(data, torch.Tensor):
                setattr(seq, attr, data.detach().cpu().numpy())
        return seq

    def torch(self, device: str | torch.device = None) -> TokSequence:
        """
        Convert ``TokSequence`` `np.ndarray` attributes into `torch.Tensor`.

        The `ìds``, ``values`` and ``interpolated`` will be transformed.

        :return: the sequence with converted attributes.
        """
        seq = replace(self)
        attributes = ["ids", "values", "interpolated"]
        for attr in attributes:
            data = getattr(self, attr)
            if data is not None:
                data = torch.from_numpy(data) if isinstance(data, np.ndarray) else data
                setattr(seq, attr, data.to(device=device))
                if attr == "values":
                    seq.values = seq.values.float()
        return seq

    def repeat(self, batch_size: int = 1):
        """
        Repeat ``TokSequence`` attributes across batch dimension.

        The `ìds``, ``values`` and ``interpolated`` will be repeated.

        :return: the sequence with batch attributes.
        """
        seq = replace(self)
        attributes = ["ids", "values", "interpolated"]
        for attr in attributes:
            data = getattr(self, attr)
            if data is not None:
                data = data[None]
                if batch_size > 1:
                    if isinstance(data, np.ndarray):
                        data = data.repeat(batch_size, axis=0)
                    else:
                        data = data.expand(batch_size, *([-1] * (len(data.shape) - 1)))
                setattr(seq, attr, data)
        return seq


class TokenizerConfig(MidiTokTokenizerConfig):
    r"""
    MIDI tokenizer base class, containing common methods and attributes for all tokenizers.
    :param special_tokens: list of special tokens. This must be given as a list of strings given
            only the names of the tokens. (default: ``["PAD", "MASK", "BOS", "EOS"]``\)
    :param **kwargs: additional parameters that will be saved in `config.additional_params`.
    """

    def __init__(
        self,
        special_tokens: Sequence[str] = SPECIAL_TOKENS,
        **kwargs,
    ):
        super().__init__(special_tokens=special_tokens, **kwargs)


SEQUENCE_TRANSFORMS = {
    SequenceType.SCORE: [
        EncodingType.SCORE,
        EncodingType.PLAIN_SCORE,
        EncodingType.PERFORMANCE,  # zero deviations and times based on score tempos
        EncodingType.REL_PERFORMANCE,  # zero deviations
        EncodingType.TIME_PERFORMANCE,  # times based on score tempos
        EncodingType.SCORE_TIME_PERFORMANCE,  # score tokens + times based on score tempos
    ],
    SequenceType.PERFORMANCE: [
        EncodingType.SCORE,
        EncodingType.PLAIN_SCORE,
        EncodingType.PERFORMANCE,
        EncodingType.REL_PERFORMANCE,
        EncodingType.TIME_PERFORMANCE,
        EncodingType.SCORE_TIME_PERFORMANCE,
    ],
    SequenceType.SYNC_PERFORMANCE: [
        EncodingType.SCORE,
        EncodingType.PLAIN_SCORE,
        EncodingType.TIME_PERFORMANCE,
        EncodingType.SCORE_TIME_PERFORMANCE,
    ],
    SequenceType.TIME_PERFORMANCE: [
        EncodingType.TIME_PERFORMANCE,
    ],
    SequenceType.PERFORMANCE_SUSTAIN: [
        EncodingType.SCORE,
        EncodingType.PLAIN_SCORE,
        EncodingType.PERFORMANCE,
        EncodingType.REL_PERFORMANCE,
        EncodingType.TIME_PERFORMANCE,
        EncodingType.SCORE_TIME_PERFORMANCE,
    ],
    SequenceType.TIME_PERFORMANCE_SUSTAIN: [
        EncodingType.TIME_PERFORMANCE,
    ],
}


@dataclass
class TokSequenceContext:
    time_signatures: tuple[np.ndarray, np.ndarray] | None = None
    tempos: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    score_ticks: np.ndarray | None = None
    note_on_ticks: np.ndarray | None = None
    note_on_times: np.ndarray | None = None
    initial_tempo: float | None = None
    onset_pairs: np.ndarray | None = None
    pedals: np.ndarray | None = None


def backend(data: TokSequence | np.ndarray | torch.Tensor):
    if isinstance(data, TokSequence):
        data = data.ids if data.ids is not None else data.values
    return torch if isinstance(data, torch.Tensor) else np
