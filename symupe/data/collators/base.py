from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from symupe.modules.constructor import Constructor


@dataclass
class SeqInputs:
    tokens: np.ndarray | torch.Tensor
    mask: np.ndarray | torch.Tensor
    values: np.ndarray | torch.Tensor | None = None


@dataclass
class SeqSegments:
    bar: np.ndarray | torch.Tensor | None = None
    beat: np.ndarray | torch.Tensor | None = None
    onset: np.ndarray | torch.Tensor | None = None


class DataCollator(Constructor):
    """
    Abstract base class for all SyMuPe data collators.

    Provides utilities for padding sequences and handling batch dictionaries.
    Inherits from Constructor to support configuration injection via factory methods.
    """

    @abstractmethod
    def __call__(self, batch: Sequence[object]) -> dataclass | dict[str, object]:
        """
        Process a list of samples into a batch of tensors.

        :param batch: list of dataset samples
        :return: Dictionary of batched tensors.
        """
        raise NotImplementedError("Collator must implement __call__")
