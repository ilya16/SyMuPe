from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


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
