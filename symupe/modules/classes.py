from __future__ import annotations

from torch import nn as nn

from symupe.utils import ExplicitEnum


class LanguageModelingMode(ExplicitEnum):
    MLM = "mlm"
    CLM = "clm"
    MixedLM = "mixlm"
    DFM = "dfm"
    FM = "fm"


class ValuePredictionMode(ExplicitEnum):
    CFM = "cfm"


class ModelWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, tokens, labels=None, **kwargs):
        ...
