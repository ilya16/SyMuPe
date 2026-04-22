from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import torch
from symusic import Score

from symupe.data.datasets import SequenceDataset
from symupe.data.datasets import SequenceTask
from symupe.data.tokenizers import (
    SyMuPe,
    SyMuPeTransformer,
    TokSequence,
    EncodingType,
    SequenceType,
)
from symupe.models import Model
from symupe.modules.constructor import Constructor


@dataclass
class GeneratorData:
    init_seq: TokSequence | None = None
    gen_seq: TokSequence | None = None

    task: SequenceTask | str | None = None
    has_sos_eos: bool = False
    reached_eos: bool = False


class Generator(Constructor, ABC):
    """Base class for generative music models."""

    def __init__(
        self,
        model: Model,
        tokenizer: SyMuPe,
        dataset: SequenceDataset | None = None,
        device: str | torch.device | None = None,
        **kwargs,
    ):
        """Initializes Generator with model and tokenizer.

        Args:
            model: :class:`Model` instance used for generation.
            tokenizer: :class:`SyMuPe` tokenizer instance for encoding/decoding.
            dataset: Optional dataset for sequence context.
            device: Target device for computation.
            **kwargs: Additional generator parameters.
        """

        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.device = device

        if self.device is not None:
            self.model = self.model.to(self.device)
        self.model.eval()

        assert isinstance(self.tokenizer, SyMuPe)
        self.token_transformer = SyMuPeTransformer(tokenizer=self.tokenizer)

        self.data: GeneratorData = GeneratorData()

    @abstractmethod
    def reset(self) -> None:
        """Resets internal Generator state and sequence data."""
        raise NotImplementedError

    @abstractmethod
    def prepare_sequence(self, **kwargs) -> GeneratorData:
        """Prepares input sequence and metadata for generation task.

        Args:
            **kwargs: Implementation-specific sequence parameters.

        Returns:
            :class:`GeneratorData` object containing prepared tensors.
        """
        raise NotImplementedError

    def generated_sequence(
        self,
        postprocess: bool = True,
        encoding: EncodingType | str | None = None,
        seq_type: SequenceType | str | None = None,
    ) -> TokSequence:
        """Retrieves and optionally post-processes generated token sequence.

        Args:
            postprocess: Whether to denormalize and decompress tokens.
            encoding: Target encoding type for output sequence.
            seq_type: Target sequence type for output metadata.

        Returns:
            Post-processed TokSequence object.
        """
        gen_seq = self.data.gen_seq[int(self.data.has_sos_eos) :].numpy()
        if seq_type is not None:
            gen_seq.type = seq_type

        if postprocess:
            gen_seq = self.tokenizer.denormalize_values(gen_seq)
            gen_seq = self.tokenizer.decompress(copy.deepcopy(gen_seq))

            if encoding is not None:
                gen_seq = self.token_transformer(gen_seq, encoding)

        return gen_seq

    def to(self, device: str | torch.device) -> Generator:
        """Moves model and internal generator tensors to specified device.

        Args:
            device: Target device for computation.

        Returns:
            :class:`Generator` instance on target device.
        """
        self.device = device
        self.model = self.model.to(device)
        return self


class Classifier(Constructor, ABC):
    """Base class for music classification models."""

    def __init__(
        self,
        model: Model,
        tokenizer: SyMuPe,
        dataset: SequenceDataset | None = None,
        device: str | torch.device | None = None,
        **kwargs,
    ):
        """Initializes classifier with model and tokenizer.

        Args:
            model: :class:`Model` instance used for generation.
            tokenizer: :class:`SyMuPe` tokenizer instance for encoding.
            dataset: Optional dataset for metadata.
            device: Target device for computation.
            **kwargs: Additional classifier parameters.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.device = device

        if self.device is not None:
            self.model = self.model.to(self.device)
        self.model.eval()

        assert isinstance(self.tokenizer, SyMuPe)
        self.token_transformer = SyMuPeTransformer(tokenizer=self.tokenizer)

    def reset(self) -> None:
        """Resets classifier state."""
        pass

    @abstractmethod
    def prepare_sequence(self, **kwargs) -> TokSequence:
        """Prepares input sequence for classification task.

        Args:
            **kwargs: Implementation-specific sequence parameters.

        Returns:
            `class:`TokSequence` prepared for model input.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        midi: str | Path | Score | TokSequence,
    ) -> dict[str, object]:
        """Performs inference on input score to predict labels.

        Args:
            midi: Input MIDI provided as path, :class:`symusic.Score` object,
                or class:`TokSequence` object.

        Returns:
            Task-specific classification result object.
        """
        raise NotImplementedError

    def to(self, device: str | torch.device) -> Classifier:
        """Moves model to specified device.

        Args:
            device: Target device for computation.

        Returns:
            :class:`Classifier` instance on target device.
        """
        self.device = device
        self.model = self.model.to(device)
        return self
