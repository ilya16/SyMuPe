from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from symusic import Score
from tqdm.auto import tqdm

from symupe.data.datasets import SequenceDataset, SequenceTask
from symupe.data.midi import preprocess_midi
from symupe.data.tokenizers import (
    SyMuPe,
    SyMuPeTransformer,
    TokSequence,
    EncodingType,
    SequenceType,
)
from symupe.models import Model
from symupe.modules.constructor import Constructor


class _InferenceWrapper(Constructor, ABC):
    """Internal class for inference wrappers."""

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
            model: :class:`Model` instance used for inference.
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

        assert isinstance(self.tokenizer, SyMuPe), "Inference wrappers require a SyMuPe tokenizer"
        self.token_transformer = SyMuPeTransformer(tokenizer=self.tokenizer)

    def reset(self) -> None:
        """Resets internal state and sequence data."""
        pass

    @abstractmethod
    def prepare_sequence(self, seq: TokSequence, **kwargs) -> TokSequence:
        """Prepares input sequence for inference task.

        Args:
            seq: :class:`TokSequence` to process.
            **kwargs: Implementation-specific sequence parameters.

        Returns:
            :class:`TokSequence` prepared for model input.
        """
        raise NotImplementedError

    @staticmethod
    def _load_midi(midi: str | Path | Score) -> Score:
        """Standardizes MIDI input loading and initial preprocessing.

        Converts various input formats (file paths, strings, or Score objects)
        into a :class:`symusic.Score` and applies pre-tokenizer preprocessing,
        such as merging tracks and removing duplicate notes.

        Args:
            midi: Input MIDI data provided as a file path, a string path,
                or an existing :class:`symusic.Score` object.

        Returns:
            Preprocessed :class:`symusic.Score` object ready for tokenization.
        """
        if isinstance(midi, (str, Path)):
            midi = Score(midi)

        midi = preprocess_midi(midi, to_single_track=True)

        return midi

    @abstractmethod
    def _tokenize_midi(self, midi: Score) -> TokSequence:
        """Converts :class:`symusic.Score` object into a token sequence.

        Args:
            midi: Preprocessed :class:`symusic.Score` object.

        Returns:
            Raw :class:`TokSequence` in performance encoding.
        """
        raise NotImplementedError

    def to(self, device: str | torch.device) -> _InferenceWrapper:
        """Moves model to specified device.

        Args:
            device: Target device for computation.

        Returns:
            Instance on target device.
        """
        self.device = device
        self.model = self.model.to(device)
        return self


@dataclass
class GeneratorData:
    init_seq: TokSequence | None = None
    gen_seq: TokSequence | None = None

    task: SequenceTask | str | None = None
    has_sos_eos: bool = False
    reached_eos: bool = False


class Generator(_InferenceWrapper):
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
            dataset: Optional dataset for sequence extraction.
            device: Target device for computation.
            **kwargs: Additional generator parameters.
        """
        super().__init__(model, tokenizer, dataset, device, **kwargs)

        self.data: GeneratorData = GeneratorData()

    @abstractmethod
    def reset(self) -> None:
        """Resets internal Generator state and sequence data."""
        raise NotImplementedError

    @abstractmethod
    def prepare_sequence(self, seq: TokSequence, **kwargs) -> GeneratorData:
        """Prepares input sequence and metadata for generation task.

        Args:
            seq: :class:`TokSequence` to process.
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
            Post-processed :class:`TokSequence` object.
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


class Classifier(_InferenceWrapper):
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
            dataset: Optional dataset for sequence extraction.
            device: Target device for computation.
            **kwargs: Additional classifier parameters.
        """
        super().__init__(model, tokenizer, dataset, device, **kwargs)

    @abstractmethod
    def prepare_sequence(self, seq: TokSequence, **kwargs) -> TokSequence:
        """Prepares input sequence for classification task.

        Args:
            seq: :class:`TokSequence` to process.
            **kwargs: Implementation-specific sequence parameters.

        Returns:
            :class:`TokSequence` prepared for model input.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Any:
        """Alias for :meth:`predict`."""
        return self.predict(*args, **kwargs)

    @abstractmethod
    def predict(
        self,
        midi: str | Path | Score | TokSequence,
        **kwargs,
    ) -> Any:
        """Performs inference on input sequence to predict labels.

        Args:
            midi: Input MIDI provided as path, :class:`symusic.Score` object,
                or :class:`TokSequence` object.

        Returns:
            Task-specific classification result object.
        """
        raise NotImplementedError


class Embedder(_InferenceWrapper):
    """Base class for musical sequence embedders."""

    def __init__(
        self,
        model: Model,
        tokenizer: SyMuPe,
        dataset: SequenceDataset | None = None,
        device: str | torch.device | None = None,
        **kwargs,
    ):
        """Initializes the Embedder for musical representation learning.

        Args:
            model: :class:`Model` instance (usually a backbone/MLM).
            tokenizer: :class:`SyMuPe` tokenizer for sequence processing.
            dataset: Optional dataset for sequence extraction.
            device: Target device for computation.
            **kwargs: Additional embedder parameters.
        """
        super().__init__(model, tokenizer, dataset, device, **kwargs)

    @abstractmethod
    def prepare_sequence(self, seq: TokSequence, **kwargs) -> TokSequence:
        """Prepares input sequence for feature extraction task.

        Args:
            seq: :class:`TokSequence` to process.
            **kwargs: Implementation-specific sequence parameters.

        Returns:
            :class:`TokSequence` prepared for model input.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Any:
        """Alias for :meth:`embed`."""
        return self.embed(*args, **kwargs)

    @abstractmethod
    def embed(
        self,
        midi: str | Path | Score | TokSequence,
        **kwargs,
    ) -> Any:
        """Performs inference on input sequence to compute embeddings.

        Args:
            midi: Input MIDI provided as path, :class:`symusic.Score` object,
                or :class:`TokSequence` object.

        Returns:
            Task-specific embedding result object.
        """
        raise NotImplementedError


class _PerformanceInference(_InferenceWrapper):
    """Internal mixin for performance-based inference tasks.

    Handles domain-specific logic for expressive music performance data,
    including time position encoding and pedal filtering.
    """

    def __init__(
        self,
        model: Model,
        tokenizer: SyMuPe,
        dataset: SequenceDataset | None = None,
        used_token_types: list[str] | None = None,
        device: str | torch.device | None = None,
        **kwargs,
    ):
        """Initializes music classifier with model and tokenizer.

        Args:
            model: :class:`Model` instance used for generation.
            tokenizer: :class:`SyMuPe` tokenizer instance for encoding.
            dataset: Optional dataset for sequence extraction.
            used_token_types: List of token types used by the model.
            device: Target device for computation.
            **kwargs: Additional classifier parameters.
        """
        super().__init__(model, tokenizer, dataset, device, **kwargs)

        self.used_token_types = used_token_types
        self.use_pedals = self.model._config.get("use_pedals", kwargs.get("use_pedals", False))

    def _tokenize_midi(self, midi: Score) -> TokSequence:
        """Converts :class:`symusic.Score` object into a performance-based token sequence.

        Args:
            midi: Preprocessed :class:`symusic.Score` object.

        Returns:
            Raw :class:`TokSequence` in performance encoding.
        """
        seq = self.tokenizer.encode_performance(midi, score_tokens=None)
        return seq

    def prepare_sequence(self, seq: TokSequence, **kwargs) -> TokSequence:
        """Aligns performance sequence with model backbone expectations.

        Applies time position tokens, filters pedals, and normalizes values
        based on the model configuration.

        Args:
            seq: Raw :class:`TokSequence` to process.
            **kwargs: Unused arguments for compatibility.

        Returns:
            Processed :class:`TokSequence` ready for model input.
        """
        # process special tokens
        if "TimePosition" in self.used_token_types:
            seq = self.tokenizer.add_time_position_tokens(seq, segment_tokens=False)

        if not self.use_pedals:
            seq = self.tokenizer.remove_pedal_tokens(seq)

        # prepare values
        seq = self.tokenizer.normalize_values(seq)
        seq = self.tokenizer.clip_values(seq)

        # compress sequence to token types the backbone expects
        seq = self.tokenizer.compress(seq, token_types=self.used_token_types)

        return seq


class _SlidingWindowInference(_InferenceWrapper):
    """Internal mixin for chunked sequence processing.

    Implements the sliding window algorithm to enable inference on arbitrarily long
    musical sequences by splitting them into overlapping mini-batches.
    """

    @abstractmethod
    def _forward_batch(self, tokens: torch.Tensor, values: torch.Tensor, **kwargs):
        """Abstract method for task-specific batch inference."""
        raise NotImplementedError

    def _sliding_window_inference(
        self,
        seq: TokSequence,
        max_seq_len: int = 512,
        hop_size: int = 256,
        batch_size: int = 16,
        show_progress: bool = True,
        **kwargs,
    ) -> tuple[list[Any], list[TokSequence], list[tuple[int, int]]]:
        """Core engine for windowed batch processing.

        Args:
            seq: Input :class:`TokSequence`.
            max_seq_len: Maximum token length for each window.
            hop_size: Number of tokens to shift between windows.
            batch_size: Number of windows processed in parallel batch.
            show_progress: Whether to display tqdm progress bar.
            **kwargs: Arguments passed to :meth:`_forward_batch`.

        Returns:
            Tuple of (raw_model_outputs, processed_sequences, window_indices).
        """
        tokens, values = seq.ids, seq.values

        # slice into windows
        total_len = len(seq)
        sequences, window_indices = [], []
        if total_len <= max_seq_len:
            # single window
            seq = self.prepare_sequence(seq).torch()
            sequences.append(seq)
            batch_tokens = seq.ids.unsqueeze(0)
            batch_values = seq.values.unsqueeze(0)
            window_indices.append((0, total_len))
        else:
            # create sliding windows
            batch_tokens, batch_values = [], []

            start_indices = list(range(0, total_len - max_seq_len + 1, hop_size))
            last_start = total_len - max_seq_len
            if start_indices[-1] != last_start:  # add the last bit
                start_indices.append(last_start)

            for i in start_indices:
                _seq = replace(
                    seq,
                    ids=tokens[i : i + max_seq_len],
                    values=values[i : i + max_seq_len],
                    token_to_note=None,
                )
                _seq = self.prepare_sequence(_seq).torch()
                sequences.append(_seq)
                batch_tokens.append(_seq.ids)
                batch_values.append(_seq.values)
                window_indices.append((i, i + max_seq_len))

            batch_tokens = torch.stack(batch_tokens)
            batch_values = torch.stack(batch_values)

        # inference in mini-batches
        pbar = None
        if show_progress:
            pbar = tqdm(total=len(batch_tokens))

        all_outputs = []
        for i in range(0, len(batch_tokens), batch_size):
            tokens = batch_tokens[i : i + batch_size].to(self.device)
            values = batch_values[i : i + batch_size].to(self.device)

            out = self._forward_batch(tokens=tokens, values=values, **kwargs)
            all_outputs.append(out)

            if pbar is not None:
                pbar.update(len(tokens))

        if pbar is not None:
            pbar.close()

        return all_outputs, sequences, window_indices
