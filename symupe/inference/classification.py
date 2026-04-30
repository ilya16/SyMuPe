from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from symusic import Score

from symupe.data.datasets import SequenceDataset
from symupe.data.tokenizers import SyMuPe, TokSequence
from symupe.models import Model
from .base import Classifier, _PerformanceInference, _SlidingWindowInference


@dataclass
class MusicClassificationResult:
    """Storage for music classification results.

    Args:
        midi: Original input MIDI object.
        seq: Token sequence used for inference.
        probabilities: Dictionary mapping label names to mean probabilities.
        prediction: Predicted class index.
        label: Predicted label name.
        all_logits: Raw logits for each sliding window.
        all_probabilities: Softmax probabilities for each sliding window.
        all_predictions: Predicted class indices for each sliding window.
        sequences: List of windowed :class:`TokSequence` objects processed by the model.
        window_indices: List of (start, end) token indices for each window.
    """

    midi: Score
    seq: TokSequence
    probabilities: dict[str, float]
    prediction: int
    label: str | None
    all_logits: torch.Tensor | None
    all_probabilities: torch.Tensor | None
    all_predictions: torch.Tensor | None
    sequences: list[TokSequence]
    window_indices: list[tuple[int, int]]


class MusicClassifier(_PerformanceInference, _SlidingWindowInference, Classifier):
    """Inference wrapper for musical sequence classification tasks.

    Handles sliding window inference and result aggregation for long musical sequences.
    This class is designed for tasks like MIDI quality assessment or style identification.
    """

    def __init__(
        self,
        model: Model,
        tokenizer: SyMuPe,
        dataset: SequenceDataset | None = None,
        used_token_types: list[str] | None = None,
        labels: dict[int, str] | None = None,
        device: str | torch.device | None = None,
        **kwargs,
    ):
        """Initializes music classifier with model and tokenizer.

        Args:
            model: :class:`Model` instance used for generation.
            tokenizer: :class:`SyMuPe` tokenizer instance for encoding.
            dataset: Optional dataset for sequence extraction.
            used_token_types: List of token types used by classifier backbone.
            labels: Optional mapping from class indices to string labels.
            device: Target device for computation.
            **kwargs: Additional classifier parameters.
        """
        used_token_types = used_token_types or list(model._config.backbone.num_tokens.keys())

        super().__init__(model, tokenizer, dataset, used_token_types, device, **kwargs)

        self.labels = (
            labels
            or getattr(model, "labels", None)
            or {i: str(i) for i in range(model.num_classes)}
        )

    def _forward_batch(self, tokens: torch.Tensor, values: torch.Tensor, **kwargs) -> torch.Tensor:
        """Internal method to perform batch inference on transformer backbone.

        Args:
            tokens: Batch of token IDs.
            values: Batch of token values.
            **kwargs: Additional arguments for the model's forward pass.

        Returns:
            Tensor of logits moved to CPU.
        """
        out = self.model(tokens=tokens, values=values, **kwargs)
        return out.logits.cpu()

    @torch.inference_mode()
    def predict(
        self,
        midi: str | Path | Score | TokSequence,
        max_seq_len: int = 512,
        hop_size: int = 256,
        batch_size: int = 16,
        show_progress: bool = True,
    ) -> MusicClassificationResult:
        """Classifies musical sequence using sliding window approach.

        Splits the sequence into overlapping chunks, performs inference on each chunk,
        and aggregates the results using soft voting (averaging probabilities).

        Args:
            midi: Input MIDI path, :class:`symusic.Score` object, or :class:`TokSequence` object.
            max_seq_len: Maximum token length for each window.
            hop_size: Number of tokens to shift between windows.
            batch_size: Number of windows processed in parallel batch.
            show_progress: Whether to display tqdm progress bar.

        Returns:
            :class:`MusicClassificationResult` containing aggregated and per-window predictions.
        """
        # load MIDI and prepare token sequence
        if not isinstance(midi, TokSequence):
            midi = self._load_midi(midi)

        seq = self._tokenize_midi(midi)

        init_seq = copy.deepcopy(seq)

        all_logits, sequences, window_indices = self._sliding_window_inference(
            seq=seq,
            max_seq_len=max_seq_len,
            hop_size=hop_size,
            batch_size=batch_size,
            show_progress=show_progress,
        )

        logits = torch.cat(all_logits)  # (num_windows, num_classes)
        probabilities = F.softmax(logits, dim=-1)

        # aggregate results (average probabilities across windows)
        probs = probabilities.mean(dim=0)
        prediction = probs.argmax().item()

        probability_dict = {
            label: prob for label, prob in zip(self.labels.values(), probs.tolist())
        }

        return MusicClassificationResult(
            midi=midi,
            seq=init_seq,
            probabilities=probability_dict,
            prediction=prediction,
            label=self.labels.get(prediction),
            all_logits=logits,
            all_probabilities=probabilities,
            all_predictions=logits.argmax(dim=-1),
            sequences=sequences,
            window_indices=window_indices,
        )


def test():
    import torch
    from symusic import Score
    from symupe.inference import AutoClassifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build Classifier by loading the model and tokenizer directly from the Hub
    classifier: MusicClassifier = AutoClassifier.from_pretrained(
        "SyMuPe/MIDI-Quality-Classifier", device=device
    )
    # model, tokenizer, labels = classifier.model, classifier.tokenizer, classifier.labels

    # Load MIDI
    midi = Score("performance.mid")

    # Classify MIDI (tokenization is handled inside)
    result = classifier(midi)
    # result is MusicClassificationResult(...) containing:
    # - midi, seq, probabilities, prediction, label, all_logits, all_probabilities, all_predictions,
    #   sequences and window_indices
    print(result.label, result.probabilities)


if __name__ == "__main__":
    test()
