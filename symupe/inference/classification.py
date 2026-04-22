from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn.functional as F
from symusic import Score
from tqdm.auto import tqdm

from symupe.data.datasets import SequenceDataset
from symupe.data.midi import preprocess_midi
from symupe.data.tokenizers import SyMuPe, TokSequence
from symupe.models import Model
from .base import Classifier


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
    """

    midi: Score
    seq: TokSequence
    probabilities: dict[str, float]
    prediction: int
    label: str | None
    all_logits: torch.Tensor | None
    all_probabilities: torch.Tensor | None
    all_predictions: torch.Tensor | None


class MusicClassifier(Classifier):
    """Inference wrapper for musical sequence classification tasks.

    Handles sliding window inference and result aggregation for long musical sequences.
    """

    def __init__(
        self,
        model: Model,
        tokenizer: SyMuPe,
        dataset: SequenceDataset | None = None,
        labels: dict[int, str] | None = None,
        device: str | torch.device | None = None,
        **kwargs,
    ):
        """Initializes music classifier with model and tokenizer.

        Args:
            model: :class:`Model` instance used for generation.
            tokenizer: :class:`SyMuPe` tokenizer instance for encoding.
            dataset: Optional dataset for metadata.
            labels: Optional mapping from class indices to string labels.
            device: Target device for computation.
            **kwargs: Additional classifier parameters.
        """
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            device=device,
            **kwargs,
        )

        self.used_token_types = list(self.model.backbone_config.num_tokens.keys())

        self.labels = (
            labels
            or getattr(model, "labels", None)
            or {i: str(i) for i in range(model.num_classes)}
        )

    def prepare_sequence(self, seq: TokSequence) -> TokSequence:
        """Prepares performance sequence for classifier input.

        Processes time positions, removes pedals, and normalizes values.

        Args:
            seq: :class:`TokSequence` to process.

        Returns:
            Processed :class:`TokSequence` ready for model backbone.
        """
        # process special tokens
        seq = self.tokenizer.add_time_position_tokens(seq, segment_tokens=False)
        seq = self.tokenizer.remove_pedal_tokens(seq)

        # prepare values
        seq = self.tokenizer.normalize_values(seq)
        seq = self.tokenizer.clip_values(seq)

        # compress sequence to token types the backbone expects
        seq = self.tokenizer.compress(seq, token_types=self.used_token_types)

        return seq

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

        Splits sequence into overlapping chunks and averages probabilities across windows.

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
        if isinstance(midi, (str, Path)):
            midi = Score(str(midi))

        if isinstance(midi, Score):
            midi = preprocess_midi(midi, to_single_track=True)

            seq = self.tokenizer.encode_performance(midi, score_tokens=None)
        else:
            seq = midi

        init_seq = copy.deepcopy(seq)
        tokens, values = seq.tokens, seq.values

        # slice into windows
        total_len = len(seq)
        if total_len <= max_seq_len:
            # single window
            seq = self.prepare_sequence(seq).torch()
            batch_tokens = seq.ids.unsqueeze(0)
            batch_values = seq.values.unsqueeze(0)
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
                )
                _seq = self.prepare_sequence(_seq).torch()
                batch_tokens.append(_seq.ids)
                batch_values.append(_seq.values)

            batch_tokens = torch.stack(batch_tokens)
            batch_values = torch.stack(batch_values)

        # inference in mini-batches
        pbar = None
        if show_progress:
            pbar = tqdm(total=len(batch_tokens))

        all_logits = []
        for i in range(0, len(batch_tokens), batch_size):
            tokens = batch_tokens[i : i + batch_size].to(self.device)
            values = batch_values[i : i + batch_size].to(self.device)

            out = self.model(tokens=tokens, values=values)
            all_logits.append(out.logits.cpu())

            if pbar is not None:
                pbar.update(len(tokens))

        if pbar is not None:
            pbar.close()

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
        )


def test():
    import torch
    from symusic import Score
    from symupe.inference import AutoClassifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Built the Generator by loading the model and tokenizer directly from the Hub
    classifier: MusicClassifier = AutoClassifier.from_pretrained(
        "SyMuPe/MIDI-Quality-Classifier", device=device
    )
    # model, tokenizer, labels = classifier.model, classifier.tokenizer, classifier.labels

    # Load MIDI
    midi = Score("performance.mid")

    # Classify MIDI (tokenization is handled inside)
    result = classifier.predict(midi=midi)
    # result is MusicClassificationResult(...) containing:
    # - midi, seq, probabilities, prediction, label, all_logits, all_probabilities, all_predictions
    print(result.label, result.probabilities)


if __name__ == "__main__":
    test()
