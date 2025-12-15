""" Embedding Classifier evaluator. """
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from symupe.modules.metrics import accuracy, accuracy_topk, Reduction, mean_reciprocal_rank


class EmbeddingClassifierEvaluator:
    def __init__(self, model, top_k: int | Sequence[int] = (3, 5), **kwargs):
        self.model = model
        self.top_k = top_k

    @torch.no_grad()
    def __call__(self, inputs, outputs) -> dict[str, torch.Tensor]:
        labels = inputs["labels"]

        # standard top-1 prediction accuracy
        predictions = torch.argmax(outputs.logits, dim=-1)
        metrics = {"accuracy": accuracy(predictions, labels, reduction=Reduction.MEAN)}

        # top-k accuracies
        for k in self.top_k:
            if outputs.logits.shape[-1] >= k:
                metrics[f"accuracy/top-{k}"] = accuracy_topk(outputs.logits, labels, k=k, reduction=Reduction.MEAN)

        # MRR from logits
        mr, mrr = mean_reciprocal_rank(outputs.logits, labels, reduction=Reduction.MEAN)
        metrics["mean_rank"] = mr
        metrics["MRR"] = mrr

        # note accuracy if `note_logits` are provided
        if outputs.note_logits is not None:
            predictions_notes = torch.argmax(outputs.note_logits, dim=-1)
            mask = inputs["mask"]
            if predictions_notes.shape[1] > mask.shape[1]:
                mask = F.pad(mask, (predictions_notes.shape[1] - mask.shape[1], 0), value=True)
            metrics["accuracy/note"] = accuracy(
                predictions_notes, labels[..., None], mask=mask, reduction=Reduction.BATCH_MEAN
            )

        return metrics


class SequenceClassifierEvaluator(EmbeddingClassifierEvaluator):
    ...
