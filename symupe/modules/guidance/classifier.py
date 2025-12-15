from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from symupe.data.collators import ScorePerformanceCollator
from symupe.data.datasets import ScorePerformanceSample
from symupe.data.tokenizers import TokSequence, SyMuPe
from symupe.models import MODELS, MusicTransformer, Seq2SeqMusicTransformer


class MLMClassifier(nn.Module):
    """
    A MIDI transformer based on a pretrained MLM for classifying segments by emotional labels.

    Developed by @realfolkcode.
    """

    def __init__(
            self,
            mlm_path: str | None,
            mlm_config: dict | None,
            nclasses: int,
            d_model: int,
            d_hid: int,
            d_mlm: int,
            output_layer: int,
            nhead: int,
            nlayers: int,
            device: str | None = None,
            checkpoint_name: str | None = None,
            dropout: float = 0.5,
            is_mlm_frozen: bool = True,
            random_smoothing: float = 0.,
            labels: dict[int, str] | None = None,
            backbone: nn.Module = Seq2SeqMusicTransformer
    ):
        """Initializes a transformer model.

        Args:
            mlm_path: The path to a pretrained MLM checkpoint.
            nclasses: The number of classes.
            d_model: The dimension of input embeddings.
            d_hid: The dimension of hidden embeddings.
            d_mlm: The dimension of MLM embeddings.
            output_layer: The output layer index of MLM embeddings to use.
            nhead: The number of heads.
            nlayers: The number of layers.
            device: The device.
            checkpoint_name: The name of the output checkpoint.
            dropout: The ratio of dropout in layers.
            is_mlm_frozen: If True, freezes MLM.
            random_smoothing: The variance of random smoothing of MLM embeddings.
            backbone: The class of the backbone that outputs embeddings.
        """
        super().__init__()
        self.nclasses = nclasses
        self.d_model = d_model
        self.device = device
        self.output_layer = output_layer
        self.checkpoint_name = checkpoint_name
        self.is_mlm_frozen = is_mlm_frozen
        self.random_smoothing = random_smoothing
        self.labels = labels

        self.backbone = backbone
        assert backbone in (MusicTransformer, Seq2SeqMusicTransformer)

        assert mlm_path is not None or mlm_config is not None
        if mlm_path is not None:
            self.mlm_model = backbone.from_pretrained(checkpoint_path=mlm_path, strict=False)
        else:
            self.mlm_model = backbone.init(mlm_config)

        if backbone is MusicTransformer:
            self.mlm_model = self.mlm_model.unwrap_model()
        elif backbone is Seq2SeqMusicTransformer:
            self.mlm_model = self.mlm_model.encoder

        self.mlm_model = self.mlm_model.to(device)
        self.mlm_model.eval()
        self.input_dropout = nn.Dropout(p=0.5)

        self.projection = nn.Linear(d_mlm, d_model, bias=False)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            d_hid,
            dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)

        self.batch_norm = nn.BatchNorm1d(d_model)

        self.classifier = nn.Linear(d_model, nclasses)

    def forward(
            self,
            x: torch.Tensor,
            values: torch.Tensor,
            mask: torch.Tensor | None = None,
            return_mlm_embeddings: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: MIDI input of shape (B, midi_len, tuple_len)
            values: MIDI values of shape (B, midi_len, tuple_len)
            mask: Attention mask of shape for `x`
            return_mlm_embeddings: whether to return MLM embeddings with the classification scores

        Returns:
            Logits of shape (B, nclasses).
        """
        if mask is None:
            mask = torch.ones_like(x[..., 0], dtype=torch.bool, device=x.device)

        if self.is_mlm_frozen:
            with torch.no_grad():
                mlm_emb = self.mlm_model(x, values=values, mask=mask, output_layer=self.output_layer)
        else:
            mlm_emb = self.mlm_model(x, values=values, mask=mask, output_layer=self.output_layer)

        mlm_emb = mlm_emb.hidden_state

        # Normalize if not the last layer
        if self.backbone == MusicTransformer:
            if self.output_layer < len(self.mlm_model.transformer.transformer.layers) - 1:
                mlm_emb = self.mlm_model.transformer.transformer.layers[self.output_layer + 1].attention_norm(mlm_emb)
        elif self.backbone == Seq2SeqMusicTransformer:
            if self.output_layer < len(self.mlm_model.transformer.layers) - 1:
                mlm_emb = self.mlm_model.transformer.layers[self.output_layer + 1].attention_norm(mlm_emb)
        else:
            raise NotImplementedError

        noise = torch.randn_like(mlm_emb).to(self.device) * self.random_smoothing
        mlm_emb = mlm_emb + noise

        x = self.input_dropout(mlm_emb)
        x = F.relu(self.projection(x))

        out = self.transformer_encoder(x, src_key_padding_mask=~mask)
        out = out.mean(dim=1)
        out = F.relu(out)
        out = self.batch_norm(out)
        out = self.classifier(out)

        if return_mlm_embeddings:
            return out, mlm_emb
        return out

    def forward_from_emb(
            self,
            mlm_emb: torch.Tensor,
            mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(mlm_emb[..., 0], dtype=torch.bool, device=mlm_emb.device)

        x = self.input_dropout(mlm_emb)
        x = F.relu(self.projection(x))

        out = self.transformer_encoder(x, src_key_padding_mask=~mask)
        out = out.mean(dim=1)
        out = F.relu(out)
        out = self.batch_norm(out)
        out = self.classifier(out)

        return out

    @classmethod
    def from_pretrained(
            cls,
            checkpoint_path: str,
            strict: bool = True
    ) -> MLMClassifier:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        config = checkpoint["config"]
        mlm_config = checkpoint["mlm_config"]
        labels = checkpoint["labels"]

        model = cls(
            mlm_path=None,
            mlm_config=mlm_config,
            nclasses=config["model"]["nclasses"],
            d_model=config["model"]["d_model"],
            d_hid=config["model"]["d_hid"],
            d_mlm=config["model"]["d_mlm"],
            output_layer=config["model"]["output_layer"],
            nhead=config["model"]["nhead"],
            nlayers=config["model"]["nlayers"],
            dropout=config["model"]["dropout"],
            is_mlm_frozen=False,
            labels=labels,
            backbone=MODELS[checkpoint.get("backbone", "MusicTransformer")]
        )

        state_dict = checkpoint["state_dict"]

        for key, weight in model.state_dict().items():
            if key not in state_dict:
                state_dict[key] = weight
        model.load_state_dict(state_dict, strict=strict)

        return model


def prepare_sample_from_sequences(
        score_seq: TokSequence,
        perf_seq: TokSequence,
        tokenizer: SyMuPe,
        used_token_types: list[str] | None = None,
        remove_special_tokens: bool = True
) -> ScorePerformanceSample:
    """Prepares a score-performance sample from the corresponding sequences.

    Args:
        score_seq: A score sequence.
        perf_seq: A performance sequence.
        tokenizer: MIDI tokenizer.

    Returns:
        A score-performance sample.
    """
    if remove_special_tokens:
        for seq in (score_seq, perf_seq):
            tokenizer.remove_pedal_tokens(seq)
            tokenizer.remove_bar_line_tokens(seq)

    for seq in (score_seq, perf_seq):
        seq.values = tokenizer.normalize_values(seq.values)

    if used_token_types is not None:
        # score_seq = tokenizer.compress(score_seq, token_types=used_token_types)
        perf_seq = tokenizer.compress(perf_seq, token_types=used_token_types)

    sample = ScorePerformanceSample(
        meta=None,
        score=score_seq,
        perf=perf_seq
    )

    return sample


def slice_sample(
        sample: ScorePerformanceSample,
        start: int,
        end: int
) -> ScorePerformanceSample:
    """Slices a score-performance sample.

    Args:
        sample: A score-performance sample.
        start: The index of a start token for slicing.
        end: The index of an end token for slicing.

    Returns:
        A sliced sample.
    """

    sample_slice = ScorePerformanceSample(
        meta=None,
        score=sample.score[start:end],
        perf=sample.perf[start:end]
    )

    return sample_slice


def get_sliding_predictions(
        classifier: MLMClassifier,
        perf_indices: list[int],
        samples_lst: list[ScorePerformanceSample],
        device: str,
        tokenizer: SyMuPe
) -> tuple[dict[int, int | Any], list[dict[str, int | list[Any]]]]:
    """Computes emotion probabilities across measures.

    Args:
        classifier: Emotion classifier.
        perf_indices: Performance indices.
        samples_lst: A list of score-performance samples.
        device: Device.
        tokenizer: MIDI tokenizer.

    Returns:
        A list containing first token positions in bars for performances, and
        a list of dictionaries with activations.
    """
    collator = ScorePerformanceCollator()
    score_seq = samples_lst[0].score

    bars = tokenizer.get_values(score_seq, "Bar", from_ids=True)
    min_bar, max_bar = 0, bars[-1]

    predictions = []
    for perf_id in perf_indices:
        pred_dict = {"perf_id": perf_id, "activations": []}  # torch.empty((num_classes, 0))}
        predictions.append(pred_dict)

    classifier.eval()

    bar_start_token_ids = {}
    start = 0
    for bar in range(min_bar, max_bar + 1):
        end = start
        while bars[end] == bar:
            end += 1
            if end >= len(bars):
                break
        if end == start:
            for i, perf_id in enumerate(perf_indices):
                predictions[i]["activations"].append(torch.zeros((1, classifier.nclasses), device=device))
            continue

        slices = []
        for sample in samples_lst:
            slices.append(slice_sample(sample, start=start, end=end))
        batch = collator(slices)

        x = batch["performances"]["tokens"].to(device)
        values = batch["performances"]["values"].to(device)
        mask = batch["performances"]["mask"].to(device)

        y_pred = classifier(x, values=values, mask=mask)  # shape: (batch_size, num_classes)
        for i, perf_id in enumerate(perf_indices):
            predictions[i]["activations"].append(y_pred[i].reshape((1, -1)))

        bar_start_token_ids[bar] = start
        start = end

    for i, perf_id in enumerate(perf_indices):
        logits = torch.cat(predictions[i]["activations"], dim=0)

        probs = F.softmax(logits, dim=-1)
        probs[torch.all(logits == 0., dim=-1)] = 0.
        predictions[i]["activations"] = probs.cpu()

    return bar_start_token_ids, predictions
