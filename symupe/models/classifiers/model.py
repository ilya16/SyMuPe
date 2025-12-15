""" Classifier Models. """
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import MISSING, DictConfig, OmegaConf

from symupe.data.collators import SequenceInputs
from symupe.data.datasets import SequenceDataset
from symupe.models.base import Model
from symupe.modules.constructor import Registry, ModuleConfig, VariableModuleConfig
from symupe.modules.metrics import masked_batch_mean
from symupe.modules.transformer import TransformerConfig, EncoderTransformer, LayerNorm
from symupe.modules.tuple_transformer import TupleTransformerConfig, TupleTransformer
from symupe.utils import asdict


@dataclass
class EmbeddingClassifierOutput:
    logits: torch.Tensor = None
    loss: torch.Tensor | None = None
    losses: dict[str, torch.Tensor] | None = None


EmbeddingClassifiersRegistry = type("_EmbeddingClassifiersRegistry", (Registry,), {})()


@dataclass
class EmbeddingClassifierConfig(VariableModuleConfig):
    input_dim: int = MISSING
    num_classes: int = MISSING
    dropout: bool = 0.
    weight: list[float] | None = None


@dataclass
class LinearEmbeddingClassifierConfig(EmbeddingClassifierConfig):
    _target_: str = "linear"
    hidden_dims: Sequence[int] | None = field(default_factory=lambda: (32,))


@EmbeddingClassifiersRegistry.register("linear")
class LinearEmbeddingClassifier(Model):
    def __init__(
            self,
            input_dim: int,
            num_classes: int,
            hidden_dims: Sequence[int] | None = (32,),
            dropout: bool = 0.,
            class_weights: list[float] | None = None,
            label_smoothing: float = 0.
    ):
        super().__init__()

        self.num_classes = num_classes

        class_weights = torch.ones(num_classes) if class_weights is None else torch.tensor(class_weights)
        self.register_buffer("class_weights", class_weights.float())

        hidden_dims = hidden_dims or []
        hidden_dims = [hidden_dims] if isinstance(hidden_dims, int) else hidden_dims
        hidden_dims = list(hidden_dims)

        in_dims = [input_dim] + hidden_dims
        out_dims = hidden_dims + [num_classes]

        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(in_dims) - 1:
                layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.label_smoothing = label_smoothing

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None):
        x = embeddings.squeeze(1) if embeddings.ndim == 3 else embeddings
        for layer in self.layers:
            x = layer(self.dropout(x))
        logits = x

        loss = losses = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights, label_smoothing=self.label_smoothing)

        return EmbeddingClassifierOutput(
            logits=logits,
            loss=loss,
            losses=losses
        )

    def prepare_inputs(self, inputs: dict[str, torch.Tensor], *args) -> dict[str, torch.Tensor]:
        return inputs


@dataclass
class SequentialEmbeddingClassifierConfig(EmbeddingClassifierConfig):
    _target_: str = "sequential"
    hidden_dim: int = 32


@EmbeddingClassifiersRegistry.register("sequential")
class SequentialEmbeddingClassifier(Model):
    def __init__(
            self,
            input_dim: int,
            num_classes: int,
            hidden_dim: int = 32,
            dropout: bool = 0.,
            class_weights: list[float] | None = None
    ):
        super().__init__()

        self.num_classes = num_classes

        class_weights = torch.ones(num_classes) if class_weights is None else torch.tensor(class_weights)
        self.register_buffer("class_weights", class_weights.float())

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            dropout=dropout
        )

        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None) -> EmbeddingClassifierOutput:
        self.gru.flatten_parameters()
        _, out = self.gru(embeddings)  # (1, b, h)
        logits = self.output(out[0])

        loss = losses = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)

        return EmbeddingClassifierOutput(
            logits=logits,
            loss=loss,
            losses=losses
        )

    def prepare_inputs(self, inputs, *args) -> dict[str, torch.Tensor]:
        return inputs


@dataclass
class MultiHeadEmbeddingClassifierOutput:
    logits: dict[str, torch.Tensor] = None
    loss: torch.Tensor | None = None
    losses: dict[str, torch.Tensor] | None = None


@dataclass
class MultiHeadEmbeddingClassifierConfig(VariableModuleConfig):
    _target_: str = "multi-head"
    input_dim: int = MISSING
    num_classes: dict[str, int] = MISSING
    classifier: LinearEmbeddingClassifierConfig = MISSING
    class_samples: dict[str, list[int]] | None = None
    weighted_classes: bool = False
    loss_weight: float = 1.
    detach_inputs: bool | float = False


@EmbeddingClassifiersRegistry.register("multi-head")
class MultiHeadEmbeddingClassifier(Model):
    def __init__(
            self,
            input_dim: int,
            num_classes: dict[str, int],
            classifier: LinearEmbeddingClassifierConfig,
            class_samples: dict[str, list[int]] | None = None,
            loss_weight: float = 1.,
            weighted_classes: bool = False,
            detach_inputs: bool | float = False
    ):
        super().__init__()

        self.num_classes = num_classes

        self.heads = nn.ModuleDict({})
        for key, num in num_classes.items():
            num_samples = class_samples.get(key, None) if class_samples is not None else None
            class_weights = self._class_weights(num_samples) if weighted_classes and num_samples is not None else None
            self.heads[key] = LinearEmbeddingClassifier.init(
                config=classifier,
                input_dim=input_dim,
                num_classes=num,
                class_weights=class_weights
            )

        self.loss_weight = loss_weight
        self.detach_inputs = float(detach_inputs)

    @staticmethod
    def _class_weights(num_samples: list[int], beta: float = 0.999, mult: int = 1e4) -> list[float]:
        num_samples = np.maximum(num_samples, 1e-6)
        effective_num = 1.0 - np.power(beta, np.array(num_samples) * mult)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(num_samples)
        return weights.tolist()

    def forward(
            self,
            embeddings: torch.Tensor,
            labels: torch.Tensor | None = None
    ) -> MultiHeadEmbeddingClassifierOutput:
        embeddings = self.detach_inputs * embeddings.detach() + (1 - self.detach_inputs) * embeddings

        logits = {}
        loss, losses = 0., {}
        for i, (key, head) in enumerate(self.heads.items()):
            out = head(embeddings, labels=labels[..., i] if labels is not None else None)
            logits[key] = out.logits

            if out.loss:
                key = "clf/" + key
                loss += out.loss
                losses[key] = out.loss

        loss = self.loss_weight * loss / len(self.heads)
        losses["clf"] = loss

        return MultiHeadEmbeddingClassifierOutput(
            logits=logits,
            loss=loss if labels is not None else None,
            losses=losses if labels is not None else None
        )

    def prepare_inputs(self, inputs, *args) -> dict[str, torch.Tensor]:
        return inputs


@dataclass
class SequenceClassifierOutput:
    logits: torch.Tensor
    note_logits: torch.Tensor | None = None
    loss: torch.Tensor | None = None
    losses: dict[str, torch.Tensor] | None = None


@dataclass
class SequenceClassifierConfig(ModuleConfig):
    num_classes: int
    backbone: DictConfig | TupleTransformerConfig | None
    backbone_checkpoint: str | None = None
    transformer: DictConfig | TransformerConfig = TransformerConfig(_target_="default")
    classifier: LinearEmbeddingClassifierConfig = LinearEmbeddingClassifierConfig(hidden_dims=None)
    note_classifier: bool = False,

    aggregation: str = "token"
    emb_norm: bool = False
    dropout: float = 0.
    context_with_memory: bool = False
    backbone_output_layer: int | None = None
    detach_inputs: bool | float = True
    label_smoothing: float = 0.


class SequenceClassifier(Model):
    def __init__(
            self,
            num_classes: int,
            backbone: DictConfig | TupleTransformerConfig | None,
            backbone_checkpoint: str | None = None,
            transformer: DictConfig | TransformerConfig = TransformerConfig(_target_="default"),
            classifier: LinearEmbeddingClassifierConfig = LinearEmbeddingClassifierConfig(hidden_dims=None),
            note_classifier: bool = False,

            aggregation: str = "token",
            emb_norm: bool = False,
            emb_dropout: float = 0.,
            clf_dropout: float = 0.,
            context_with_memory: bool = False,
            backbone_output_layer: int | None = None,
            detach_inputs: bool | float = True,
            label_smoothing: float = 0.
    ):
        super().__init__()

        assert backbone is not None or backbone_checkpoint is not None

        self.backbone = None
        if backbone is not None:
            self.backbone = TupleTransformer.init(backbone)
            self.backbone_config = backbone

        if backbone_checkpoint is not None:
            from symupe.models import MODELS, MusicTransformer, Seq2SeqMusicTransformer

            checkpoint = torch.load(backbone_checkpoint, map_location="cpu", weights_only=True)

            self.backbone_config = OmegaConf.create(checkpoint["model"]["config"])

            backbone_cls = MODELS[checkpoint["model"]["config"]["_name_"]]
            backbone_model = backbone_cls.from_pretrained(checkpoint_path=backbone_checkpoint, strict=False)

            if isinstance(backbone_model, MusicTransformer):
                self.backbone = backbone_model.unwrap_model().transformer
            elif isinstance(backbone_model, Seq2SeqMusicTransformer):
                self.backbone = backbone_model.encoder
            else:
                raise ValueError()

        self.backbone_output_layer = backbone_output_layer
        self.context_with_memory = context_with_memory
        self.detach_inputs = float(detach_inputs)

        self.dim = transformer.dim

        self.emb_norm = LayerNorm(self.backbone.dim) if emb_norm else nn.Identity()
        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self.project_emb = nn.Linear(self.backbone.dim, self.dim, bias=False)

        assert aggregation in ("mean", "token")
        self.aggregation = aggregation

        self.transformer = EncoderTransformer.init(
            transformer,
            memory_tokens=1 if aggregation == "token" else 0
        )

        self.dropout = nn.Dropout(p=clf_dropout)

        self.num_classes = num_classes
        self.classifier = LinearEmbeddingClassifier.init(
            config=classifier,
            input_dim=self.dim,
            num_classes=num_classes,
            label_smoothing=label_smoothing
        )

        self.note_classifier = nn.Linear(self.dim, num_classes, bias=True) if note_classifier else None

        self.label_smoothing = label_smoothing

    def forward(
            self,
            tokens: torch.Tensor,
            values: torch.Tensor | None = None,
            mask: torch.Tensor | None = None,
            labels: torch.Tensor | None = None
    ) -> SequenceClassifierOutput:
        if self.detach_inputs:
            with torch.no_grad():
                backbone_out = self.backbone(
                    tokens, values=values, mask=mask,
                    output_layer=self.backbone_output_layer, return_embeddings=True
                )
        else:
            backbone_out = self.backbone(
                tokens, values=values, mask=mask,
                output_layer=self.backbone_output_layer, return_embeddings=True
            )

        embeddings = backbone_out.hidden_state

        if self.backbone_output_layer and self.backbone_output_layer < len(self.backbone.transformer.layers) - 1:
            embeddings = self.backbone.transformer.layers[self.backbone_output_layer + 1].attention_norm(embeddings)

        if self.context_with_memory:
            mask = torch.ones_like(embeddings[..., 0]).bool() if mask is None else mask

            if backbone_out.memory_state is not None:
                embeddings = torch.cat((backbone_out.memory_state, embeddings), dim=1)
                mask = F.pad(mask, (backbone_out.memory_state.shape[1], 0), value=True)

        embeddings = self.emb_norm(embeddings)
        embeddings = self.emb_dropout(embeddings)
        embeddings = self.project_emb(embeddings)

        out = self.transformer(embeddings, mask=mask)

        if self.aggregation == "token":
            embedding = out.memory_tokens[:, 0]
        else:
            embeddings = out.out
            if mask is None:
                embedding = embeddings.mean(dim=1)
            else:
                embeddings = embeddings * mask[..., None]
                embedding = embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]

        # embedding = F.gelu(embedding)

        clf_out = self.classifier(self.dropout(embedding), labels=labels)
        loss = clf_out.loss
        losses = {"clf": clf_out.loss} if clf_out.loss else None

        note_logits = None
        if self.note_classifier is not None:
            note_logits = self.note_classifier(self.dropout(out.out))

            if labels is not None:
                note_loss = masked_batch_mean(
                    F.cross_entropy(
                        note_logits.transpose(1, 2), labels[:, None].expand(-1, note_logits.shape[1]),
                        label_smoothing=self.label_smoothing
                    ),
                    mask=mask
                )
                loss = loss + note_loss
                losses["clf/note"] = note_loss

        return SequenceClassifierOutput(
            logits=clf_out.logits,
            note_logits=note_logits,
            loss=loss,
            losses=losses
        )

    def prepare_inputs(
            self,
            inputs: dict | SequenceInputs,
            ema_model: nn.Module | None = None
    ) -> dict[str, torch.Tensor]:
        if isinstance(inputs, SequenceInputs):
            inputs = asdict(inputs)

        seq_key = "sequences" if "sequences" in inputs else "performances"

        inputs_dict = {
            "tokens": inputs[seq_key]["tokens"],
            "values": inputs[seq_key]["values"],
            "mask": inputs[seq_key]["mask"]
        }

        if inputs.get("emotion_labels", None) is not None:
            inputs_dict["labels"] = inputs["emotion_labels"]
        elif inputs.get("sequence_labels", None) is not None:
            inputs_dict["labels"] = inputs["sequence_labels"]

        return inputs_dict

    @staticmethod
    def inject_data_config(
            config: DictConfig | SequenceClassifierConfig | None,
            dataset: SequenceDataset | None
    ) -> DictConfig | ModuleConfig | None:
        if config["backbone_checkpoint"] is not None:
            checkpoint = torch.load(config["backbone_checkpoint"], map_location="cpu", weights_only=True)
            backbone_config = OmegaConf.create(checkpoint["model"]["config"])

            backbone_cls = checkpoint["model"]["config"]["_name_"]
            if backbone_cls == "MusicTransformer":
                config["backbone"] = backbone_config["transformer"]
            elif backbone_cls == "Seq2SeqMusicTransformer":
                config["backbone"] = backbone_config["encoder"]
            else:
                raise NotImplementedError()

            config["backbone"]["num_tokens"] = backbone_config["num_tokens"]
        else:
            config["backbone"]["num_tokens"] = dataset.token_sizes

        return config
