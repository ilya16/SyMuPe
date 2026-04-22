from __future__ import annotations

import os

import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

from symupe.utils import load_json
from .base import Model, Evaluator


class AutoModel:
    """Factory class to initialize models from pretrained weights or checkpoints."""

    MODELS: dict[str, type[Model]] = {}

    @classmethod
    def register(cls, model_cls: type[Model]):
        """Registers class into :class:`AutoModel` registry.

        Args:
            model_cls: Model class to register.
        """
        cls.MODELS[model_cls.__name__] = model_cls

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        from_ema: bool = False,
        load_weights: bool = True,
        strict: bool = True,
    ) -> Model:
        """Initializes model from monolithic training checkpoint (.pt).

        Args:
            checkpoint_path: Path to checkpoint file.
            from_ema: Whether to load model from EMA checkpoint.
            load_weights: Whether to load weights or only initialize architecture.
            strict: Whether to strictly match checkpoint keys.

        Returns:
            Initialized model instance.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if "model" not in checkpoint or "config" not in checkpoint["model"]:
            raise ValueError(f"Checkpoint at {checkpoint_path} is missing the model config.")

        config = checkpoint["model"]["config"]
        config_dict = OmegaConf.to_container(config) if hasattr(config, "to_container") else config

        return cls._get_model_class(config_dict.get("_name_")).from_checkpoint(
            checkpoint_path, from_ema=from_ema, load_weights=load_weights, strict=strict
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        strict: bool = True,
        device: str | torch.device | None = None,
        **kwargs,
    ) -> Model:
        """Initializes model from Hugging Face Hub or local path using config name.

        Args:
            pretrained_path: Path or Hub repo ID for pretrained model.
            strict: Whether to strictly match checkpoint keys.
            device: Target device for model loading.

        Returns:
            Initialized model instance.
        """
        if os.path.isdir(pretrained_path):
            config_path = os.path.join(pretrained_path, Model.CONFIG_NAME)
        else:
            config_path = hf_hub_download(
                repo_id=pretrained_path, filename=Model.CONFIG_NAME, **kwargs
            )

        config = load_json(config_path)
        model = cls._get_model_class(config.get("_name_")).from_pretrained(
            pretrained_path, strict=strict, device=device, **kwargs
        )
        model.eval()
        return model

    @classmethod
    def _get_model_class(cls, model_name: str | None) -> type[Model]:
        """Routes configuration names to registered Python classes.

        Args:
            model_name: Name of model class.

        Returns:
            Registered model class type.
        """
        if model_name in cls.MODELS:
            return cls.MODELS[model_name]

        raise ValueError(
            f"Model class `{model_name}` not found in memory. "
            f"Ensure the model module is imported in symupe.models.__init__"
        )


class AutoEvaluator:
    """Factory class to manage model evaluators."""

    EVALUATORS: dict[str, type[Evaluator]] = {}

    @classmethod
    def register(cls, evaluator_cls: type[Evaluator]):
        """Registers class into AutoEvaluator registry.

        Args:
            evaluator_cls: Evaluator class to register.
        """
        cls.EVALUATORS[evaluator_cls.__name__] = evaluator_cls
