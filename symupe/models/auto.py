from __future__ import annotations

import os

import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

from symupe.utils import load_json
from .base import Model, Evaluator


class AutoModel:
    MODELS: dict[str, type[Model]] = {}

    @classmethod
    def register(cls, model_cls: type[Model]):
        """ Registers a class into the AutoModel registry. """
        cls.MODELS[model_cls.__name__] = model_cls

    @classmethod
    def from_checkpoint(
            cls,
            checkpoint_path: str,
            from_ema: bool = False,
            load_weights: bool = True,
            strict: bool = True
    ) -> Model:
        """
        Load a model from a monolithic training checkpoint (.pt).

        :param checkpoint_path: path to checkpoint file
        :param from_ema: whether to load the model from an EMA checkpoint
        :param load_weights: whether to load the weights from the checkpoint or only initialize the model
        :param strict: whether to strictly enforce that the model keys match the keys in the checkpoint
        :return: loaded model
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
            device: str = "cpu",
            **kwargs
    ) -> Model:
        """
        Load a model from the Hugging Face Hub or a local directory.

        :param pretrained_path: path to pretrained model
        :param strict: whether to strictly enforce that the model keys match the keys in the checkpoint
        :param device: device to load the model
        :return: loaded model
        """
        if os.path.isdir(pretrained_path):
            config_path = os.path.join(pretrained_path, Model.CONFIG_NAME)
        else:
            config_path = hf_hub_download(repo_id=pretrained_path, filename=Model.CONFIG_NAME, **kwargs)

        config = load_json(config_path)
        model = cls._get_model_class(config.get("_name_")).from_pretrained(
            pretrained_path, strict=strict, device=device, **kwargs
        )
        model.eval()
        return model

    @classmethod
    def _get_model_class(cls, model_name: str | None) -> type[Model]:
        """Internal helper to route config names to Python classes."""
        if model_name in cls.MODELS:
            return cls.MODELS[model_name]

        raise ValueError(
            f"Model class `{model_name}` not found in memory. "
            f"Ensure the model module is imported in symupe.models.__init__"
        )


class AutoEvaluator:
    EVALUATORS: dict[str, type[Evaluator]] = {}

    @classmethod
    def register(cls, evaluator_cls: type[Evaluator]):
        """ Registers a class into the AutoEvaluator registry. """
        cls.EVALUATORS[evaluator_cls.__name__] = evaluator_cls
