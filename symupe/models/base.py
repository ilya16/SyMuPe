"""Base Model class."""

from __future__ import annotations

import os
from abc import abstractmethod

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, HfApi
from huggingface_hub.utils import SoftTemporaryDirectory
from loguru import logger
from miditok import MusicTokenizer
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_model, save_model
from torch.utils.data import Dataset

from symupe import __version__
from symupe.modules.constructor import Constructor, ModuleConfig
from symupe.utils import load_json, dump_json


class Model(nn.Module, Constructor):
    """Base Model class for SyMuPe framework models.

    Integrates :class:`torch.nn.Module` with factory-based Constructor
    and Hugging Face-style serialization.

    Provides unified methods for:
        - injection and cleanup of data configurations,
        - preparation of inputs,
        - saving and loading model weights from training checkpoints (.pt)
        and model artifacts (safetensors).
    """

    CONFIG_NAME = "config.json"
    SAFETENSORS_FILE_NAME = "model.safetensors"
    TOKENIZER_FILE_NAME = "tokenizer.json"

    GENERATOR_CLASS = None
    CLASSIFIER_CLASS = None

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Performs forward pass logic for model."""
        ...

    @abstractmethod
    def prepare_inputs(
        self, inputs: object, ema_model: nn.Module | None = None
    ) -> dict[str, torch.Tensor]:
        """Formats raw input data into dictionary of tensors for model.

        Args:
            inputs: Raw input data.
            ema_model: Exponential moving average model used inside model.

        Returns:
            Dictionary of tensors for model.
        """
        ...

    @staticmethod
    def allocate_inputs(
        inputs_dict: dict[str, torch.Tensor],
        device: str | torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Moves dictionary of tensors to the specified device.

        Args:
            inputs_dict: Dictionary of tensors.
            device: Target torch device.

        Returns:
            Dictionary of tensors on target device.
        """
        return {key: value.to(device, non_blocking=True) for key, value in inputs_dict.items()}

    @staticmethod
    def inject_data_config(
        config: DictConfig | ModuleConfig | None, dataset: Dataset | None
    ) -> DictConfig | ModuleConfig | None:
        """Updates model configuration with data-specific attributes.

        Args:
            config: Model configuration.
            dataset: Dataset object providing metadata.

        Returns:
            Updated model configuration.
        """
        return config

    @staticmethod
    def cleanup_config(
        config: DictConfig | ModuleConfig | None,
    ) -> DictConfig | ModuleConfig | None:
        """Removes extra fields from configuration.

        Args:
            config: Model configuration.

        Returns:
            Cleaned model configuration.
        """
        return config

    def load(
        self,
        state_dict: dict[str, torch.Tensor],
        ignore_layers: list[str] | None = None,
        ignore_mismatched_keys: bool = False,
    ) -> nn.Module:
        """Loads weights from state dictionary into model instance.

        Args:
            state_dict: Model state dictionary.
            ignore_layers: List of layers to exclude.
            ignore_mismatched_keys: Whether to ignore keys with shape mismatches.

        Returns:
            Loaded model instance.
        """
        return load_state_dict(self, state_dict, ignore_layers, ignore_mismatched_keys)

    def freeze(self, exception_list: list[str] | None = None) -> None:
        """Freezes model parameters for fine-tuning.

        Args:
            exception_list: List of layers to ignore, prefixed with '!' to force freeze.
        """
        not_frozen = []
        exception_list = exception_list or []
        finetune_list = [layer for layer in exception_list if layer[0] != "!"]
        freeze_list = [layer[1:] for layer in exception_list if layer[0] == "!"]
        for name, param in self.named_parameters():
            param.requires_grad = any(name.startswith(layer) for layer in finetune_list) or all(
                not name.startswith(layer) for layer in freeze_list
            )
            if param.requires_grad:
                not_frozen.append(name)
        logger.info(
            f"The model graph has been frozen, except for the following parameters: {not_frozen}"
        )

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

        model_cfg = OmegaConf.create(checkpoint["model"]["config"])
        model = cls.init(model_cfg)

        model._config = model_cfg

        if not load_weights:
            return model

        if from_ema:
            assert "ema_model" in checkpoint
            state_dict = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model"]["ema_model"].items()
                if k.startswith("ema_model.")
            }
        else:
            state_dict = checkpoint["model"]["state_dict"]

        for key, weight in model.state_dict().items():
            if key not in state_dict:
                state_dict[key] = weight
        model.load_state_dict(state_dict, strict=strict)

        return model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        strict: bool = True,
        device: str | torch.device | None = None,
        **kwargs,
    ) -> Model:
        """Initializes model from Hugging Face Hub or local directory.

        Args:
            pretrained_path: Path or Hub repo ID for pretrained model.
            strict: Whether to strictly match checkpoint keys.
            device: Target device for model loading.

        Returns:
            Initialized model instance.
        """
        if os.path.isdir(pretrained_path):
            config_path = os.path.join(pretrained_path, cls.CONFIG_NAME)
            weights_path = os.path.join(pretrained_path, cls.SAFETENSORS_FILE_NAME)
        else:
            config_path = hf_hub_download(
                repo_id=pretrained_path, filename=cls.CONFIG_NAME, **kwargs
            )
            weights_path = hf_hub_download(
                repo_id=pretrained_path, filename=cls.SAFETENSORS_FILE_NAME, **kwargs
            )

        model_cfg = OmegaConf.create(load_json(config_path))
        model = cls.init(model_cfg)

        model._config = model_cfg

        load_model(model, weights_path, strict=strict, device=device)

        return model

    def save_pretrained(
        self,
        save_directory: str,
        *,
        config: dict | DictConfig | None = None,
        tokenizer: MusicTokenizer | None = None,
        model_version: str = "1.0",
    ) -> None:
        """Exports model and configuration in Hugging Face compatible format.

        Args:
            save_directory: Path to export directory.
            config: Optional configuration to overwrite internal config.
            tokenizer: Optional tokenizer to save alongside model.
            model_version: Version string for saved model.
        """
        os.makedirs(save_directory, exist_ok=True)

        config = config or getattr(self, "_config", None)

        if config is not None:
            if isinstance(config, (DictConfig, ModuleConfig)):
                config_dict = OmegaConf.to_container(config, resolve=True)
            else:
                config_dict = config

            metadata = {
                "_name_": config_dict.pop("_name_", self.__class__.__name__),
                "_model_version_": config_dict.pop("_version_", model_version),
                "_symupe_version_": __version__,
            }

            config_dict = {**metadata, **config_dict}

            dump_json(config_dict, os.path.join(save_directory, self.CONFIG_NAME), indent=4)
        else:
            logger.warning(f"No configuration found to save for {self.__class__.__name__}")

        # save model
        save_model(self, os.path.join(save_directory, self.SAFETENSORS_FILE_NAME))

        if tokenizer is not None:
            tokenizer.save(os.path.join(save_directory, self.TOKENIZER_FILE_NAME))

        logger.info(f"Model and config saved to {save_directory}")

    def push_to_hub(
        self,
        repo_id: str,
        *,
        config: dict | DictConfig | None = None,
        tokenizer: MusicTokenizer | None = None,
        commit_message: str = "Push model using huggingface_hub",
        private: bool | None = None,
        token: str | None = None,
        branch: str | None = None,
    ) -> str:
        """Uploads model weights, config, and tokenizer to Hugging Face Hub.

        Args:
            repo_id: Target repository ID.
            config: Optional configuration to overwrite internal `model._config``.
            tokenizer: Optional tokenizer to upload 'tokenizer.json'.
            commit_message: Commit message to push model weights to.
            private: Whether to push model weights to private repo.
            token: Hugging Face authentication token.
            branch: Target git branch.

        Returns:
            URL of uploaded repository commit.
        """
        api = HfApi(token=token)
        repo_id = api.create_repo(repo_id=repo_id, private=private, exist_ok=True).repo_id

        with SoftTemporaryDirectory() as tmp:
            self.save_pretrained(tmp, config=config, tokenizer=tokenizer)

            return api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=tmp,
                commit_message=commit_message,
                revision=branch,
            )


def load_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    ignore_layers: list[str] | None = None,
    ignore_mismatched_keys: bool = False,
) -> nn.Module:
    """Loads model weights from state dictionary into model instance.

    Args:
        model: Model instance to load weights into.
        state_dict: Dictionary containing model weights.
        ignore_layers: List of layer names to exclude from loading.
        ignore_mismatched_keys: Whether to ignore keys with incompatible tensor shapes.

    Returns:
        Model instance with loaded weights.
    """
    ignore_layers = ignore_layers or []

    model_state = model.state_dict()

    extra_keys = [k for k in state_dict.keys() if k not in model_state]
    if extra_keys:
        logger.warning(
            f"The following checkpoint keys are not presented in the model "
            f"and will be ignored: {extra_keys}"
        )
        state_dict = {k: v for k, v in state_dict.items() if k not in extra_keys}

    ignored_keys = []
    if ignore_mismatched_keys:
        auto_ignore_layers = []
        for k, v in state_dict.items():
            if v.data.shape != model_state[k].data.shape:
                auto_ignore_layers.append(k)
        logger.info(
            f"Automatically found the checkpoint keys "
            f"incompatible with the model: {auto_ignore_layers}"
        )
        ignored_keys.extend(auto_ignore_layers)

    if ignore_layers:
        for k, v in state_dict.items():
            if any(layer in k for layer in ignore_layers):
                ignored_keys.append(k)

    if ignored_keys:
        state_dict = {k: v for k, v in state_dict.items() if all(k != key for key in ignored_keys)}
        logger.info(f"The following checkpoint keys were ignored: {ignored_keys}")

    model_state.update(state_dict)
    model.load_state_dict(model_state)

    return model


class Evaluator(Constructor):
    """Base class for all model evaluators."""

    def __init__(self, model: Model, **kwargs):
        """Initializes evaluator with model instance.

        Args:
            model: Model instance to evaluate.
        """
        self.model = model

    @abstractmethod
    @torch.no_grad()
    def __call__(self, inputs: object, outputs: object, **kwargs) -> dict[str, torch.Tensor]:
        """Computes metrics from inputs and model outputs.

        Args:
            inputs: Raw input data.
            outputs: Model forward pass outputs.

        Returns:
            Dictionary of metric names and their :class:`torch.Tensor` values.
        """
        raise NotImplementedError
