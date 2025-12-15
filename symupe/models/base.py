""" Base Model class. """
from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

from symupe.modules.constructor import Constructor, ModuleConfig


class Model(nn.Module, Constructor):
    @abstractmethod
    def forward(self, *args, **kwargs):
        ...

    @abstractmethod
    def prepare_inputs(self, inputs, ema_model: nn.Module | None = None) -> dict[str, torch.Tensor]:
        ...

    @staticmethod
    def allocate_inputs(inputs_dict: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
        return {key: value.to(device, non_blocking=True) for key, value in inputs_dict.items()}

    @staticmethod
    def inject_data_config(
            config: DictConfig | ModuleConfig | None,
            dataset: Dataset | None
    ) -> DictConfig | ModuleConfig | None:
        return config

    @staticmethod
    def cleanup_config(
            config: DictConfig | ModuleConfig | None
    ) -> DictConfig | ModuleConfig | None:
        return config

    @classmethod
    def from_pretrained(
            cls,
            checkpoint_path: str,
            from_ema: bool = False,
            load_weights: bool = True,
            strict: bool = True
    ) -> Model:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        model_cfg = OmegaConf.create(checkpoint["model"]["config"])
        model = cls.init(model_cfg)

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

    def load(
            self,
            state_dict: dict[str, torch.Tensor],
            ignore_layers: list[str] | None = None,
            ignore_mismatched_keys: bool = False
    ) -> nn.Module:
        return load_state_dict(self, state_dict, ignore_layers, ignore_mismatched_keys)

    def freeze(self, exception_list: list[str] | None = None) -> None:
        not_frozen = []
        exception_list = exception_list or []
        finetune_list = [layer for layer in exception_list if layer[0] != "!"]
        freeze_list = [layer[1:] for layer in exception_list if layer[0] == "!"]
        for name, param in self.named_parameters():
            param.requires_grad = (
                    any(name.startswith(layer) for layer in finetune_list)
                    or all(not name.startswith(layer) for layer in freeze_list)
            )
            if param.requires_grad:
                not_frozen.append(name)
        logger.info(f"The model graph has been frozen, except for the following parameters: {not_frozen}")



def load_state_dict(
        model: nn.Module,
        state_dict: dict[str, torch.Tensor],
        ignore_layers: list[str] | None = None,
        ignore_mismatched_keys: bool = False
) -> nn.Module:
    ignore_layers = ignore_layers or []

    model_state = model.state_dict()

    extra_keys = [k for k in state_dict.keys() if k not in model_state]
    if extra_keys:
        logger.warning(f"The following checkpoint keys are not presented in the model "
                       f"and will be ignored: {extra_keys}")
        state_dict = {k: v for k, v in state_dict.items() if k not in extra_keys}

    ignored_keys = []
    if ignore_mismatched_keys:
        auto_ignore_layers = []
        for k, v in state_dict.items():
            if v.data.shape != model_state[k].data.shape:
                auto_ignore_layers.append(k)
        logger.info(f"Automatically found the checkpoint keys "
                    f"incompatible with the model: {auto_ignore_layers}")
        ignored_keys.extend(auto_ignore_layers)

    if ignore_layers:
        for k, v in state_dict.items():
            if any(layer in k for layer in ignore_layers):
                ignored_keys.append(k)

    if ignored_keys:
        state_dict = {k: v for k, v in state_dict.items()
                      if all(k != key for key in ignored_keys)}
        logger.info(f"The following checkpoint keys were ignored: {ignored_keys}")

    model_state.update(state_dict)
    model.load_state_dict(model_state)

    return model