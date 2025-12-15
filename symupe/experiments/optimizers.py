""" Optimizers and Learning Rate Schedulers. """
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import torch
from accelerate import Accelerator
from omegaconf import DictConfig, MISSING

from symupe.modules.constructor import Constructor, VariableModuleConfig, Registry
from symupe.utils import Config


def group_weight_decayable_params(params):
    wd_params, no_wd_params = [], []
    for param in params:
        param_list = no_wd_params if param.squeeze().ndim < 2 else wd_params
        param_list.append(param)
    return wd_params, no_wd_params


@dataclass
class OptimizerParams(VariableModuleConfig):
    params: Iterable[torch.Tensor] | Iterable[dict] = MISSING
    lr: float = MISSING
    weight_decay: float = 0.


class _OptimizerRegistry(Registry):
    def instantiate(self, config: OptimizerParams | DictConfig, **kwargs):
        module = self.get(config._target_)

        if config.weight_decay > 0. and kwargs.pop("group_wd_params", True):
            wd_params, no_wd_params = group_weight_decayable_params(config.get("params", kwargs.get("params")))

            kwargs["params"] = [
                {"params": wd_params},
                {"params": no_wd_params, "weight_decay": 0.},
            ]

        module = module.init(config=config, **kwargs)

        return module


OptimizerRegistry = _OptimizerRegistry()


@dataclass
class SGDParams(OptimizerParams):
    _target_: str = "sgd"
    momentum: float = 0
    dampening: float = 0
    nesterov: bool = False


@OptimizerRegistry.register("sgd")
class SGD(torch.optim.SGD, Constructor):
    ...


@dataclass
class AdamParams(OptimizerParams):
    _target_: str = "adam"
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False


@OptimizerRegistry.register("adam")
class Adam(torch.optim.Adam, Constructor):
    ...


@dataclass
class AdamWParams(OptimizerParams):
    _target_: str = "adamw"
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    amsgrad: bool = False


@OptimizerRegistry.register("adamw")
class AdamW(torch.optim.AdamW, Constructor):
    ...


LRSchedulersRegistry = type("_LRSchedulersRegistry", (Registry, ), {})()


@dataclass
class LRSchedulerParams(VariableModuleConfig):
    optimizer: torch.optim.Optimizer = MISSING
    verbose: bool = False


@dataclass
class FakeSchedulerParams(LRSchedulerParams):
    _target_: str = "none"


@LRSchedulersRegistry.register("none")
class FakeScheduler(torch.optim.lr_scheduler._LRScheduler, Constructor):
    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]


@dataclass
class ExponentialLRParams(LRSchedulerParams):
    gamma: float = MISSING
    _target_: str = "exponential"
    last_epoch: int = -1


@LRSchedulersRegistry.register("exponential")
class ExponentialLR(torch.optim.lr_scheduler.ExponentialLR, Constructor):
    ...


@dataclass
class ExponentialLRParams(ExponentialLRParams):
    _target_: str = "exponential-step"


@LRSchedulersRegistry.register("exponential-step")
class ExponentialStepLR(torch.optim.lr_scheduler.ExponentialLR, Constructor):
    ...


@dataclass
class WarmUpAnnealLRParams(LRSchedulerParams):
    _target_: str = "warmup"
    warmup_steps: int = 1000
    anneal_steps: Sequence[int] | None = None
    anneal_rate: float = 0.9
    last_epoch: int = -1


@LRSchedulersRegistry.register("warmup")
class WarmUpAnnealLR(torch.optim.lr_scheduler.LRScheduler, Constructor):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            warmup_steps: int = 1000,
            anneal_steps: Sequence[int] | None = None,
            anneal_rate: float = 0.9,
            last_epoch: int = -1,
            verbose: bool = False
    ):
        self.optimizer = optimizer

        self.warmup_steps = warmup_steps
        self.anneal_steps = anneal_steps
        self.anneal_rate = anneal_rate
        self._scale = warmup_steps ** 0.5 if warmup_steps > 0 else 1.

        super().__init__(optimizer, last_epoch, verbose)

    def anneal_function(self, step):
        if self.warmup_steps == 0:
            return self._scale
        elif step > self.warmup_steps:
            return self._scale / (step ** 0.5)
        else:
            return self._scale * step / (self.warmup_steps ** 1.5)

    def get_lr(self):
        step = self._step_count
        scale = self.anneal_function(step)

        if self.anneal_steps:
            for s in self.anneal_steps:
                if step > s:
                    scale *= self.anneal_rate

        return [group["initial_lr"] * scale for group in self.optimizer.param_groups]


@dataclass
class CosineAnnealLRParams(LRSchedulerParams):
    _target_: str = "cosine"
    total_steps: int = MISSING
    warmup_steps: int = 1000
    lr_min_ratio: float = 0.1
    cycle_length: float = 1.0


@LRSchedulersRegistry.register("cosine")
class CosineAnnealLR(torch.optim.lr_scheduler.LRScheduler, Constructor):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            total_steps: int,
            warmup_steps: int = 1000,
            lr_min_ratio: float = 0.1,
            cycle_length: float = 1.0
    ):
        self.optimizer = optimizer

        self.warmup_steps = warmup_steps
        assert self.warmup_steps >= 0
        self.total_steps = total_steps
        assert self.total_steps >= 0
        self.lr_min_ratio = lr_min_ratio
        self.cycle_length = cycle_length

        super().__init__(optimizer)

    def anneal_function(self, step):
        if step < self.warmup_steps:
            return step / self.warmup_steps
        elif step <= self.total_steps:
            s = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.lr_min_ratio + 0.5 * (1 - self.lr_min_ratio) * (1. + math.cos(math.pi * s / self.cycle_length))
        else:
            return self.lr_min_ratio

    def get_lr(self):
        step = self._step_count
        scale = self.anneal_function(step)

        return [group["initial_lr"] * scale for group in self.optimizer.param_groups]


@dataclass
class OptimizationConfig(Config):
    optimizer: OptimizerParams | dict[str, object] | DictConfig = field(
        default_factory=lambda: {"_target_": "adamw", "lr": 1e-3},
        metadata={"help": "Optimizer and its parameters (including the learning rate). "
                          "Defaults to {\"_target_\": \"adamw\", \"lr\": 1e-3}"}
    )
    lr_scheduler: LRSchedulerParams | dict[str, object] | DictConfig | None = field(
        default=None, metadata={"help": "Learning rate scheduler and tis parameters. Defaults to None"}
    )
    grad_clip: float | None = field(
        default=None, metadata={"help": "Gradient clipping threshold. Defaults to None"}
    )
    grad_accum_steps: int | None = field(
        default=1, metadata={"help": "Gradient accumulation steps. Defaults to 1"}
    )
    group_wd_params: bool = field(
        default=True, metadata={"help": "Remove biases from weight decay regularization. Defaults to True"}
    )


class Optimizer:
    """ Combined Optimizer and Scheduler. """

    def __init__(
            self,
            accelerator: Accelerator,
            config: OptimizationConfig,
            model: torch.nn.Module = None,
            parameters: list = None,
            num_train_steps: int | None = None
    ):
        self.accelerator = accelerator
        self.config = config

        self.optimizer = OptimizerRegistry.instantiate(
            DictConfig(config.optimizer) if isinstance(config.optimizer, dict) else config.optimizer,
            params=model.parameters() if model is not None else parameters,
            group_wd_params=config.group_wd_params
        )

        self.lr_scheduler = None
        if config.lr_scheduler is not None:
            self.lr_scheduler = LRSchedulersRegistry.instantiate(
                DictConfig(config.lr_scheduler) if isinstance(config.lr_scheduler, dict) else config.lr_scheduler,
                optimizer=self.optimizer,
                total_steps=num_train_steps
            )

            self.is_step_lr_scheduler = isinstance(
                self.lr_scheduler,
                (LRSchedulersRegistry.get("warmup"),
                 LRSchedulersRegistry.get("cosine"),
                 LRSchedulersRegistry.get("exponential-step"))
            )

        self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.optimizer, self.lr_scheduler
        )

        self.grad_clip = config.grad_clip
        self.grad_accum_steps = config.grad_accum_steps

    def step(self, loss_value, step_optimizer=True):
        self.accelerator.backward(loss_value / self.grad_accum_steps)

        grad_norm = None
        if step_optimizer:
            if self.grad_clip is not None:
                parameters = self.optimizer.param_groups[0]["params"]
                grad_norm = self.accelerator.clip_grad_norm_(parameters, self.grad_clip)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    grad_norm = None

            self.optimizer.step()
            self.optimizer.zero_grad()

        return grad_norm

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def anneal_on_epoch_end(self, *args):
        if self.lr_scheduler is not None:
            if not self.is_step_lr_scheduler:
                if isinstance(self.lr_scheduler.scheduler, LRSchedulersRegistry.get("plateau")):
                    self.lr_scheduler.step(*args)
                else:
                    self.lr_scheduler.step()

    def anneal_on_step_end(self, *args):
        if self.lr_scheduler is not None:
            if self.is_step_lr_scheduler:
                self.lr_scheduler.step()

    def get_last_lr(self):
        return self.lr_scheduler.get_last_lr()

    def load_state_dict(self, state_dict, restore_lr: bool = True):
        self.optimizer.load_state_dict(state_dict["optimizer"])

        if restore_lr:
            lr_scheduler_params = state_dict.get("lr_scheduler", None)
            if lr_scheduler_params is not None:
                self.lr_scheduler.load_state_dict(lr_scheduler_params)
        else:
            base_lr = self.lr_scheduler.get_last_lr()
            for lr, param_group in zip(base_lr, self.optimizer.param_groups):
                param_group["lr"] = lr

    def state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict()
        }

    def set_progress(self, iteration, epoch):
        for key, value in self.optimizer.state.items():
            if "step" in value:
                self.optimizer.state[key]["step"] = iteration

        self.lr_scheduler.last_epoch = epoch
        self.lr_scheduler._step_count = epoch + 1

    def __repr__(self):
        return f"Optimizer(optimizer={self.optimizer}, lr_scheduler={self.lr_scheduler})"
