"""
Trainer.

As simple as it could be.
"""
from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
from pathlib import Path
from shutil import copyfile

import numpy as np
import torch
from accelerate import Accelerator
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from symupe import __version__ as symupe_version
from symupe.models.base import Model, load_state_dict
from symupe.utils import count_parameters, set_random_seed
from .callbacks import (
    TrainerState, TrainerControl,
    TrainerCallback, CallbackHandler, CallbacksRegistry,
    PrinterCallback, DefaultFlowCallback, ProgressCallback, TrackerCallback
)
from .modules import ExperimentConfig
from .console_logger import setup_logger
from .optimizers import Optimizer
from .trainer_config import TrainerConfig
from .trainer_utils import Accumulator, IntervalStrategy

TRAINER_STATE_NAME = "trainer_state.json"
BEST_CHECKPOINT_NAME = "checkpoint_best.pt"
FINAL_CHECKPOINT_NAME = "checkpoint_last.pt"

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback


class Trainer:
    def __init__(
            self,
            model: Model,
            config: TrainerConfig | ExperimentConfig,
            train_dataset: Dataset | None = None,
            eval_dataset: Dataset | None = None,
            collator: Callable[any, any] | None = None,
            optimizer: Optimizer | None = None,
            evaluator: Callable[any, any] | None = None
    ):
        self.model = model

        self.exp_config = None
        if isinstance(config, ExperimentConfig):
            self.exp_config = config
            config = config.trainer
            config = config if isinstance(config, TrainerConfig) else TrainerConfig(**config)

        self.config: TrainerConfig = config

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.collator = collator
        self.evaluator = evaluator
        self.train_dataloader = None
        self.eval_dataloader = None

        self.optimizer = optimizer
        self.callbacks = None
        self.is_in_train = False

        self.accelerator = None

        self._init()

    def _init(self):
        # accelerator
        self._setup_accelerator()

        # state and control
        self.state = TrainerState(
            is_main_process=self.accelerator.is_main_process,
            is_local_main_process=self.accelerator.is_local_main_process,
        )
        self.control = TrainerControl()

        # directories and logger
        self._setup_directories()

        # random effects
        self._set_random_seed()

        # callbacks
        self._setup_callbacks()

        # dashboard logger
        self._setup_trackers()

        # prepare model
        self._setup_model()

        # dataloaders
        self.build_dataloaders()

        # optimizer with lr scheduler
        self.build_optimizer()

        self.accelerator.wait_for_everyone()
        logger.info(f"Built Trainer!")

    def _setup_directories(self):
        # output results directory
        Path(self.config.output_dir).mkdir(mode=0o775, parents=True, exist_ok=True)

        # logging directory and loggers
        Path(self.config.log_dir).mkdir(mode=0o775, parents=True, exist_ok=True)

        file_log = Path(self.config.log_dir, "trainer.log") if self.config.log_to_file else None
        setup_logger(
            file=file_log,
            is_main_process=self.state.is_main_process,
            process=f"[{os.getpid()}]",
            node_info=f"[{self.config.device}]",
            message_level=""
        )

        logger.info(f"Initialized output directory: {self.config.output_dir}")  # , message_level="INIT")
        logger.info(f"Initialized logs directory: {self.config.log_dir}")

    def _setup_accelerator(self):
        self.accelerator = Accelerator(
            mixed_precision=self.config.accelerator.mixed_precision,
            cpu=self.config.accelerator.cpu,
            log_with=self.config.accelerator.log_with,
            project_dir=self.config.accelerator.project_dir,
            step_scheduler_with_optimizer=False,
            **self.config.accelerator.kwargs,
        )

        if self.accelerator.num_processes == 1:
            device = torch.device(self.config.device)
            if device.type == "cuda" and device.index is not None and torch.cuda.is_available():
                self.accelerator.state.device = device
                torch.cuda.set_device(device)

        self.device = self.accelerator.device

    def _set_random_seed(self):
        set_random_seed(seed=self.config.seed)

    def _setup_callbacks(self):
        self.callback_handler = CallbackHandler(DEFAULT_CALLBACKS, self.model, self.optimizer)

        callbacks = self.config.callbacks or {}
        for name, config in callbacks.items():
            self.add_callback(CallbacksRegistry.instantiate(config))

        self.add_callback(PrinterCallback if self.config.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        logger.info("Built CallbackHandler")

    def add_callback(self, callback: TrainerCallback):
        self.callback_handler.add_callback(callback)

    def pop_callback(self, callback: TrainerCallback):
        return self.callback_handler.pop_callback(callback)

    def remove_callback(self, callback: TrainerCallback):
        self.callback_handler.remove_callback(callback)

    def _setup_trackers(self):
        if not self.state.is_main_process:
            return

        # override `self.accelerator.init_trackers` for `tensorboard` logging_dir without `project_name`
        for tracker in self.accelerator.log_with:
            from accelerate.tracking import GeneralTracker
            if issubclass(type(tracker), GeneralTracker):
                # custom trackers are already initialized
                self.accelerator.trackers.append(tracker)
            else:
                from accelerate.tracking import LOGGER_TYPE_TO_CLASS
                tracker_init = LOGGER_TYPE_TO_CLASS[str(tracker)]
                tracker_kwargs = self.config.tracker_kwargs.get(str(tracker), {})
                if tracker.value == "tensorboard":
                    self.accelerator.trackers.append(tracker_init("", self.accelerator.logging_dir, **tracker_kwargs))
                elif tracker.value == "wandb":
                    self.accelerator.trackers.append(tracker_init(self.config.project_name, **tracker_kwargs))

        if not self.callback_handler.has_callback(TrackerCallback):
            self.add_callback(TrackerCallback(accelerator=self.accelerator))
            logger.info(f"Built TrackerCallback")

    def _setup_model(self):
        # prepare model
        self.model = self.accelerator.prepare(self.model)

        # prepare ema
        self.ema_model = None
        if self.config.use_ema:
            from ema_pytorch import EMA
            self.ema_model = EMA(
                self.accelerator.unwrap_model(self.model),
                beta=self.config.ema_beta,
                update_after_step=self.config.ema_update_after_step,
                update_every=self.config.ema_update_every,
                include_online_model=False
            )
            self.ema_model.buffer_names = {}

    def build_dataloader(self, dataset: Dataset, is_train: bool = True):
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size if is_train else self.config.eval_batch_size,
            shuffle=is_train and self.config.shuffle and not self.config.eval_mode,
            num_workers=self.config.num_workers,
            collate_fn=self.collator,
            pin_memory=self.config.pin_memory
        )

    def build_dataloaders(self):
        self.phases = []
        self.train_dataloader = None
        if self.train_dataset is not None:
            self.train_dataloader = self.build_dataloader(self.train_dataset, is_train=self.config.do_train)
            self.callback_handler.train_dataloader = self.train_dataloader
            self.phases.append("train")
            self.train_dataloader = self.accelerator.prepare(self.train_dataloader)
            logger.info("Built Training DataLoader")

        self.eval_dataloader = None
        if self.eval_dataset is not None:
            self.eval_dataloader = self.build_dataloader(self.eval_dataset, is_train=False)
            self.callback_handler.eval_dataloader = self.eval_dataloader
            self.phases.append("eval")
            self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)
            logger.info("Built Evaluation DataLoader")

        assert len(self.phases), "One of `train_dataset`/`eval_dataset` should be explicitly provided."

    def build_optimizer(self):
        if self.optimizer is None and self.config.do_train:
            assert self.train_dataloader is not None

            num_update_steps_per_epoch = max(len(self.train_dataloader) // self.config.optimization.grad_accum_steps, 1)
            if self.config.max_steps > 0:
                num_train_steps = self.config.max_steps
            else:
                num_train_steps = math.ceil(self.config.epochs * num_update_steps_per_epoch)

            self.optimizer = Optimizer(
                accelerator=self.accelerator,
                config=self.config.optimization,
                model=self.model,
                num_train_steps=num_train_steps
            )
            logger.info(f"Built Optimizer")

        self.grad_accum_steps = self.optimizer.grad_accum_steps if self.optimizer is not None else 1

    def train(self, resume_from_checkpoint: str | float | None = None):
        try:
            # catch any exceptions or interruptions  inside
            self._train(resume_from_checkpoint=resume_from_checkpoint)
        except Exception as e:
            raise e
        finally:
            if self.state.is_local_main_process:
                logger.info("Trying to save final checkpoint before exit...")
                self.state.save_to_json(os.path.join(self.config.output_dir, TRAINER_STATE_NAME))
                self._save_checkpoint(os.path.join(self.config.output_dir, FINAL_CHECKPOINT_NAME), minimal=False)

    def _train(self, resume_from_checkpoint: str | float | None = None):
        if not self.config.do_train:
            logger.warning("`do_train` is False, halting training.")
            return
        elif self.train_dataloader is None:
            logger.warning("No `train_dataloader` is provided, halting training.")
            return

        # load model and training state if needed
        self._maybe_load_checkpoint(resume_from_checkpoint=resume_from_checkpoint)

        # prepare for training
        config = self.config
        self.is_in_train = True

        self.callback_handler.on_train_begin(self.config, self.state, self.control, exp_config=self.exp_config)

        if config.max_steps > 0:
            logger.info("`max_steps` is given, it will override any value given in `epochs`")

        total_train_batch_size = config.batch_size * self.accelerator.num_processes * self.grad_accum_steps
        num_update_steps_per_epoch = max(len(self.train_dataloader) // self.grad_accum_steps, 1)

        if config.max_steps > 0:
            max_steps = config.max_steps
            num_train_epochs = config.max_steps // num_update_steps_per_epoch \
                               + int(config.max_steps % num_update_steps_per_epoch > 0)
            num_train_samples = max_steps * total_train_batch_size  # note: last batch might be incomplete
        else:
            max_steps = math.ceil(config.epochs * num_update_steps_per_epoch)
            num_train_epochs = math.ceil(config.epochs)
            num_train_samples = len(self.train_dataloader.dataset) * config.epochs

        num_examples = len(self.train_dataloader.dataset)

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num epochs = {num_train_epochs}")
        logger.info(f"  Batch size per process = {config.batch_size}")
        logger.info(f"  Num processes = {self.accelerator.num_processes}")
        logger.info(f"  Gradient accumulation steps = {self.grad_accum_steps}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(f"  Model parameters = {count_parameters(self.model):_}")
        logger.info(f"  Optimized model parameters = {count_parameters(self.model, requires_grad=True):_}")
        logger.complete()

        epochs_trained = self.state.global_step // num_update_steps_per_epoch
        self.state.num_train_epochs = num_train_epochs
        self.state.max_steps = max_steps

        # possible pre-evaluation
        self._maybe_log_save_evaluate()

        if self.config.do_train and self.train_dataloader is not None:
            for epoch in range(epochs_trained, self.state.num_train_epochs):
                metrics = self.run_epoch(self.train_dataloader, is_train=True)

                if self.state.global_step != 0:
                    self.optimizer.anneal_on_epoch_end(metrics["loss"])

        self.callback_handler.on_train_end(self.config, self.state, self.control)
        self._save_checkpoint(os.path.join(config.output_dir, FINAL_CHECKPOINT_NAME), minimal=False)
        self.is_in_train = False

    def evaluate(self, eval_dataset: Dataset | None = None):
        if eval_dataset is not None:
            eval_dataloader = self.build_dataloader(eval_dataset, is_train=False)
            eval_dataloader = self.accelerator.prepare(eval_dataloader)
        elif self.eval_dataloader is not None:
            eval_dataloader = self.eval_dataloader
        elif self.config.do_eval:
            logger.warning("Trainer has no `eval_dataloader` and `eval_dataset` is not provided. Skipping evaluation.")
            return
        else:
            return

        logger.info(f"*** Running evaluation ***")

        metrics = self.run_epoch(eval_dataloader, is_train=False)

        if self.is_in_train:
            self.model.train()

        return metrics

    def _save_checkpoint(self, checkpoint_path, minimal: bool = False):
        if not self.state.is_main_process:
            return

        checkpoint = {
            "experiment": {
                "config": self.exp_config.to_json_string() if self.exp_config is not None else None,
                "trainer": self.config.to_json_string(),
                "state": self.state.to_json_string(),
            },
            "model": {
                "config": OmegaConf.to_container(self.exp_config.model, resolve=True)
                if self.exp_config is not None else None,
                "state_dict": self.accelerator.get_state_dict(self.model),
            },
        }

        if self.config.use_ema and self.ema_model is not None:
            checkpoint["ema_model"] = {
                "ema_model": self.ema_model.state_dict(),
            }

        dataset = self.train_dataset if self.train_dataset is not None else self.eval_dataset
        if dataset is not None and hasattr(dataset, "tokenizer"):
            checkpoint["tokenizer"] = {
                "config": dataset.tokenizer.config.to_dict(),
            }

        if not minimal:
            checkpoint["optimizer"] = self.optimizer.state_dict()

        checkpoint["version"] = symupe_version

        logger.info(f"*** Saving checkpoint {checkpoint_path} ***")
        torch.save(checkpoint, checkpoint_path)

    def save_checkpoint(self, metrics=None):
        if not self.state.is_main_process:
            return

        config = self.config

        if config.save_strategy == IntervalStrategy.STEPS:
            step = f"s{self.state.global_step:d}"
        else:
            step = f"e{math.ceil(self.state.epoch):d}"
        checkpoint_path = os.path.join(config.output_dir, f"checkpoint_{step}.pt")
        last_checkpoint_path = self.state.last_model_checkpoint

        is_best = False
        if metrics is not None and self.config.metric_for_best_model in metrics:
            eval_metric = metrics[self.config.metric_for_best_model]
            operator = np.greater if self.config.metric_maximize else np.less

            if self.state.best_metric is None or operator(eval_metric, self.state.best_metric):
                message = f"Metric improvement on evaluation set ({self.config.metric_for_best_model}: "
                message += f"{self.state.best_metric:7.5f} -> " if self.state.best_metric is not None else ""
                message += f"{eval_metric:7.5f})"
                logger.info(message)
                self.state.best_metric = eval_metric
                self.state.best_model_checkpoint = checkpoint_path
                is_best = True

        self.state.save_to_json(os.path.join(config.output_dir, TRAINER_STATE_NAME))

        if not config.save_best_only or (metrics is not None and is_best):
            self._save_checkpoint(checkpoint_path, minimal=not self.config.save_optimizer)
            self.state.last_model_checkpoint = checkpoint_path

            if config.save_rewrite_checkpoint and last_checkpoint_path and os.path.exists(last_checkpoint_path):
                os.remove(last_checkpoint_path)

        if is_best:
            copyfile(checkpoint_path, os.path.join(config.output_dir, BEST_CHECKPOINT_NAME))

    def _maybe_log_save_evaluate(
            self,
            eval_dataset: Dataset | None = None,
            logs: dict | None = None,
    ):
        if self.control.should_log:
            self.callback_handler.on_log(self.config, self.state, self.control, logs=logs)

        should_save = self.control.should_save
        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(eval_dataset=eval_dataset)

        if should_save:
            self.accelerator.wait_for_everyone()
            self.save_checkpoint(metrics=metrics)
            self.callback_handler.on_save(self.config, self.state, self.control)

    def _maybe_load_checkpoint(self, resume_from_checkpoint: str | float | None = None):
        resume_from_checkpoint = resume_from_checkpoint or self.config.resume_from_checkpoint

        if isinstance(resume_from_checkpoint, bool):
            if resume_from_checkpoint:
                resume_from_checkpoint = os.path.join(self.config.output_dir, FINAL_CHECKPOINT_NAME)
                assert os.path.exists(resume_from_checkpoint), \
                    f"`resume_from_checkpoint` is set to True, " \
                    f"but checkpoint {resume_from_checkpoint} is not found in `output_dir`"
            else:
                resume_from_checkpoint = None

        if resume_from_checkpoint is not None:
            self.load_checkpoint(
                checkpoint_path=resume_from_checkpoint,
                warm_start=self.config.warm_start
            )

        if self.config.finetune_layers:
            self.model.freeze(exception_list=self.config.finetune_layers)

    def load_checkpoint(self, checkpoint_path: str, warm_start: bool = False):
        logger.info("*** Loading checkpoint ***")
        logger.info(f"Checkpoint path: `{checkpoint_path}`")

        with self.accelerator.main_process_first():
            checkpoint_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

            model = self.accelerator.unwrap_model(self.model)
            model_state = checkpoint_dict["model"]["state_dict"]
            if warm_start:
                logger.info("Warm start is enabled.")
                model.load(model_state, self.config.ignore_layers, self.config.ignore_mismatched_keys)

                if "optimizer" in checkpoint_dict and self.config.restore_optimizer:
                    self.optimizer.load_state_dict(checkpoint_dict["optimizer"], self.config.restore_lr)
            else:
                model.load(model_state, None, False)

                if "optimizer" in checkpoint_dict:
                    self.optimizer.load_state_dict(checkpoint_dict["optimizer"], self.config.restore_lr)

                trainer_state_path = os.path.join(self.config.output_dir, TRAINER_STATE_NAME)
                if "experiment" in checkpoint_dict and "state" in checkpoint_dict["experiment"]:
                    self.state = TrainerState.from_json_string(checkpoint_dict["experiment"]["state"])
                    logger.info(f"Loaded trainer state from the checkpoint.")
                elif os.path.exists(trainer_state_path):
                    self.state = self.state.load_from_json(trainer_state_path)
                    logger.info(f"Loaded trainer state from `{trainer_state_path}`.")
                else:
                    logger.warning("`trainer_state_path` is not found, the training progress will start from scratch "
                                   "and might be not the expected behaviour.")

                self.state.is_main_process = self.accelerator.is_main_process
                self.state.is_local_main_process = self.accelerator.is_local_main_process

            if self.ema_model and self.ema_model is not None and "ema_model" in checkpoint_dict:
                load_state_dict(self.ema_model, checkpoint_dict["ema_model"]["ema_model"], ignore_mismatched_keys=True)

        logger.info(f"Loaded checkpoint `{checkpoint_path}`.")

        return self.model

    def run_epoch(self, dataloader, is_train=False):
        self.control.is_train = is_train

        config = self.config

        model = self.model
        model.train() if is_train else model.eval()

        if is_train:
            assert self.optimizer is not None
        if self.optimizer is not None:
            self.optimizer.zero_grad()

        # torch.cuda.empty_cache()

        epoch_stats = Accumulator()
        stats_accumulator = Accumulator()  # accumulating metrics for multiple gradient accumulation steps

        accum_steps = self.grad_accum_steps if is_train else 1
        batches_in_epoch = len(dataloader)
        if not is_train and config.eval_batches:
            batches_in_epoch = max(1, min(len(dataloader), config.eval_batches))
        steps_in_epoch = math.ceil(batches_in_epoch / accum_steps)

        self.callback_handler.on_epoch_begin(self.config, self.state, self.control, steps_in_epoch=steps_in_epoch)

        _epoch_step = self.state.epoch_step
        with torch.set_grad_enabled(is_train):
            epoch = math.floor(self.state.epoch)
            self.state.epoch_step = 0

            start_time = time.perf_counter()
            self.callback_handler.on_step_begin(self.config, self.state, self.control)
            for idx, batch in enumerate(dataloader):
                time_data = time.perf_counter() - start_time
                step_end = (idx + 1) % accum_steps == 0 or (idx + 1) == batches_in_epoch

                start_time_model = time.perf_counter()
                inputs = self.accelerator.unwrap_model(model).prepare_inputs(
                    batch, ema_model=self.ema_model.model if self.ema_model is not None else None
                )
                outputs = model(**inputs)
                loss, losses = outputs.loss, outputs.losses

                grad_norm = None
                if is_train:
                    grad_norm = self.optimizer.step(loss, step_optimizer=step_end)

                time_model = time.perf_counter() - start_time_model

                stats_accumulator.update_value("loss", loss.detach())
                if losses is not None:
                    stats_accumulator.update_values({f"loss/{key}": value.detach() for key, value in losses.items()})

                if self.evaluator is not None:
                    metrics = self.evaluator(inputs, outputs)
                    stats_accumulator.update_values({key: value.detach() for key, value in metrics.items()})

                self.callback_handler.on_substep_end(self.config, self.state, self.control)

                if step_end:
                    self.state.epoch_step += 1

                    if is_train:
                        self.state.global_step += 1
                        self.state.epoch = epoch + self.state.epoch_step / steps_in_epoch

                        self.optimizer.anneal_on_step_end()

                        if self.state.is_main_process and self.config.use_ema:
                            self.ema_model.update()

                    epoch_stats.update_values(stats_accumulator.mean_values)

                    self.callback_handler.on_step_end(
                        self.config, self.state, self.control,
                        epoch_stats=epoch_stats.mean_values,
                        lr=self.optimizer.get_last_lr()[0] if self.optimizer is not None else None,
                        grad_norm=grad_norm
                    )

                    if is_train:
                        # training step logs
                        logs = {}
                        if self.control.should_log:
                            logs = {
                                key: value.mean().item()
                                for key, value in self.accelerator.gather(stats_accumulator.mean_values).items()
                            }
                            logs.update(**{
                                "stats/time": time.perf_counter() - start_time,
                                "stats/time/data": time_data,
                                "stats/time/model": time_model,
                                "stats/learning_rate": self.optimizer.get_last_lr()[0],
                                "stats/grad_norm": float(grad_norm) if grad_norm is not None else None
                            })
                            logs = {f"train_step/{key}": value for key, value in logs.items()}

                        # potential model evaluation
                        self._maybe_log_save_evaluate(logs=logs)

                        # reset back to training
                        self.control.is_train = True
                        self.control.should_epoch_stop = False  # might still do training
                        self.state.epoch_step = int((idx + 1) / accum_steps)  # might be reset during evaluation

                    stats_accumulator.reset()

                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        break

                    start_time = time.perf_counter()
                    self.callback_handler.on_step_begin(self.config, self.state, self.control)

        prefix = "train" if is_train else "eval"
        metrics = {
            key: value.mean().item()
            for key, value in self.accelerator.gather(epoch_stats.mean_values).items()
        }
        logs = {f"{prefix}/{key}": value for key, value in metrics.items()}

        # eval_logs = None
        # if self.evaluator is not None and not is_train:
        #     eval_logs = self.evaluator.on_eval_epoch_end(inputs=batch, outputs=outputs)

        self.callback_handler.on_log(self.config, self.state, self.control, logs=logs)  # , eval_logs=eval_logs)

        self.callback_handler.on_epoch_end(self.config, self.state, self.control, metrics=metrics)

        if is_train:
            self._maybe_log_save_evaluate()
        else:
            self.state.epoch_step = _epoch_step
        # torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()

        return metrics
