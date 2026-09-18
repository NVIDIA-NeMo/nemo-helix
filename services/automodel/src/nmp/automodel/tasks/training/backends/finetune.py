# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Automodel training subprocess entry point.

Wraps nemo_automodel recipes with Jobs-service progress reporting (SFT, KD, embedding).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.training.step_scheduler import StepScheduler
from nemo_automodel.recipes.llm.kd import KnowledgeDistillationRecipeForNextTokenPrediction
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction
from nemo_automodel.recipes.retrieval.train_bi_encoder import TrainBiEncoderRecipe
from nemo_automodel.recipes.retrieval.train_cross_encoder import TrainCrossEncoderRecipe
from nmp.automodel.tasks.tqdm_logging import install_line_tqdm
from nmp.automodel.tasks.training.progress import JobsServiceProgressReporter
from nmp.customization_common.service.context import NMPJobContext
from nmp.customization_common.training.callbacks import DatasetQualifier, TrainingProgressCallback
from nmp.customization_common.training.reporting import (
    DEFAULT_MIN_REPORT_INTERVAL_SECONDS,
    DIAGNOSTIC_TIME_SERIES,
)

logger = logging.getLogger(__name__)

CHECKPOINT_STATS_FILENAME = "checkpoint_stats.json"


def _json_metric(value: Any) -> Any:
    """Convert tensor/numpy scalar metric values into JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(key): _json_metric(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_metric(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    return value


@runtime_checkable
class AutomodelRecipe(Protocol):
    """Protocol defining the interface we need from Automodel recipes.

    This makes the dependencies explicit and enables type checking, unlike
    the previous mixin approach that relied on implicit attributes.
    """

    cfg: Any
    step_scheduler: StepScheduler
    checkpointer: Checkpointer
    dist_env: Any

    def setup(self) -> None:
        """Build all components needed for training."""
        ...

    def run_train_validation_loop(self) -> None:
        """Run the main training/validation loop."""
        ...

    def log_train_metrics(self, log_data: Any) -> None:
        """Log training metrics."""
        ...

    def log_val_metrics(self, *args: Any, **kwargs: Any) -> None:
        """Log validation metrics.

        Note: Signature varies across Automodel recipes:
        - LLM/KD: (val_name, log_data, metric_logger=None)
        - VLM/biencoder/seq_cls: (log_data)
        """
        ...

    def save_checkpoint(
        self,
        epoch: int,
        step: int,
        train_loss: float,
        val_loss: dict[str, float] | None = None,
        best_metric_key: str = "default",
    ) -> None:
        """Save a checkpoint."""
        ...


def strip_val_prefix(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Drop the ``val_`` the Automodel recipes already put on a metric name.

    The shared callback's naming rule is that the backend supplies its
    framework's own name and the phase supplies the prefix, so a name that
    arrives pre-prefixed comes back doubled. Stripping is what makes ``val_loss``
    land as ``val_loss`` rather than ``val_val_loss``.

    It applies to every name, not just ``val_loss``, because the recipes are
    inconsistent about which metrics they prefix: ``train_ft`` reports
    ``val_loss`` alongside a bare ``lr`` and ``num_label_tokens``, and
    ``train_bi_encoder`` adds ``val_acc1`` and ``val_mrr``. Only the callback
    should be deciding the phase, so the prefix comes off wherever the recipe
    happened to put one.

    Stripping can collide: a dict carrying both ``val_loss`` and ``loss`` maps
    them onto one name, and whichever lands second silently replaces the other.
    None of today's recipes do that, but nothing stops one from starting, and the
    failure would read as a validation curve charting the wrong quantity. The
    prefixed name wins, being the one the recipe marked as validation, and the
    collision is logged rather than swallowed.
    """
    stripped: dict[str, Any] = {}
    for name, value in metrics.items():
        bare = name.removeprefix("val_")
        if bare not in stripped:
            stripped[bare] = value
            continue
        # One of the two is the prefixed name -- they cannot both be, having come
        # from one dict -- so whether this one wins is just whether it is that one.
        prefixed, dropped = (name, bare) if bare != name else (f"val_{bare}", name)
        logger.warning(
            f"Validation metrics carry both {prefixed!r} and {bare!r}, which strip to one name; "
            f"reporting the {prefixed!r} value and dropping {dropped!r}."
        )
        if bare != name:
            stripped[bare] = value
    return stripped


def _reporting_block(recipe: AutomodelRecipe) -> Any:
    """The ``_progress_reporting`` block config.py compiled in, if it is there.

    Ours rather than the recipe's, which is why every read of it is defensive:
    the recipe config is also loadable from a hand-written YAML, and a run whose
    config predates this block should report at the shared defaults rather than
    fail to start over a reporting knob. This runs from the wrapper's
    constructor, outside any try, so raising here kills the training process.
    """
    cfg = getattr(recipe, "cfg", None)
    return cfg.get("_progress_reporting") if hasattr(cfg, "get") else None


def _resolve_time_series_metrics(recipe: AutomodelRecipe) -> tuple[str, ...] | list[str]:
    """Which metrics get a stored series, defaulting to the diagnostic set.

    Automodel is the backend this matters most for. Its recipes report eight
    metrics on the train path and four on validation, and seven of those twelve
    series are throughput and accounting counters, whose current value is all
    anyone reads: ``mem``, ``tps``, ``tps_per_gpu``, ``num_tokens_per_step`` and
    ``num_label_tokens`` on train, then ``mem`` and ``num_label_tokens`` a second
    time on validation. Keeping the loss, learning rate and gradient norm takes
    the stored blob from twelve series to five.

    An absent or null list means "the default", not "everything": ``["*"]`` is
    how a user asks for every metric. An empty list means no series at all and is
    honoured as written.

    A non-list, or a list with anything but strings in it, falls back to the
    default whole. Taking the usable half of a malformed list would produce a
    silently arbitrary set of curves, which is worse than a stated one.
    """
    block = _reporting_block(recipe)
    names = block.get("time_series_metrics") if hasattr(block, "get") else None
    if isinstance(names, (list, tuple)) and all(isinstance(name, str) for name in names):
        return list(names)
    if names is not None:
        logger.warning(
            f"Ignoring an unusable progress_reporting.time_series_metrics ({names!r}); "
            f"recording the default set instead: {', '.join(DIAGNOSTIC_TIME_SERIES)}."
        )
    return DIAGNOSTIC_TIME_SERIES


def _resolve_min_report_interval(recipe: AutomodelRecipe) -> float:
    """Least seconds between reports, defaulting to the shared value.

    Read as defensively as its neighbour and for the same reason: this comes off
    a config file that a person can hand-write, and it is consumed in the
    wrapper's constructor with nothing catching underneath. A negative value is
    left to the limiter, which clamps rather than raises -- a reporting knob must
    not be able to stop a run from starting.
    """
    block = _reporting_block(recipe)
    interval = block.get("min_report_interval_seconds") if hasattr(block, "get") else None
    if isinstance(interval, (int, float)) and not isinstance(interval, bool):
        return float(interval)
    if interval is not None:
        logger.warning(
            f"Ignoring an unusable progress_reporting.min_report_interval_seconds ({interval!r}); "
            f"reporting at most every {DEFAULT_MIN_REPORT_INTERVAL_SECONDS}s."
        )
    return DEFAULT_MIN_REPORT_INTERVAL_SECONDS


class AutomodelRecipeWrapper:
    """Wraps an Automodel recipe with Jobs-service progress reporting."""

    def __init__(self, recipe: AutomodelRecipe, job_ctx: NMPJobContext | None = None):
        """Initialize the wrapper with an Automodel recipe.

        Args:
            recipe: Any recipe implementing the AutomodelRecipe protocol
                    (SFT, KD, biencoder, etc.).
            job_ctx: NeMo Platform job context for progress reporting (optional,
                     defaults to environment variables).
        """
        self._job_ctx = job_ctx or NMPJobContext.from_env()

        self._recipe = recipe
        self._recipe.setup()

        self.max_steps = self._recipe.step_scheduler.max_steps
        self.num_epochs = getattr(self._recipe.step_scheduler, "num_epochs", None) or 1

        #: Keeps a second validation dataset's metrics out of the first's series.
        self._val_datasets = DatasetQualifier()

        # A local, not an attribute: the callback owns the reporter from here on,
        # including closing it in run_train_validation_loop. Nothing else in this
        # class reports directly.
        #
        # This used to be built above `setup()` so it could report an
        # `automodel_recipe_setup` phase before the model loaded. That report is
        # gone: the runner already reports `training` before it spawns this
        # subprocess, so the phase went Training -> Recipe Setup -> Training, and a
        # phase that regresses reads worse than no phase. It was also a one-shot
        # with no heartbeat, so a hang early in setup looked exactly like a hang an
        # hour in. Covering that window wants a heartbeat, not a marker.
        reporter = JobsServiceProgressReporter(self._job_ctx)
        self.callback = TrainingProgressCallback(
            reporter,
            time_series_metrics=_resolve_time_series_metrics(recipe),
            min_report_interval_seconds=_resolve_min_report_interval(recipe),
        )
        logger.info(f"Automodel recipe wrapper initialized: max_steps={self.max_steps}, num_epochs={self.num_epochs}")

        # Store original methods before patching
        self._original_log_train_metrics = recipe.log_train_metrics
        self._original_log_val_metrics = recipe.log_val_metrics
        self._original_save_checkpoint = recipe.save_checkpoint
        self._checkpoint_stats: list[dict[str, Any]] = []

        # Monkey-patch the recipe's methods to add our callbacks
        recipe.log_train_metrics = self._log_train_metrics  # type: ignore[method-assign]
        recipe.log_val_metrics = self._log_val_metrics  # type: ignore[method-assign]
        recipe.save_checkpoint = self._save_checkpoint  # type: ignore[method-assign]

    @property
    def recipe(self) -> AutomodelRecipe:
        """Access the underlying recipe."""
        return self._recipe

    def run_train_validation_loop(self) -> None:
        """Run training and close the progress callback."""
        try:
            self.callback.report_training_start(self.max_steps, self.num_epochs)
            self._recipe.run_train_validation_loop()
        finally:
            if self.callback:
                self.callback.close()
                logger.info("Training progress callback closed")

    def _log_train_metrics(self, log_data: Any) -> None:
        """Wrapped log_train_metrics with Jobs-service reporting."""
        self._original_log_train_metrics(log_data)
        if self.callback and log_data:
            try:
                metrics = getattr(log_data, "metrics", {})
                self.callback.report_train_step(
                    step=getattr(log_data, "step", 0) + 1,  # Convert to 1-based
                    epoch=getattr(log_data, "epoch", 0) + 1,  # Convert to 1-based
                    metrics=dict(metrics),
                )
            except Exception as e:
                logger.warning(f"Failed to report training progress: {e}")

            try:
                if self._recipe.step_scheduler.is_last_batch:
                    self.callback.report_epoch_end(
                        step=self._recipe.step_scheduler.step + 1,
                        epoch=self._recipe.step_scheduler.epoch + 1,
                    )
            except Exception as e:
                logger.warning(f"Failed to report epoch end: {e}")

    def _log_val_metrics(self, *args: Any, **kwargs: Any) -> None:
        """Wrapped log_val_metrics with Jobs-service reporting.

        Handles different Automodel recipe signatures:
        - LLM/KD: (val_name, log_data, metric_logger=None)
        - VLM/biencoder/seq_cls: (log_data)
        """
        # Call original method first with whatever args were passed
        self._original_log_val_metrics(*args, **kwargs)

        # Extract log_data from args (it's always the last positional arg before kwargs)
        # LLM signature: (val_name, log_data, metric_logger=None) -> log_data is args[1]
        # VLM/biencoder signature: (log_data) -> log_data is args[0]
        log_data = None
        val_name = None
        if len(args) >= 2:
            # LLM/KD style: (val_name, log_data, ...)
            val_name, log_data = args[0], args[1]
        elif len(args) == 1:
            # VLM/biencoder style: (log_data)
            log_data = args[0]

        if self.callback and log_data:
            try:
                metrics = strip_val_prefix(getattr(log_data, "metrics", {}))
                # `run_train_validation_loop` iterates `val_dataloaders` and calls
                # this once per entry, all at one step. `val_name` is the only
                # thing distinguishing them, and discarding it made every dataset
                # report as `val_loss` -- two datasets interleaved as two points
                # at one step in one series, with Studio's step-keyed chart
                # silently picking a winner. The first dataset keeps the bare
                # names so the ordinary single-dataset run is unchanged.
                if val_name is not None:
                    metrics = self._val_datasets.qualify(str(val_name), str(val_name), metrics)
                self.callback.report_validation(
                    step=getattr(log_data, "step", 0) + 1,  # Convert to 1-based
                    epoch=getattr(log_data, "epoch", 0) + 1,  # Convert to 1-based
                    metrics=metrics,
                )
            except Exception as e:
                logger.warning(f"Failed to report validation progress: {e}")

    def _save_checkpoint(
        self,
        epoch: int,
        step: int,
        train_loss: float,
        val_loss: dict[str, float] | None = None,
        best_metric_key: str = "default",
    ) -> None:
        """Wrapped save_checkpoint with Jobs-service reporting."""
        self._original_save_checkpoint(epoch, step, train_loss, val_loss, best_metric_key)
        checkpoint_dir = getattr(
            getattr(self._recipe.checkpointer, "config", None),
            "checkpoint_dir",
            None,
        )
        if self.callback:
            try:
                self.callback.report_checkpoint_saved(
                    step=step + 1,  # Convert to 1-based
                    epoch=epoch + 1,  # Convert to 1-based
                    checkpoint_path=str(checkpoint_dir) if checkpoint_dir else None,
                )
            except Exception as e:
                logger.warning(f"Failed to report checkpoint save: {e}")

        if checkpoint_dir and int(os.environ.get("RANK", "0")) == 0:
            try:
                self._checkpoint_stats.append(
                    {
                        "epoch": epoch + 1,
                        "step": step + 1,
                        "train_loss": _json_metric(train_loss),
                        "val_loss": _json_metric(val_loss),
                        "best_metric_key": best_metric_key,
                    }
                )
                stats_path = Path(checkpoint_dir) / CHECKPOINT_STATS_FILENAME
                stats_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = stats_path.with_suffix(".tmp")
                temp_path.write_text(
                    json.dumps({"checkpoints": self._checkpoint_stats}, indent=2) + "\n",
                    encoding="utf-8",
                )
                temp_path.replace(stats_path)
            except Exception as e:
                logger.warning(f"Failed to write checkpoint statistics: {e}")


def _is_kd_config(cfg: Any) -> bool:
    """Check if config is for knowledge distillation."""
    return cfg.get("teacher_model") is not None or cfg.get("kd_ratio") is not None


def _model_target_contains(cfg: Any, needle: str) -> bool:
    """Check a resolved model target's module/name for a recipe marker.

    Note: ConfigNode automatically resolves _target_ to the actual function/class,
    so inspect the function's module and qualified name.
    """
    try:
        model_cfg = cfg.get("model", {})
        if model_cfg is None:
            return False

        target = model_cfg.get("_target_")
        if target is None:
            return False

        # target is resolved to the actual function/class by ConfigNode
        # Check its module path or qualified name
        module = getattr(target, "__module__", "") or ""
        qualname = getattr(target, "__qualname__", "") or ""
        needle = needle.lower()
        return needle in module.lower() or needle in qualname.lower()
    except (AttributeError, TypeError):
        return False


def create_automodel_recipe(cfg: Any) -> AutomodelRecipeWrapper:
    """Create a progress-reporting wrapper for the recipe implied by *cfg*."""
    if _model_target_contains(cfg, "biencoder"):
        logger.info("Detected biencoder config, using embedding model recipe")
        base_recipe = TrainBiEncoderRecipe(cfg)
    elif _model_target_contains(cfg, "crossencoder"):
        logger.info("Detected cross-encoder config, using reranking recipe")
        base_recipe = TrainCrossEncoderRecipe(cfg)
    elif _is_kd_config(cfg):
        logger.info("Detected Knowledge Distillation config, using KD recipe")
        base_recipe = KnowledgeDistillationRecipeForNextTokenPrediction(cfg)
    else:
        logger.info("Using SFT fine-tuning recipe")
        base_recipe = TrainFinetuneRecipeForNextTokenPrediction(cfg)

    return AutomodelRecipeWrapper(base_recipe)


def main() -> None:
    # Automodel's base recipe imports tqdm lazily when it builds the bar, so this
    # takes effect here despite the recipe imports above. Without it the bar's
    # carriage-return refreshes arrive as partial per-line log records.
    install_line_tqdm()
    cfg = parse_args_and_load_config()
    recipe = create_automodel_recipe(cfg)
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()
