# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Automodel training entry point's metric naming.

The shared callback's rule is that the backend passes its framework's own metric
name and the phase supplies the prefix. Automodel is the one backend whose
framework already prefixes some of its validation metrics, so it has to undo
that first -- otherwise the prefix arrives twice.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest

#: The leaf modules finetune.py imports from. nemo_automodel exists only inside
#: the training image, so the module cannot be imported in a plain checkout.
#: Only the leaves are needed: `from a.b.c import D` consults sys.modules for
#: `a.b.c` before it ever reaches for the parents.
_STUBBED_MODULES = (
    "nemo_automodel.components.checkpoint.checkpointing",
    "nemo_automodel.components.config._arg_parser",
    "nemo_automodel.components.training.step_scheduler",
    "nemo_automodel.recipes.llm.kd",
    "nemo_automodel.recipes.llm.train_ft",
    "nemo_automodel.recipes.retrieval.train_bi_encoder",
    "nemo_automodel.recipes.retrieval.train_cross_encoder",
)
_FINETUNE = "nmp.automodel.tasks.training.backends.finetune"


@pytest.fixture
def finetune(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """Import finetune.py under throwaway stubs.

    The stubs go in via monkeypatch rather than a bare ``sys.modules``
    assignment so they are torn down with the test: a module-scope stub outlives
    the file that installed it and leaks into whatever else shares the xdist
    worker. finetune itself is popped for the same reason -- left behind, it is
    a cached module holding references to mocks.
    """
    for name in _STUBBED_MODULES:
        monkeypatch.setitem(sys.modules, name, MagicMock())
    monkeypatch.delitem(sys.modules, _FINETUNE, raising=False)
    yield importlib.import_module(_FINETUNE)
    sys.modules.pop(_FINETUNE, None)


def test_the_recipes_own_val_prefix_comes_off(finetune: ModuleType) -> None:
    """Otherwise the phase prefix arrives twice: `val_loss` as `val_val_loss`."""
    assert finetune.strip_val_prefix({"val_loss": 0.5}) == {"loss": 0.5}


def test_every_prefixed_name_is_stripped_not_just_the_loss(finetune: ModuleType) -> None:
    """train_bi_encoder reports val_acc1 and val_mrr alongside val_loss.

    Special-casing `val_loss` fixes the one curve Studio charts and leaves the
    rest as `val_val_acc1` / `val_val_mrr`.
    """
    stripped = finetune.strip_val_prefix({"val_loss": 0.5, "val_acc1": 0.8, "val_mrr": 0.7})

    assert stripped == {"loss": 0.5, "acc1": 0.8, "mrr": 0.7}


def test_cross_encoder_target_is_detected(finetune: ModuleType) -> None:
    def NeMoAutoModelCrossEncoder() -> None:
        pass

    cfg = {"model": {"_target_": NeMoAutoModelCrossEncoder}}

    assert finetune._model_target_contains(cfg, "crossencoder") is True
    assert finetune._model_target_contains(cfg, "biencoder") is False


def test_an_unprefixed_name_is_left_alone(finetune: ModuleType) -> None:
    """The recipes are inconsistent: train_ft pairs `val_loss` with a bare `lr`."""
    stripped = finetune.strip_val_prefix({"val_loss": 0.5, "lr": 5e-06, "num_label_tokens": 128, "mem": 4.2})

    assert stripped == {"loss": 0.5, "lr": 5e-06, "num_label_tokens": 128, "mem": 4.2}


def test_only_a_leading_occurrence_is_removed(finetune: ModuleType) -> None:
    """`removeprefix`, not a replace: an interior `val_` is part of the name."""
    assert finetune.strip_val_prefix({"interval_val_x": 1.0}) == {"interval_val_x": 1.0}


def test_an_empty_metric_dict_survives(finetune: ModuleType) -> None:
    assert finetune.strip_val_prefix({}) == {}


def test_a_collision_keeps_the_prefixed_value(finetune: ModuleType) -> None:
    """Stripping can map two names onto one, and the wrong winner is undetectable.

    No recipe reports both today, but the loser vanishes silently and the
    validation curve then charts whatever else was in the dict. The prefixed name
    wins, being the one the recipe marked as validation.
    """
    assert finetune.strip_val_prefix({"val_loss": 0.5, "loss": 0.7}) == {"loss": 0.5}
    assert finetune.strip_val_prefix({"loss": 0.7, "val_loss": 0.5}) == {"loss": 0.5}


def test_a_collision_is_logged(finetune: ModuleType, caplog: pytest.LogCaptureFixture) -> None:
    """Silently dropping a metric is how this would go unnoticed for a release."""
    with caplog.at_level(logging.WARNING):
        finetune.strip_val_prefix({"val_loss": 0.5, "loss": 0.7})

    assert "val_loss" in caplog.text


# --------------------------------------------------------------------------- #
# The reporting budget, read back off the compiled recipe config
# --------------------------------------------------------------------------- #


class _Recipe:
    """A recipe carrying only the config attribute the resolver reads."""

    def __init__(self, cfg: object) -> None:
        self.cfg = cfg


def test_the_compiled_metric_list_is_used(finetune: ModuleType) -> None:
    recipe = _Recipe({"_progress_reporting": {"time_series_metrics": ["train_loss", "*_lr"]}})

    assert finetune._resolve_time_series_metrics(recipe) == ["train_loss", "*_lr"]


def test_an_empty_list_is_kept_rather_than_read_as_absent(finetune: ModuleType) -> None:
    """`[]` records nothing and absent takes the default; they must not collapse.

    An empty list is a legitimate request -- current values with no history at
    all -- and it is the one value a truthiness check would silently convert into
    its opposite.
    """
    assert finetune._resolve_time_series_metrics(_Recipe({"_progress_reporting": {"time_series_metrics": []}})) == []


@pytest.mark.parametrize(
    "cfg",
    [
        {},  # a config compiled before the knob existed
        {"_progress_reporting": {}},
        {"_progress_reporting": {"time_series_metrics": None}},  # stated explicitly
        None,
    ],
)
def test_an_absent_list_takes_the_backend_default(finetune: ModuleType, cfg: object) -> None:
    """Absent means the backend's default, not everything.

    `["*"]` is how a user asks for every metric, which matters because the two
    now differ: the default drops automodel's five throughput and accounting
    counters.
    """
    assert finetune._resolve_time_series_metrics(_Recipe(cfg)) == finetune.DIAGNOSTIC_TIME_SERIES


def test_the_default_keeps_the_diagnostic_metrics_and_drops_the_counters(finetune: ModuleType) -> None:
    """Pinned against automodel's real metric names by running the real callback.

    Train comes from `train_ft.py` `_run_train_optim_step`, validation from
    `_run_validation_epoch` after `strip_val_prefix`. An earlier version of this
    test reimplemented `fnmatchcase` over the same hardcoded list and asserted
    against the module constant, so it executed no production code at all and
    would have passed against any matching implementation.
    """
    from typing import cast

    from nmp.customization_common.training.callbacks import TrainingProgressCallback
    from nmp.customization_common.training.progress import JobsServiceProgressReporter

    class _Reporter:
        def __init__(self) -> None:
            self.reports: list[dict[str, Any]] = []

        def fetch_current_metrics(self) -> dict[str, list[dict[str, float]]]:
            return {}

        def configure_progress_tracking(self, max_steps: int, num_epochs: int) -> None:
            pass

        def report_running(self, phase: str, **details: Any) -> None:
            self.reports.append(details)

        def close(self) -> None:
            pass

    reporter = _Reporter()
    callback = TrainingProgressCallback(
        cast(JobsServiceProgressReporter, reporter),
        time_series_metrics=finetune.DIAGNOSTIC_TIME_SERIES,
        min_report_interval_seconds=0,
    )
    callback.report_train_step(
        step=1,
        epoch=1,
        metrics={
            "loss": 0.5,
            "grad_norm": 1.2,
            "lr": 1e-5,
            "mem": 12.5,
            "tps": 4821.0,
            "tps_per_gpu": 4821.0,
            "num_tokens_per_step": 8192,
            "num_label_tokens": 4096,
        },
    )
    callback.report_validation(
        step=1, epoch=1, metrics={"loss": 0.4, "lr": 1e-5, "num_label_tokens": 4096, "mem": 12.5}
    )

    recorded = {name for name, points in reporter.reports[-1]["metrics"].items() if points}
    assert recorded == {"train_loss", "train_grad_norm", "train_lr", "val_loss", "val_lr"}
    # ...and the counters still report a latest value, which is the whole point
    # of leaving them out rather than dropping them.
    assert reporter.reports[0]["train_tps"] == 4821.0


@pytest.mark.parametrize(
    "names",
    ["train_loss", {"train_loss": True}, ["train_loss", 3], [None], 7],
)
def test_an_unusable_list_falls_back_to_the_default_and_says_so(
    finetune: ModuleType, caplog: pytest.LogCaptureFixture, names: object
) -> None:
    """Discarded whole: half a malformed list is a silently arbitrary set of curves.

    A bare string is the trap worth naming -- `time_series_metrics: train_loss`
    in YAML is iterable, and taken as a list it would look for metrics named
    `t`, `r`, `a` and so on.
    """
    recipe = _Recipe({"_progress_reporting": {"time_series_metrics": names}})
    with caplog.at_level(logging.WARNING):
        assert finetune._resolve_time_series_metrics(recipe) == finetune.DIAGNOSTIC_TIME_SERIES

    assert "time_series_metrics" in caplog.text


# --------------------------------------------------------------------------- #
# Validation over more than one dataset
# --------------------------------------------------------------------------- #


class _LogData:
    def __init__(self, step: int, epoch: int, metrics: dict[str, float]) -> None:
        self.step, self.epoch, self.metrics = step, epoch, metrics


class _RecordingCallback:
    def __init__(self) -> None:
        self.validations: list[dict[str, object]] = []

    def report_validation(self, step: int, epoch: int, metrics: dict[str, float]) -> None:
        self.validations.append({"step": step, "metrics": metrics})


def _wrapper_with(finetune: ModuleType, callback: _RecordingCallback) -> Any:
    """An AutomodelRecipeWrapper with only the collaborators _log_val_metrics touches.

    Built without __init__ because the real one calls recipe.setup() and builds a
    live reporter; what is under test is the validation-reporting path alone.
    """
    wrapper = finetune.AutomodelRecipeWrapper.__new__(finetune.AutomodelRecipeWrapper)
    wrapper.callback = callback
    wrapper._original_log_val_metrics = lambda *a, **k: None
    wrapper._val_datasets = finetune.DatasetQualifier()
    return wrapper


def test_a_second_validation_dataset_does_not_share_the_first_ones_series(finetune: ModuleType) -> None:
    """`run_train_validation_loop` iterates `val_dataloaders` and logs once per entry.

    The wrapper discarded `val_name`, so every dataset reported as `val_loss` --
    two datasets interleaved as two points at one step in one series. Studio keys
    its loss chart by step, so one silently won and the chart showed whichever
    dataset iterated last, with no indication the other existed.
    """
    callback = _RecordingCallback()
    wrapper = _wrapper_with(finetune, callback)

    wrapper._log_val_metrics("train_ds", _LogData(step=99, epoch=0, metrics={"val_loss": 0.4}))
    wrapper._log_val_metrics("heldout", _LogData(step=99, epoch=0, metrics={"val_loss": 0.9}))

    assert callback.validations[0]["metrics"] == {"loss": 0.4}, "the first dataset keeps the bare names"
    assert callback.validations[1]["metrics"] == {"heldout_loss": 0.9}


def test_one_validation_dataset_keeps_the_bare_metric_names(finetune: ModuleType) -> None:
    """The ordinary run must be untouched -- automodel names that dataloader too,
    so keying on "a name arrived" would rename val_loss and take Studio's curve."""
    callback = _RecordingCallback()
    wrapper = _wrapper_with(finetune, callback)

    for step in (99, 199):
        wrapper._log_val_metrics("train_ds", _LogData(step=step, epoch=0, metrics={"val_loss": 0.4}))

    assert [v["metrics"] for v in callback.validations] == [{"loss": 0.4}, {"loss": 0.4}]


def test_a_recipe_that_passes_no_dataset_name_is_unchanged(finetune: ModuleType) -> None:
    """The VLM/biencoder signature is (log_data) with no val_name to qualify by."""
    callback = _RecordingCallback()
    wrapper = _wrapper_with(finetune, callback)

    wrapper._log_val_metrics(_LogData(step=99, epoch=0, metrics={"val_loss": 0.4, "val_acc1": 0.8}))

    assert callback.validations[0]["metrics"] == {"loss": 0.4, "acc1": 0.8}


def test_the_compiled_report_interval_is_used(finetune: ModuleType) -> None:
    recipe = _Recipe({"_progress_reporting": {"min_report_interval_seconds": 30}})

    assert finetune._resolve_min_report_interval(recipe) == 30.0


@pytest.mark.parametrize(
    "cfg",
    [
        {},  # a config compiled before the knob existed
        {"_progress_reporting": {}},
        {"_progress_reporting": {"min_report_interval_seconds": None}},
        {"_progress_reporting": {"min_report_interval_seconds": "soon"}},
        {"_progress_reporting": {"min_report_interval_seconds": True}},  # bool is an int subclass
        None,
    ],
)
def test_an_unusable_report_interval_falls_back(finetune: ModuleType, cfg: object) -> None:
    """Read as defensively as its neighbour: this runs in the wrapper's
    constructor, outside any try, so raising would kill the training process."""
    assert finetune._resolve_min_report_interval(_Recipe(cfg)) == finetune.DEFAULT_MIN_REPORT_INTERVAL_SECONDS


def test_a_zero_report_interval_is_kept_rather_than_read_as_absent(finetune: ModuleType) -> None:
    """0 means "send every report" and is a legitimate request, not a missing value."""
    assert (
        finetune._resolve_min_report_interval(_Recipe({"_progress_reporting": {"min_report_interval_seconds": 0}}))
        == 0.0
    )


# --------------------------------------------------------------------------- #
# AutomodelRecipeWrapper end to end
#
# The code that actually runs in the training loop, and it had no coverage at
# all: a review found eight mutations to it that the whole suite survived,
# including deleting the config arguments to the callback and dropping
# `strip_val_prefix` at its call site. `strip_val_prefix` itself is thoroughly
# tested above -- it was only its *call site* that nothing reached.
# --------------------------------------------------------------------------- #


class _Scheduler:
    def __init__(self, max_steps: int = 40, num_epochs: int = 2) -> None:
        self.max_steps, self.num_epochs = max_steps, num_epochs
        self.step, self.epoch, self.is_last_batch = 0, 0, False


class _Sample:
    def __init__(self, step: int, epoch: int, metrics: dict[str, float]) -> None:
        self.step, self.epoch, self.metrics = step, epoch, metrics


class _CheckpointerConfig:
    def __init__(self, checkpoint_dir: str) -> None:
        self.checkpoint_dir = checkpoint_dir


class _Checkpointer:
    def __init__(self, checkpoint_dir: str) -> None:
        self.config = _CheckpointerConfig(checkpoint_dir)


class _FakeRecipe:
    """A recipe with the surface the wrapper touches, and nothing else."""

    def __init__(self, cfg: object | None = None, checkpoint_dir: str = "/ckpt") -> None:
        self.cfg = cfg if cfg is not None else {}
        self.step_scheduler = _Scheduler()
        self.checkpointer = _Checkpointer(checkpoint_dir)
        self.dist_env = None
        self.calls: list[str] = []

    def setup(self) -> None:
        self.calls.append("setup")

    def run_train_validation_loop(self) -> None:
        self.calls.append("loop")

    def log_train_metrics(self, log_data: object) -> None:
        self.calls.append("orig_train")

    def log_val_metrics(self, *args: object, **kwargs: object) -> None:
        self.calls.append("orig_val")

    def save_checkpoint(self, epoch, step, train_loss, val_loss=None, best_metric_key="default") -> None:
        self.calls.append("orig_save")


def _wrapper(finetune: ModuleType, monkeypatch: pytest.MonkeyPatch, recipe: _FakeRecipe) -> tuple[Any, list]:
    """Build the real wrapper over a fake recipe, capturing what it reports."""
    reports: list[tuple[str, dict]] = []

    class _Reporter:
        def __init__(self, *a: object, **k: object) -> None: ...
        def fetch_current_metrics(self):
            return {}

        def configure_progress_tracking(self, max_steps, num_epochs): ...
        def report_running(self, phase, **details):
            reports.append((phase, details))

        def close(self): ...

    monkeypatch.setattr(finetune, "JobsServiceProgressReporter", _Reporter)
    monkeypatch.setattr(finetune.NMPJobContext, "from_env", classmethod(lambda cls: object()))
    return finetune.AutomodelRecipeWrapper(recipe), reports


def test_the_wrapper_reports_nothing_while_it_builds(finetune: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """Construction is not an event; the first report is the first real one.

    The runner reports `training` before it spawns this subprocess, and
    `report_training_start` reports `training` again once setup finishes. Anything
    sent from the constructor lands between the two, so the phase visibly goes
    backwards -- Studio renders Training, then Recipe Setup, then Training. An
    `automodel_recipe_setup` report used to sit here and did exactly that.

    Covering the model-load window is a real want, but it needs a heartbeat across
    `setup()`; a one-shot marker cannot tell a hang at second 30 from one an hour in.
    """
    _, reports = _wrapper(finetune, monkeypatch, _FakeRecipe())

    assert reports == [], f"reported {[phase for phase, _ in reports]} before training started"


def test_the_wrapper_reports_a_train_step_one_based(finetune: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """The recipe counts steps from 0; status_details is 1-based throughout."""
    recipe = _FakeRecipe()
    wrapper, reports = _wrapper(finetune, monkeypatch, recipe)

    recipe.log_train_metrics(_Sample(step=0, epoch=0, metrics={"loss": 0.5, "lr": 1e-5}))

    assert "orig_train" in recipe.calls, "the recipe's own logging still happens"
    training = [d for phase, d in reports if phase == "training" and "step" in d]
    assert training[-1]["step"] == 1
    assert training[-1]["epoch"] == 1
    assert training[-1]["train_loss"] == 0.5


def test_the_wrapper_strips_the_recipes_val_prefix_at_the_call_site(
    finetune: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without this the phase prefix arrives twice, as `val_val_loss`."""
    recipe = _FakeRecipe()
    wrapper, reports = _wrapper(finetune, monkeypatch, recipe)

    recipe.log_val_metrics("train_ds", _Sample(step=9, epoch=0, metrics={"val_loss": 0.4}))

    validation = [d for phase, d in reports if phase == "validation"]
    assert validation[-1]["val_loss"] == 0.4
    assert "val_val_loss" not in validation[-1]


def test_the_wrapper_passes_the_compiled_reporting_config_to_the_callback(
    finetune: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both arguments were deletable without a single test noticing."""
    recipe = _FakeRecipe({"_progress_reporting": {"time_series_metrics": ["*_loss"], "min_report_interval_seconds": 0}})
    wrapper, reports = _wrapper(finetune, monkeypatch, recipe)

    # Two steps back to back. The interval argument is only observable across
    # more than one report: the first always sends whatever the limit is, so a
    # single-step test passes just as happily with the argument dropped and the
    # ten-second default in force.
    recipe.log_train_metrics(_Sample(step=0, epoch=0, metrics={"loss": 0.5, "tps": 4821.0}))
    recipe.log_train_metrics(_Sample(step=1, epoch=0, metrics={"loss": 0.4, "tps": 4822.0}))

    training = [d for phase, d in reports if phase == "training" and "step" in d]
    assert [d["step"] for d in training] == [1, 2], "interval 0 sends both"
    assert training[-1]["train_tps"] == 4822.0, "reported as a latest value"
    assert "train_tps" not in training[-1]["metrics"], "but given no history, per the config"


def test_a_reporting_failure_does_not_stop_training(finetune: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every wrapped hook runs inline in the training loop.

    The three `try/except` blocks are the only thing between a reporting bug and
    a dead run, and all three survived being turned into bare `raise`.
    """
    recipe = _FakeRecipe()
    wrapper, _ = _wrapper(finetune, monkeypatch, recipe)

    def boom(*a: object, **k: object) -> None:
        raise RuntimeError("jobs service unreachable")

    monkeypatch.setattr(wrapper.callback, "report_train_step", boom)
    monkeypatch.setattr(wrapper.callback, "report_validation", boom)
    monkeypatch.setattr(wrapper.callback, "report_checkpoint_saved", boom)
    monkeypatch.setattr(wrapper.callback, "report_epoch_end", boom)

    recipe.log_train_metrics(_Sample(step=0, epoch=0, metrics={"loss": 0.5}))
    recipe.log_val_metrics("ds", _Sample(step=0, epoch=0, metrics={"val_loss": 0.4}))
    recipe.save_checkpoint(epoch=0, step=9, train_loss=0.5)

    assert recipe.calls.count("orig_train") == 1, "the recipe's own work still ran"
    assert recipe.calls.count("orig_val") == 1
    assert recipe.calls.count("orig_save") == 1


def test_the_wrapper_states_the_schedule_and_closes_the_callback(
    finetune: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`report_training_start` is what tells the reporter the run length, and
    `close()` is in a `finally` so a crashing loop still flushes."""
    recipe = _FakeRecipe()
    wrapper, reports = _wrapper(finetune, monkeypatch, recipe)
    closed: list[bool] = []
    monkeypatch.setattr(wrapper.callback, "close", lambda: closed.append(True))

    def _raise_cuda_oom() -> None:
        raise RuntimeError("cuda oom")

    monkeypatch.setattr(wrapper._recipe, "run_train_validation_loop", _raise_cuda_oom)
    with pytest.raises(RuntimeError, match="cuda oom"):
        wrapper.run_train_validation_loop()

    start = [d for phase, d in reports if phase == "training" and "max_steps" in d]
    assert start[-1] == {"max_steps": 40, "num_epochs": 2}
    assert closed == [True], "closed even though the loop raised"


def test_the_wrapper_reports_a_checkpoint_one_based_with_its_path(
    finetune: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = _FakeRecipe()
    wrapper, reports = _wrapper(finetune, monkeypatch, recipe)

    recipe.save_checkpoint(epoch=1, step=19, train_loss=0.5)

    saved = [d for phase, d in reports if phase == "checkpoint_saved"]
    assert saved[-1]["step"] == 20
    assert saved[-1]["epoch"] == 2
    assert saved[-1]["checkpoint_path"] == "/ckpt"


def test_the_wrapper_writes_validation_stats_for_each_checkpoint(
    finetune: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recipe = _FakeRecipe(checkpoint_dir=str(tmp_path))
    _wrapper(finetune, monkeypatch, recipe)

    recipe.save_checkpoint(epoch=0, step=9, train_loss=0.6, val_loss={"default": 0.5})
    recipe.save_checkpoint(epoch=1, step=19, train_loss=0.4, val_loss={"default": 0.3})

    stats = json.loads((tmp_path / "checkpoint_stats.json").read_text())
    assert stats["checkpoints"] == [
        {
            "epoch": 1,
            "step": 10,
            "train_loss": 0.6,
            "val_loss": {"default": 0.5},
            "best_metric_key": "default",
        },
        {
            "epoch": 2,
            "step": 20,
            "train_loss": 0.4,
            "val_loss": {"default": 0.3},
            "best_metric_key": "default",
        },
    ]
