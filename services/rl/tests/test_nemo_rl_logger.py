# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for NemoRLLogger's translation of NeMo-RL metrics into Jobs Service reports.

The metric dicts here mirror what NeMo-RL actually hands the logger, including the
non-scalar entries (``Histogram`` objects, tables, nested dicts) that share the dict
with the numbers, so they are exercised rather than sanitised away.

The logger is an adapter and nothing else: it decides *whether* a call is a report
and what to call the metrics in it. Which entries survive, and *when* a report is
admitted, are both the shared callback's business -- see
``packages/nhx_customization_common/tests/training/test_callbacks.py``, which is
where this file's throttle and flush coverage moved when the cadence stopped being
implemented here. The stub below cannot observe a throttle by construction, which
is the point: there is no longer one to observe on this side.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from typing import Any

import pytest
from nhx.customization_common.training.reporting import DIAGNOSTIC_TIME_SERIES

# NeMo-RL is only installed inside the training image, so the LoggerInterface import
# at nemo_rl_logger module scope fails in a plain repo checkout. Stub just enough to
# import the module under test; when the real package IS present (the in-image smoke
# run) this is skipped and the genuine base class is used.
#
# The stubs are removed again as soon as the import that needs them is done. Left in
# sys.modules they outlived this file and leaked a hollow `nemo_rl` package into
# every other test sharing the xdist worker -- a sibling test_dpo_config had already
# had to be written around exactly that. nemo_rl_logger binds LoggerInterface at
# import time, so nothing downstream needs the stub to still be there.
_stubbed: list[str] = []
if importlib.util.find_spec("nemo_rl") is None:  # pragma: no cover - env dependent

    class LoggerInterface:  # minimal stand-in for the abstract base
        pass

    _logger_mod = types.ModuleType("nemo_rl.utils.logger")
    setattr(_logger_mod, "LoggerInterface", LoggerInterface)
    for _name, _module in (
        ("nemo_rl", types.ModuleType("nemo_rl")),
        ("nemo_rl.utils", types.ModuleType("nemo_rl.utils")),
        ("nemo_rl.utils.logger", _logger_mod),
    ):
        if _name not in sys.modules:
            sys.modules[_name] = _module
            _stubbed.append(_name)

from nhx.rl.tasks.training.backends.nemo_rl import nemo_rl_logger  # noqa: E402
from nhx.rl.tasks.training.backends.nemo_rl.nemo_rl_logger import (  # noqa: E402
    NemoRLLogger,
    resolve_steps_per_epoch,
)

for _name in _stubbed:
    del sys.modules[_name]


class _RecordingCallback:
    """Stands in for TrainingProgressCallback, capturing what the logger forwards."""

    def __init__(self) -> None:
        self.train_steps: list[dict[str, Any]] = []
        self.validations: list[dict[str, Any]] = []
        self.training_starts: list[dict[str, Any]] = []
        #: Report kinds in arrival order -- the real callback prunes against the
        #: step of the last one it saw, so the sequence is part of the contract.
        self.order: list[str] = []
        self.closed = False
        self.closes = 0
        #: The reporting config the logger built us with, so the plumbing from
        #: the job config down to the gate is assertable from this side.
        self.time_series_metrics: Any = None
        self.min_report_interval_seconds: Any = None

    def report_training_start(self, max_steps: int, num_epochs: int, *, run_facts=None) -> None:
        self.training_starts.append({"max_steps": max_steps, "num_epochs": num_epochs, "run_facts": run_facts})

    def report_train_step(self, step, epoch, metrics, *, backend=None):
        self.train_steps.append({"step": step, "epoch": epoch, "metrics": metrics})
        self.order.append("train")

    def report_validation(self, step, epoch, metrics, *, backend=None):
        self.validations.append({"step": step, "epoch": epoch, "metrics": metrics})
        self.order.append("validation")

    def close(self) -> None:
        self.closed = True
        self.closes += 1


@pytest.fixture
def callback(monkeypatch: pytest.MonkeyPatch) -> _RecordingCallback:
    """Build a NemoRLLogger whose reporter/callback are inert local objects."""
    recorder = _RecordingCallback()

    def _build(
        _reporter: Any,
        *,
        time_series_metrics: Any = None,
        min_report_interval_seconds: Any = None,
    ) -> _RecordingCallback:
        recorder.time_series_metrics = time_series_metrics
        recorder.min_report_interval_seconds = min_report_interval_seconds
        return recorder

    monkeypatch.setattr(nemo_rl_logger, "JobsServiceProgressReporter", lambda *a, **k: object())
    monkeypatch.setattr(nemo_rl_logger, "TrainingProgressCallback", _build)
    return recorder


def _make_logger(**kwargs: Any) -> NemoRLLogger:
    params: dict[str, Any] = {"steps_per_epoch": 10}
    params.update(kwargs)
    return NemoRLLogger(**params)


def _driver_steps(max_steps: int) -> range:
    """The step sequence an N-step run actually produces.

    dpo.py logs `total_steps + 1` with total_steps 0-based and incremented after
    the log, so an N-step run emits 1..N -- not 0..N-1. Tests that use range(N)
    directly would validate against a convention no caller uses.
    """
    return range(1, max_steps + 1)


class _RecordingReporter:
    """A JobsServiceProgressReporter that keeps every report instead of sending it.

    Lets a test drive the real TrainingProgressCallback, so what gets recorded as a
    series is decided by the production `_records_history`/`_qualify_metric_names`
    rather than by the test restating the matching rule.

    `fetch_current_metrics` returns `{}` -- nothing stored -- which is the first-run
    case. Returning None would mean the seed *read failed*, which suppresses the
    `metrics` payload entirely and would make every assertion here vacuous.
    """

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


def _recording_callback(time_series_metrics: Any) -> tuple[Any, _RecordingReporter]:
    """A real TrainingProgressCallback over a _RecordingReporter."""
    from typing import cast

    from nhx.customization_common.training.callbacks import TrainingProgressCallback
    from nhx.customization_common.training.progress import JobsServiceProgressReporter

    reporter = _RecordingReporter()
    callback = TrainingProgressCallback(
        cast(JobsServiceProgressReporter, reporter),
        time_series_metrics=time_series_metrics,
        min_report_interval_seconds=0,
    )
    return callback, reporter


class _Histogram:
    """Stand-in for a non-numeric metric value — NaN-hostile, like the real thing."""


# A DPO `train` dict: the scalars, plus the non-scalars that ride along with them
# in a real NeMo-RL metric dict.
TRAIN_METRICS: dict[str, Any] = {
    "loss": 0.5,
    "lr": 1e-5,
    "grad_norm": 0.9,
    "preference_loss": 0.42,
    "rewards_rejected_mean": -0.3,
    "num_valid_samples": 8,
    "global_valid_seqs": 8.0,
    "global_valid_toks": 1024.0,
    "some/histogram": _Histogram(),
    "generation_logger_metrics": {"inflight": [1, 2, 3]},
    "per_worker_token_counts": [{0: 100, 1: 120}],
}

VALIDATION_METRICS: dict[str, Any] = {"loss": 0.25, "num_valid_samples": 8}

# GRPO/PPO log this under `prefix="train"` at the same step as TRAIN_METRICS,
# from the rollout rather than the training step. Generation stats only: no loss.
ROLLOUT_METRICS: dict[str, Any] = {
    "total_turns": 64,
    "avg_turns_per_sample": 1.0,
    "mean_gen_tokens_per_sample": 412.5,
    "truncation_rate": 0.02,
}

# What GRPO's validate() returns. It scores on rewards, so there is no loss in it.
REWARD_VALIDATION_METRICS: dict[str, Any] = {"accuracy": 0.61, "avg_length": 412.5}


# --------------------------------------------------------------------------- #
# Module stub hygiene
# --------------------------------------------------------------------------- #


def test_the_import_stub_does_not_outlive_this_module() -> None:
    """A stub left in sys.modules leaks a hollow nemo_rl across the xdist worker.

    Worth asserting because the failure is silent and lands somewhere else: an
    unrelated `import nemo_rl.algorithms.dpo` would get a module with no
    attributes rather than a clean ImportError, and which tests broke would
    depend on collection order. Vacuously true in the training image, where
    nothing was stubbed because the real package is installed.
    """
    assert all(name not in sys.modules for name in _stubbed)


# --------------------------------------------------------------------------- #
# Train metrics
# --------------------------------------------------------------------------- #


def test_the_metric_dict_is_forwarded_whole(callback: _RecordingCallback) -> None:
    """No allow-list: a metric NeMo-RL adds charts without a change here.

    The old list silently dropped anything never added to it -- DPO's `accuracy`,
    `sft_loss` and `rewards_chosen_mean` among them. Deciding which entries are
    chartable is the callback's job, not a second gate doing a weaker version of
    the same check.
    """
    _make_logger().log_metrics(TRAIN_METRICS, step=0, prefix="train")

    assert callback.train_steps[0]["metrics"] == TRAIN_METRICS


def test_the_forwarded_dict_is_a_copy(callback: _RecordingCallback) -> None:
    """NeMo-RL reuses its metric dict across steps; a reference would alias it."""
    metrics = dict(TRAIN_METRICS)
    _make_logger().log_metrics(metrics, step=0, prefix="train")
    metrics["loss"] = 99.0

    assert callback.train_steps[0]["metrics"]["loss"] == 0.5


def test_train_call_without_a_loss_is_ignored(callback: _RecordingCallback) -> None:
    """A `train` log without a loss is a partial, mid-step call rather than a step."""
    rollout_only = {k: v for k, v in TRAIN_METRICS.items() if k != "loss"}
    _make_logger().log_metrics(rollout_only, step=0, prefix="train")

    assert callback.train_steps == []


def test_the_rollout_log_does_not_produce_a_second_train_report(callback: _RecordingCallback) -> None:
    """GRPO and PPO log twice under `prefix="train"` at one step.

    grpo.py logs `rollout_metrics` and then the training `metrics`, both at
    `total_steps + 1`. Only the second has a loss, which is why the branch keys
    on one: without it the step reports twice -- each resending every series --
    and a throttled step ends up pending as the rollout half, losing the loss.
    """
    logger = _make_logger()
    logger.log_metrics(ROLLOUT_METRICS, step=1, prefix="train")
    logger.log_metrics(TRAIN_METRICS, step=1, prefix="train")

    assert len(callback.train_steps) == 1
    assert callback.train_steps[0]["metrics"]["loss"] == 0.5


def test_every_train_scalar_is_forwarded(callback: _RecordingCallback) -> None:
    """Including the ones the old allow-list had no entry for."""
    _make_logger().log_metrics(TRAIN_METRICS, step=0, prefix="train")

    reported = callback.train_steps[0]["metrics"]
    assert reported["loss"] == 0.5
    assert reported["preference_loss"] == 0.42
    assert reported["rewards_rejected_mean"] == -0.3
    assert reported["global_valid_toks"] == 1024.0


# --------------------------------------------------------------------------- #
# Teardown reaches the callback, whichever name NeMo-RL calls
# --------------------------------------------------------------------------- #


def test_finish_closes_like_close(callback: _RecordingCallback) -> None:
    """`finish` is the name NeMo-RL's composite Logger actually dispatches.

    nemo_rl.utils.logger.Logger has no close(); its teardown fan-out is
    `getattr(logger, "finish", None)`. Without this alias the composite skips us
    entirely, the callback is never closed, and the step its gate withheld is
    never flushed. The flush itself is the callback's now -- what has to hold
    here is only that teardown reaches it.
    """
    logger = _make_logger()
    logger.log_metrics(TRAIN_METRICS, step=1, prefix="train")

    logger.finish()

    assert callback.closed


def test_finish_is_reachable_through_the_composite_dispatch(callback: _RecordingCallback) -> None:
    """Mirrors Logger.finish()'s exact lookup, so a rename here fails loudly."""
    logger = _make_logger()

    finish = getattr(logger, "finish", None)
    assert callable(finish)
    finish()

    assert callback.closed


def test_finish_then_close_closes_once(callback: _RecordingCallback) -> None:
    """Both the composite and the driver may call in; teardown runs once.

    The callback's close() is idempotent as well, so this is belt and braces --
    but a second close here would also mean a second flush there, and the pending
    report would land twice.
    """
    logger = _make_logger()
    logger.log_metrics(TRAIN_METRICS, step=1, prefix="train")

    logger.finish()
    logger.close()

    assert callback.closes == 1


# --------------------------------------------------------------------------- #
# Schedule resolution — one formula for both drivers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("val_period", [None, 0, 1, 5, 100, 20_000])
def test_the_schedule_no_longer_reads_the_validation_cadence(
    callback: _RecordingCallback, val_period: int | None
) -> None:
    """How often a run validates says nothing about how often it should report.

    The two were coupled, and the coupling ran the wrong way: validating less
    often -- what you do when validation is expensive -- made the training curve
    coarser. That term is gone, and then the whole cadence moved to the shared
    callback; `val_period` survives only as an accepted-and-ignored argument, so
    what is pinned here is that no configuration of it changes the logger built.
    """
    logger = NemoRLLogger.for_schedule(max_steps=20_000, num_epochs=1, val_period=val_period)

    assert logger._max_steps == 20_000
    assert logger._steps_per_epoch == 20_000


@pytest.mark.parametrize(
    "max_steps,num_epochs,explicit,expected",
    [
        (100, 4, None, 25),
        (100, None, None, 100),
        (100, 0, None, 100),  # guard against a zero divisor
        (3, 10, None, 1),  # floors to 0 -> clamped
        (100, 4, 40, 40),  # explicit wins (DPO carries steps_per_epoch)
        (100, 4, 0, 25),  # ...unless it is unusable
    ],
)
def test_resolve_steps_per_epoch(max_steps: int, num_epochs: int | None, explicit: int | None, expected: int) -> None:
    assert resolve_steps_per_epoch(max_steps, num_epochs, explicit) == expected


def test_for_schedule_builds_a_consistent_logger(callback: _RecordingCallback) -> None:
    """The schedule the logger still owns: epoch derivation and the run length."""
    logger = NemoRLLogger.for_schedule(max_steps=100, num_epochs=4, val_period=100)

    assert logger._steps_per_epoch == 25
    assert logger._max_steps == 100
    assert logger._num_epochs == 4


def test_the_charted_metric_names_reach_the_callback(callback: _RecordingCallback) -> None:
    NemoRLLogger.for_schedule(
        max_steps=20_000, num_epochs=1, val_period=100, time_series_metrics=["*_loss", "*_accuracy"]
    )

    assert callback.time_series_metrics == ["*_loss", "*_accuracy"]


def test_the_report_interval_reaches_the_callback(callback: _RecordingCallback) -> None:
    NemoRLLogger.for_schedule(max_steps=20_000, num_epochs=1, val_period=100, min_report_interval_seconds=30)

    assert callback.min_report_interval_seconds == 30


def test_an_absent_report_interval_takes_the_shared_default(callback: _RecordingCallback) -> None:
    """The DPO block carries it as an undeclared extra, so it can be absent.

    A config compiled before the knob existed omits it, and a run must start and
    report rather than fail on a missing reporting field.
    """
    NemoRLLogger.for_schedule(max_steps=20_000, num_epochs=1, val_period=100)

    assert callback.min_report_interval_seconds == nemo_rl_logger.DEFAULT_MIN_REPORT_INTERVAL_SECONDS


def test_an_absent_list_takes_the_callers_default(callback: _RecordingCallback) -> None:
    """Absent means the calling driver's algorithm set, not everything.

    A config compiled before this knob existed omits it, and a user who wants
    every metric asks with ``["*"]`` -- the two are different requests.
    """
    NemoRLLogger.for_schedule(
        max_steps=20_000,
        num_epochs=1,
        val_period=100,
        default_time_series_metrics=nemo_rl_logger.GRPO_DEFAULT_TIME_SERIES_METRICS,
    )

    assert callback.time_series_metrics == nemo_rl_logger.GRPO_DEFAULT_TIME_SERIES_METRICS


def test_a_caller_that_states_no_default_gets_the_diagnostics_only(callback: _RecordingCallback) -> None:
    """A driver that forgets its algorithm list loses history, not reporting.

    The fallback is deliberately the algorithm-agnostic diagnostic set rather than
    any one algorithm's: a new driver wired up without a default keeps the loss/lr/
    grad_norm curves and drops its own metrics' history, which shows up in the log,
    instead of silently inheriting patterns belonging to a different algorithm.
    """
    NemoRLLogger.for_schedule(max_steps=20_000, num_epochs=1, val_period=100)

    assert callback.time_series_metrics == DIAGNOSTIC_TIME_SERIES


def test_an_empty_list_is_honoured_rather_than_defaulted(callback: _RecordingCallback) -> None:
    """The one value a truthiness check would flip into its opposite."""
    NemoRLLogger.for_schedule(max_steps=20_000, num_epochs=1, val_period=100, time_series_metrics=[])

    assert callback.time_series_metrics == []


def test_the_default_set_covers_dpos_diagnostics_and_drops_its_counters() -> None:
    """Pinned against DPO's real metric names by running the real callback.

    Train comes from `dpo.py` plus the dtensor worker's `loss_metrics`;
    validation from `DPOValMetrics`. An earlier version reimplemented
    `fnmatchcase` over the same hardcoded list and asserted against the module
    constant, executing no production code at all.
    """
    train = {
        "loss": 0.69,
        "grad_norm": 1.2,
        "lr": 5e-7,
        "sft_loss": 0.0,
        "preference_loss": 0.69,
        "accuracy": 0.5,
        "rewards_chosen_mean": 0.01,
        "rewards_rejected_mean": -0.01,
        "num_valid_samples": 64.0,
        "global_valid_seqs": 64.0,
        "global_valid_toks": 16384.0,
    }
    val = {k: v for k, v in train.items() if k not in ("grad_norm", "lr")}

    callback, reporter = _recording_callback(nemo_rl_logger.DPO_DEFAULT_TIME_SERIES_METRICS)
    callback.report_train_step(step=1, epoch=1, metrics=train)
    callback.report_validation(step=1, epoch=1, metrics=val)

    recorded = {name for name, points in reporter.reports[-1]["metrics"].items() if points}
    dropped = {f"train_{n}" for n in train} | {f"val_{n}" for n in val}
    dropped -= recorded

    assert len(recorded) == 14, "twenty series become fourteen"
    assert dropped == {
        "train_global_valid_seqs",
        "train_global_valid_toks",
        "train_num_valid_samples",
        "val_global_valid_seqs",
        "val_global_valid_toks",
        "val_num_valid_samples",
    }
    assert reporter.reports[0]["train_global_valid_toks"] == 16384.0, "still a latest value"


def test_the_default_set_covers_grpos_reward_and_off_policy_diagnostics() -> None:
    """Pinned against GRPO's real metric names, the same way as DPO's above.

    Names taken from `grpo.py`'s train dict at the `prefix="train"` log: the loss
    function's `all_mb_metrics`, the NeMo-Gym rollout aggregate (whose families
    carry `/mean` and friends), and `validate()`'s `accuracy`/`avg_length`.

    Before the GRPO entries existed this recorded four series out of eighty-odd
    metrics and none of them was a reward, which is the one curve a policy-gradient
    run is read on. The `/max /min /median` siblings staying out is the other half
    of the intent: a family costs one series, not six. `/stddev` is the exception,
    and only for reward, where the spread is the band Studio draws around the mean.
    """
    train = {
        "loss": -0.041,
        "grad_norm": 0.87,
        "lr": 8.4e-07,
        "reward": 0.617,
        "kl_penalty": 0.0031,
        "approx_entropy": 0.49,
        "token_mult_prob_error": 1.011,
        "gen_kl_error": 0.0041,
        "policy_kl_error": 0.00071,
        "js_divergence_error": 6.47e-05,
        "sampling_importance_ratio": 1.0000072,
        "advantages/mean": 0.003,
        "advantages/max": 2.49,
        "baseline_reward/pct_0": 46.09,
        "baseline_reward/pct_1": 0.78,
        "baseline_reward/pct_mixed": 53.13,
        "total_reward/mean": 0.617,
        "total_reward/stddev": 0.486,
        "total_reward/p25": 0.0,
        "total_reward/p75": 1.0,
        "total_reward/median": 0.0,
        "gen_tokens_per_sample/mean": 689.3,
        "gen_tokens_per_sample/max": 2048.0,
        "turns_per_sample/mean": 3.41,
        "turns_per_sample/p95": 7.0,
        "truncation_rate": 0.039,
        "natural_termination_rate": 0.961,
        "mean_prompt_length": 753.8,
        "global_valid_toks": 176432.0,
    }
    val = {
        "accuracy": 0.703,
        "avg_length": 604.2,
        "total_reward/mean": 0.703,
        "total_reward/stddev": 0.457,
        "total_reward/p25": 0.0,
        "total_reward/p75": 1.0,
        "truncation_rate": 0.023,
    }

    callback, reporter = _recording_callback(nemo_rl_logger.GRPO_DEFAULT_TIME_SERIES_METRICS)
    callback.report_train_step(step=1, epoch=1, metrics=train)
    callback.report_validation(step=1, epoch=1, metrics=val)

    recorded = {name for name, points in reporter.reports[-1]["metrics"].items() if points}

    assert recorded == {
        # Shared diagnostics.
        "train_loss",
        "train_grad_norm",
        "train_lr",
        # Reward, on both phases -- the point of the change.
        "train_reward",
        "val_accuracy",
        # The raw verifier reward. Equal to `train_reward` here, and to `train_filtered_reward`
        # when dynamic sampling drops zero-variance groups; the two separate as soon as reward
        # shaping or dynamic sampling is enabled, which is when the pair earns its series.
        # `val_total_reward/mean` stays out: it is the last batch's mean, not the whole pass,
        # so `val_accuracy` is the honest validation number and this would shadow it.
        "train_total_reward/mean",
        # Reward dispersion, which is the band around the mean. Its `/median` and
        # `/max` siblings above stay out.
        "train_total_reward/stddev",
        "val_total_reward/stddev",
        "train_total_reward/p25",
        "train_total_reward/p75",
        "val_total_reward/p25",
        "val_total_reward/p75",
        # Group diversity: how many prompt groups had anything to learn from.
        "train_baseline_reward/pct_0",
        "train_baseline_reward/pct_1",
        "train_baseline_reward/pct_mixed",
        # Off-policy drift and advantage centering.
        "train_kl_penalty",
        "train_gen_kl_error",
        "train_policy_kl_error",
        "train_js_divergence_error",
        "train_sampling_importance_ratio",
        "train_token_mult_prob_error",
        "train_advantages/mean",
        "train_approx_entropy",
        # Length and termination behaviour.
        "train_gen_tokens_per_sample/mean",
        "train_truncation_rate",
        "train_turns_per_sample/mean",
        "val_avg_length",
        "val_truncation_rate",
    }
    # Excluded from the series, still reported as a current value -- dropping a
    # metric's history never drops the metric.
    assert reporter.reports[-1]["train_total_reward/mean"] == 0.617
    assert reporter.reports[-1]["train_total_reward/median"] == 0.0
    assert reporter.reports[-1]["train_global_valid_toks"] == 176432.0
    # The redundancy the exclusion rests on: same number, two names.
    assert reporter.reports[-1]["train_total_reward/mean"] == reporter.reports[-1]["train_reward"]


def test_the_run_length_reaches_the_callback_that_gates_on_it(callback: _RecordingCallback) -> None:
    """The cadence is set from `report_training_start`, so it has to state the run.

    The logger no longer throttles, which makes this its whole remaining
    contribution to the cadence: forward `max_steps` before the first metric
    report. NeMo-RL calls `log_hyperparams` on the composite exactly once, before
    the loop.
    """
    logger = NemoRLLogger.for_schedule(max_steps=20_000, num_epochs=2, val_period=100)
    logger.log_hyperparams({})

    assert callback.training_starts == [{"max_steps": 20_000, "num_epochs": 2, "run_facts": {}}]


# --------------------------------------------------------------------------- #
# Step and epoch arithmetic
# --------------------------------------------------------------------------- #


def test_step_is_reported_as_the_caller_numbered_it(callback: _RecordingCallback) -> None:
    """The caller's step is already 1-indexed; re-incrementing shifted the curve."""
    logger = _make_logger(max_steps=23)
    for step in _driver_steps(23):
        logger.log_metrics(TRAIN_METRICS, step=step, prefix="train")

    reported = [r["step"] for r in callback.train_steps]
    assert reported[0] == 1, "an N-step run starts at 1"
    assert reported[-1] == 23, "...and ends at N, not N+1"


@pytest.mark.parametrize(
    "step,expected_epoch",
    [
        (0, 1),  # validate-at-start, before any training
        (1, 1),
        (10, 1),  # last step of epoch 1 at steps_per_epoch=10
        (11, 2),  # first of epoch 2
        (20, 2),
        (21, 3),
    ],
)
def test_epoch_boundaries(callback: _RecordingCallback, step: int, expected_epoch: int) -> None:
    """Epoch flips on the step after a full epoch, not the last step of one."""
    _make_logger().log_metrics({"loss": 0.5}, step=step, prefix="train")

    assert callback.train_steps[0]["epoch"] == expected_epoch


def test_validate_at_start_reports_step_zero(callback: _RecordingCallback) -> None:
    """Both algorithms run an optional validation pass at step 0 before training."""
    _make_logger().log_metrics({"loss": 0.5}, step=0, prefix="validation")

    reported = callback.validations[0]
    assert reported["step"] == 0
    assert reported["epoch"] == 1


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def test_validation_forwards_the_dict_whole(callback: _RecordingCallback) -> None:
    """`loss` is forwarded under its own name; the phase prefix makes it val_loss."""
    _make_logger().log_metrics(VALIDATION_METRICS, step=10, prefix="validation")

    reported = callback.validations[0]["metrics"]
    assert reported["loss"] == 0.25
    assert reported["num_valid_samples"] == 8


def test_validation_with_nothing_usable_is_not_reported(callback: _RecordingCallback) -> None:
    """An empty or all-non-scalar dict must not produce a hollow report."""
    _make_logger().log_metrics({}, step=9, prefix="validation")
    _make_logger().log_metrics({"histogram/x": _Histogram()}, step=9, prefix="validation")

    assert callback.validations == []


def test_a_validation_pass_without_a_loss_is_still_reported(callback: _RecordingCallback) -> None:
    """No metric is privileged, so a loss is not what makes a pass worth recording.

    GRPO scores validation on rewards: its validate() returns `accuracy` and
    `avg_length` and no loss at all. Gating on one dropped every validation pass
    it ran -- not the loss curve, the whole pass.
    """
    _make_logger().log_metrics(REWARD_VALIDATION_METRICS, step=10, prefix="validation")

    assert callback.validations[0]["metrics"] == REWARD_VALIDATION_METRICS


def test_a_loss_free_validation_is_still_a_reported_pass(callback: _RecordingCallback) -> None:
    """A pass that scores on rewards alone reports, like any other."""
    logger = _make_logger()
    logger.log_metrics({"loss": 0.4}, step=10, prefix="validation")
    logger.log_metrics(REWARD_VALIDATION_METRICS, step=20, prefix="validation")

    assert len(callback.validations) == 2


# --------------------------------------------------------------------------- #
# Step timing — arrives under its own prefix
# --------------------------------------------------------------------------- #


def test_step_timings_are_reported_under_the_timing_prefix(callback: _RecordingCallback) -> None:
    """NeMo-RL logs these separately from the train metrics, at the same step.

    Dropped whole before this, which is why a job could report eighty metrics and
    not how long a step took.
    """
    _make_logger().log_metrics(
        {"total_step_time": 41.2, "generation": 28.0, "policy_training": 9.1},
        step=3,
        prefix="timing/train",
    )

    reported = callback.train_steps[0]["metrics"]
    assert reported["timing/total_step_time"] == 41.2
    assert reported["timing/generation"] == 28.0
    assert callback.train_steps[0]["step"] == 3


def test_step_timings_do_not_displace_the_train_metrics_at_that_step(
    callback: _RecordingCallback,
) -> None:
    """Both arrive at one step; the store merges by name, so both survive."""
    logger = _make_logger()
    logger.log_metrics(TRAIN_METRICS, step=3, prefix="train")
    logger.log_metrics({"total_step_time": 41.2}, step=3, prefix="timing/train")

    assert [r["step"] for r in callback.train_steps] == [3, 3]
    assert "loss" in callback.train_steps[0]["metrics"]
    assert "timing/total_step_time" in callback.train_steps[1]["metrics"]


def test_timing_names_keep_their_prefix_so_durations_are_not_mistaken_for_counts(
    callback: _RecordingCallback,
) -> None:
    """Bare, `generation` and `policy_training` read like counts rather than seconds."""
    _make_logger().log_metrics({"generation": 28.0}, step=3, prefix="timing/train")

    reported = callback.train_steps[0]["metrics"]
    assert "generation" not in reported
    assert reported["timing/generation"] == 28.0


def test_other_timing_prefixes_are_still_ignored(callback: _RecordingCallback) -> None:
    """Only the per-step train timer is charted; setup and validation timings are not."""
    logger = _make_logger()
    logger.log_metrics({"total_validation_time": 12.0}, step=3, prefix="timing/validation")
    logger.log_metrics({"policy_init_time_s": 90.0}, step=0, prefix="timing/setup")

    assert callback.train_steps == []
    assert callback.validations == []


# --------------------------------------------------------------------------- #
# Run facts — constants the driver states once, at training start
# --------------------------------------------------------------------------- #


def test_run_facts_ride_along_with_the_training_start_report(callback: _RecordingCallback) -> None:
    """Which algorithm ran, and how many rollouts a step makes, are constants.

    They belong on the one report that fires before the first step rather than on
    every metric report after it.
    """
    facts = {"training_type": "grpo", "rollouts_per_step": 128}
    logger = NemoRLLogger.for_schedule(max_steps=30, num_epochs=1, run_facts=facts)
    logger.log_hyperparams({})

    assert callback.training_starts[0]["run_facts"] == facts


def test_run_facts_are_optional(callback: _RecordingCallback) -> None:
    """A driver that states none reports the schedule alone, as before."""
    NemoRLLogger.for_schedule(max_steps=30, num_epochs=1).log_hyperparams({})

    assert callback.training_starts[0]["run_facts"] == {}


def test_run_facts_reach_the_report_intact(callback: _RecordingCallback) -> None:
    """Whatever the driver states is forwarded as-is; nothing filters it here."""
    NemoRLLogger.for_schedule(
        max_steps=30, num_epochs=1, run_facts={"training_type": "grpo", "rollouts_per_step": 128}
    ).log_hyperparams({})

    assert callback.training_starts[0]["run_facts"] == {"training_type": "grpo", "rollouts_per_step": 128}


def test_one_dataset_keeps_the_bare_metric_names(callback: _RecordingCallback) -> None:
    """NeMo-RL names the dataloader even when there is only one, so val_loss must survive."""
    logger = _make_logger()
    logger.log_metrics({"loss": 0.4, "accuracy": 0.9}, step=10, prefix="validation-train_ds")

    assert callback.validations[0]["metrics"] == {"loss": 0.4, "accuracy": 0.9}


def test_a_second_dataset_does_not_share_the_first_ones_series(callback: _RecordingCallback) -> None:
    """validate() logs once per dataloader, all at one step, all under `validation-*`.

    Passed through as-is, two datasets' `loss` land as two points at the same
    step in one val_loss series -- the collision the <phase>_ prefix rule exists
    to prevent, a level further down.
    """
    logger = _make_logger()
    logger.log_metrics({"loss": 0.4}, step=10, prefix="validation-train_ds")
    logger.log_metrics({"loss": 0.9}, step=10, prefix="validation-heldout")

    assert callback.validations[0]["metrics"] == {"loss": 0.4}
    assert callback.validations[1]["metrics"] == {"heldout_loss": 0.9}


def test_a_dataset_keeps_the_namespace_it_started_in(callback: _RecordingCallback) -> None:
    """Otherwise a curve would change names partway through the run."""
    logger = _make_logger()
    for step in (10, 20):
        logger.log_metrics({"loss": 0.4}, step=step, prefix="validation-train_ds")
        logger.log_metrics({"loss": 0.9}, step=step, prefix="validation-heldout")

    assert [set(v["metrics"]) for v in callback.validations] == [
        {"loss"},
        {"heldout_loss"},
        {"loss"},
        {"heldout_loss"},
    ]


def test_every_validation_pass_reaches_the_callback(callback: _RecordingCallback) -> None:
    """The logger forwards every pass; thinning them is the callback's decision.

    This used to be counted here, against a per-run budget, and the counter was
    wrong in a way the move fixes: it counted *reports*, and `validate()` logs
    once per dataloader at a single step, so N dataloaders advanced it N times
    per pass and could split one pass across the admit/hold boundary. The gate
    keys on the distinct step instead.
    """
    logger = NemoRLLogger.for_schedule(max_steps=1_000, num_epochs=1, val_period=1)
    for step in _driver_steps(1_000):
        logger.log_metrics({"loss": 0.5}, step=step, prefix="validation")

    assert [v["step"] for v in callback.validations] == list(_driver_steps(1_000))


@pytest.mark.parametrize(
    "prefix",
    [
        "validation",  # sft, grpo, ppo, distillation: one val set, bare prefix
        "validation-default",  # dpo/rm, single val set -- the dict is keyed "default"
        "validation-HelpSteer3",  # dpo/rm, keyed by dataset name
        "validation/nemo_gym",  # no caller emits this; lstrip("-/") covers it anyway
    ],
)
def test_all_validation_prefixes_are_handled(callback: _RecordingCallback, prefix: str) -> None:
    """DPO and RM suffix the prefix per dataloader; every form must route to validation."""
    _make_logger().log_metrics(VALIDATION_METRICS, step=9, prefix=prefix)

    assert len(callback.validations) == 1


# --------------------------------------------------------------------------- #
# Prefixes we intentionally ignore
# --------------------------------------------------------------------------- #


# `timing/train` used to be here and is now routed; see the step-timing section.
@pytest.mark.parametrize("prefix", ["timing/validation", "timing/setup", "performance", "refit", ""])
def test_unhandled_prefixes_produce_no_reports(callback: _RecordingCallback, prefix: str) -> None:
    _make_logger().log_metrics({"loss": 0.1, "total_step_time": 12.0}, step=0, prefix=prefix)

    assert callback.train_steps == []
    assert callback.validations == []
