# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared TrainingProgressCallback.

Focused on the contract every backend depends on. There is one naming rule --
a metric is stored and reported as ``<phase>_<name>`` -- and no metric is
privileged, so most of this file is about proving the rule holds with no
exceptions hiding in it. The rest covers the transport: the Jobs service merges
``status_details`` key-wise but shallowly, so a report either resends a series in
full or leaves the key out. Scalars ride on every report: they are a handful of
numbers, and restating them is what makes a dropped report cost nothing.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, cast

import pytest
from nhx.customization_common.training.callbacks import TrainingProgressCallback, is_chartable
from nhx.customization_common.training.progress import JobsServiceProgressReporter


class _RecordingReporter:
    """Stands in for JobsServiceProgressReporter, capturing each report payload."""

    def __init__(self, prior: dict[str, list[dict[str, Any]]] | None = None) -> None:
        #: `{}` is what the real `fetch_current_metrics` returns for a task with
        #: nothing stored. Defaulting to `{"train_loss": [], "val_loss": []}` --
        #: which is what `_build_metrics_summary` is supposed to guarantee --
        #: meant the stub supplied the property under test, and removing the
        #: seeding from the callback left every `metrics["train_loss"] == []`
        #: assertion still passing.
        self._prior = {} if prior is None else prior
        self.reports: list[dict[str, Any]] = []
        self.tracking: tuple[int, int] | None = None
        self.closed = False

    def fetch_current_metrics(self) -> dict[str, list[dict[str, Any]]] | None:
        return self._prior

    def configure_progress_tracking(self, max_steps: int, num_epochs: int) -> None:
        self.tracking = (max_steps, num_epochs)

    def report_running(self, phase: str, **details: Any) -> None:
        self.reports.append({"phase": phase, **details})

    def close(self) -> None:
        self.closed = True


class _UnreadableReporter(_RecordingReporter):
    """A reporter whose one read failed, as opposed to finding nothing stored."""

    def fetch_current_metrics(self) -> dict[str, list[dict[str, Any]]] | None:
        return None


@pytest.fixture
def reporter() -> _RecordingReporter:
    return _RecordingReporter()


def _make_callback(reporter: _RecordingReporter, **kwargs: Any) -> TrainingProgressCallback:
    """Build the callback over a duck-typed reporter, narrowing the type once here.

    Rate limiting is off unless a test asks for it. Almost everything in this
    file is about what a report *contains* -- the naming rule, the payload shape,
    what the seed does -- and those tests report several steps in the same
    instant, which the default interval would withhold. The limiter has its own
    section, with a clock it controls.
    """
    kwargs.setdefault("min_report_interval_seconds", 0)
    return TrainingProgressCallback(cast(JobsServiceProgressReporter, reporter), **kwargs)


def _train_steps(reporter: _RecordingReporter) -> list[int]:
    return [r["step"] for r in reporter.reports if r["phase"] == "training" and "step" in r]


def _val_steps(reporter: _RecordingReporter) -> list[int]:
    return [r["step"] for r in reporter.reports if r["phase"] == "validation"]


# --------------------------------------------------------------------------- #
# One naming rule, no exceptions
# --------------------------------------------------------------------------- #


def test_the_phase_supplies_the_prefix(reporter: _RecordingReporter) -> None:
    """Backends pass their framework's own names; the phase qualifies them."""
    callback = _make_callback(reporter)
    callback.report_train_step(step=1, epoch=1, metrics={"loss": 0.5, "lr": 5e-06, "grad_norm": 1.9})
    callback.report_validation(step=1, epoch=1, metrics={"loss": 0.4, "accuracy": 0.9})

    metrics = reporter.reports[-1]["metrics"]
    assert metrics["train_loss"] == [{"step": 1, "epoch": 1, "value": 0.5}]
    assert metrics["train_lr"] == [{"step": 1, "epoch": 1, "value": 5e-06}]
    assert metrics["train_grad_norm"] == [{"step": 1, "epoch": 1, "value": 1.9}]
    assert metrics["val_loss"] == [{"step": 1, "epoch": 1, "value": 0.4}]
    assert metrics["val_accuracy"] == [{"step": 1, "epoch": 1, "value": 0.9}]


def test_train_loss_and_val_loss_fall_out_of_the_rule(reporter: _RecordingReporter) -> None:
    """The two names Studio charts are what `loss` produces, not special cases.

    They used to be carved out of the prefixing scheme because backends passed
    them pre-prefixed. Passing the framework's own `loss` regenerates both names
    from the ordinary rule, which is why nothing downstream had to change.
    """
    callback = _make_callback(reporter)
    callback.report_train_step(step=1, epoch=1, metrics={"loss": 0.5})
    callback.report_validation(step=1, epoch=1, metrics={"loss": 0.4})

    metrics = reporter.reports[-1]["metrics"]
    assert metrics["train_loss"] == [{"step": 1, "epoch": 1, "value": 0.5}]
    assert metrics["val_loss"] == [{"step": 1, "epoch": 1, "value": 0.4}]


def test_the_same_name_in_both_phases_stays_separate(reporter: _RecordingReporter) -> None:
    """DPO reports `accuracy` in both dicts; one series would interleave them."""
    callback = _make_callback(reporter)
    callback.report_train_step(step=1, epoch=1, metrics={"loss": 0.5, "accuracy": 0.18})
    callback.report_validation(step=1, epoch=1, metrics={"accuracy": 0.04})

    metrics = reporter.reports[-1]["metrics"]
    assert metrics["train_accuracy"] == [{"step": 1, "epoch": 1, "value": 0.18}]
    assert metrics["val_accuracy"] == [{"step": 1, "epoch": 1, "value": 0.04}]


def test_a_pre_prefixed_name_is_not_re_interpreted(reporter: _RecordingReporter) -> None:
    """A backend passing `val_loss` on a train step gets `train_val_loss`.

    The rule is mechanical: it never inspects the name for meaning, so a train
    step cannot reach the validation curve however its metric is spelled.
    """
    _make_callback(reporter).report_train_step(step=1, epoch=1, metrics={"loss": 0.5, "val_loss": 99.0})

    metrics = reporter.reports[-1]["metrics"]
    assert metrics["val_loss"] == []
    assert metrics["train_val_loss"] == [{"step": 1, "epoch": 1, "value": 99.0}]


def test_the_current_value_uses_the_series_name(reporter: _RecordingReporter) -> None:
    """One name per metric, whether you read the curve or the latest point."""
    _make_callback(reporter).report_train_step(step=1, epoch=1, metrics={"loss": 0.5, "lr": 5e-06, "reward": 0.62})

    report = reporter.reports[-1]
    assert report["train_loss"] == 0.5
    assert report["train_lr"] == 5e-06
    assert report["train_reward"] == 0.62
    assert "loss" not in report and "lr" not in report and "reward" not in report


def test_reserved_names_cannot_be_reached_by_a_metric(reporter: _RecordingReporter) -> None:
    """Prefixing removes the collision class outright rather than filtering it.

    `phase` is report_running's own parameter and used to raise TypeError into
    the training loop; `step`, `epoch` and `metrics` are this callback's own keys
    and used to need splat ordering to avoid being shadowed.
    """
    callback = _make_callback(reporter)
    collide = {"phase": 1.0, "step": 2.0, "epoch": 3.0, "metrics": 4.0, "loss": 0.5}
    callback.report_train_step(step=10, epoch=1, metrics=collide)
    callback.report_validation(step=10, epoch=1, metrics=collide)

    for report in reporter.reports:
        assert report["step"] == 10, "the real step survives"
        assert report["epoch"] == 1
        assert isinstance(report["metrics"], dict), "the series payload survives"
    assert [r["phase"] for r in reporter.reports] == ["training", "validation"]
    assert reporter.reports[0]["train_phase"] == 1.0, "the metric is kept, under its qualified name"
    assert reporter.reports[0]["train_step"] == 2.0


# --------------------------------------------------------------------------- #
# Nothing is required
# --------------------------------------------------------------------------- #


def test_a_step_without_a_loss_reports_the_rest(reporter: _RecordingReporter) -> None:
    """No metric is privileged, so none of them is mandatory either."""
    _make_callback(reporter).report_train_step(step=1, epoch=1, metrics={"reward": 0.62})

    report = reporter.reports[-1]
    assert report["train_reward"] == 0.62
    assert report["metrics"]["train_loss"] == []


def test_validation_without_a_loss_leaves_the_curve_empty(reporter: _RecordingReporter) -> None:
    """An algorithm may validate on task metrics alone; a null charts as zero."""
    _make_callback(reporter).report_validation(step=1, epoch=1, metrics={"accuracy": 0.75})

    report = reporter.reports[-1]
    assert "val_loss" not in report
    assert report["val_accuracy"] == 0.75
    assert report["metrics"]["val_loss"] == []


def test_an_empty_metric_dict_still_reports_progress(reporter: _RecordingReporter) -> None:
    """step/epoch are progress, not metrics; they land with or without a curve.

    The series are left out rather than resent: nothing was added to them, the
    stored copy already matches, and the merge leaves an unmentioned key
    standing. Resending would pay the full payload for a report that changed
    nothing -- up to 413 KB by this module's own measurements.
    """
    _make_callback(reporter).report_train_step(step=7, epoch=2, metrics={})

    report = reporter.reports[-1]
    assert report["step"] == 7
    assert report["epoch"] == 2
    assert "metrics" not in report


# --------------------------------------------------------------------------- #
# What cannot be charted is dropped from both places
# --------------------------------------------------------------------------- #


def test_unchartable_metrics_are_dropped_from_the_series(reporter: _RecordingReporter) -> None:
    """Histograms and tables ride in the same dict as the scalars upstream."""
    _make_callback(reporter).report_train_step(
        step=1,
        epoch=1,
        metrics={"loss": 0.5, "histogram": object(), "nested": {"a": 1}, "flag": True, "missing": float("nan")},
    )

    metrics = reporter.reports[-1]["metrics"]
    assert set(metrics) == {"train_loss", "val_loss"}


def test_a_dropped_metric_costs_only_itself(reporter: _RecordingReporter) -> None:
    """It must not ride along in the payload either, or the whole report dies.

    A `Histogram` in status_details makes the SDK update fail to serialize, and
    `update_task` swallows that error -- so every metric in the report is lost
    while the job goes on looking healthy. Verified against a live platform
    before this filter existed: the step never landed.
    """
    _make_callback(reporter).report_train_step(step=1, epoch=1, metrics={"hist": object(), "reward": 0.62})

    report = reporter.reports[-1]
    assert "hist" not in report and "train_hist" not in report
    assert report["train_reward"] == 0.62, "a well-behaved metric in the same report still lands"


def test_non_finite_scalars_are_omitted(reporter: _RecordingReporter) -> None:
    """A NaN grad_norm is routine on a skipped step; the SDK sends it as null."""
    _make_callback(reporter).report_train_step(
        step=1, epoch=1, metrics={"loss": 0.5, "lr": float("inf"), "grad_norm": float("nan"), "absent": None}
    )

    report = reporter.reports[-1]
    assert "train_lr" not in report
    assert "train_grad_norm" not in report
    assert "train_absent" not in report
    assert report["train_loss"] == 0.5


def test_counts_stay_integers(reporter: _RecordingReporter) -> None:
    """`num_valid_samples: 8` should not chart as 8.0."""
    _make_callback(reporter).report_train_step(step=1, epoch=1, metrics={"num_valid_samples": 8})

    assert reporter.reports[-1]["train_num_valid_samples"] == 8


def test_numpy_scalars_are_coerced(reporter: _RecordingReporter) -> None:
    """numpy satisfies numbers.Real but is not JSON-serializable."""
    np = pytest.importorskip("numpy")
    _make_callback(reporter).report_train_step(step=1, epoch=1, metrics={"loss": np.float32(0.5), "count": np.int64(3)})

    report = reporter.reports[-1]
    assert type(report["train_loss"]) is float
    assert type(report["train_count"]) is int


class _Histogram:
    """Stand-in for a non-numeric metric value -- NaN-hostile, like the real thing."""

    def __float__(self) -> float:
        raise TypeError("Histogram is not a scalar")


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.5, True),
        (0, True),
        (-1.5, True),
        (float("nan"), False),
        (float("inf"), False),
        (float("-inf"), False),
        (None, False),
        # Non-scalars that genuinely appear in NeMo-RL metric dicts. Each raises
        # TypeError under a bare math.isfinite, which is the regression guarded here.
        (_Histogram(), False),
        ({"inflight": [1, 2]}, False),
        ([1, 2, 3], False),
        ("0.5", False),
        # bool is an int subclass; charting a flag as 0/1 is not wanted.
        (True, False),
        (False, False),
        # float() overflows here; the predicate must classify, never propagate.
        (10**400, False),
    ],
)
def test_is_chartable(value: Any, expected: bool) -> None:
    assert is_chartable(value) is expected


def test_is_chartable_accepts_numpy_scalars() -> None:
    np = pytest.importorskip("numpy")
    assert is_chartable(np.float32(0.5)) is True
    assert is_chartable(np.float64(0.5)) is True
    assert is_chartable(np.int64(3)) is True
    assert is_chartable(np.float32("nan")) is False


# --------------------------------------------------------------------------- #
# Which reports carry the series
# --------------------------------------------------------------------------- #


def test_step_reports_carry_the_series(reporter: _RecordingReporter) -> None:
    """The reports that move a curve send every curve, since the merge is shallow."""
    callback = _make_callback(reporter)
    callback.report_train_step(step=1, epoch=1, metrics={"loss": 0.5})
    callback.report_validation(step=1, epoch=1, metrics={"loss": 0.45})

    assert len(reporter.reports) == 2
    for report in reporter.reports:
        assert "metrics" in report, report["phase"]
        assert set(report["metrics"]) == {"train_loss", "val_loss"}


def test_non_step_reports_omit_the_series(reporter: _RecordingReporter) -> None:
    """The server merges, so a report with nothing to add leaves the stored curve alone.

    Sending it anyway is pure upload on every checkpoint, and at training start
    it is worse than useless: the accumulator is empty there, and a shallow merge
    would replace a resumed job's stored series with two empty lists.
    """
    callback = _make_callback(reporter)
    callback.report_training_start(max_steps=10, num_epochs=1)
    callback.report_checkpoint_saved(step=1, epoch=1, checkpoint_path="/ckpt")
    callback.report_epoch_end(step=1, epoch=1)

    assert len(reporter.reports) == 3
    for report in reporter.reports:
        assert "metrics" not in report, report["phase"]


def test_training_start_does_not_resend_seeded_metrics() -> None:
    """It fires before the first step, so it has nothing to say about the curves."""
    prior = {"train_loss": [{"step": 1, "epoch": 1, "value": 0.9}], "val_loss": []}
    reporter = _RecordingReporter(prior)
    callback = _make_callback(reporter)
    callback.report_training_start(max_steps=10, num_epochs=1)

    assert "metrics" not in reporter.reports[0]

    # ...and the seeded series is still there for the first step that does report.
    callback.report_train_step(step=2, epoch=1, metrics={"loss": 0.8})
    assert reporter.reports[-1]["metrics"]["train_loss"] == [
        {"step": 1, "epoch": 1, "value": 0.9},
        {"step": 2, "epoch": 1, "value": 0.8},
    ]


def test_series_survive_a_resume_beyond_the_loss_curves() -> None:
    """A resumed job must continue every curve, not just train_loss."""
    prior = {
        "train_loss": [{"step": 1, "epoch": 1, "value": 0.9}],
        "train_reward": [{"step": 1, "epoch": 1, "value": 0.2}],
    }
    reporter = _RecordingReporter(prior)
    _make_callback(reporter).report_train_step(step=2, epoch=1, metrics={"loss": 0.8, "reward": 0.3})

    assert reporter.reports[-1]["metrics"]["train_reward"] == [
        {"step": 1, "epoch": 1, "value": 0.2},
        {"step": 2, "epoch": 1, "value": 0.3},
    ]


# --------------------------------------------------------------------------- #
# Seeded points, and the run they came from
# --------------------------------------------------------------------------- #


def test_validation_and_training_at_the_same_step_both_land(reporter: _RecordingReporter) -> None:
    """NeMo-RL validates at step N before logging train N; both belong on the curves."""
    callback = _make_callback(reporter)
    callback.report_validation(step=10, epoch=1, metrics={"loss": 0.9})
    callback.report_train_step(step=10, epoch=1, metrics={"loss": 0.8})

    metrics = reporter.reports[-1]["metrics"]
    assert metrics["val_loss"] == [{"step": 10, "epoch": 1, "value": 0.9}]
    assert metrics["train_loss"] == [{"step": 10, "epoch": 1, "value": 0.8}]


def test_a_restart_appends_to_the_previous_runs_points() -> None:
    """Known issue, pinned so it cannot change without someone deciding to.

    A replaced pod -- a suspended and resumed Kubernetes Job, or a Volcano
    restart where an execution profile raises maxRetry above zero -- seeds itself
    from the old run's points and appends its own. This is the shape it takes
    when the backend does not resume: automodel and unsloth never do, and a
    NeMo-RL run that has not written a checkpoint yet has none to return to. The
    series then carries both runs end to end.

    The other shape is a NeMo-RL run that does resume -- dpo.setup() loads the
    latest checkpoint unconditionally -- and replays the steps it had already
    reported since that checkpoint, so its curve doubles back across that range
    instead of restarting. Not pinned here.

    Accepted rather than solved either way: telling the runs apart needs a run
    identifier on each point, and dropping the superseded ones is right for a
    replay and destructive for a restart. See the seeding note in callbacks.py.
    """
    prior = {"train_loss": [{"step": 50, "epoch": 1, "value": 1.0}]}
    reporter = _RecordingReporter(prior)
    _make_callback(reporter).report_train_step(step=1, epoch=1, metrics={"loss": 2.0})

    assert reporter.reports[-1]["metrics"]["train_loss"] == [
        {"step": 50, "epoch": 1, "value": 1.0},
        {"step": 1, "epoch": 1, "value": 2.0},
    ]


def test_a_malformed_stored_point_cannot_raise_into_the_training_loop() -> None:
    """The blob is read back from the server, so its shape is not guaranteed.

    report_train_step is called straight from NeMo-RL's log_metrics with nothing
    catching underneath, so a stored point that is not a {step, ...} dict has to
    be dropped rather than raise.
    """
    # Cast because the shape is the point: this is what a corrupted blob reads
    # back as, and no annotation describes it honestly.
    prior = cast(
        dict[str, list[dict[str, Any]]],
        {"train_loss": [{"step": 100, "epoch": 1, "value": 1.0}, "junk", {"epoch": 1}, {"step": "eight"}]},
    )
    reporter = _RecordingReporter(prior)
    _make_callback(reporter).report_train_step(step=110, epoch=1, metrics={"loss": 0.7})

    assert reporter.reports[-1]["metrics"]["train_loss"] == [
        {"step": 100, "epoch": 1, "value": 1.0},
        {"step": 110, "epoch": 1, "value": 0.7},
    ]


def test_a_report_that_changes_no_curve_omits_the_series(reporter: _RecordingReporter) -> None:
    """Nothing recorded means nothing to say: the stored copy already matches."""
    callback = _make_callback(reporter)
    callback.report_train_step(step=1, epoch=1, metrics={"loss": 0.5})
    callback.report_train_step(step=2, epoch=1, metrics={"grad_norm": float("nan")})

    assert "metrics" in reporter.reports[0]
    assert "metrics" not in reporter.reports[-1]


# --------------------------------------------------------------------------- #
# Training start states where the run actually starts
# --------------------------------------------------------------------------- #


def test_training_start_states_the_schedule_and_no_position(reporter: _RecordingReporter) -> None:
    """It fires before the first step, so it has no position to report yet."""
    _make_callback(reporter).report_training_start(max_steps=940, num_epochs=2)

    report = reporter.reports[-1]
    assert report["max_steps"] == 940
    assert report["num_epochs"] == 2
    assert "step" not in report


def test_run_facts_ride_along_with_the_schedule(reporter: _RecordingReporter) -> None:
    """Constants about the run share the report that states the schedule."""
    _make_callback(reporter).report_training_start(
        max_steps=940, num_epochs=2, run_facts={"training_type": "grpo", "rollouts_per_step": 128}
    )

    report = reporter.reports[-1]
    assert report["training_type"] == "grpo"
    assert report["rollouts_per_step"] == 128
    assert report["max_steps"] == 940


def test_run_facts_cannot_overwrite_the_schedule(reporter: _RecordingReporter) -> None:
    """The reported schedule has to be the one progress tracking runs on.

    Both leave in the same call -- as `configure_progress_tracking` arguments and
    as two entries in the same blob. A run fact shadowing either lets them
    disagree: progress still tracked against the real max_steps, the blob beside
    it advertising another, and a job rendering as `step 30 / 999` at 100%.
    Writing the schedule last makes that unreachable, whatever a caller passes.
    """
    _make_callback(reporter).report_training_start(
        max_steps=940, num_epochs=2, run_facts={"max_steps": 999, "num_epochs": 7, "training_type": "grpo"}
    )

    report = reporter.reports[-1]
    assert report["max_steps"] == 940, "the argument wins"
    assert report["num_epochs"] == 2, "the argument wins"
    assert report["training_type"] == "grpo", "everything else still rides along"
    assert reporter.tracking == (940, 2), "and it matches what progress tracking was given"


def test_training_start_does_not_reset_the_stored_progress() -> None:
    """report_running derives percentage_done from a stated step, and it merges.

    A literal 0 here wrote 0% over whatever progress the task had stored, and
    Studio rendered `2/2 (0%)` until the first train step landed -- many minutes
    later at log_interval=10. Stating nothing leaves the stored value standing.
    """
    prior = {"train_loss": [{"step": 470, "epoch": 2, "value": 0.5}]}
    reporter = _RecordingReporter(prior)

    _make_callback(reporter).report_training_start(max_steps=940, num_epochs=2)

    assert "step" not in reporter.reports[-1]
    assert "percentage_done" not in reporter.reports[-1]


# --------------------------------------------------------------------------- #
# A seed that could not be read is not a seed that found nothing
# --------------------------------------------------------------------------- #


def test_a_failed_seed_read_withholds_the_series() -> None:
    """The accumulator is known incomplete, and the merge replaces a sent key whole.

    Sending it would overwrite the stored history with the fraction of it this
    process happens to have. Omitting the key leaves the history standing, which
    costs this run's curves and is the recoverable half of the trade.
    """
    reporter = _UnreadableReporter()
    callback = _make_callback(reporter)
    callback.report_train_step(step=1, epoch=1, metrics={"loss": 0.5})
    callback.report_validation(step=1, epoch=1, metrics={"loss": 0.4})

    assert all("metrics" not in report for report in reporter.reports)


def test_a_failed_seed_read_still_reports_progress_and_current_values() -> None:
    """Only the accumulated curves are withheld; the scalars merge harmlessly."""
    reporter = _UnreadableReporter()
    _make_callback(reporter).report_train_step(step=7, epoch=2, metrics={"loss": 0.5, "lr": 5e-06})

    report = reporter.reports[-1]
    assert report["step"] == 7
    assert report["epoch"] == 2
    assert report["train_loss"] == 0.5
    assert report["train_lr"] == 5e-06


def test_an_empty_seed_read_still_reports_the_series(reporter: _RecordingReporter) -> None:
    """The other half of the distinction: nothing stored is safe to write over."""
    _make_callback(reporter).report_train_step(step=1, epoch=1, metrics={"loss": 0.5})

    assert reporter.reports[-1]["metrics"]["train_loss"] == [{"step": 1, "epoch": 1, "value": 0.5}]


def test_series_are_snapshots_not_live_references(reporter: _RecordingReporter) -> None:
    """A shared list would retroactively mutate already-sent payloads."""
    callback = _make_callback(reporter)
    callback.report_train_step(step=1, epoch=1, metrics={"loss": 0.5})
    first_payload = reporter.reports[-1]["metrics"]["train_loss"]
    callback.report_train_step(step=2, epoch=1, metrics={"loss": 0.4})

    assert len(first_payload) == 1


# --------------------------------------------------------------------------- #
# Optional checkpoint_path
# --------------------------------------------------------------------------- #


def test_checkpoint_report_without_a_path_omits_the_key(reporter: _RecordingReporter) -> None:
    """A null would overwrite the last known checkpoint instead of leaving it.

    The server merges key-wise, so omitting the key leaves the stored path
    standing while an explicit null replaces it. automodel and unsloth both pass
    None when their framework hands back no path.
    """
    _make_callback(reporter).report_checkpoint_saved(step=1, epoch=1)

    assert "checkpoint_path" not in reporter.reports[-1]


def test_checkpoint_report_with_a_path_states_it(reporter: _RecordingReporter) -> None:
    _make_callback(reporter).report_checkpoint_saved(step=1, epoch=1, checkpoint_path="/ckpt")

    assert reporter.reports[-1]["checkpoint_path"] == "/ckpt"


# --------------------------------------------------------------------------- #
# Backend stamping
# --------------------------------------------------------------------------- #


def test_the_default_backend_is_stamped_on_every_report(reporter: _RecordingReporter) -> None:
    class _Stamped(TrainingProgressCallback):
        _default_backend: ClassVar[str | None] = "test-backend"

    callback = _Stamped(cast(JobsServiceProgressReporter, reporter))
    callback.report_train_step(step=1, epoch=1, metrics={"loss": 0.5, "reward": 0.62})

    report = reporter.reports[-1]
    assert report["backend"] == "test-backend"
    assert report["train_reward"] == 0.62


def test_no_backend_field_when_the_default_is_unset(reporter: _RecordingReporter) -> None:
    """automodel and NeMo-RL both depend on this: their reports carry no `backend`.

    Neither subclasses this class, so the absence of the key is a property of the
    default here rather than of anything on their side. unsloth opts in.
    """
    callback = _make_callback(reporter)
    callback.report_training_start(max_steps=10, num_epochs=1)
    callback.report_train_step(step=1, epoch=1, metrics={"loss": 0.5})
    callback.report_validation(step=1, epoch=1, metrics={"loss": 0.4})
    callback.report_checkpoint_saved(step=1, epoch=1, checkpoint_path="/ckpt")
    callback.report_epoch_end(step=1, epoch=1)

    assert reporter.reports, "expected reports to assert against"
    assert all("backend" not in report for report in reporter.reports)


def test_a_per_call_backend_overrides_the_default(reporter: _RecordingReporter) -> None:
    """unsloth's HF trainer callback passes it per call."""
    _make_callback(reporter).report_train_step(step=1, epoch=1, metrics={"loss": 0.5}, backend="unsloth")

    assert reporter.reports[-1]["backend"] == "unsloth"


def test_close_delegates_to_the_reporter(reporter: _RecordingReporter) -> None:
    _make_callback(reporter).close()

    assert reporter.closed


# --------------------------------------------------------------------------- #
# The reporting cadence
#
# Moved here from services/rl/tests/test_nemo_rl_logger.py, which is where this
# was implemented for one backend while the other two reported every step. The
# invariants are that backend's; the coverage is now everyone's.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Elapsed steps, not a modulus
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Each path carries its own budget
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# The decimation backstop
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Never lose the tail
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# The gate's state is whatever the seeded series says it is
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# What a withheld report costs
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Which metrics get a stored time series
#
# The other bound on the payload, and on most backends the larger one: what is
# stored is `series x max_points`. Patterns match the qualified name, the one
# that appears in status_details, so what a user writes is what they read back.
# --------------------------------------------------------------------------- #


def test_every_metric_is_recorded_by_default(reporter: _RecordingReporter) -> None:
    """None means everything at this layer; a narrower default is each backend's call."""
    _make_callback(reporter).report_train_step(step=1, epoch=1, metrics={"loss": 0.5, "tps": 4821.0})

    assert set(reporter.reports[-1]["metrics"]) == {"train_loss", "train_tps", "val_loss"}


def test_only_the_named_metrics_get_a_series(reporter: _RecordingReporter) -> None:
    callback = _make_callback(reporter, time_series_metrics=["train_loss", "train_lr"])
    callback.report_train_step(step=1, epoch=1, metrics={"loss": 0.5, "lr": 1e-5, "tps": 4821.0, "mem": 12.5})

    metrics = reporter.reports[-1]["metrics"]
    assert metrics["train_loss"] == [{"step": 1, "epoch": 1, "value": 0.5}]
    assert metrics["train_lr"] == [{"step": 1, "epoch": 1, "value": 1e-5}]
    assert "train_tps" not in metrics
    assert "train_mem" not in metrics


def test_an_excluded_metric_still_reports_its_current_value(reporter: _RecordingReporter) -> None:
    """The distinction that makes this safe where a report-time allow-list was not.

    That one dropped metrics outright, and DPO's `accuracy`, `sft_loss` and
    `rewards_chosen_mean` went missing for a release because nobody had added
    them to it. Leaving a metric out here costs its history, never its
    visibility.
    """
    callback = _make_callback(reporter, time_series_metrics=["*_loss"])
    callback.report_train_step(step=1, epoch=1, metrics={"loss": 0.5, "tps": 4821.0})

    report = reporter.reports[-1]
    assert report["train_tps"] == 4821.0, "still a current value"
    assert "train_tps" not in report["metrics"], "just no history"


def test_a_qualified_name_selects_one_phase(reporter: _RecordingReporter) -> None:
    """What qualification buys that unqualified names could not express at all."""
    callback = _make_callback(reporter, time_series_metrics=["train_loss"])
    callback.report_train_step(step=1, epoch=1, metrics={"loss": 0.5})
    callback.report_validation(step=1, epoch=1, metrics={"loss": 0.4})

    train_report, val_report = reporter.reports
    assert train_report["metrics"]["train_loss"] == [{"step": 1, "epoch": 1, "value": 0.5}]
    assert train_report["metrics"]["val_loss"] == []
    # The validation pass recorded nothing, so it says nothing about the series
    # rather than resending them -- it still reports its current value.
    assert val_report["val_loss"] == 0.4
    assert "metrics" not in val_report


def test_a_wildcard_covers_both_phases_and_the_whole_family(reporter: _RecordingReporter) -> None:
    """`*_loss` is how one entry says what `loss` used to, and more.

    It picks up the algorithm-specific members of the family too, which is what
    keeps a default set short for a backend like DPO.
    """
    callback = _make_callback(reporter, time_series_metrics=["*_loss"])
    callback.report_train_step(step=1, epoch=1, metrics={"loss": 0.5, "sft_loss": 0.1, "accuracy": 0.9})
    callback.report_validation(step=1, epoch=1, metrics={"loss": 0.4, "preference_loss": 0.2})

    metrics = reporter.reports[-1]["metrics"]
    assert {"train_loss", "train_sft_loss", "val_loss", "val_preference_loss"} <= set(metrics)
    assert "train_accuracy" not in metrics


def test_a_wildcard_reaches_a_dataset_qualified_name(reporter: _RecordingReporter) -> None:
    """The caveat that qualification removed rather than documented.

    NeMo-RL folds the dataloader name into the metric names of every validation
    set past the first, so a second set's loss arrives as `val_heldout_loss`. No
    unqualified spelling could reach it; `*_loss` does, without anyone having to
    know the dataset's name.
    """
    callback = _make_callback(reporter, time_series_metrics=["*_loss"])
    callback.report_validation(step=1, epoch=1, metrics={"heldout_loss": 0.4})

    assert reporter.reports[-1]["metrics"]["val_heldout_loss"] == [{"step": 1, "epoch": 1, "value": 0.4}]


def test_a_star_records_everything(reporter: _RecordingReporter) -> None:
    """The explicit spelling of the default, and how a user opts out of a backend's set."""
    callback = _make_callback(reporter, time_series_metrics=["*"])
    callback.report_train_step(step=1, epoch=1, metrics={"loss": 0.5, "tps": 4821.0})

    assert {"train_loss", "train_tps"} <= set(reporter.reports[-1]["metrics"])


def test_the_matcher_is_fnmatchcase_and_not_fnmatch() -> None:
    """An identity assertion, because on this platform nothing else can be.

    `fnmatch` applies `os.path.normcase`, which is the identity on POSIX and
    lowercases on Windows. So the two functions are *behaviourally
    indistinguishable* on Linux: swapping one for the other in the source changes
    no observable behaviour here, and every behavioural test of the swap passes.
    Verified by mutation -- `from fnmatch import fnmatch as fnmatchcase` left the
    whole suite green, including a test asserting case-sensitive matching.

    The bug it guards against is therefore invisible on the machine most likely
    to run this: a pattern that matches on Linux and not on macOS. Since no
    behaviour distinguishes them, the dependency itself is what gets pinned.
    """
    import fnmatch

    from nhx.customization_common.training import callbacks as under_test

    assert under_test.fnmatchcase is fnmatch.fnmatchcase


def test_a_pattern_of_the_wrong_case_matches_nothing(reporter: _RecordingReporter) -> None:
    """The behaviour that follows, which does hold on every platform."""
    callback = _make_callback(reporter, time_series_metrics=["TRAIN_LOSS"])
    callback.report_train_step(step=1, epoch=1, metrics={"loss": 0.5})

    assert "metrics" not in reporter.reports[-1]


def test_a_report_with_no_recorded_metric_omits_the_series(reporter: _RecordingReporter) -> None:
    """Nothing was added, so the stored copy already matches and is left standing.

    Resending it would pay the full payload to say nothing, which is the same
    rule an all-unchartable report already follows.
    """
    callback = _make_callback(reporter, time_series_metrics=["*_loss"])
    callback.report_train_step(step=1, epoch=1, metrics={"tps": 4821.0})

    report = reporter.reports[-1]
    assert report["train_tps"] == 4821.0
    assert "metrics" not in report


def test_a_pattern_that_never_arrives_is_harmless(reporter: _RecordingReporter) -> None:
    """A default set has to survive backends that produce different metrics.

    unsloth never reports `rewards_chosen_mean`; NeMo-RL's default names it. The
    same fragment has to be safe in both.
    """
    callback = _make_callback(reporter, time_series_metrics=["*_loss", "*_rewards_chosen_mean"])
    callback.report_train_step(step=1, epoch=1, metrics={"loss": 0.5})

    assert set(reporter.reports[-1]["metrics"]) == {"train_loss", "val_loss"}


def test_an_empty_list_records_nothing_but_still_reports(reporter: _RecordingReporter) -> None:
    """Distinct from None, and a legitimate way to ask for values without history."""
    callback = _make_callback(reporter, time_series_metrics=[])
    callback.report_train_step(step=1, epoch=1, metrics={"loss": 0.5})

    report = reporter.reports[-1]
    assert report["train_loss"] == 0.5
    assert report["step"] == 1
    assert "metrics" not in report


def test_the_excluded_names_are_logged_once(reporter: _RecordingReporter, caplog: pytest.LogCaptureFixture) -> None:
    """ "Why is there no train_tps history" needs an answer in the job log.

    Once per name rather than once per step: this fires from the report path, and
    a 20,000-step run would otherwise say it 20,000 times.
    """
    callback = _make_callback(reporter, time_series_metrics=["*_loss"])
    with caplog.at_level(logging.INFO):
        for step in range(1, 4):
            callback.report_train_step(step=step, epoch=1, metrics={"loss": 0.5, "tps": 4821.0})

    excluded = [r for r in caplog.records if "no stored history" in r.getMessage()]
    assert len(excluded) == 1
    assert "train_tps" in excluded[0].getMessage()


def test_a_metric_that_appears_later_is_logged_when_it_does(
    reporter: _RecordingReporter, caplog: pytest.LogCaptureFixture
) -> None:
    """Backends omit metrics on some steps, so the full set is not known at the first."""
    callback = _make_callback(reporter, time_series_metrics=["*_loss"])
    with caplog.at_level(logging.INFO):
        callback.report_train_step(step=1, epoch=1, metrics={"loss": 0.5, "tps": 4821.0})
        callback.report_train_step(step=2, epoch=1, metrics={"loss": 0.5, "grad_norm": 1.2})

    logged = " ".join(r.getMessage() for r in caplog.records if "no stored history" in r.getMessage())
    assert "train_tps" in logged
    assert "train_grad_norm" in logged


# --------------------------------------------------------------------------- #
# A pattern that matched nothing is a misconfiguration, not a preference
# --------------------------------------------------------------------------- #


def test_an_unqualified_name_matches_nothing_and_says_so(
    reporter: _RecordingReporter, caplog: pytest.LogCaptureFixture
) -> None:
    """The likeliest mistake once names are qualified, and it is silent without this.

    `loss` selects nothing, because the metric is `train_loss`. The run then
    reports no history and looks exactly like one configured not to.
    """
    callback = _make_callback(reporter, time_series_metrics=["loss"])
    callback.report_train_step(step=1, epoch=1, metrics={"loss": 0.5})
    with caplog.at_level(logging.WARNING):
        callback.close()

    warned = [r for r in caplog.records if "matched no metric" in r.getMessage()]
    assert len(warned) == 1
    assert "loss" in warned[0].getMessage()
    assert "train_loss" in warned[0].getMessage(), "names what did arrive, so the fix is obvious"


def test_patterns_that_all_matched_say_nothing(reporter: _RecordingReporter, caplog: pytest.LogCaptureFixture) -> None:
    callback = _make_callback(reporter, time_series_metrics=["*_loss"])
    callback.report_train_step(step=1, epoch=1, metrics={"loss": 0.5})
    with caplog.at_level(logging.WARNING):
        callback.close()

    assert not [r for r in caplog.records if "matched no metric" in r.getMessage()]


def test_a_pattern_matching_nothing_is_only_judged_at_the_end(
    reporter: _RecordingReporter, caplog: pytest.LogCaptureFixture
) -> None:
    """A name that has not arrived yet is not yet wrong.

    Validation metrics do not exist until the first pass, so checking up front
    would warn about `val_loss` on every run that validates late.
    """
    callback = _make_callback(reporter, time_series_metrics=["*_loss", "val_accuracy"])
    with caplog.at_level(logging.WARNING):
        callback.report_train_step(step=1, epoch=1, metrics={"loss": 0.5})
        assert not [r for r in caplog.records if "matched no metric" in r.getMessage()]
        callback.report_validation(step=1, epoch=1, metrics={"loss": 0.4, "accuracy": 0.9})
        callback.close()

    assert not [r for r in caplog.records if "matched no metric" in r.getMessage()]


def test_a_run_that_reported_no_metrics_is_not_warned_about(
    reporter: _RecordingReporter, caplog: pytest.LogCaptureFixture
) -> None:
    """Every pattern trivially matched nothing; saying so is noise on a real failure."""
    callback = _make_callback(reporter, time_series_metrics=["*_loss"])
    with caplog.at_level(logging.WARNING):
        callback.close()

    assert not [r for r in caplog.records if "matched no metric" in r.getMessage()]


# --------------------------------------------------------------------------- #
# Ship-blockers found by independent review
#
# Every test below fails on the implementation that preceded it. They are
# grouped because they share a cause: the gate was written for one report per
# step per path, and validation produces one report per *dataloader* per step.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# A misconfigured pattern list must not reach the training loop
#
# These values come off a config file, and on the NeMo-RL path straight out of a
# YAML with no schema in between. Matching runs inline in a training step.
# --------------------------------------------------------------------------- #


def test_a_non_string_entry_does_not_raise_into_the_training_loop(
    reporter: _RecordingReporter, caplog: pytest.LogCaptureFixture
) -> None:
    """`fnmatchcase(name, None)` raises TypeError, and nothing catches it.

    `report_train_step` is called straight from a backend's logging hook with no
    try around it, so this used to end the run: "object of type 'NoneType' has
    no len()", out of a reporting knob.
    """
    with caplog.at_level(logging.WARNING):
        callback = _make_callback(reporter, time_series_metrics=["*_loss", None])
    callback.report_train_step(step=1, epoch=1, metrics={"loss": 0.5})

    assert reporter.reports[-1]["metrics"]["train_loss"] == [{"step": 1, "epoch": 1, "value": 0.5}]
    assert "non-string" in caplog.text


def test_a_bare_string_is_one_pattern_not_ten(reporter: _RecordingReporter) -> None:
    """`str` satisfies Collection[str] as a collection of its characters.

    `time_series_metrics: train_loss` in YAML became the patterns 't', 'r', 'a'
    and so on, which match nothing -- a run that silently recorded no history at
    all. There is only one thing it can have meant.
    """
    callback = _make_callback(reporter, time_series_metrics="train_loss")
    callback.report_train_step(step=1, epoch=1, metrics={"loss": 0.5})

    assert reporter.reports[-1]["metrics"]["train_loss"] == [{"step": 1, "epoch": 1, "value": 0.5}]


@pytest.mark.parametrize("value", [[None, 3], 7, object()])
def test_an_entirely_unusable_list_records_everything(reporter: _RecordingReporter, value: object) -> None:
    """A broken config should cost noise, not data.

    Distinct from `[]`, which is a legitimate request for no history and is
    honoured -- see the neighbouring test.
    """
    callback = _make_callback(reporter, time_series_metrics=value)
    callback.report_train_step(step=1, epoch=1, metrics={"loss": 0.5, "tps": 4821.0})

    assert set(reporter.reports[-1]["metrics"]) >= {"train_loss", "train_tps"}


def test_an_explicitly_empty_list_is_still_honoured(reporter: _RecordingReporter) -> None:
    """The one input the "unusable falls back to everything" rule must not catch."""
    callback = _make_callback(reporter, time_series_metrics=[])
    callback.report_train_step(step=1, epoch=1, metrics={"loss": 0.5})

    assert "metrics" not in reporter.reports[-1]


def test_a_typo_is_reported_even_when_every_metric_matched(
    reporter: _RecordingReporter, caplog: pytest.LogCaptureFixture
) -> None:
    """The case the warning was gated out of, and the only one that matters.

    It used to require that some metric had been *excluded* before it would
    speak. A run whose metrics all matched something excluded nothing, so the
    misspelling beside them passed in silence -- which is every ordinary run.
    """
    callback = _make_callback(reporter, time_series_metrics=["*_loss", "val_accuarcy"])
    callback.report_train_step(step=1, epoch=1, metrics={"loss": 0.5})
    with caplog.at_level(logging.WARNING):
        callback.close()

    warned = [r for r in caplog.records if "matched no metric" in r.getMessage()]
    assert len(warned) == 1
    assert "val_accuarcy" in warned[0].getMessage()
    assert "train_loss" in warned[0].getMessage(), "names what did arrive, so the fix is obvious"


# --------------------------------------------------------------------------- #
# Rate limiting
#
# What it bounds is how often the accumulator goes on the wire. What it must
# never touch is what the accumulator holds: a withheld report's points are
# already recorded, and ship with the next request that does go.
# --------------------------------------------------------------------------- #


def _merged(reporter: _RecordingReporter) -> dict[str, Any]:
    """What the server ends up holding, applying its key-wise merge to the reports.

    Asserting on the last report alone would miss the defect these tests cover:
    a value lost from one report is not restored by a later one, so only the
    merged view distinguishes "sent earlier and still standing" from "never sent".
    """
    merged: dict[str, Any] = {}
    for report in reporter.reports:
        merged.update(report)
    return merged


class _FakeClock:
    """A monotonic clock the test advances by hand."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_reports_inside_the_interval_are_withheld(reporter: _RecordingReporter) -> None:
    clock = _FakeClock()
    callback = _make_callback(reporter, min_report_interval_seconds=10, clock=clock)

    for step in range(1, 6):
        callback.report_train_step(step=step, epoch=1, metrics={"loss": 0.5})

    assert _train_steps(reporter) == [1], "the first always goes; the rest are inside the interval"


def test_a_report_goes_once_the_interval_has_passed(reporter: _RecordingReporter) -> None:
    clock = _FakeClock()
    callback = _make_callback(reporter, min_report_interval_seconds=10, clock=clock)

    for step in range(1, 61):
        callback.report_train_step(step=step, epoch=1, metrics={"loss": 0.5})
        clock.advance(1)

    assert _train_steps(reporter) == [1, 11, 21, 31, 41, 51]


def test_a_withheld_report_loses_no_points(reporter: _RecordingReporter) -> None:
    """The whole difference from the point cap this replaced.

    Steps 2 through 5 are never sent as reports of their own, but their points
    are on the curve by the time a report does go.
    """
    clock = _FakeClock()
    callback = _make_callback(reporter, min_report_interval_seconds=10, clock=clock)

    for step in range(1, 6):
        callback.report_train_step(step=step, epoch=1, metrics={"loss": float(step)})
    clock.advance(10)
    callback.report_train_step(step=6, epoch=1, metrics={"loss": 6.0})

    stored = reporter.reports[-1]["metrics"]["train_loss"]
    assert [p["step"] for p in stored] == [1, 2, 3, 4, 5, 6]
    assert [p["value"] for p in stored] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_a_withheld_scalar_survives_a_validation_report(reporter: _RecordingReporter) -> None:
    """The bug the single pending slot had: a disjoint key set took the slot.

    A withheld report used to be held whole, so a second withheld report replaced
    it outright. A validation report carries `val_` names -- a key set disjoint
    from `train_` -- so it carried nothing fresher for the train scalars it
    displaced, and they were never sent at all. The store merges by name now, so
    the two cannot collide.

    `tps` is a scalar here, not a series, which is what makes it vulnerable:
    series points were always safe, because they accumulate by name already.
    """
    clock = _FakeClock()
    callback = _make_callback(reporter, min_report_interval_seconds=10, clock=clock, time_series_metrics=("*_loss",))

    callback.report_train_step(step=1, epoch=1, metrics={"loss": 1.0})  # goes
    callback.report_train_step(step=2, epoch=1, metrics={"loss": 0.9, "tps": 1234.0})  # withheld
    callback.report_validation(step=2, epoch=1, metrics={"loss": 0.8})  # withheld
    callback.close()

    stored = _merged(reporter)
    assert stored["train_tps"] == 1234.0, "the validation report displaced it"
    assert stored["val_loss"] == 0.8


def test_a_withheld_scalar_survives_a_later_report_that_omits_it(reporter: _RecordingReporter) -> None:
    """Same defect without crossing phases: backends omit metrics on some steps.

    Two withheld train reports where the second does not carry `tps`. Under a
    whole-report slot the second replaced the first and the value was lost.
    """
    clock = _FakeClock()
    callback = _make_callback(reporter, min_report_interval_seconds=10, clock=clock, time_series_metrics=("*_loss",))

    callback.report_train_step(step=1, epoch=1, metrics={"loss": 1.0})  # goes
    callback.report_train_step(step=2, epoch=1, metrics={"loss": 0.9, "tps": 1234.0})  # withheld
    callback.report_train_step(step=3, epoch=1, metrics={"loss": 0.8})  # withheld
    callback.close()

    assert _merged(reporter)["train_tps"] == 1234.0


def test_the_end_of_a_run_reports_the_final_train_scalars(reporter: _RecordingReporter) -> None:
    """The ordinary shutdown sequence, which is where this bug actually bit.

    Last train step, then a final validation pass, then close() -- all inside one
    interval. The train scalars sent must be the last ones the run produced, not
    whichever the limiter last happened to admit.
    """
    clock = _FakeClock()
    callback = _make_callback(reporter, min_report_interval_seconds=10, clock=clock, time_series_metrics=("*_loss",))

    callback.report_train_step(step=1, epoch=1, metrics={"loss": 1.0, "tps": 100.0})
    clock.advance(10)
    for step in (2, 3):
        callback.report_train_step(step=step, epoch=1, metrics={"loss": 0.5, "tps": float(step)})
    callback.report_validation(step=3, epoch=1, metrics={"loss": 0.4})
    callback.close()

    assert _merged(reporter)["train_tps"] == 3.0, "stale: an earlier admitted report's value"


def test_close_sends_what_the_limiter_withheld(reporter: _RecordingReporter) -> None:
    """Load-bearing for data, not only for freshness.

    Points recorded since the last send exist nowhere but this process until
    something carries them, so a run that ends mid-interval must not simply stop.
    """
    clock = _FakeClock()
    callback = _make_callback(reporter, min_report_interval_seconds=10, clock=clock)
    for step in range(1, 6):
        callback.report_train_step(step=step, epoch=1, metrics={"loss": float(step)})

    assert _train_steps(reporter) == [1]

    callback.close()

    assert _train_steps(reporter) == [1, 5]
    assert [p["step"] for p in reporter.reports[-1]["metrics"]["train_loss"]] == [1, 2, 3, 4, 5]


def test_close_sends_nothing_when_the_last_report_already_went(reporter: _RecordingReporter) -> None:
    clock = _FakeClock()
    callback = _make_callback(reporter, min_report_interval_seconds=10, clock=clock)
    callback.report_train_step(step=1, epoch=1, metrics={"loss": 0.5})

    callback.close()

    assert _train_steps(reporter) == [1], "no duplicate of a report that already went"


def test_a_send_carries_points_recorded_by_reports_that_did_not(reporter: _RecordingReporter) -> None:
    """The report that passes the limiter may itself add nothing chartable.

    Asking whether *this* report recorded a point would strand everything the
    withheld ones recorded until some later report happened to add one of its
    own. The question is whether anything is unsent.
    """
    clock = _FakeClock()
    callback = _make_callback(reporter, min_report_interval_seconds=10, clock=clock)
    callback.report_train_step(step=1, epoch=1, metrics={"loss": 0.5})
    callback.report_train_step(step=2, epoch=1, metrics={"loss": 0.6})
    clock.advance(10)
    callback.report_train_step(step=3, epoch=1, metrics={"grad_norm": float("nan")})

    assert [p["step"] for p in reporter.reports[-1]["metrics"]["train_loss"]] == [1, 2]


def test_the_two_paths_share_one_interval(reporter: _RecordingReporter) -> None:
    """The cost being limited is a request, and both paths send the same blob.

    A per-path limit would let a validating run send twice as often for the same
    reason a single-path one sends once.
    """
    clock = _FakeClock()
    callback = _make_callback(reporter, min_report_interval_seconds=10, clock=clock)
    callback.report_train_step(step=1, epoch=1, metrics={"loss": 0.5})
    callback.report_validation(step=1, epoch=1, metrics={"loss": 0.4})

    assert len(reporter.reports) == 1


def test_events_are_never_withheld(reporter: _RecordingReporter) -> None:
    """Checkpoints and epoch ends are events, not samples.

    A checkpoint report carries the only record of where the checkpoint landed,
    so dropping one loses it rather than delaying it.
    """
    clock = _FakeClock()
    callback = _make_callback(reporter, min_report_interval_seconds=10, clock=clock)
    callback.report_train_step(step=1, epoch=1, metrics={"loss": 0.5})
    for step in (2, 3, 4):
        callback.report_checkpoint_saved(step=step, epoch=1, checkpoint_path=f"/ckpt/{step}")
        callback.report_epoch_end(step=step, epoch=1)

    assert len([r for r in reporter.reports if r["phase"] == "checkpoint_saved"]) == 3
    assert len([r for r in reporter.reports if r["phase"] == "epoch_end"]) == 3


def test_zero_sends_every_report(reporter: _RecordingReporter) -> None:
    clock = _FakeClock()
    callback = _make_callback(reporter, min_report_interval_seconds=0, clock=clock)

    for step in range(1, 6):
        callback.report_train_step(step=step, epoch=1, metrics={"loss": 0.5})

    assert _train_steps(reporter) == [1, 2, 3, 4, 5]


def test_a_negative_interval_is_clamped_rather_than_raising(reporter: _RecordingReporter) -> None:
    """This arrives from a job config; a reporting knob must not stop a run."""
    callback = _make_callback(reporter, min_report_interval_seconds=-5, clock=_FakeClock())
    callback.report_train_step(step=1, epoch=1, metrics={"loss": 0.5})
    callback.report_train_step(step=2, epoch=1, metrics={"loss": 0.5})

    assert _train_steps(reporter) == [1, 2]
