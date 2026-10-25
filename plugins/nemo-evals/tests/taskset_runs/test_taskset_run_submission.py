# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failure tolerances, operator budgets and pure deadline arithmetic."""

from datetime import datetime, timezone, tzinfo
from zoneinfo import ZoneInfo

import pytest
from nemo_evals.config import EvaluatorConfig, TasksetRunsConfig
from nemo_evals.taskset_runs.submission import TimeoutAboveMaximumError, check_timeout, resolve_deadlines
from pydantic import ValidationError


class _TimezoneWithoutOffset(tzinfo):
    """Represent a tzinfo object that leaves its datetime naive."""

    def utcoffset(self, dt: datetime | None) -> None:
        """Return no offset for the supplied datetime.

        Args:
            dt: Datetime whose awareness is being checked, or None.

        Returns:
            None, making the datetime naive despite its non-null tzinfo.
        """
        return None

    def tzname(self, dt: datetime | None) -> None:
        """Return no zone name; ``tzinfo`` leaves this abstract."""
        return None

    def dst(self, dt: datetime | None) -> None:
        """Return no DST offset; ``tzinfo`` leaves this abstract."""
        return None


def test_operator_budget_defaults():
    runs = TasksetRunsConfig()
    assert check_timeout(None, runs) == 86_400
    assert check_timeout(259_200, runs) == 259_200
    assert runs.queue_timeout_seconds == 172_800
    assert runs.finalization_grace_seconds == 600
    accepted = datetime(2026, 10, 12, 12, tzinfo=timezone.utc)
    assert resolve_deadlines(accepted, 3600, runs.finalization_grace_seconds) == (
        datetime(2026, 10, 12, 13, tzinfo=timezone.utc),
        datetime(2026, 10, 12, 13, 10, tzinfo=timezone.utc),
    )


@pytest.mark.parametrize(
    "field", ["default_timeout_seconds", "max_timeout_seconds", "queue_timeout_seconds", "finalization_grace_seconds"]
)
@pytest.mark.parametrize("value", [0, -1])
def test_operator_budget_requires_positive_values(field, value):
    with pytest.raises(ValidationError):
        TasksetRunsConfig(**{field: value})


def test_operator_default_must_fit_within_maximum():
    with pytest.raises(ValidationError) as exc:
        TasksetRunsConfig(default_timeout_seconds=101, max_timeout_seconds=100)
    assert (
        exc.value.errors()[0]["msg"] == "Value error, runs.default_timeout_seconds must be ≤ runs.max_timeout_seconds"
    )
    assert check_timeout(None, TasksetRunsConfig(default_timeout_seconds=100, max_timeout_seconds=100)) == 100


def test_operator_budget_rejects_unknown_keys():
    with pytest.raises(ValidationError, match="typo"):
        TasksetRunsConfig.model_validate({"typo": 1})


def test_evaluator_settings_have_independent_run_defaults(monkeypatch):
    monkeypatch.delenv("NEMO_EVALS_RUNS", raising=False)
    first, second = EvaluatorConfig(), EvaluatorConfig()
    first.runs.default_timeout_seconds = 7
    assert check_timeout(None, second.runs) == 86_400


def test_evaluator_settings_read_json_environment_override(monkeypatch):
    monkeypatch.setenv("NEMO_EVALS_RUNS", '{"default_timeout_seconds":7}')
    assert check_timeout(None, EvaluatorConfig().runs) == 7


def test_evaluator_settings_reject_unknown_nested_environment_keys(monkeypatch):
    monkeypatch.setenv("NEMO_EVALS_RUNS", '{"default_timeout_seconds":7,"typo":1}')
    with pytest.raises(ValidationError, match="typo"):
        EvaluatorConfig()


@pytest.mark.parametrize("requested,expected", [(None, 7), (1, 1), (100, 100)])
def test_timeout_selects_caller_budget_or_default(requested, expected):
    assert check_timeout(requested, TasksetRunsConfig(default_timeout_seconds=7, max_timeout_seconds=100)) == expected


def test_timeout_above_operator_maximum_has_typed_error():
    runs = TasksetRunsConfig(default_timeout_seconds=7, max_timeout_seconds=100)
    with pytest.raises(TimeoutAboveMaximumError, match="^timeout_seconds 101 exceeds the operator maximum 100$"):
        check_timeout(101, runs)


@pytest.mark.parametrize(
    "accepted",
    [datetime(2026, 10, 12, 12), datetime(2026, 10, 12, 12, tzinfo=_TimezoneWithoutOffset())],
)
def test_deadlines_reject_naive_acceptance_timestamp(accepted: datetime) -> None:
    with pytest.raises(ValueError, match="^start must be timezone-aware$"):
        resolve_deadlines(accepted, 3600, 600)


@pytest.mark.parametrize(
    "accepted,timeout,grace,execution,finalization",
    [
        ("2026-10-12T12:00:00+00:00", 3600, 600, "2026-10-12T13:00:00+00:00", "2026-10-12T13:10:00+00:00"),
        ("2026-10-12T23:59:00+00:00", 120, 30, "2026-10-13T00:01:00+00:00", "2026-10-13T00:01:30+00:00"),
        ("2026-10-12T23:59:00+02:00", 120, 30, "2026-10-13T00:01:00+02:00", "2026-10-13T00:01:30+02:00"),
    ],
)
def test_deadlines_preserve_timezone_and_roll_over_midnight(accepted, timeout, grace, execution, finalization):
    actual = resolve_deadlines(datetime.fromisoformat(accepted), timeout, grace)
    assert tuple(value.isoformat() for value in actual) == (execution, finalization)
    assert all(value.tzinfo is not None for value in actual)


@pytest.mark.parametrize(
    "accepted,timeout,grace",
    [
        (datetime(2026, 11, 1, 1, 30, tzinfo=ZoneInfo("America/New_York"), fold=0), 3600, 600),
        (datetime(2026, 3, 8, 1, 30, tzinfo=ZoneInfo("America/New_York")), 3600, 600),
        (datetime(2026, 11, 1, 1, 0, tzinfo=ZoneInfo("America/New_York"), fold=0), 1800, 3600),
    ],
)
def test_deadlines_measure_elapsed_seconds_across_dst(accepted, timeout, grace):
    execution, finalization = resolve_deadlines(accepted, timeout, grace)
    assert execution.timestamp() - accepted.timestamp() == timeout
    assert finalization.timestamp() - execution.timestamp() == grace
    assert execution.tzinfo is accepted.tzinfo
    assert finalization.tzinfo is accepted.tzinfo


def test_controller_caps_have_defaults_and_refuse_a_creator_cap_above_the_cluster_cap():
    controller = TasksetRunsConfig().controller
    assert (controller.max_cluster_concurrent_jobs, controller.default_max_concurrent_task_jobs) == (50, 20)
    assert controller.max_concurrent_task_jobs_per_creator is None and controller.create_grace_seconds == 600
    assert TasksetRunsConfig.model_validate(
        {"controller": {"max_cluster_concurrent_jobs": 25, "max_concurrent_task_jobs_per_creator": 25}}
    )
    with pytest.raises(ValidationError, match="max_concurrent_task_jobs_per_creator must be ≤"):
        TasksetRunsConfig.model_validate(
            {"controller": {"max_cluster_concurrent_jobs": 25, "max_concurrent_task_jobs_per_creator": 26}}
        )
    with pytest.raises(ValidationError, match="typo"):
        TasksetRunsConfig.model_validate({"controller": {"typo": 1}})
    # Shorter than a create may take, a 404 could be read while the timed-out create is still saving.
    with pytest.raises(ValidationError, match="create_grace_seconds"):
        TasksetRunsConfig.model_validate({"controller": {"create_grace_seconds": 30}})
