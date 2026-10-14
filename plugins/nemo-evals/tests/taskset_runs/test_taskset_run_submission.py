# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failure tolerances, operator budgets and pure deadline arithmetic."""

from datetime import datetime, timezone, tzinfo

import pytest
from nemo_evals.api.taskset_run_schemas import HarborTaskJobsExecutionInput
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


@pytest.mark.parametrize(
    "threshold,task_count,expected",
    [
        (None, 100, None),
        (10, 100, 10),
        ("10%", 100, 10),
        ("10%", 74, 7),
        ("10%", 5, 0),
        ("100%", 74, 74),
        (0, 74, 0),
        ("0%", 74, 0),
        (100, 5, 100),
        ("10%", 0, 0),
    ],
)
def test_failure_threshold_resolves_to_count(threshold, task_count, expected):
    execution = HarborTaskJobsExecutionInput(sandbox_policy="harbor-standard", max_failed_tasks=threshold)
    assert execution.resolve_max_failed_tasks(task_count) == expected


def test_operator_budget_defaults():
    runs = TasksetRunsConfig()
    assert check_timeout(None, runs) == 86_400
    assert check_timeout(259_200, runs) == 259_200
    accepted = datetime(2026, 10, 12, 12, tzinfo=timezone.utc)
    assert resolve_deadlines(accepted, 3600, runs) == (
        datetime(2026, 10, 12, 13, tzinfo=timezone.utc),
        datetime(2026, 10, 12, 13, 10, tzinfo=timezone.utc),
    )


@pytest.mark.parametrize("field", ["default_timeout_seconds", "max_timeout_seconds", "finalization_grace_seconds"])
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
    with pytest.raises(ValueError, match="^accepted_at must be timezone-aware$"):
        resolve_deadlines(accepted, 3600, TasksetRunsConfig())


@pytest.mark.parametrize(
    "accepted,timeout,grace,execution,finalization",
    [
        ("2026-10-12T12:00:00+00:00", 3600, 600, "2026-10-12T13:00:00+00:00", "2026-10-12T13:10:00+00:00"),
        ("2026-10-12T23:59:00+00:00", 120, 30, "2026-10-13T00:01:00+00:00", "2026-10-13T00:01:30+00:00"),
        ("2026-10-12T23:59:00+02:00", 120, 30, "2026-10-13T00:01:00+02:00", "2026-10-13T00:01:30+02:00"),
    ],
)
def test_deadlines_preserve_timezone_and_roll_over_midnight(accepted, timeout, grace, execution, finalization):
    actual = resolve_deadlines(
        datetime.fromisoformat(accepted), timeout, TasksetRunsConfig(finalization_grace_seconds=grace)
    )
    assert tuple(value.isoformat() for value in actual) == (execution, finalization)
    assert all(value.tzinfo is not None for value in actual)
