# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock, patch

from nhx.core.jobs.controllers.diagnostics import _MAX_ERROR_STACK_CHARS, collect_job_diagnostics

from services.core.jobs.tests.controllers.client_mocks import data_response


def _make_jobs_client_with_logs() -> Mock:
    """Build a mock typed ``JobsClient`` with response-shaped method results."""
    jobs = Mock()
    jobs.get_job.return_value = data_response(
        SimpleNamespace(
            name="job-1",
            status="error",
            status_details={},
            error_details={},
        )
    )
    jobs.get_job_status.return_value = data_response(
        SimpleNamespace(
            status="error",
            status_details={},
            error_details={},
            steps=[],
        )
    )
    jobs.get_job_step.return_value = data_response(
        SimpleNamespace(
            name="step-1",
            status="error",
            status_details={},
            error_details={},
        )
    )
    jobs.list_job_step_tasks.return_value = data_response(SimpleNamespace(data=[]))
    jobs.list_job_logs.return_value.page.return_value = SimpleNamespace(
        items=[SimpleNamespace(message="secret-token=abc123"), SimpleNamespace(message="another line")]
    )
    return jobs


def test_collect_job_diagnostics_omits_raw_job_logs_by_default() -> None:
    sdk = Mock()
    jobs = _make_jobs_client_with_logs()

    with (
        patch("nhx.core.jobs.controllers.diagnostics.client_from_platform", return_value=jobs),
        patch("nhx.core.jobs.controllers.diagnostics.config.include_job_logs_in_diagnostics", False),
    ):
        diagnostics = collect_job_diagnostics(
            sdk,
            workspace="default",
            job_name="job-1",
            step_name="step-1",
            context="test",
        )

    assert "job_logs" not in diagnostics
    jobs.list_job_logs.assert_not_called()


def test_collect_job_diagnostics_includes_raw_job_logs_when_enabled() -> None:
    sdk = Mock()
    jobs = _make_jobs_client_with_logs()

    with (
        patch("nhx.core.jobs.controllers.diagnostics.client_from_platform", return_value=jobs),
        patch("nhx.core.jobs.controllers.diagnostics.config.include_job_logs_in_diagnostics", True),
    ):
        diagnostics = collect_job_diagnostics(
            sdk,
            workspace="default",
            job_name="job-1",
            step_name="step-1",
            context="test",
        )

    assert diagnostics["job_logs"] == ["secret-token=abc123", "another line"]
    jobs.list_job_logs.assert_called_once_with(workspace="default", name="job-1")


def test_collect_job_diagnostics_trims_long_error_details_tracebacks() -> None:
    sdk = Mock()
    jobs = Mock()
    long_error = "traceback-" + ("x" * (_MAX_ERROR_STACK_CHARS + 50))
    expected_trimmed = long_error[-_MAX_ERROR_STACK_CHARS:]
    error_details = {"message": "boom", "error": long_error, "other": "keep"}
    jobs.get_job.return_value = data_response(
        SimpleNamespace(
            name="job-1",
            status="error",
            status_details={},
            error_details=error_details,
        )
    )
    jobs.get_job_status.return_value = data_response(
        SimpleNamespace(
            status="error",
            status_details={},
            error_details=error_details,
            steps=[
                SimpleNamespace(
                    name="step-1",
                    status="error",
                    status_details={},
                    error_details=error_details,
                    tasks=[
                        SimpleNamespace(
                            name="task-1",
                            status="error",
                            status_details={},
                            error_details=error_details,
                            error_stack=long_error,
                        )
                    ],
                )
            ],
        )
    )
    jobs.get_job_step.return_value = data_response(
        SimpleNamespace(
            name="step-1",
            status="error",
            status_details={},
            error_details=error_details,
        )
    )
    jobs.list_job_step_tasks.return_value = data_response(
        SimpleNamespace(
            data=[
                SimpleNamespace(
                    name="task-1",
                    status="error",
                    status_details={},
                    error_details=error_details,
                    error_stack=long_error,
                )
            ]
        )
    )

    with patch("nhx.core.jobs.controllers.diagnostics.client_from_platform", return_value=jobs):
        diagnostics = collect_job_diagnostics(
            sdk,
            workspace="default",
            job_name="job-1",
            step_name="step-1",
            context="test",
        )

    assert diagnostics["job"]["error_details"]["error"] == expected_trimmed
    assert diagnostics["status_api"]["error_details"]["error"] == expected_trimmed
    assert diagnostics["status_api"]["steps"][0]["error_details"]["error"] == expected_trimmed
    assert diagnostics["status_api"]["steps"][0]["tasks"][0]["error_details"]["error"] == expected_trimmed
    assert diagnostics["step"]["error_details"]["error"] == expected_trimmed
    assert diagnostics["tasks_api"][0]["error_details"]["error"] == expected_trimmed
