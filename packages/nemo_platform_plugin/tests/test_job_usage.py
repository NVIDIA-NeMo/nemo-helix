# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the task-facing job usage reporters."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from nemo_platform_plugin.job_usage import JobTokenUsage, LocalJobUsageReporter, PlatformJobUsageReporter
from pydantic import ValidationError


@pytest.mark.parametrize("value", [-1, True, 1.5, "1"])
def test_job_token_usage_rejects_noncanonical_totals(value: object) -> None:
    with pytest.raises(ValidationError):
        JobTokenUsage(input_tokens=value)  # ty: ignore[invalid-argument-type]


def test_job_token_usage_requires_at_least_one_total() -> None:
    with pytest.raises(ValidationError, match="at least one token total"):
        JobTokenUsage()


def test_local_reporter_retains_zero_and_partial_totals() -> None:
    reporter = LocalJobUsageReporter()

    reporter.report_totals(output_tokens=0)

    assert reporter.latest == JobTokenUsage(output_tokens=0)


def test_local_reporter_replaces_cumulative_totals() -> None:
    reporter = LocalJobUsageReporter()
    reporter.report_totals(input_tokens=2, output_tokens=3)

    reporter.report_totals(input_tokens=5, output_tokens=8)

    assert reporter.latest == JobTokenUsage(input_tokens=5, output_tokens=8)


def test_platform_reporter_patches_canonical_status_details() -> None:
    jobs_client = MagicMock()
    reporter = PlatformJobUsageReporter(job_name="job-a", workspace="ws", jobs_client=jobs_client)

    reporter.report_totals(input_tokens=12, output_tokens=7)

    jobs_client.update_status_details.assert_called_once_with(
        "job-a",
        workspace="ws",
        body={"input_tokens": 12, "output_tokens": 7},
    )


def test_platform_reporter_omits_unknown_dimension() -> None:
    jobs_client = MagicMock()
    reporter = PlatformJobUsageReporter(job_name="job-a", workspace="ws", jobs_client=jobs_client)

    reporter.report_totals(output_tokens=7)

    jobs_client.update_status_details.assert_called_once_with(
        "job-a",
        workspace="ws",
        body={"output_tokens": 7},
    )


def test_platform_reporter_does_not_fail_job_on_transport_error() -> None:
    jobs_client = MagicMock()
    jobs_client.update_status_details.side_effect = RuntimeError("unavailable")
    reporter = PlatformJobUsageReporter(job_name="job-a", workspace="ws", jobs_client=jobs_client)

    with patch("nemo_platform_plugin.job_usage.logger.warning") as warning:
        reporter.report_totals(input_tokens=1)

    warning.assert_called_once()
