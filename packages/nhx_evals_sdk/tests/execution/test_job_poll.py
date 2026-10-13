# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for async job polling helpers."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from unittest.mock import AsyncMock

import pytest
from nhx_evals_sdk.execution.job_poll import async_poll_until_terminal
from pytest_mock import MockerFixture


def _status_getter(statuses: Iterator[str]):
    """Return an async status getter backed by the provided iterator."""

    async def get_status() -> str:
        """Return the next status value."""
        return next(statuses)

    return get_status


@pytest.mark.asyncio
class TestAsyncPollUntilTerminal:
    """Coverage for async poll progress logging."""

    async def test_logs_each_non_terminal_and_terminal_status_as_json(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Every poll attempt should emit a JSON info log with the current status."""
        with caplog.at_level(logging.INFO, logger="nhx_evals_sdk.execution.job_poll"):
            result = await async_poll_until_terminal(
                _status_getter(iter(["running", "active", "completed"])),
                status_value=lambda status: status,
                job_name="metric job 'abc'",
                terminal=frozenset({"completed"}),
                timeout=10.0,
                pending_timeout=10.0,
                poll_interval=0.0,
            )

        assert result == "completed"
        payloads = [json.loads(record.getMessage()) for record in caplog.records]
        assert payloads == [
            {"job_name": "metric job 'abc'", "status": "running", "elapsed_s": 0.0, "poll_interval_s": 0.0},
            {"job_name": "metric job 'abc'", "status": "active", "elapsed_s": 0.0, "poll_interval_s": 0.0},
            {"job_name": "metric job 'abc'", "status": "completed", "elapsed_s": 0.0, "poll_interval_s": 0.0},
        ]
        assert all("pending_elapsed_s" not in payload for payload in payloads)

    async def test_pending_progress_logs_include_pending_elapsed(self, caplog: pytest.LogCaptureFixture) -> None:
        """Pending status logs should include pending elapsed progress."""
        with caplog.at_level(logging.INFO, logger="nhx_evals_sdk.execution.job_poll"):
            await async_poll_until_terminal(
                _status_getter(iter(["pending", "completed"])),
                status_value=lambda status: status,
                job_name="metric job 'pending-job'",
                terminal=frozenset({"completed"}),
                timeout=10.0,
                pending_timeout=10.0,
                poll_interval=0.0,
            )

        payloads = [json.loads(record.getMessage()) for record in caplog.records]
        assert payloads[0]["job_name"] == "metric job 'pending-job'"
        assert payloads[0]["status"] == "pending"
        assert payloads[0]["pending_elapsed_s"] == 0.0
        assert "pending_elapsed_s" not in payloads[1]

    async def test_logs_status_details_as_nested_json(self, caplog: pytest.LogCaptureFixture) -> None:
        """Poll details should be emitted as a nested status_details object."""
        with caplog.at_level(logging.INFO, logger="nhx_evals_sdk.execution.job_poll"):
            await async_poll_until_terminal(
                _status_getter(iter(["active", "completed"])),
                status_value=lambda status: status,
                details_value=lambda status: {"message": "Job is running"} if status == "active" else None,
                job_name="metric job 'details-job'",
                terminal=frozenset({"completed"}),
                timeout=10.0,
                pending_timeout=10.0,
                poll_interval=0.0,
            )

        payloads = [json.loads(record.getMessage()) for record in caplog.records]
        assert payloads[0]["status_details"] == {"message": "Job is running"}
        assert "status_details" not in payloads[1]

    async def test_timeout_behavior_remains_unchanged(self, caplog: pytest.LogCaptureFixture) -> None:
        """Timeouts should still raise with the current status in the error."""
        with caplog.at_level(logging.INFO, logger="nhx_evals_sdk.execution.job_poll"):
            with pytest.raises(
                TimeoutError,
                match=r"'metric job 'slow'' timed out after 0\.0s\. Status: running",
            ):
                await async_poll_until_terminal(
                    _status_getter(iter(["running"])),
                    status_value=lambda status: status,
                    job_name="metric job 'slow'",
                    terminal=frozenset({"completed"}),
                    timeout=0.0,
                    pending_timeout=10.0,
                    poll_interval=0.0,
                )

        payloads = [json.loads(record.getMessage()) for record in caplog.records]
        assert any(
            payload["job_name"] == "metric job 'slow'" and payload["status"] == "running" for payload in payloads
        )

    async def test_timeout_does_not_sleep_when_get_status_already_exceeds_budget(self, mocker: MockerFixture) -> None:
        """Timeout must fire before sleeping when ``get_status`` already exhausted the budget."""
        fake_now = [0.0]

        async def slow_status() -> str:
            """Simulate a slow status call by advancing the fake clock past the timeout."""
            fake_now[0] += 2.0
            return "running"

        mocker.patch("nhx_evals_sdk.execution.job_poll.time.monotonic", side_effect=lambda: fake_now[0])
        pause_mock = mocker.patch("nhx_evals_sdk.execution.job_poll._async_pause", new=AsyncMock())

        with pytest.raises(TimeoutError, match=r"timed out after 1\.0s.*Status: running"):
            await async_poll_until_terminal(
                slow_status,
                status_value=lambda status: status,
                job_name="slow",
                terminal=frozenset({"completed"}),
                timeout=1.0,
                pending_timeout=600.0,
                poll_interval=10.0,
            )

        pause_mock.assert_not_awaited()
