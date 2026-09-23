# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Unit tests for job-waiting utilities.

Tests the refactored wait_for_platform_job() which delegates to
poll_until_terminal() so that image-pull time (pending status) is not
counted against the main job-execution timeout.
"""

import time
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import httpx
import pytest
from nemo_helix_plugin.client.errors import ConflictError, NotFoundError
from nhx.testing.e2e.jobs import (
    TERMINAL_STATUSES,
    cleanup_platform_job,
    poll_until_terminal,
    wait_budget,
    wait_for_platform_job,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resp(data):
    """Wrap a payload in a NemoResponse-like object whose ``.data()`` returns it.

    Production now consumes typed-client responses via ``client.<op>(...).data()``,
    so mocked jobs-client methods must return an object with a ``.data()`` accessor
    rather than the payload directly.
    """
    m = MagicMock()
    m.data.return_value = data
    return m


def _http_error(error_type, status_code: int, method: str = "GET"):
    request = httpx.Request(method, "http://localhost/apis/jobs/v2/workspaces/ws/jobs/my-job")
    response = httpx.Response(status_code, request=request, json={"detail": f"HTTP {status_code}"})
    return error_type(response)


def _make_jobs_client(*statuses: str) -> MagicMock:
    """Return a typed jobs-client mock whose get_job() cycles through *statuses*.

    Each ``get_job`` call returns a ``_resp(job)`` where ``job.status`` is the next
    status in *statuses*. ``get_job_status`` returns a ``_resp`` around a model with
    an empty ``model_dump``.
    """
    jobs_client = MagicMock()
    responses = []
    for s in statuses:
        j = MagicMock()
        j.status = s
        responses.append(_resp(j))
    jobs_client.get_job.side_effect = responses
    jobs_client.get_job_status.return_value = _resp(MagicMock(model_dump=MagicMock(return_value={})))
    jobs_client.cancel_job.return_value = _resp(None)
    jobs_client.delete_job.return_value = _resp(None)
    return jobs_client


@contextmanager
def _patch_client(jobs_client: MagicMock):
    """Patch ``client_from_platform`` in the production module to return *jobs_client*."""
    with patch("nhx.testing.e2e.jobs.client_from_platform", return_value=jobs_client):
        yield


def _make_sdk(*statuses: str) -> MagicMock:
    """Return an SDK mock (unused by production routing, kept for call signatures)."""
    return MagicMock()


# ---------------------------------------------------------------------------
# Basic terminal-status behaviour
# ---------------------------------------------------------------------------


class TestWaitForHelixJobTerminalStatus:
    """Tests that wait_for_platform_job returns on terminal statuses."""

    def test_returns_immediately_on_completed(self):
        """Returns as soon as job is 'completed'."""
        jobs_client = _make_jobs_client("completed")
        with _patch_client(jobs_client):
            job = wait_for_platform_job(_make_sdk(), "my-job", "ws", timeout=5.0)
        assert job.status == "completed"

    def test_returns_immediately_on_error(self):
        """Returns (without raising) when job is 'error'."""
        jobs_client = _make_jobs_client("error")
        with _patch_client(jobs_client):
            job = wait_for_platform_job(_make_sdk(), "my-job", "ws", timeout=5.0)
        assert job.status == "error"

    def test_returns_immediately_on_cancelled(self):
        """Returns when job is 'cancelled'."""
        jobs_client = _make_jobs_client("cancelled")
        with _patch_client(jobs_client):
            job = wait_for_platform_job(_make_sdk(), "my-job", "ws", timeout=5.0)
        assert job.status == "cancelled"

    def test_returns_job_object(self):
        """Returns the actual job object from get_job().data() (not a copy)."""
        expected_job = MagicMock()
        expected_job.status = "completed"
        jobs_client = MagicMock()
        jobs_client.get_job.return_value = _resp(expected_job)
        jobs_client.get_job_status.return_value = _resp(MagicMock(model_dump=MagicMock(return_value={})))
        with _patch_client(jobs_client):
            job = wait_for_platform_job(_make_sdk(), "my-job", "ws", timeout=5.0)
        assert job is expected_job


# ---------------------------------------------------------------------------
# status_to_check behaviour
# ---------------------------------------------------------------------------


class TestWaitForHelixJobStatusToCheck:
    """Tests that status_to_check stops the loop at a non-terminal status."""

    def test_stops_on_status_to_check(self):
        """Returns when the job reaches status_to_check before terminal."""
        jobs_client = _make_jobs_client("created", "pending", "active")
        with _patch_client(jobs_client):
            job = wait_for_platform_job(_make_sdk(), "my-job", "ws", timeout=5.0, status_to_check="active")
        assert job.status == "active"

    def test_also_stops_on_terminal_when_status_to_check_set(self):
        """If job reaches a terminal status before status_to_check, still returns."""
        jobs_client = _make_jobs_client("created", "error")
        with _patch_client(jobs_client):
            job = wait_for_platform_job(_make_sdk(), "my-job", "ws", timeout=5.0, status_to_check="active")
        assert job.status == "error"

    def test_terminal_set_includes_status_to_check(self):
        """poll_until_terminal is called with status_to_check merged into terminal set."""
        jobs_client = _make_jobs_client("paused")
        with _patch_client(jobs_client), patch("nhx.testing.e2e.jobs.poll_until_terminal") as mock_poll:
            # Simulate poll_until_terminal calling get_status once
            def fake_poll(get_status, label, terminal, timeout, image_pull_timeout, poll_interval):
                get_status()

            mock_poll.side_effect = fake_poll
            wait_for_platform_job(_make_sdk(), "my-job", "ws", status_to_check="paused")

        _, kwargs = mock_poll.call_args
        terminal_used = mock_poll.call_args[1]["terminal"] if mock_poll.call_args[1] else mock_poll.call_args[0][2]
        assert "paused" in terminal_used
        for ts in TERMINAL_STATUSES:
            assert ts in terminal_used


# ---------------------------------------------------------------------------
# image_pull_timeout behaviour
# ---------------------------------------------------------------------------


class TestWaitForHelixJobImagePullTimeout:
    """Tests that pending time is handled by poll_until_terminal's image_pull_timeout."""

    def test_pending_status_does_not_consume_main_timeout(self):
        """A job stuck in pending does not exhaust the execution timeout."""
        # pending -> completed: pending time should NOT count against timeout=5.0
        jobs_client = _make_jobs_client("pending", "completed")
        with _patch_client(jobs_client):
            job = wait_for_platform_job(_make_sdk(), "my-job", "ws", timeout=5.0, image_pull_timeout=60.0)
        assert job.status == "completed"

    def test_image_pull_timeout_parameter_passed_to_poll_until_terminal(self):
        """image_pull_timeout is forwarded to poll_until_terminal."""
        jobs_client = _make_jobs_client("completed")
        with _patch_client(jobs_client), patch("nhx.testing.e2e.jobs.poll_until_terminal") as mock_poll:

            def fake_poll(get_status, label, terminal, timeout, image_pull_timeout, poll_interval):
                get_status()

            mock_poll.side_effect = fake_poll
            wait_for_platform_job(_make_sdk(), "my-job", "ws", timeout=30.0, image_pull_timeout=999.0)

        args = mock_poll.call_args
        # image_pull_timeout may be positional or keyword
        if args[1]:
            assert args[1]["image_pull_timeout"] == 999.0
        else:
            assert args[0][4] == 999.0

    def test_default_image_pull_timeout_is_600(self):
        """Default image_pull_timeout is 600 seconds."""
        jobs_client = _make_jobs_client("completed")
        with _patch_client(jobs_client), patch("nhx.testing.e2e.jobs.poll_until_terminal") as mock_poll:

            def fake_poll(get_status, label, terminal, timeout, image_pull_timeout, poll_interval):
                get_status()

            mock_poll.side_effect = fake_poll
            wait_for_platform_job(_make_sdk(), "my-job", "ws")

        args = mock_poll.call_args
        if args[1]:
            assert args[1]["image_pull_timeout"] == 600.0
        else:
            assert args[0][4] == 600.0


# ---------------------------------------------------------------------------
# Timeout error enrichment
# ---------------------------------------------------------------------------


class TestWaitForHelixJobTimeoutError:
    """Tests that TimeoutError from poll_until_terminal is enriched with context."""

    def test_raises_timeout_error_when_poll_times_out(self):
        """TimeoutError propagates when poll_until_terminal raises it."""
        jobs_client = _make_jobs_client("created")
        with _patch_client(jobs_client), patch("nhx.testing.e2e.jobs.poll_until_terminal") as mock_poll:
            mock_poll.side_effect = TimeoutError("'my-job' timed out after 5.0s. Status: created")
            with pytest.raises(TimeoutError):
                wait_for_platform_job(_make_sdk(), "my-job", "ws", timeout=5.0)

    def test_timeout_error_includes_status_history(self):
        """TimeoutError message includes the accumulated status history."""
        jobs_client = _make_jobs_client("created", "pending")

        def fake_poll(get_status, label, terminal, timeout, image_pull_timeout, poll_interval):
            # Call get_status twice to populate history, then timeout
            get_status()
            get_status()
            raise TimeoutError(f"'{label}' timed out after {timeout}s. Status: pending")

        with _patch_client(jobs_client), patch("nhx.testing.e2e.jobs.poll_until_terminal", side_effect=fake_poll):
            with pytest.raises(TimeoutError) as exc_info:
                wait_for_platform_job(_make_sdk(), "my-job", "ws", timeout=5.0)

        assert "Status history:" in str(exc_info.value)
        assert "created" in str(exc_info.value)
        assert "pending" in str(exc_info.value)

    def test_timeout_error_includes_job_status_details(self):
        """TimeoutError message includes detailed job status from get_job_status API."""
        jobs_client = _make_jobs_client("pending")
        jobs_client.get_job_status.return_value = _resp(
            MagicMock(model_dump=MagicMock(return_value={"status": "pending", "message": "pulling image"}))
        )

        def fake_poll(get_status, label, terminal, timeout, image_pull_timeout, poll_interval):
            get_status()
            raise TimeoutError(f"'{label}' timed out")

        with _patch_client(jobs_client), patch("nhx.testing.e2e.jobs.poll_until_terminal", side_effect=fake_poll):
            with pytest.raises(TimeoutError) as exc_info:
                wait_for_platform_job(_make_sdk(), "my-job", "ws", timeout=5.0)

        assert "Job status details:" in str(exc_info.value)


class TestCleanupHelixJob:
    def test_cancels_waits_and_deletes_non_terminal_job(self):
        jobs_client = _make_jobs_client("active", "cancelled")
        with _patch_client(jobs_client):
            cleanup_platform_job(_make_sdk(), "my-job", "ws", timeout=5.0, poll_interval=0.0)

        jobs_client.cancel_job.assert_called_once_with(name="my-job", workspace="ws")
        jobs_client.delete_job.assert_called_once_with(name="my-job", workspace="ws")

    def test_missing_job_is_already_cleaned_up(self):
        jobs_client = _make_jobs_client()
        jobs_client.get_job.side_effect = _http_error(NotFoundError, 404)

        with _patch_client(jobs_client):
            cleanup_platform_job(_make_sdk(), "my-job", "ws", timeout=5.0, poll_interval=0.0)

        jobs_client.cancel_job.assert_not_called()
        jobs_client.delete_job.assert_not_called()

    def test_retries_delete_conflict_until_success(self):
        jobs_client = _make_jobs_client("completed")
        jobs_client.delete_job.side_effect = [_http_error(ConflictError, 409, method="DELETE"), _resp(None)]

        with _patch_client(jobs_client):
            cleanup_platform_job(_make_sdk(), "my-job", "ws", timeout=5.0, poll_interval=0.0)

        assert jobs_client.delete_job.call_count == 2

    def test_delete_conflict_timeout_raises_timeout(self):
        jobs_client = _make_jobs_client("completed")
        jobs_client.delete_job.side_effect = _http_error(ConflictError, 409, method="DELETE")

        with _patch_client(jobs_client):
            with pytest.raises(TimeoutError, match="could not be deleted"):
                cleanup_platform_job(_make_sdk(), "my-job", "ws", timeout=0.0, poll_interval=0.0)


# ---------------------------------------------------------------------------
# Total wall-clock budget
# ---------------------------------------------------------------------------


class TestTotalWallClockBudget:
    """A stuck job must fail its own test, never the whole session.

    Neither ``timeout`` nor ``image_pull_timeout`` bounds wall clock alone, so
    these pin the cap that keeps the helper's diagnostic ahead of pytest's kill.
    """

    def test_pending_job_stops_at_the_published_budget(self):
        """A job stuck in 'pending' raises once the harness budget expires."""
        started = time.monotonic()
        with wait_budget(0.3), pytest.raises(TimeoutError) as excinfo:
            poll_until_terminal(
                lambda: "pending",
                label="stuck-job",
                terminal=TERMINAL_STATUSES,
                timeout=300.0,
                image_pull_timeout=600.0,
                poll_interval=0.01,
            )

        assert time.monotonic() - started < 10.0, "the wait outran its budget"
        assert "total wall-clock budget" in str(excinfo.value)

    def test_budget_is_ignored_when_image_pull_timeout_is_tighter(self):
        """The cap is a backstop; a tighter per-status budget still wins."""
        with wait_budget(60.0), pytest.raises(TimeoutError) as excinfo:
            poll_until_terminal(
                lambda: "pending",
                label="stuck-job",
                terminal=TERMINAL_STATUSES,
                timeout=300.0,
                image_pull_timeout=0.05,
                poll_interval=0.01,
            )

        assert "stuck in pending" in str(excinfo.value)

    def test_terminal_status_still_returns_under_a_budget(self):
        """The cap does not disturb a job that finishes."""
        jobs_client = _make_jobs_client("completed")
        with _patch_client(jobs_client), wait_budget(30.0):
            job = wait_for_platform_job(_make_sdk(), "my-job", "ws", timeout=5.0)

        assert job.status == "completed"

    def test_stuck_pending_job_fails_with_full_diagnostics(self):
        """The exact CI failure: an unpullable image parks the job in 'pending'.

        The run reported no status history at all, because pytest killed the
        process first. The helper's own error must carry it.
        """
        job = MagicMock()
        job.status = "pending"
        jobs_client = MagicMock()
        jobs_client.get_job.return_value = _resp(job)
        jobs_client.get_job_status.return_value = _resp(
            MagicMock(model_dump=MagicMock(return_value={"steps": [{"status": "pending"}]}))
        )

        with _patch_client(jobs_client), wait_budget(0.3), pytest.raises(TimeoutError) as excinfo:
            wait_for_platform_job(_make_sdk(), "my-job", "ws", timeout=300.0, poll_interval=0.01)

        message = str(excinfo.value)
        assert "total wall-clock budget" in message
        assert "Status history: pending" in message
        assert "Job status details" in message

    def test_wait_budget_nests_and_none_clears_the_cap(self):
        """``None`` clears the cap for non-pytest callers; the outer cap returns."""

        def poll(image_pull_timeout: float) -> str:
            with pytest.raises(TimeoutError) as excinfo:
                poll_until_terminal(
                    lambda: "pending",
                    label="stuck-job",
                    terminal=TERMINAL_STATUSES,
                    timeout=300.0,
                    image_pull_timeout=image_pull_timeout,
                    poll_interval=0.01,
                )
            return str(excinfo.value)

        with wait_budget(2.0):
            # Cleared: only image_pull_timeout bounds the wait.
            with wait_budget(None):
                assert "stuck in pending" in poll(0.05)
            # Restored: the outer cap bounds a wait that image_pull_timeout would not.
            assert "total wall-clock budget" in poll(600.0)

        # Left behind: no cap leaks out to later tests.
        assert "stuck in pending" in poll(0.05)

    def test_a_nested_budget_cannot_extend_the_enclosing_one(self):
        """The outer cap is what keeps the wait inside pytest's timeout."""
        started = time.monotonic()
        with wait_budget(0.3), wait_budget(30.0), pytest.raises(TimeoutError) as excinfo:
            poll_until_terminal(
                lambda: "pending",
                label="stuck-job",
                terminal=TERMINAL_STATUSES,
                timeout=300.0,
                image_pull_timeout=600.0,
                poll_interval=0.01,
            )

        assert time.monotonic() - started < 10.0, "the inner budget extended the outer one"
        assert "total wall-clock budget" in str(excinfo.value)

    def test_the_wait_does_not_sleep_past_its_deadline(self):
        """A poll interval longer than the remaining budget must not overshoot it.

        Overshooting by a whole interval is exactly the margin that decides
        whether this raises its own error or pytest kills the process first.
        """
        started = time.monotonic()
        with wait_budget(0.3), pytest.raises(TimeoutError):
            poll_until_terminal(
                lambda: "pending",
                label="stuck-job",
                terminal=TERMINAL_STATUSES,
                timeout=300.0,
                image_pull_timeout=600.0,
                poll_interval=30.0,
            )

        assert time.monotonic() - started < 5.0, "slept past the deadline"
