# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job waiting utilities for E2E tests.

Provides functions for waiting on job completion across any NeMo Helix service
that implements the standard jobs API pattern.

Why these waits are capped on wall clock
----------------------------------------

``pytest.ini`` sets ``timeout_method = thread``, which cannot unwind a test
blocked in a poll loop: it calls ``os._exit(1)``, losing fixture teardown, job
cleanup, the JUnit report and every later test.  A wait that outlives pytest's
budget therefore destroys the run rather than failing one test, so each wait is
capped below it via :func:`wait_budget` and raises its own diagnostic first.
"""

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from nemo_helix import NeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.errors import ConflictError, NotFoundError
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.schemas import HelixJobLogPage
from nemo_helix_plugin.jobs.types import HelixJobResponse

logger = logging.getLogger(__name__)

# Absolute ``time.monotonic()`` instant past which no wait below may run, or
# ``None`` when the harness has not published a budget.
_wait_deadline: float | None = None


@contextmanager
def wait_budget(seconds: float | None) -> Iterator[None]:
    """Cap every job wait in this block to *seconds* of wall clock.

    The e2e harness publishes each test's share of its pytest budget here.
    ``None`` clears the cap, for callers outside pytest.

    A nested block can only tighten the cap, never extend it: the outer budget
    is what keeps the wait inside pytest's own timeout, so letting an inner one
    reach past it would defeat the point.
    """
    global _wait_deadline
    previous = _wait_deadline
    if seconds is None:
        _wait_deadline = None
    else:
        deadline = time.monotonic() + seconds
        _wait_deadline = deadline if previous is None else min(previous, deadline)
    try:
        yield
    finally:
        _wait_deadline = previous


def poll_until_terminal(
    get_status: Callable[[], str],
    label: str,
    terminal: set[str],
    timeout: float,
    image_pull_timeout: float,
    poll_interval: float,
) -> None:
    """Poll *get_status* until it returns a value in *terminal* or a timeout fires.

    Time spent in ``pending`` status is not counted against *timeout*; it is
    instead capped by the separate *image_pull_timeout*.  *get_status* must
    return a **lowercase** status string each call.

    Total wall-clock time is capped at whichever comes first of ``timeout +
    image_pull_timeout`` and any budget published via :func:`wait_budget`.
    Neither per-status budget bounds wall clock alone, because a job parked in
    ``pending`` never advances *timeout*.

    Raises:
        TimeoutError: When *timeout* is exceeded (excluding pending time),
            *image_pull_timeout* is exceeded while in pending status, or the
            total wall-clock budget is exhausted.
    """
    start = time.monotonic()
    budget = timeout + image_pull_timeout
    harness_deadline = _wait_deadline
    if harness_deadline is not None:
        budget = min(budget, harness_deadline - start)
    deadline = start + budget

    elapsed = 0.0
    pending_elapsed = 0.0
    pending_logged = False
    last_logged_period = -1

    while True:
        poll_start = time.time()
        status = get_status()

        if status in terminal:
            return

        if elapsed >= timeout:
            raise TimeoutError(f"'{label}' timed out after {timeout}s. Status: {status}")

        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"'{label}' exhausted its total wall-clock budget of {budget:.0f}s "
                f"(job timeout {timeout}s, image pull timeout {image_pull_timeout}s). Status: {status}"
            )

        # Never sleep past the deadline: overshooting it by a poll interval is
        # exactly the margin that decides whether this raises or pytest kills us.
        time.sleep(max(0.0, min(poll_interval, deadline - time.monotonic())))
        poll_duration = time.time() - poll_start

        if status == "pending":
            pending_elapsed += poll_duration
            if not pending_logged:
                logger.debug(
                    f"'{label}' is pending (image may be pulling); "
                    f"this time is not counted against the {timeout}s job timeout "
                    f"(image pull timeout: {image_pull_timeout}s)."
                )
                pending_logged = True
            else:
                current_period = int(pending_elapsed / 30)
                if current_period > last_logged_period:
                    logger.info(f"'{label}' still pending after {pending_elapsed:.0f}s.")
                    last_logged_period = current_period
            if pending_elapsed >= image_pull_timeout:
                raise TimeoutError(
                    f"'{label}' stuck in pending after {image_pull_timeout}s (image pull may have failed or stalled)."
                )
        else:
            if pending_logged and pending_elapsed > 0:
                logger.debug(f"'{label}' left pending after {pending_elapsed:.1f}s; now counting toward job timeout.")
                pending_logged = False
            elapsed += poll_duration


# Terminal statuses for platform jobs
TERMINAL_STATUSES = {"completed", "error", "cancelled"}


def _job_status_value(status: object) -> str:
    value = getattr(status, "value", status)
    return str(value or "").lower()


def wait_for_platform_job(
    sdk: NeMoHelix,
    job_name: str,
    workspace: str,
    timeout: float = 120.0,
    image_pull_timeout: float = 600.0,
    poll_interval: float = 1.0,
    status_to_check: str = "",
) -> HelixJobResponse:
    """Wait for a platform job to reach a terminal state.

    Uses the SDK's jobs API to poll for job status until it reaches
    a terminal state (completed, error, or cancelled).

    Time spent in ``pending`` status (e.g. pulling a container image) is not
    counted against *timeout*. A separate *image_pull_timeout* caps how long
    the job may remain pending before the test fails.

    Args:
        sdk: The NeMo Helix SDK client.
        job_name: The platform job name.
        workspace: The workspace name.
        timeout: Maximum time to wait in seconds (excluding image-pull time).
        image_pull_timeout: Maximum time to wait while the job is in
            ``pending`` status before raising ``TimeoutError``.
        poll_interval: Time between status checks in seconds.
        status_to_check: If set, also stop when the job reaches this specific
            status (e.g. ``"active"`` or ``"paused"``).  Terminal statuses
            always stop the loop regardless.

    Returns:
        The final job object from the SDK.

    Raises:
        TimeoutError: If the job doesn't complete within the timeout, or if
            the job is stuck in pending longer than *image_pull_timeout*.
    """
    terminal = ({status_to_check} | TERMINAL_STATUSES) if status_to_check else TERMINAL_STATUSES
    last_job = None
    status_history: list[str] = []

    def get_status() -> str:
        nonlocal last_job
        last_job = client_from_platform(sdk, JobsClient).get_job(name=job_name, workspace=workspace).data()
        current = _job_status_value(last_job.status)
        if not status_history or status_history[-1] != current:
            status_history.append(current)
        return current

    try:
        poll_until_terminal(
            get_status,
            label=job_name,
            terminal=terminal,
            timeout=timeout,
            image_pull_timeout=image_pull_timeout,
            poll_interval=poll_interval,
        )
    except TimeoutError as e:
        error_parts = [str(e), f"Status history: {' -> '.join(status_history)}"]
        try:
            job_status = client_from_platform(sdk, JobsClient).get_job_status(name=job_name, workspace=workspace).data()
            error_parts.append(f"Job status details: {job_status.model_dump()}")
        except Exception as detail_err:
            error_parts.append(f"Failed to get job status: {detail_err}")
        raise TimeoutError("\n".join(error_parts)) from e

    # poll_until_terminal calls get_status (which sets last_job) at least once before returning.
    assert last_job is not None
    return last_job


def wait_for_platform_job_terminal_or_absent(
    sdk: NeMoHelix,
    job_name: str,
    workspace: str,
    timeout: float = 120.0,
    poll_interval: float = 1.0,
    terminal_statuses: set[str] | None = None,
) -> HelixJobResponse | None:
    """Wait until a platform job is terminal or already absent."""
    jobs = client_from_platform(sdk, JobsClient)
    terminal = terminal_statuses or TERMINAL_STATUSES
    deadline = time.monotonic() + timeout
    last_status = ""

    while True:
        try:
            job = jobs.get_job(name=job_name, workspace=workspace).data()
        except NotFoundError:
            return None

        last_status = _job_status_value(job.status)
        if last_status in terminal:
            return job

        if time.monotonic() >= deadline:
            raise TimeoutError(f"'{job_name}' did not reach a terminal status within {timeout}s. Status: {last_status}")
        time.sleep(poll_interval)


def cleanup_platform_job(
    sdk: NeMoHelix,
    job_name: str,
    workspace: str,
    timeout: float = 120.0,
    poll_interval: float = 1.0,
) -> None:
    """Cancel, wait for terminal state, and delete a platform job.

    A missing job is treated as already cleaned up. A delete conflict is not
    swallowed; the helper keeps polling DELETE until it succeeds, the job is
    absent, or the bounded timeout expires.
    """
    jobs = client_from_platform(sdk, JobsClient)
    try:
        job = jobs.get_job(name=job_name, workspace=workspace).data()
    except NotFoundError:
        return

    if _job_status_value(job.status) not in TERMINAL_STATUSES:
        try:
            jobs.cancel_job(name=job_name, workspace=workspace).data()
        except NotFoundError:
            return
        except ConflictError:
            pass
        wait_for_platform_job_terminal_or_absent(
            sdk,
            job_name,
            workspace,
            timeout=timeout,
            poll_interval=poll_interval,
        )

    deadline = time.monotonic() + timeout
    while True:
        try:
            jobs.delete_job(name=job_name, workspace=workspace).data()
            return
        except NotFoundError:
            return
        except ConflictError as exc:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"'{job_name}' could not be deleted within {timeout}s because it remained non-terminal"
                ) from exc
            time.sleep(poll_interval)


def wait_for_job_completion(
    sdk: NeMoHelix,
    service: str,
    workspace: str,
    job_name: str,
    timeout: float = 120.0,
    image_pull_timeout: float = 600.0,
    poll_interval: float = 0.5,
) -> dict:
    """Wait for a job to complete and return the final status.

    Works with any NeMo Helix service that implements the standard jobs API pattern
    at `/apis/{service}/v2/workspaces/{workspace}/jobs/{job_name}`.

    Time spent in ``pending`` status (e.g. pulling a container image) is not
    counted against *timeout*. A separate *image_pull_timeout* caps how long
    the job may remain pending before the test fails.

    Args:
        sdk: The NeMo Helix SDK client.
        service: The service name (e.g., "hello-world", "evaluator", "customizer").
        workspace: The workspace name.
        job_name: The name of the job to wait for.
        timeout: Maximum time to wait in seconds (excluding image pull time).
        image_pull_timeout: Maximum time to wait in pending status before failing.
        poll_interval: Time between status checks in seconds.

    Returns:
        The final job status response.

    Raises:
        TimeoutError: If the job doesn't complete within the timeout.
    """
    base_path = f"/apis/{service}/v2/workspaces/{workspace}/jobs/{job_name}"
    last_status: dict | None = None
    status_history: list[str] = []
    terminal = {"completed", "error", "failed", "cancelled"}

    def get_status() -> str:
        nonlocal last_status
        response = sdk._client.get(f"{base_path}/status")
        assert response.status_code == 200, f"Failed to get job status: {response.text}"
        last_status = response.json()
        current = (last_status.get("status") or "unknown").lower()
        if not status_history or status_history[-1] != current:
            status_history.append(current)
        return current

    try:
        poll_until_terminal(
            get_status,
            label=f"{job_name} ({service})",
            terminal=terminal,
            timeout=timeout,
            image_pull_timeout=image_pull_timeout,
            poll_interval=poll_interval,
        )
    except TimeoutError as e:
        # Re-raise with additional context for job-timeout failures.
        error_parts = [str(e), f"Status history: {' -> '.join(status_history)}"]
        if last_status:
            error_parts.append(f"Last status response: {last_status}")
        job_response = sdk._client.get(base_path)
        if job_response.status_code == 200:
            error_parts.append(f"Full job details: {job_response.json()}")
        raise TimeoutError("\n".join(error_parts)) from e

    # poll_until_terminal calls get_status (which sets last_status) at least once before returning.
    assert last_status is not None
    return last_status


def wait_for_job_logs(
    sdk: NeMoHelix,
    job_name: str,
    workspace: str,
    min_log_count: int = 1,
    timeout: float = 60.0,
    poll_interval: float = 0.5,
):
    """Wait for job logs to be available.

    OTLP logs are batched and may not be immediately available after job
    completion. This function retries until logs appear or timeout.

    Args:
        sdk: The NeMo Helix SDK client.
        job_name: The platform job name.
        workspace: The workspace name.
        min_log_count: Minimum number of logs expected.
        timeout: Maximum time to wait in seconds.
        poll_interval: Time between status checks in seconds.

    Returns:
        The logs pagination object from the SDK.

    Raises:
        TimeoutError: If logs don't appear within the timeout.
    """
    start_time = time.time()
    logs = None

    while time.time() - start_time < timeout:
        page = client_from_platform(sdk, JobsClient).list_job_logs(workspace=workspace, name=job_name).page()
        logs = HelixJobLogPage(data=page.items, **page.metadata)
        if len(logs.data) >= min_log_count:
            return logs
        time.sleep(poll_interval)

    elapsed = time.time() - start_time
    raise TimeoutError(
        f"Job {job_name} logs not available after {elapsed:.1f}s. "
        f"Expected at least {min_log_count} logs, got {len(logs.data) if logs else 0}"
    )
