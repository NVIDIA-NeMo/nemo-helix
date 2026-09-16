# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared wait helpers for CLI commands."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from datetime import datetime
from typing import Any

from nemo_platform import APIConnectionError, APIStatusError, APITimeoutError, NotFoundError
from nemo_platform_plugin.client.response import NemoPaginatedResponse, NemoResponse
from nemo_platform_plugin.client.types import CursorPagination
from nemo_platform_plugin.jobs.client import JobsWatchClient
from nemo_platform_plugin.jobs.schemas import PlatformJobLog, PlatformJobStatusResponse
from nemo_platform_plugin.jobs.types import JobLogsQueryParams
from nemo_platform_plugin.jobs.watch_types import (
    JobStatusEvent,
    JobWarningEvent,
    JobWatchEvent,
    JobWatchTimeoutError,
)
from rich.console import Console
from rich.live import Live
from rich.text import Text

logger = logging.getLogger(__name__)
console = Console()

_TRANSIENT_GATEWAY_STATUS_CODES = {429, 502, 503, 504}


def _pause(seconds: float) -> None:
    time.sleep(seconds)


class _WatchedJobsClient:
    def __init__(self, jobs_client: JobsWatchClient) -> None:
        self._jobs_client = jobs_client
        self.last_status: PlatformJobStatusResponse | None = None

    def get_job_status(
        self,
        *,
        workspace: str | None = None,
        name: str,
    ) -> NemoResponse[PlatformJobStatusResponse]:
        response = self._jobs_client.get_job_status(workspace=workspace, name=name)
        self.last_status = response.data()
        return response

    def list_job_logs(
        self,
        *,
        workspace: str | None = None,
        name: str,
        query_params: JobLogsQueryParams | None = None,
    ) -> NemoPaginatedResponse[PlatformJobLog, CursorPagination]:
        return self._jobs_client.list_job_logs(workspace=workspace, name=name, query_params=query_params)

    def watch_job(
        self,
        name: str,
        *,
        workspace: str | None = None,
        poll_interval: float = 3,
        timeout: float | None = None,
        include_history: bool = True,
        include_logs: bool = True,
        attempt_id: int | None = None,
        step_id: str | None = None,
        task_id: str | None = None,
        limit: int | None = None,
        page_cursor: str | None = None,
    ) -> Iterator[JobWatchEvent]:
        from nemo_platform_plugin.jobs.watch import watch_job

        return watch_job(
            self,
            name,
            workspace=workspace,
            poll_interval=poll_interval,
            timeout=timeout,
            include_history=include_history,
            include_logs=include_logs,
            attempt_id=attempt_id,
            step_id=step_id,
            task_id=task_id,
            limit=limit,
            page_cursor=page_cursor,
        )


def _seconds_since_creation(entry_timestamp: datetime | str | None, created_at: datetime | None) -> int | None:
    if created_at is None or entry_timestamp is None:
        return None
    if isinstance(entry_timestamp, str):
        try:
            entry_timestamp = datetime.fromisoformat(entry_timestamp.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
    if not hasattr(entry_timestamp, "timestamp") or not hasattr(created_at, "timestamp"):
        return None
    try:
        return int(entry_timestamp.timestamp() - created_at.timestamp())
    except (TypeError, OSError):
        return None


def _status_text(status: Any) -> str:
    return str(status or "")


def _make_history_line(
    timestamp_str: str,
    seconds_since_creation: int | None,
    status: str,
    status_message: str = "",
) -> Text:
    text = Text()
    text.append(f"  [{timestamp_str}] ", style="dim")
    if seconds_since_creation is not None:
        text.append(f"(+{seconds_since_creation}s) ", style="dim")
    text.append("Status: ")
    text.append(status, style="cyan bold")
    if status_message:
        text.append(f" - {status_message}", style="dim")
    return text


def _make_live_display(
    polling_time: str,
    timeout: int,
    poll_interval: int,
    wait_elapsed: int,
) -> Text:
    text = Text()
    text.append("\n")
    text.append(
        f"Polling: {polling_time} | Timeout: {timeout}s | Poll interval: {poll_interval}s | Wait: {wait_elapsed}s",
        style="dim",
    )
    return text


class _PlatformJobWaitLiveDisplay:
    def __init__(self, *, start_time: float, timeout: int, poll_interval: int) -> None:
        self.start_time = start_time
        self.timeout = timeout
        self.poll_interval = poll_interval

    def snapshot(self) -> tuple[str, int]:
        return datetime.now().strftime("%H:%M:%S"), int(time.time() - self.start_time)

    def __rich__(self) -> Text:
        polling_time, wait_elapsed = self.snapshot()
        return _make_live_display(polling_time, self.timeout, self.poll_interval, wait_elapsed)


def _sleep_until_next_poll(start_time: float, timeout: float, poll_interval: int) -> bool:
    if poll_interval <= 0:
        raise ValueError(f"_sleep_until_next_poll poll_interval must be greater than 0, got {poll_interval}")
    remaining = timeout - (time.time() - start_time)
    if remaining <= 0:
        return False
    _pause(min(poll_interval, remaining))
    return True


def _gateway_provider_ref(deployment: Any, fallback_name: str, fallback_workspace: str) -> tuple[str, str]:
    model_provider_id = getattr(deployment, "model_provider_id", None)
    if isinstance(model_provider_id, str):
        workspace, separator, provider_name = model_provider_id.partition("/")
        if separator and workspace and provider_name:
            return provider_name, workspace
    return fallback_name, fallback_workspace


def _print_transient_wait_error(live: Live, resource_label: str, error: Exception) -> None:
    live.stop()
    console.print(f"\n[yellow]Transient {resource_label} check failed: {error}[/yellow]")
    live.start()


def wait_for_inference_deployment(
    client: Any,
    name: str,
    *,
    workspace: str | None = None,
    status: str = "READY",
    timeout: int = 1200,
    poll_interval: int = 3,
    check_gateway: bool = True,
    verbose: bool = True,
) -> bool:
    """Wait for an inference deployment to reach the requested status."""
    if workspace is None:
        workspace = client._get_workspace_path_param()

    start_time = time.time()
    last_history_len = 0
    last_status = ""
    last_message = ""

    if verbose:
        console.print(f"[bold]Waiting for deployment '{name}' to reach status: {status}[/bold]\n")

    with Live(console=console, refresh_per_second=4, transient=True) as live:
        while time.time() - start_time < timeout:
            wait_elapsed = int(time.time() - start_time)
            polling_time = datetime.now().strftime("%H:%M:%S")
            if verbose:
                live.update(_make_live_display(polling_time, timeout, poll_interval, wait_elapsed))

            try:
                deployment = client.inference.deployments.retrieve(name, workspace=workspace)
                history = getattr(deployment, "status_history", None)
                created_at = getattr(deployment, "created_at", None)
                if history and len(history) > 0:
                    last_entry = history[-1]
                    current_status = _status_text(getattr(last_entry, "status", getattr(deployment, "status", "")))
                    current_message = getattr(last_entry, "status_message", "") or ""
                else:
                    current_status = _status_text(getattr(deployment, "status", ""))
                    current_message = getattr(deployment, "status_message", "") or ""
                last_status = current_status
                last_message = current_message

                if verbose and history and len(history) > last_history_len:
                    live.stop()
                    if last_history_len == 0:
                        console.print()
                    for i in range(last_history_len, len(history)):
                        entry = history[i]
                        ts = getattr(entry, "timestamp", None)
                        ts_str = ts.strftime("%H:%M:%S") if hasattr(ts, "strftime") else str(ts) if ts else ""
                        st = _status_text(getattr(entry, "status", ""))
                        msg = getattr(entry, "status_message", "") or ""
                        secs = _seconds_since_creation(ts, created_at)
                        console.print(_make_history_line(ts_str, secs, st, msg))
                    last_history_len = len(history)
                    console.print()
                    live.start()

                if verbose:
                    live.update(_make_live_display(polling_time, timeout, poll_interval, wait_elapsed))

                if current_status == status and status != "DELETED":
                    live.stop()
                    if verbose:
                        console.print(f"\n[green]✓ Deployment reached {status} status![/green]")
                    if status == "READY" and check_gateway:
                        remaining_timeout = timeout - (time.time() - start_time)
                        if remaining_timeout <= 0:
                            console.print("\n[red]✗ Timeout before gateway readiness check could complete[/red]")
                            return False
                        provider_name, provider_workspace = _gateway_provider_ref(deployment, name, workspace)
                        return wait_for_gateway(
                            client,
                            provider_name,
                            provider_workspace,
                            timeout=remaining_timeout,
                            poll_interval=poll_interval,
                            verbose=verbose,
                        )
                    return True

                if current_status in {"ERROR", "LOST"} and status not in {"ERROR", "LOST"}:
                    live.stop()
                    console.print(f"\n[red]✗ Deployment entered {current_status} state: {current_message}[/red]")
                    return False

            except NotFoundError:
                if status == "DELETED":
                    live.stop()
                    console.print(f"\n[green]✓ Deployment {status}![/green]")
                    return True
                live.stop()
                console.print("\n[red]✗ Deployment not found[/red]")
                return False
            except (APIConnectionError, APITimeoutError) as exc:
                if verbose:
                    _print_transient_wait_error(live, "deployment status", exc)
            except APIStatusError as exc:
                if exc.status_code not in _TRANSIENT_GATEWAY_STATUS_CODES:
                    raise
                if verbose:
                    _print_transient_wait_error(live, "deployment status", exc)

            if not _sleep_until_next_poll(start_time, timeout, poll_interval):
                break

    wait_elapsed = int(time.time() - start_time)
    detail = f"Last status: {last_status}"
    if last_message:
        detail += f" - {last_message}"
    console.print(f"\n[red]✗ Timeout after {wait_elapsed}s. {detail}[/red]")
    return False


def wait_for_platform_job(
    jobs_client: JobsWatchClient,
    name: str,
    *,
    workspace: str | None = None,
    resource_label: str = "job",
    timeout: int = 1200,
    poll_interval: int = 3,
) -> bool:
    """Wait for a platform job resource to complete."""
    start_time = time.time()
    last_status = ""
    jobs = _WatchedJobsClient(jobs_client)

    console.print(f"[bold]Waiting for {resource_label} '{name}' to complete[/bold]\n")

    live_display = _PlatformJobWaitLiveDisplay(start_time=start_time, timeout=timeout, poll_interval=poll_interval)
    with Live(live_display, console=console, refresh_per_second=4, transient=True) as live:
        try:
            for event in jobs.watch_job(
                name,
                workspace=workspace,
                poll_interval=poll_interval,
                timeout=timeout,
                include_logs=False,
            ):
                polling_time, wait_elapsed = live_display.snapshot()
                live.update(live_display)

                if isinstance(event, JobWarningEvent):
                    live.stop()
                    console.print(f"\n[yellow]{event.message}[/yellow]")
                    live.start()
                    continue

                if not isinstance(event, JobStatusEvent):
                    continue

                current_status = event.status
                if current_status != last_status:
                    live.stop()
                    console.print(_make_history_line(polling_time, wait_elapsed, current_status))
                    last_status = current_status
                    console.print()
                    live.start()

                if event.terminal:
                    live.stop()
                    if event.successful:
                        console.print(f"\n[green]✓ {resource_label.title()} completed![/green]")
                        return True
                    console.print(f"\n[red]✗ {resource_label.title()} entered {current_status} state[/red]")
                    return False
        except NotFoundError:
            live.stop()
            console.print(f"\n[red]✗ {resource_label.title()} not found[/red]")
            return False
        except JobWatchTimeoutError:
            pass

    wait_elapsed = int(time.time() - start_time)
    detail = f"Last status: {last_status}" if last_status else "No status returned"
    console.print(f"\n[red]✗ Timeout after {wait_elapsed}s. {detail}[/red]")
    return False


def wait_for_gateway(
    client: Any,
    provider_name: str,
    workspace: str,
    timeout: float = 60,
    poll_interval: int = 1,
    verbose: bool = True,
) -> bool:
    """Wait for the inference gateway to be able to route to a provider."""
    start_time = time.time()
    start_timestamp = datetime.now().strftime("%H:%M:%S")

    if verbose:
        console.print(f"[bold]Waiting for gateway to be ready for provider '{provider_name}'[/bold]\n")

    def _make_gateway_display(polling_time: str, elapsed: int, status: str) -> Text:
        text = Text()
        text.append(f"Polling: {polling_time} | Timeout: {timeout}s | Poll interval: {poll_interval}s\n", style="dim")
        text.append(f"  [{start_timestamp}] ", style="dim")
        text.append(f"({elapsed}s) ", style="dim")
        text.append(status)
        return text

    with Live(console=console, refresh_per_second=4, transient=True) as live:
        while time.time() - start_time < timeout:
            elapsed = int(time.time() - start_time)
            polling_time = datetime.now().strftime("%H:%M:%S")
            if verbose:
                live.update(_make_gateway_display(polling_time, elapsed, "Checking gateway..."))

            try:
                client.inference.gateway.provider.ready(provider_name, workspace=workspace)
                live.stop()
                if verbose:
                    console.print(f"  [{polling_time}] ({elapsed}s) [green]Gateway is ready![/green]")
                return True
            except NotFoundError:
                pass
            except (APIConnectionError, APITimeoutError):
                pass
            except APIStatusError as exc:
                if exc.status_code in _TRANSIENT_GATEWAY_STATUS_CODES:
                    pass
                else:
                    live.stop()
                    console.print(f"\n[red]✗ Gateway readiness failed: {exc}[/red]")
                    return False

            if not _sleep_until_next_poll(start_time, timeout, poll_interval):
                break

    elapsed = int(time.time() - start_time)
    console.print(f"\n[red]✗ Gateway timeout after {elapsed}s[/red]")
    return False
