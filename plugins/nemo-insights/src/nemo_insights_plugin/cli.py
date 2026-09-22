# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Insights scheduling and AnalysisRun CLI."""

import asyncio
import json
import os
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from datetime import datetime
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import httpx
import typer
from nemo_insights_plugin.contracts.profile import (
    DEFAULT_BASE_URL,
)
from nemo_insights_plugin.platform_client import make_client
from nemo_insights_plugin.sdk_resources.analysis_runs import (
    DEFAULT_POLL_INTERVAL,
    DEFAULT_WAIT_TIMEOUT,
    AnalysisRunNotSubmittedError,
    AnalysisRunTimeoutError,
)
from nemo_platform import AsyncNeMoPlatform, NeMoPlatformError
from nemo_platform_plugin.cli import NemoCLI
from nemo_platform_plugin.cli_options import WORKSPACE_FLAGS, workspace_help
from nemo_platform_plugin.cli_state import resolve_cli_workspace
from nemo_platform_plugin.jobs.schemas import PlatformJobStatus
from nemo_platform_plugin.nooa_model_client import configured_model_refs


def _one_line_error(exc: BaseException) -> str:
    """Collapse expected CLI failures to one readable terminal line."""
    message = " ".join(str(exc).splitlines()).strip() or type(exc).__name__
    if isinstance(exc, httpx.HTTPStatusError):
        # The default message names the status and URL but not the reason the
        # service gave, which is the only part a caller can act on.
        detail = " ".join(exc.response.text.splitlines()).strip()
        if detail:
            message = f"{message}: {detail}"
    return message


_T = TypeVar("_T")


def _run_command(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run one analysis-run command, turning expected failures into exit 1."""
    try:
        return asyncio.run(coro)
    except (
        ValueError,
        AnalysisRunNotSubmittedError,
        AnalysisRunTimeoutError,
        NeMoPlatformError,
        httpx.HTTPError,
    ) as exc:
        typer.echo(f"Error: {_one_line_error(exc)}", err=True)
        raise typer.Exit(1) from None


class InsightsCLI(NemoCLI):
    """``nemo insights ...`` subcommands."""

    name: ClassVar[str] = "insights"
    description: ClassVar[str] = "Analyze agent telemetry and act on insights."

    def get_cli(self) -> typer.Typer:
        app = typer.Typer(help=self.description, no_args_is_help=True)

        @app.callback()
        def _root() -> None:
            """Force subcommand dispatch even when only one verb is registered."""

        analysis_app = typer.Typer(
            help="Manage periodic agent analysis opt-in state.",
            no_args_is_help=True,
        )
        app.add_typer(analysis_app, name="analysis")

        @analysis_app.command("enable")
        def enable_analysis(
            typer_ctx: typer.Context,
            agent: str = typer.Option(
                ...,
                "--agent",
                help="Name of the agent to opt in to periodic analysis.",
            ),
            workspace: str | None = typer.Option(
                None,
                *WORKSPACE_FLAGS,
                help=workspace_help("Workspace the agent belongs to."),
            ),
            base_url: str = typer.Option(
                os.environ.get("NMP_BASE_URL", DEFAULT_BASE_URL),
                "--base-url",
                help="Base URL of the running NMP instance.",
                envvar="NMP_BASE_URL",
            ),
        ) -> None:
            """Enable periodic analysis for an agent."""
            workspace = resolve_cli_workspace(typer_ctx, workspace)
            typer.echo(
                asyncio.run(
                    _analysis_config_command(
                        action="enable",
                        agent=agent,
                        workspace=workspace,
                        base_url=base_url,
                    )
                )
            )

        @analysis_app.command("disable")
        def disable_analysis(
            typer_ctx: typer.Context,
            agent: str = typer.Option(
                ...,
                "--agent",
                help="Name of the agent to opt out of periodic analysis.",
            ),
            workspace: str | None = typer.Option(
                None,
                *WORKSPACE_FLAGS,
                help=workspace_help("Workspace the agent belongs to."),
            ),
            base_url: str = typer.Option(
                os.environ.get("NMP_BASE_URL", DEFAULT_BASE_URL),
                "--base-url",
                help="Base URL of the running NMP instance.",
                envvar="NMP_BASE_URL",
            ),
        ) -> None:
            """Disable periodic analysis for an agent."""
            workspace = resolve_cli_workspace(typer_ctx, workspace)
            typer.echo(
                asyncio.run(
                    _analysis_config_command(
                        action="disable",
                        agent=agent,
                        workspace=workspace,
                        base_url=base_url,
                    )
                )
            )

        @analysis_app.command("status")
        def analysis_status(
            typer_ctx: typer.Context,
            agent: str | None = typer.Option(
                None,
                "--agent",
                help="Optional agent name. Omit to list all analysis configs.",
            ),
            workspace: str | None = typer.Option(
                None,
                *WORKSPACE_FLAGS,
                help=workspace_help("Workspace to inspect."),
            ),
            base_url: str = typer.Option(
                os.environ.get("NMP_BASE_URL", DEFAULT_BASE_URL),
                "--base-url",
                help="Base URL of the running NMP instance.",
                envvar="NMP_BASE_URL",
            ),
        ) -> None:
            """Show periodic analysis opt-in state."""
            workspace = resolve_cli_workspace(typer_ctx, workspace)
            typer.echo(
                asyncio.run(
                    _analysis_config_command(
                        action="status",
                        agent=agent,
                        workspace=workspace,
                        base_url=base_url,
                    )
                )
            )

        runs_app = typer.Typer(
            help="Submit and inspect on-demand analysis runs.",
            no_args_is_help=True,
        )
        app.add_typer(runs_app, name="analysis-runs")

        @runs_app.command("create")
        def create_analysis_run(
            typer_ctx: typer.Context,
            agent: str = typer.Option(
                ...,
                "--agent",
                help="Name of the agent whose telemetry should be analyzed.",
            ),
            workspace: str | None = typer.Option(
                None,
                *WORKSPACE_FLAGS,
                help=workspace_help("Workspace the agent belongs to."),
            ),
            base_url: str = typer.Option(
                os.environ.get("NMP_BASE_URL", DEFAULT_BASE_URL),
                "--base-url",
                help="Base URL of the running NMP instance.",
                envvar="NMP_BASE_URL",
            ),
            default_model: str | None = typer.Option(
                None,
                "--default-model",
                help="Model Entity ref for analysis work. Default: the configured default model.",
            ),
            fast_model: str | None = typer.Option(
                None,
                "--fast-model",
                help="Model Entity ref for context summarization. Default: the configured fast model.",
            ),
            since: str | None = typer.Option(
                None,
                "--since",
                help="ISO-8601 lower bound enforced on the analyst's trace/span reads.",
            ),
            ethos: Path | None = typer.Option(
                None,
                "--ethos",
                help="Path to the agent's Ethos Markdown. Its contents are sent with the run.",
            ),
            evaluation_id: str | None = typer.Option(
                None,
                "--evaluation-id",
                help="Restrict the run to spans from one evaluation.",
            ),
            timeout_seconds: float | None = typer.Option(
                None,
                "--timeout-seconds",
                help="Timeout applied to the backing execute-agent job.",
            ),
            wait: bool = typer.Option(
                False,
                "--wait",
                help="Poll the run until its backing job reaches a terminal state.",
            ),
            poll_timeout: float = typer.Option(
                DEFAULT_WAIT_TIMEOUT,
                "--poll-timeout",
                help="How long --wait polls before giving up.",
            ),
            poll_interval: float = typer.Option(
                DEFAULT_POLL_INTERVAL,
                "--poll-interval",
                help="Seconds between --wait polls.",
            ),
        ) -> None:
            """Submit an analysis run for an agent.

            The run is backed by an agents.execute job that shares its name.
            With --wait, exits non-zero if that job does not complete.
            """
            workspace = resolve_cli_workspace(typer_ctx, workspace)
            payload, completed = _run_command(
                _create_analysis_run(
                    agent=agent,
                    workspace=workspace,
                    base_url=base_url,
                    default_model=default_model,
                    fast_model=fast_model,
                    ethos=ethos,
                    since=since,
                    evaluation_id=evaluation_id,
                    timeout_seconds=timeout_seconds,
                    wait=wait,
                    poll_timeout=poll_timeout,
                    poll_interval=poll_interval,
                )
            )
            typer.echo(payload)
            if not completed:
                raise typer.Exit(1)

        @runs_app.command("list")
        def list_analysis_runs(
            typer_ctx: typer.Context,
            agent: str | None = typer.Option(
                None,
                "--agent",
                help="Only list runs that analyzed this agent.",
            ),
            workspace: str | None = typer.Option(
                None,
                *WORKSPACE_FLAGS,
                help=workspace_help("Workspace to inspect."),
            ),
            base_url: str = typer.Option(
                os.environ.get("NMP_BASE_URL", DEFAULT_BASE_URL),
                "--base-url",
                help="Base URL of the running NMP instance.",
                envvar="NMP_BASE_URL",
            ),
            page: int = typer.Option(1, "--page", help="Page number (1-indexed)."),
            page_size: int = typer.Option(20, "--page-size", help="Items per page."),
            sort: str = typer.Option(
                "-created_at",
                "--sort",
                help="Sort field; prefix with '-' for descending.",
            ),
        ) -> None:
            """List analysis runs. Job state is not joined — read one run to get it."""
            workspace = resolve_cli_workspace(typer_ctx, workspace)
            typer.echo(
                _run_command(
                    _list_analysis_runs(
                        agent=agent,
                        workspace=workspace,
                        base_url=base_url,
                        page=page,
                        page_size=page_size,
                        sort=sort,
                    )
                )
            )

        @runs_app.command("get")
        def get_analysis_run(
            typer_ctx: typer.Context,
            name: str = typer.Argument(..., help="Name of the analysis run."),
            workspace: str | None = typer.Option(
                None,
                *WORKSPACE_FLAGS,
                help=workspace_help("Workspace the run belongs to."),
            ),
            base_url: str = typer.Option(
                os.environ.get("NMP_BASE_URL", DEFAULT_BASE_URL),
                "--base-url",
                help="Base URL of the running NMP instance.",
                envvar="NMP_BASE_URL",
            ),
            wait: bool = typer.Option(
                False,
                "--wait",
                help="Poll until the run's backing job reaches a terminal state.",
            ),
            poll_timeout: float = typer.Option(
                DEFAULT_WAIT_TIMEOUT,
                "--poll-timeout",
                help="How long --wait polls before giving up.",
            ),
            poll_interval: float = typer.Option(
                DEFAULT_POLL_INTERVAL,
                "--poll-interval",
                help="Seconds between --wait polls.",
            ),
        ) -> None:
            """Get one analysis run, joined with the live state of its backing job.

            A null job means submission never landed: no job exists under the
            run's name, and the run can be resubmitted.
            """
            workspace = resolve_cli_workspace(typer_ctx, workspace)
            payload, completed = _run_command(
                _get_analysis_run(
                    name=name,
                    workspace=workspace,
                    base_url=base_url,
                    wait=wait,
                    poll_timeout=poll_timeout,
                    poll_interval=poll_interval,
                )
            )
            typer.echo(payload)
            if not completed:
                raise typer.Exit(1)

        for entry_point in sorted(entry_points(group="nemo.insights.commands"), key=lambda item: item.name):
            app.add_typer(entry_point.load()(), name=entry_point.name)
        return app


async def _analysis_config_command(
    *,
    action: str,
    agent: str | None,
    workspace: str,
    base_url: str,
) -> str:
    """Run one analysis-config CLI action and return JSON for stdout."""
    model_refs = None
    if action == "enable":
        if agent is None:
            raise ValueError("agent is required for enable")
        model_refs = configured_model_refs()

    client = make_client(base_url)
    try:
        if action == "enable":
            assert agent is not None
            assert model_refs is not None
            result = await client.insights.analysis_configs.enable(
                workspace=workspace,
                agent=agent,
                default_model=model_refs.default,
                fast_model=model_refs.fast,
            )
            return _json(result.model_dump(mode="json"))
        if action == "disable":
            if agent is None:
                raise ValueError("agent is required for disable")
            result = await client.insights.analysis_configs.disable(workspace=workspace, agent=agent)
            return _json(result.model_dump(mode="json"))
        if action == "status":
            if agent:
                result = await client.insights.analysis_configs.get(workspace=workspace, agent=agent)
                return _json(result.model_dump(mode="json"))
            page = await client.insights.analysis_configs.list_configs(workspace=workspace, page_size=100)
            return _json(page.model_dump(mode="json"))
        raise ValueError(f"Unknown analysis config action: {action}")
    finally:
        await client.close()


@asynccontextmanager
async def _client(base_url: str) -> AsyncIterator[AsyncNeMoPlatform]:
    """Open a platform client for one CLI command and always close it."""
    client = make_client(base_url)
    try:
        yield client
    finally:
        await client.close()


def _parse_since(since: str | None) -> datetime | None:
    """Parse ``--since`` here so a bad value fails before anything is submitted."""
    if since is None:
        return None
    try:
        return datetime.fromisoformat(since)
    except ValueError:
        raise ValueError(f"--since must be an ISO-8601 timestamp, got {since!r}") from None


def _read_ethos_file(ethos: Path | None) -> str | None:
    """Inline the Ethos here: the job's Fabric adapter has no Files access.

    Unlike preflight's tolerant read, an explicit ``--ethos`` that cannot be
    read is fatal — submitting the run without it would silently analyze the
    agent against no contract at all.
    """
    if ethos is None:
        return None
    try:
        content = ethos.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"--ethos could not be read: {exc}") from None
    if not content.strip():
        raise ValueError(f"--ethos is empty: {ethos}")
    return content


def _resolve_model_refs(default_model: str | None, fast_model: str | None) -> tuple[str, str]:
    """Fill either model ref from the operator's CLI config when not given.

    The Platform process cannot read that config, so the request has to carry
    the pair; the CLI is where it is known.
    """
    if default_model and fast_model:
        return default_model, fast_model
    configured = configured_model_refs()
    return default_model or configured.default, fast_model or configured.fast


def _status_reporter() -> Callable[[str | None], None]:
    """Report each job-status change on stderr, keeping stdout pure JSON."""

    def report(status: str | None) -> None:
        typer.echo(f"  status: {status}", err=True)

    return report


async def _create_analysis_run(
    *,
    agent: str,
    workspace: str,
    base_url: str,
    default_model: str | None,
    fast_model: str | None,
    ethos: Path | None,
    since: str | None,
    evaluation_id: str | None,
    timeout_seconds: float | None,
    wait: bool,
    poll_timeout: float,
    poll_interval: float,
) -> tuple[str, bool]:
    """Submit a run and return its JSON plus whether it completed.

    Without ``--wait`` there is nothing to have failed yet, so the run counts
    as completed for exit-code purposes.
    """
    parsed_since = _parse_since(since)
    ethos_content = _read_ethos_file(ethos)
    resolved_default, resolved_fast = _resolve_model_refs(default_model, fast_model)
    async with _client(base_url) as client:
        response = await client.insights.analysis_runs.create(
            workspace=workspace,
            agent=agent,
            default_model=resolved_default,
            fast_model=resolved_fast,
            ethos=ethos_content,
            since=parsed_since,
            evaluation_id=evaluation_id,
            timeout_seconds=timeout_seconds,
        )
        if not wait:
            return _json(response.model_dump(mode="json")), True
        typer.echo(f"Created analysis run '{response.run.name}'.", err=True)
        return await _wait_for_run(
            client,
            workspace=workspace,
            name=response.run.name,
            poll_timeout=poll_timeout,
            poll_interval=poll_interval,
        )


async def _list_analysis_runs(
    *,
    agent: str | None,
    workspace: str,
    base_url: str,
    page: int,
    page_size: int,
    sort: str,
) -> str:
    async with _client(base_url) as client:
        result = await client.insights.analysis_runs.list_runs(
            workspace=workspace,
            agent=agent,
            page=page,
            page_size=page_size,
            sort=sort,
        )
    return _json(result.model_dump(mode="json"))


async def _get_analysis_run(
    *,
    name: str,
    workspace: str,
    base_url: str,
    wait: bool,
    poll_timeout: float,
    poll_interval: float,
) -> tuple[str, bool]:
    async with _client(base_url) as client:
        if wait:
            return await _wait_for_run(
                client,
                workspace=workspace,
                name=name,
                poll_timeout=poll_timeout,
                poll_interval=poll_interval,
            )
        response = await client.insights.analysis_runs.get(workspace=workspace, name=name)
    return _json(response.model_dump(mode="json")), True


async def _wait_for_run(
    client: AsyncNeMoPlatform,
    *,
    workspace: str,
    name: str,
    poll_timeout: float,
    poll_interval: float,
) -> tuple[str, bool]:
    """Poll one run to a terminal job state, reporting status changes on stderr."""
    response = await client.insights.analysis_runs.wait(
        workspace=workspace,
        name=name,
        timeout=poll_timeout,
        poll_interval=poll_interval,
        on_status=_status_reporter(),
    )
    return _json(response.model_dump(mode="json")), response.job_status == PlatformJobStatus.COMPLETED.value


def _json(payload: object) -> str:
    """Serialize a CLI payload with stable indentation."""
    return json.dumps(payload, indent=2)
