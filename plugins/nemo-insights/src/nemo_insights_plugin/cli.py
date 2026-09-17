# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Insights CLI and contributed subcommands.

The module-level :func:`analyze` and :func:`doctor` callbacks are the verb bodies for
:class:`nemo_insights_plugin.analyst.cli.AnalystCLI` (``nemo agents analyst run`` /
``nemo agents analyst doctor``). This module's ``InsightsCLI`` keeps the periodic
``analysis`` surface and does not mount those agent verbs.
"""

import asyncio
import json
import os
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import httpx
import typer
from nemo_insights_plugin.analyst.run import ClientConstructionError, run_analyst
from nemo_insights_plugin.contracts.checks import CheckResult, advisories, format_report, required_failures
from nemo_insights_plugin.contracts.insights import InsightsFileError, validate_insights_file
from nemo_insights_plugin.contracts.profile import (
    DEFAULT_BASE_URL,
    EnvFileError,
    ProfileError,
    discover_profile,
    load_env_file,
    resolve_base_url,
)
from nemo_insights_plugin.platform_client import make_client
from nemo_insights_plugin.preflight import (
    AnalysisProbes,
    check_environment,
    check_models,
    check_profile,
    read_ethos,
)
from nemo_insights_plugin.profile import AnalysisProfile, load_profile, pick_ethos
from nemo_insights_plugin.sdk_resources.analysis_runs import (
    DEFAULT_POLL_INTERVAL,
    DEFAULT_WAIT_TIMEOUT,
    AnalysisRunNotSubmittedError,
    AnalysisRunTimeoutError,
)
from nemo_platform import AsyncNeMoPlatform, NeMoPlatformError
from nemo_platform_plugin.cli import NemoCLI
from nemo_platform_plugin.jobs.schemas import PlatformJobStatus
from nemo_platform_plugin.nooa_model_client import configured_model_refs
from nooa import GenerationError

DEFAULT_WORKSPACE = "default"
_PREFLIGHT_PROBES: AnalysisProbes | None = None


@dataclass(frozen=True)
class _ResolvedAnalysis:
    agent: str
    ethos: str | None
    workspace: str
    base_url: str
    insights_output: Path | None
    profile_dir: Path | None
    ethos_checks: tuple[CheckResult, ...]


def _load_profile_or_error(profile_path: Path | None) -> tuple[AnalysisProfile | None, str | None]:
    """Load an explicit or discovered profile, preserving non-explicit failures."""
    found = profile_path or discover_profile()
    if found is None:
        return None, None
    try:
        profile = load_profile(found)
    except ProfileError as exc:
        if profile_path is not None:
            raise
        loaded = load_env_file(found.parent / ".env")
        if loaded:
            typer.echo(f"Loaded .env from {found.parent / '.env'} ({len(loaded)} vars)", err=True)
        return None, str(exc)
    if profile_path is None:
        typer.echo(f"Using profile: {found} (agent: {profile.agent})", err=True)
    loaded = load_env_file(found.parent / ".env")
    if loaded:
        typer.echo(f"Loaded .env from {found.parent / '.env'} ({len(loaded)} vars)", err=True)
    return profile, None


def _preflight_or_exit(checks: list[CheckResult]) -> None:
    """Print blockers and stop before an analyst run."""
    if required_failures(checks):
        typer.echo(format_report(checks), err=True)
        raise typer.Exit(code=1)
    warnings = advisories(checks)
    if warnings:
        typer.echo(format_report(warnings), err=True)


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


def _resolve_analysis(
    *,
    agent: str | None,
    ethos: Path | None,
    workspace: str | None,
    base_url: str | None,
    profile_path: Path | None,
    insights_output: Path | None,
) -> _ResolvedAnalysis:
    profile, profile_error = _load_profile_or_error(profile_path)
    if profile_error is not None:
        if agent is None:
            raise ProfileError(profile_error)
        typer.echo(f"warning: ignoring discovered profile: {profile_error}", err=True)

    resolved_agent = agent if agent is not None else (profile.agent if profile is not None else None)
    if resolved_agent is None:
        raise ProfileError(
            "No --agent given and no optimizer.yaml profile found. Pass --agent or run from a directory with a profile."
        )
    if workspace is not None:
        resolved_workspace = workspace
    else:
        resolved_workspace = profile.workspace if profile is not None else DEFAULT_WORKSPACE

    ethos_path = ethos
    ethos_error: str | None = None
    if ethos_path is None and profile is not None:
        try:
            ethos_path = pick_ethos(profile)
        except ProfileError as exc:
            ethos_error = str(exc)
    ethos_content, ethos_checks = read_ethos(ethos_path, ethos_error)

    resolved_base_url = resolve_base_url(base_url)
    validate_insights_file(insights_output)

    return _ResolvedAnalysis(
        agent=resolved_agent,
        ethos=ethos_content,
        workspace=resolved_workspace,
        base_url=resolved_base_url,
        insights_output=insights_output,
        profile_dir=profile.profile_dir if profile is not None else None,
        ethos_checks=tuple(ethos_checks),
    )


def _prepare_mirror(insights_output: Path | None) -> Path | None:
    """Ready the mirror's directory, dropping the mirror if that fails.

    The mirror is a convenience beside the platform, which is the source of
    truth, so an unusable local path must not cost the user the analysis. This
    matches how a failed mirror *write* is reported — a warning on the run
    report rather than a failed run.
    """
    if insights_output is None:
        return None
    try:
        insights_output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        typer.echo(
            f"warning: insights mirror disabled — could not create {insights_output.parent}: "
            f"{_one_line_error(exc)}. Insights are still written to the platform.",
            err=True,
        )
        return None
    typer.echo(f"Insights file (mirror of the platform): {insights_output}", err=True)
    return insights_output


async def _run_analysis(analysis: _ResolvedAnalysis, *, verbose: bool) -> str:
    checks = list(analysis.ethos_checks)
    checks.extend(check_models())
    _preflight_or_exit(checks)

    insights_output = _prepare_mirror(analysis.insights_output)
    try:
        try:
            client = make_client(analysis.base_url)
        except (RuntimeError, ValueError) as exc:
            raise ClientConstructionError(str(exc)) from None
        return await run_analyst(
            agent=analysis.agent,
            ethos=analysis.ethos,
            workspace=analysis.workspace,
            base_url=analysis.base_url,
            client=client,
            insights_output=insights_output,
            verbose=verbose,
        )
    except GenerationError as exc:
        detail = _one_line_error(exc).rstrip(".")
        typer.echo(
            f"Error: analyst run failed: {detail}. "
            "Check inference model access and credentials, "
            "then retry or adjust usage limits.",
            err=True,
        )
        raise typer.Exit(1) from None
    except (ClientConstructionError, NeMoPlatformError, httpx.HTTPError) as exc:
        detail = _one_line_error(exc).rstrip(".")
        typer.echo(
            f"Error: analysis failed: {detail}. Check --base-url/NMP_BASE_URL, "
            "authentication, workspace, and Intake availability.",
            err=True,
        )
        raise typer.Exit(1) from None


def analyze(
    agent: str | None = typer.Option(
        None,
        "--agent",
        help="Name of the agent (agent under test) the analyst should focus on.",
    ),
    ethos: Path | None = typer.Option(
        None,
        "--ethos",
        help="Path to a markdown file describing the agent under test (its Ethos).",
        exists=True,
        readable=True,
    ),
    workspace: str | None = typer.Option(
        None,
        "--workspace",
        help="Workspace the analyst should operate in.",
    ),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        help="Base URL of the running NMP instance the analyst's tools should call.",
    ),
    profile_path: Path | None = typer.Option(
        None,
        "--profile",
        help="Path to optimizer.yaml. Default: discovered by walking up from cwd.",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    insights_output: Path | None = typer.Option(
        None,
        "--insights-file-output",
        help=(
            "Also write insights to this local YAML file. Insights always go "
            "to the platform first; the file mirrors what was stored, "
            "platform ids included, and each run merges into it."
        ),
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help=(
            "Stream the analyst's tool calls and reasoning to stderr "
            "while it runs. Off by default so that stdout stays clean "
            "for piping the final answer."
        ),
    ),
) -> None:
    """Run the analyst agent against a running NMP instance.

    Builds the analyst agent with ``--agent`` (and optional ``--ethos``)
    formatted into its instructions and tools scoped
    to ``--agent`` / ``--workspace`` / ``--base-url``, runs it, and
    prints whatever the agent returns. Insights are written to the
    platform, and mirrored to ``--insights-file-output`` when given.
    """
    try:
        analysis = _resolve_analysis(
            agent=agent,
            ethos=ethos,
            workspace=workspace,
            base_url=base_url,
            profile_path=profile_path,
            insights_output=insights_output,
        )
        output = asyncio.run(_run_analysis(analysis, verbose=verbose))
    except (ProfileError, EnvFileError, InsightsFileError, OSError, UnicodeError) as exc:
        typer.echo(f"Error: {_one_line_error(exc)}", err=True)
        raise typer.Exit(1) from None
    typer.echo(output)


def doctor(
    profile_path: Path | None = typer.Option(
        None,
        "--profile",
        help="Path to optimizer.yaml. Default: discovered by walking up from cwd.",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        help="Base URL of the running NMP instance to check.",
    ),
) -> None:
    """Check whether the current profile is ready for analysis."""
    try:
        try:
            profile, profile_error = _load_profile_or_error(profile_path)
        except ProfileError as exc:
            profile, profile_error = None, str(exc)
        ethos_path: Path | None = None
        ethos_error: str | None = None
        if profile is not None:
            try:
                ethos_path = pick_ethos(profile)
            except ProfileError as exc:
                ethos_error = str(exc)
        _, ethos_results = read_ethos(ethos_path, ethos_error)

        async def _flow() -> list[CheckResult]:
            results = check_profile(profile, profile_error)
            results.extend(ethos_results)
            results.extend(
                await check_environment(
                    agent=profile.agent if profile is not None else None,
                    workspace=profile.workspace if profile is not None else None,
                    base_url=resolve_base_url(base_url),
                    profile_dir=profile.profile_dir if profile is not None else None,
                    probes=_PREFLIGHT_PROBES,
                )
            )
            return results

        results = asyncio.run(_flow())
    except (EnvFileError, OSError, UnicodeError) as exc:
        typer.echo(f"Error: {_one_line_error(exc)}", err=True)
        raise typer.Exit(1) from None
    typer.echo(format_report(results))
    if required_failures(results):
        raise typer.Exit(code=1)


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
            agent: str = typer.Option(
                ...,
                "--agent",
                help="Name of the agent to opt in to periodic analysis.",
            ),
            workspace: str = typer.Option(
                DEFAULT_WORKSPACE,
                "--workspace",
                help="Workspace the agent belongs to.",
            ),
            base_url: str = typer.Option(
                os.environ.get("NMP_BASE_URL", DEFAULT_BASE_URL),
                "--base-url",
                help="Base URL of the running NMP instance.",
                envvar="NMP_BASE_URL",
            ),
        ) -> None:
            """Enable periodic analysis for an agent."""
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
            agent: str = typer.Option(
                ...,
                "--agent",
                help="Name of the agent to opt out of periodic analysis.",
            ),
            workspace: str = typer.Option(
                DEFAULT_WORKSPACE,
                "--workspace",
                help="Workspace the agent belongs to.",
            ),
            base_url: str = typer.Option(
                os.environ.get("NMP_BASE_URL", DEFAULT_BASE_URL),
                "--base-url",
                help="Base URL of the running NMP instance.",
                envvar="NMP_BASE_URL",
            ),
        ) -> None:
            """Disable periodic analysis for an agent."""
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
            agent: str | None = typer.Option(
                None,
                "--agent",
                help="Optional agent name. Omit to list all analysis configs.",
            ),
            workspace: str = typer.Option(
                DEFAULT_WORKSPACE,
                "--workspace",
                help="Workspace to inspect.",
            ),
            base_url: str = typer.Option(
                os.environ.get("NMP_BASE_URL", DEFAULT_BASE_URL),
                "--base-url",
                help="Base URL of the running NMP instance.",
                envvar="NMP_BASE_URL",
            ),
        ) -> None:
            """Show periodic analysis opt-in state."""
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
            agent: str = typer.Option(
                ...,
                "--agent",
                help="Name of the agent whose telemetry should be analyzed.",
            ),
            workspace: str = typer.Option(
                DEFAULT_WORKSPACE,
                "--workspace",
                help="Workspace the agent belongs to.",
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
            agent: str | None = typer.Option(
                None,
                "--agent",
                help="Only list runs that analyzed this agent.",
            ),
            workspace: str = typer.Option(
                DEFAULT_WORKSPACE,
                "--workspace",
                help="Workspace to inspect.",
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
            name: str = typer.Argument(..., help="Name of the analysis run."),
            workspace: str = typer.Option(
                DEFAULT_WORKSPACE,
                "--workspace",
                help="Workspace the run belongs to.",
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
