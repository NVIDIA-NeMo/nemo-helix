# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Insights scheduling and AnalysisRun CLI."""

from collections.abc import Callable
from datetime import datetime
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, ClassVar

import click
import typer
from nemo_helix_plugin.cli import NemoCLI, create_typer_app
from nemo_helix_plugin.cli_codegen import handle_code_generation
from nemo_helix_plugin.cli_error_handling import handle_errors
from nemo_helix_plugin.cli_options import (
    WORKSPACE_FLAGS,
    AllPagesOption,
    EntityOutputFormatOption,
    ListOutputFormat,
    ListOutputFormatOption,
    NoTruncateOption,
    OutputColumnsOption,
    workspace_help,
)
from nemo_helix_plugin.cli_output import Column, check_output_columns_with_format, format_output
from nemo_helix_plugin.cli_pagination import PaginationType, collect_offset_pages, warn_if_more_pages
from nemo_helix_plugin.cli_state import CLIState, cli_state, resolve_cli_workspace, resolve_output_format
from nemo_helix_plugin.cli_warnings import collect_warnings
from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from nemo_helix_plugin.nooa_model_client import configured_fast_model, configured_model_refs
from nemo_insights_plugin.client import InsightsClient
from nemo_insights_plugin.schema import AnalysisRunResponse, EnableAnalysisConfigRequest
from nemo_insights_plugin.sdk_resources.analysis_configs import _list_params as _config_list_params
from nemo_insights_plugin.sdk_resources.analysis_runs import (
    DEFAULT_POLL_INTERVAL,
    DEFAULT_WAIT_TIMEOUT,
    AnalysisRunNotSubmittedError,
    AnalysisRunTimeoutError,
    _build_create_body,
    poll_until_terminal,
)
from nemo_insights_plugin.sdk_resources.analysis_runs import _list_params as _run_list_params

_RUN_COLUMNS = [Column("name"), Column("agent"), Column("evaluation_id"), Column("created_at")]
_CONFIG_COLUMNS = [
    Column("agent"),
    Column("enabled"),
    Column("default_model"),
    Column("fast_model"),
    Column("updated_at"),
]


class InsightsCLI(NemoCLI):
    """``nemo insights ...`` subcommands."""

    name: ClassVar[str] = "insights"
    description: ClassVar[str] = "Analyze agent telemetry and act on insights."

    def get_cli(self) -> typer.Typer:
        app = create_typer_app(help=self.description)

        @app.callback()
        def _root() -> None:
            """Force subcommand dispatch even when only one verb is registered."""

        analysis_app = create_typer_app(help="Manage periodic agent analysis opt-in state.")
        app.add_typer(analysis_app, name="analysis")

        @analysis_app.command("enable")
        @collect_warnings
        @handle_errors
        def enable_analysis(
            ctx: typer.Context,
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
            output_format: EntityOutputFormatOption = None,
        ) -> None:
            """Enable periodic analysis for an agent."""
            state = cli_state(ctx)
            resolved_output_format = resolve_output_format(ctx, output_format)
            default_model, fast_model = _resolve_model_refs(default_model, fast_model)
            kwargs = {
                "workspace": resolve_cli_workspace(ctx, workspace),
                "agent": agent,
                "body": EnableAnalysisConfigRequest(default_model=default_model, fast_model=fast_model),
            }
            if handle_code_generation(InsightsClient, "enable_analysis_config", kwargs, resolved_output_format, state):
                return
            response = state.typed_client(InsightsClient).enable_analysis_config(**kwargs)
            format_output(response, output_format=resolved_output_format)

        @analysis_app.command("disable")
        @collect_warnings
        @handle_errors
        def disable_analysis(
            ctx: typer.Context,
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
            output_format: EntityOutputFormatOption = None,
        ) -> None:
            """Disable periodic analysis for an agent."""
            state = cli_state(ctx)
            resolved_output_format = resolve_output_format(ctx, output_format)
            kwargs = {"workspace": resolve_cli_workspace(ctx, workspace), "agent": agent}
            if handle_code_generation(InsightsClient, "disable_analysis_config", kwargs, resolved_output_format, state):
                return
            response = state.typed_client(InsightsClient).disable_analysis_config(**kwargs)
            format_output(response, output_format=resolved_output_format)

        @analysis_app.command("status")
        @collect_warnings
        @handle_errors
        def analysis_status(
            ctx: typer.Context,
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
            page: int = typer.Option(1, "--page", help="Page number (1-indexed). Ignored with --agent."),
            page_size: int = typer.Option(100, "--page-size", help="Items per page. Ignored with --agent."),
            all_pages: AllPagesOption = False,
            output_format: ListOutputFormatOption = None,
            no_truncate: NoTruncateOption = None,
            columns: OutputColumnsOption = None,
        ) -> None:
            """Show periodic analysis opt-in state."""
            state = cli_state(ctx)
            resolved_output_format = resolve_output_format(ctx, output_format)
            workspace = resolve_cli_workspace(ctx, workspace)
            if agent:
                kwargs: dict[str, Any] = {"workspace": workspace, "agent": agent}
                if handle_code_generation(InsightsClient, "get_analysis_config", kwargs, resolved_output_format, state):
                    return
                response = state.typed_client(InsightsClient).get_analysis_config(**kwargs)
                format_output(response, output_format=resolved_output_format)
                return

            check_output_columns_with_format(columns, resolved_output_format)
            kwargs = {
                "workspace": workspace,
                "query_params": _config_list_params(page=page, page_size=page_size, sort="-created_at", enabled=None),
            }
            if handle_code_generation(
                InsightsClient,
                "list_analysis_configs",
                kwargs,
                resolved_output_format,
                state,
                result="all-pages" if all_pages else "list",
            ):
                return
            response = state.typed_client(InsightsClient).list_analysis_configs(**kwargs)
            _print_list(
                state,
                collect_offset_pages(response, all_pages=all_pages),
                output_format=resolved_output_format,
                columns=columns,
                default_columns=_CONFIG_COLUMNS,
                no_truncate=no_truncate,
                all_pages=all_pages,
            )

        runs_app = create_typer_app(help="Submit and inspect on-demand analysis runs.")
        app.add_typer(runs_app, name="analysis-runs")

        @runs_app.command("create")
        @collect_warnings
        @handle_errors
        def create_analysis_run(
            ctx: typer.Context,
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
            output_format: EntityOutputFormatOption = None,
        ) -> None:
            """Submit an analysis run for an agent.

            The run is backed by an agents.execute job that shares its name.
            With --wait, exits non-zero if that job does not complete.
            """
            state = cli_state(ctx)
            resolved_output_format = resolve_output_format(ctx, output_format)
            _reject_wait_with_code(wait, resolved_output_format)
            workspace = resolve_cli_workspace(ctx, workspace)
            resolved_default, resolved_fast = _resolve_model_refs(default_model, fast_model)
            kwargs = {
                "workspace": workspace,
                "body": _build_create_body(
                    agent=agent,
                    default_model=resolved_default,
                    fast_model=resolved_fast,
                    ethos=_read_ethos_file(ethos),
                    since=_parse_since(since),
                    evaluation_id=evaluation_id,
                    timeout_seconds=timeout_seconds,
                ),
            }
            if handle_code_generation(InsightsClient, "create_analysis_run", kwargs, resolved_output_format, state):
                return
            client = state.typed_client(InsightsClient)
            response = client.create_analysis_run(**kwargs).data()
            if not wait:
                format_output(response, output_format=resolved_output_format)
                return
            typer.echo(f"Created analysis run '{response.run.name}'.", err=True)
            _print_waited_run(
                client,
                workspace=workspace,
                name=response.run.name,
                poll_timeout=poll_timeout,
                poll_interval=poll_interval,
                output_format=resolved_output_format,
            )

        @runs_app.command("list")
        @collect_warnings
        @handle_errors
        def list_analysis_runs(
            ctx: typer.Context,
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
            page: int = typer.Option(1, "--page", help="Page number (1-indexed)."),
            page_size: int = typer.Option(20, "--page-size", help="Items per page."),
            sort: str = typer.Option(
                "-created_at",
                "--sort",
                help="Sort field; prefix with '-' for descending.",
            ),
            all_pages: AllPagesOption = False,
            output_format: ListOutputFormatOption = None,
            no_truncate: NoTruncateOption = None,
            columns: OutputColumnsOption = None,
        ) -> None:
            """List analysis runs. Job state is not joined — read one run to get it."""
            state = cli_state(ctx)
            resolved_output_format = resolve_output_format(ctx, output_format)
            check_output_columns_with_format(columns, resolved_output_format)
            kwargs = {
                "workspace": resolve_cli_workspace(ctx, workspace),
                "query_params": _run_list_params(page=page, page_size=page_size, sort=sort, agent=agent),
            }
            if handle_code_generation(
                InsightsClient,
                "list_analysis_runs",
                kwargs,
                resolved_output_format,
                state,
                result="all-pages" if all_pages else "list",
            ):
                return
            response = state.typed_client(InsightsClient).list_analysis_runs(**kwargs)
            _print_list(
                state,
                collect_offset_pages(response, all_pages=all_pages),
                output_format=resolved_output_format,
                columns=columns,
                default_columns=_RUN_COLUMNS,
                no_truncate=no_truncate,
                all_pages=all_pages,
            )

        @runs_app.command("get")
        @collect_warnings
        @handle_errors
        def get_analysis_run(
            ctx: typer.Context,
            name: str = typer.Argument(..., help="Name of the analysis run."),
            workspace: str | None = typer.Option(
                None,
                *WORKSPACE_FLAGS,
                help=workspace_help("Workspace the run belongs to."),
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
            output_format: EntityOutputFormatOption = None,
        ) -> None:
            """Get one analysis run, joined with the live state of its backing job.

            A null job means submission never landed: no job exists under the
            run's name, and the run can be resubmitted.
            """
            state = cli_state(ctx)
            resolved_output_format = resolve_output_format(ctx, output_format)
            _reject_wait_with_code(wait, resolved_output_format)
            workspace = resolve_cli_workspace(ctx, workspace)
            kwargs = {"workspace": workspace, "name": name}
            if handle_code_generation(InsightsClient, "get_analysis_run", kwargs, resolved_output_format, state):
                return
            client = state.typed_client(InsightsClient)
            if not wait:
                format_output(client.get_analysis_run(**kwargs), output_format=resolved_output_format)
                return
            _print_waited_run(
                client,
                workspace=workspace,
                name=name,
                poll_timeout=poll_timeout,
                poll_interval=poll_interval,
                output_format=resolved_output_format,
            )

        for entry_point in sorted(entry_points(group="nemo.insights.commands"), key=lambda item: item.name):
            app.add_typer(entry_point.load()(), name=entry_point.name)
        return app


def _print_list(
    state: CLIState,
    result: Any,
    *,
    output_format: ListOutputFormat,
    columns: str | None,
    default_columns: list[Column],
    no_truncate: bool | None,
    all_pages: bool,
) -> None:
    """Render one page (or all pages) of a list command, then warn if more pages remain."""
    output_columns: str | list[Column] = default_columns
    if columns is not None and columns.strip() != "default":
        output_columns = columns
    format_output(
        result,
        is_list=True,
        output_format=output_format,
        output_columns=output_columns,
        no_truncate=state.get_no_truncate(no_truncate),
        timestamp_format=state.get_timestamp_format(),
    )
    if not all_pages:
        warn_if_more_pages(result, PaginationType.PAGE_NUMBER)


def _reject_wait_with_code(wait: bool, output_format: ListOutputFormat) -> None:
    """``-f code`` prints one client call; it cannot express the --wait poll loop."""
    if wait and output_format == "code":
        raise click.UsageError("--wait cannot be combined with --output-format code.")


def _print_waited_run(
    client: InsightsClient,
    *,
    workspace: str,
    name: str,
    poll_timeout: float,
    poll_interval: float,
    output_format: ListOutputFormat,
) -> None:
    """Poll one run to a terminal job state, print it, and exit non-zero unless it completed."""
    try:
        response = poll_until_terminal(
            lambda: client.get_analysis_run(workspace=workspace, name=name).data(),
            timeout=poll_timeout,
            poll_interval=poll_interval,
            on_status=_status_reporter(),
        )
    except (AnalysisRunNotSubmittedError, AnalysisRunTimeoutError) as exc:
        raise click.ClickException(str(exc)) from None
    format_output(response, output_format=output_format)
    if not _completed(response):
        raise typer.Exit(1)


def _completed(response: AnalysisRunResponse) -> bool:
    return response.job_status == HelixJobStatus.COMPLETED.value


def _parse_since(since: str | None) -> datetime | None:
    """Parse ``--since`` here so a bad value fails before anything is submitted."""
    if since is None:
        return None
    try:
        return datetime.fromisoformat(since)
    except ValueError:
        raise click.BadParameter(f"must be an ISO-8601 timestamp, got {since!r}", param_hint="'--since'") from None


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
        raise click.BadParameter(f"could not be read: {exc}", param_hint="'--ethos'") from None
    if not content.strip():
        raise click.BadParameter(f"is empty: {ethos}", param_hint="'--ethos'")
    return content


def _resolve_model_refs(default_model: str | None, fast_model: str | None) -> tuple[str, str]:
    """Fill either model ref from the operator's CLI config when not given.

    The Helix process cannot read that config, so the request has to carry
    the pair; the CLI is where it is known.
    """
    if default_model and fast_model:
        return default_model, fast_model
    if default_model:
        return default_model, configured_fast_model() or default_model
    try:
        configured = configured_model_refs()
    except ValueError as exc:
        raise click.ClickException(f"{exc} Or pass --default-model and --fast-model.") from None
    return default_model or configured.default, fast_model or configured.fast


def _status_reporter() -> Callable[[str | None], None]:
    """Report each job-status change on stderr, keeping stdout for the command's output."""

    def report(status: str | None) -> None:
        typer.echo(f"  status: {status}", err=True)

    return report
