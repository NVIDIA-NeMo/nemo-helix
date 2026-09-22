# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo experiments`` command group, backed by the typed Intake client."""

from __future__ import annotations

from typing import Annotated, Any, Literal, cast

import typer
from nemo_platform_ext.cli.core.api import build_kwargs, merge_filter_dict
from nemo_platform_ext.cli.core.code_generator import handle_code_generation
from nemo_platform_ext.cli.core.context import CLIContext
from nemo_platform_ext.cli.core.errors import handle_errors
from nemo_platform_ext.cli.core.formatters import (
    Column,
    check_output_columns_with_format,
    format_output,
    validate_stream_output_format,
)
from nemo_platform_ext.cli.core.help_formatter import collect_warnings, create_typer_app
from nemo_platform_ext.cli.core.pagination import PaginationType, collect_offset_pages, warn_if_more_pages
from nemo_platform_ext.cli.core.stdin_utils import (
    build_request_body,
    read_data_input_with_flags,
    read_payload,
    validate_required_fields,
)
from nemo_platform_ext.cli.core.types import (
    EntityOutputFormatOption,
    ListOutputFormatOption,
    NoTruncateOption,
    OutputColumnsOption,
    StreamOutputOption,
)
from nemo_platform_plugin.intake.client import IntakeClient
from nemo_platform_plugin.intake.types import (
    ExperimentCreateRequest,
    ExperimentUpdateRequest,
    ListExperimentsQueryParams,
)
from nmp.intake.cli_commands.common import list_query_params
from pydantic import TypeAdapter

app = create_typer_app(name="experiments", help="Manage experiments")

# --input-data may carry exist_ok as a JSON/YAML string; parse it like the server would.
_BOOL = TypeAdapter(bool)

_COLUMN_LAYOUT_HELP = (
    "A saved table layout for a group's evaluations list: column order and which columns are hidden. Column ids "
    "are Studio's and cannot be enumerated here — the table builds a column per evaluator and metadata key found "
    "in the rows — so ids are stored and echoed back unvalidated. Visibility is stored as the _hidden_ ids rather "
    "than a map over every column, so a column that appears later (a new evaluator, a new metadata key) shows up "
    "by default. (JSON string)"
)
_DEFAULT_SORT_HELP = (
    "Default sort for this experiment's evaluations list, as a `sort`-param string: a comma-separated, ordered "
    "list of fields where the first is the primary sort and the rest break ties (leading '-' on a field = "
    "descending), e.g. '-evaluators.reward.mean,cost_usd.mean'. Defaults to '-created_at'. Accepts any field the "
    "evaluations list `sort` param does; clients apply it as the list `sort` param."
)
_PARETO_HELP = (
    "Default X/Y metrics for a group's cost-vs-accuracy Pareto view. Metric ids use the same vocabulary as the "
    "evaluations list sort/filter fields — `cost_usd`, `latency_ms`, or `evaluators.<name>`. Defaults to cost (x) "
    "vs latency (y): both exist for every group, so the chart always has something to render before anyone "
    "customizes it. (JSON string)"
)
_IS_FAVORITE_HELP = (
    "Whether this Experiment is marked as a favorite. Defaults to false on create; omit on update to preserve "
    "the existing value."
)
_SHOW_OVER_TIME_HELP = (
    "Whether Studio should display this Experiment's Evaluation results over time. Defaults to false on create; "
    "omit on update to preserve the existing value."
)


@app.command("create")
@collect_warnings
@handle_errors
def create_experiments(
    ctx: typer.Context,
    name: Annotated[str | None, typer.Argument(help="Workspace-unique experiment name. (required)")] = None,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    column_layout: Annotated[str | None, typer.Option("--column-layout", help=_COLUMN_LAYOUT_HELP)] = None,
    default_sort: Annotated[str | None, typer.Option("--default-sort", help=_DEFAULT_SORT_HELP)] = None,
    description: Annotated[
        str | None, typer.Option("--description", help="Human-readable purpose of the experiment.")
    ] = None,
    insight_id: Annotated[
        str | None,
        typer.Option("--insight-id", help="Reference to an external insight that seeded this experiment, if any."),
    ] = None,
    is_favorite: Annotated[bool | None, typer.Option("--is-favorite", help=_IS_FAVORITE_HELP)] = None,
    metadata: Annotated[
        str | None, typer.Option("--metadata", help="Free-form producer metadata for the experiment. (JSON string)")
    ] = None,
    pareto: Annotated[str | None, typer.Option("--pareto", help=_PARETO_HELP)] = None,
    show_evaluations_over_time: Annotated[
        bool | None, typer.Option("--show-evaluations-over-time", help=_SHOW_OVER_TIME_HELP)
    ] = None,
    summary: Annotated[
        str | None, typer.Option("--summary", help="Human- or agent-authored summary of the experiment's findings.")
    ] = None,
    exist_ok: Annotated[
        bool | None,
        typer.Option(
            "--exist-ok", help="Do not raise an error if the resource already exists. Returns the existing resource."
        ),
    ] = None,
    input_file: Annotated[
        str | None,
        typer.Option("--input-file", help="Path to JSON file (use '-' for stdin)", rich_help_panel="Input Options"),
    ] = None,
    input_data: Annotated[
        str | None,
        typer.Option("--input-data", help="Input data for the request (JSON or YAML)", rich_help_panel="Input Options"),
    ] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Create Experiment

    [bold red]Required fields:[/] name

    [green]Examples:[/]
    nemo experiments create <name> --input-file config.json
    nemo experiments create <name> --input-data '{"name": "value"}'
    echo '{"json": "data"}' | nemo experiments create <name> --input-file -
    nemo experiments create <name> --<option> "value"
    """
    # Read base input (optional if all fields provided via flags)
    if input_file or input_data:
        input_payload = read_data_input_with_flags(input_file=input_file, input_data=input_data)
    else:
        input_payload = {}

    # Apply CLI flag overrides (flags take precedence)
    if workspace is not None:
        input_payload["workspace"] = workspace
    if name is not None:
        input_payload["name"] = name
    if column_layout is not None:
        input_payload["column_layout"] = read_payload("column_layout", column_layout)
    if default_sort is not None:
        input_payload["default_sort"] = default_sort
    if description is not None:
        input_payload["description"] = description
    if insight_id is not None:
        input_payload["insight_id"] = insight_id
    if is_favorite is not None:
        input_payload["is_favorite"] = is_favorite
    if metadata is not None:
        input_payload["metadata"] = read_payload("metadata", metadata)
    if pareto is not None:
        input_payload["pareto"] = read_payload("pareto", pareto)
    if show_evaluations_over_time is not None:
        input_payload["show_evaluations_over_time"] = show_evaluations_over_time
    if summary is not None:
        input_payload["summary"] = summary
    if exist_ok is not None:
        input_payload["exist_ok"] = exist_ok
    # Validate required fields are present after merging
    validate_required_fields(
        input_payload,
        ["name"],
        "experiments create",
        {
            "name": "Workspace-unique experiment name. (required)",
        },
    )

    body = build_request_body(
        ExperimentCreateRequest, input_payload, exclude={"workspace", "exist_ok"}, command_name="experiments create"
    )
    resolved_exist_ok = _BOOL.validate_python(input_payload.get("exist_ok", False))
    kwargs = build_kwargs(
        workspace=input_payload.get("workspace"),
        body=body,
        exist_ok=resolved_exist_ok or None,
    )
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    if handle_code_generation(IntakeClient, "create_experiment", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(IntakeClient).create_experiment(**kwargs)

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@app.command("delete")
@collect_warnings
@handle_errors
def delete_experiments(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
) -> None:
    """Delete Experiment"""
    state: CLIContext = ctx.obj
    state.typed_client(IntakeClient).delete_experiment(name=name, workspace=workspace)

    typer.echo("✓ Deleted successfully")


@app.command("list")
@collect_warnings
@handle_errors
def list_experiments(
    ctx: typer.Context,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    filter: Annotated[
        str | None,
        typer.Option(
            "--filter",
            metavar="FILTER_JSON",
            help="Use --filter with JSON for complex/nested queries, or --filter.FIELD options for simple fields. Both can be combined, with field options taking precedence.\nJSON-only fields:\n  metadata: dict[str, str]\n\nFilter experiments by name, insight_id, is_favorite, show_evaluations_over_time, baseline_evaluation_name, is_deleted, or a metadata key/value (filter[metadata.<key>]=<value>). Pass is_deleted=true to return only soft-deleted experiments; omit to see only live ones.",
            rich_help_panel="Filter Options",
        ),
    ] = None,
    filter_baseline_evaluation_name: Annotated[
        str | None, typer.Option("--filter.baseline-evaluation-name", rich_help_panel="Filter Options")
    ] = None,
    filter_insight_id: Annotated[
        str | None, typer.Option("--filter.insight-id", rich_help_panel="Filter Options")
    ] = None,
    filter_is_deleted: Annotated[
        bool | None, typer.Option("--filter.is-deleted", rich_help_panel="Filter Options")
    ] = None,
    filter_is_favorite: Annotated[
        bool | None, typer.Option("--filter.is-favorite", rich_help_panel="Filter Options")
    ] = None,
    filter_name: Annotated[str | None, typer.Option("--filter.name", rich_help_panel="Filter Options")] = None,
    filter_show_evaluations_over_time: Annotated[
        bool | None, typer.Option("--filter.show-evaluations-over-time", rich_help_panel="Filter Options")
    ] = None,
    page: Annotated[int | None, typer.Option("--page", help="Page number.")] = None,
    page_size: Annotated[int | None, typer.Option("--page-size", help="Page size.")] = None,
    sort: Annotated[
        Literal["-created_at", "created_at", "-updated_at", "updated_at", "-name", "name"] | None,
        typer.Option("--sort", help="Sort field; prefix with '-' for descending."),
    ] = None,
    output_format: ListOutputFormatOption = None,
    no_truncate: NoTruncateOption = None,
    columns: OutputColumnsOption = None,
    stream: StreamOutputOption = False,
    all_pages: Annotated[bool, typer.Option("--all-pages", help="Fetch all pages")] = False,
) -> None:
    """List Experiments"""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)
    validate_stream_output_format(resolved_output_format, stream)

    check_output_columns_with_format(columns, resolved_output_format)

    default_columns = [
        Column("name", None),
        Column("workspace", None),
        Column("created_at", None),
    ]
    output_columns: str | list[Column] | None = columns
    if columns is None or str(columns).strip() == "default":
        output_columns = default_columns

    query_params = cast(
        ListExperimentsQueryParams | None,
        list_query_params(
            filter=merge_filter_dict(
                filter,
                baseline_evaluation_name=filter_baseline_evaluation_name,
                insight_id=filter_insight_id,
                is_deleted=filter_is_deleted,
                is_favorite=filter_is_favorite,
                name=filter_name,
                show_evaluations_over_time=filter_show_evaluations_over_time,
            ),
            page=page,
            page_size=page_size,
            sort=sort,
        ),
    )
    kwargs = build_kwargs(workspace=workspace, query_params=query_params)

    if handle_code_generation(IntakeClient, "list_experiments", kwargs, resolved_output_format, state, result="list"):
        return

    response = state.typed_client(IntakeClient).list_experiments(workspace=workspace, query_params=query_params)
    pagination_type = PaginationType.PAGE_NUMBER
    items = collect_offset_pages(response, all_pages=all_pages)

    format_output(
        items,
        is_list=True,
        output_format=resolved_output_format,
        output_columns=output_columns,
        no_truncate=state.get_no_truncate(no_truncate),
        timestamp_format=state.get_timestamp_format(),
        stream=stream,
    )
    if not all_pages:
        warn_if_more_pages(items, pagination_type)


@app.command("get")
@collect_warnings
@handle_errors
def retrieve_experiments(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Get Experiment"""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(name=name, workspace=workspace)
    if handle_code_generation(IntakeClient, "get_experiment", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(IntakeClient).get_experiment(name=name, workspace=workspace)

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@app.command("update")
@collect_warnings
@handle_errors
def update_experiments(
    ctx: typer.Context,
    path_name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    body_name: Annotated[
        str | None, typer.Option("--body-name", help="Workspace-unique experiment name. (required)")
    ] = None,
    baseline_evaluation_name: Annotated[
        str | None,
        typer.Option(
            "--baseline-evaluation-name",
            help="Name of this Experiment's baseline Evaluation. The Evaluation must already be a live member of the Experiment. Set null to clear the selected baseline.",
        ),
    ] = None,
    column_layout: Annotated[str | None, typer.Option("--column-layout", help=_COLUMN_LAYOUT_HELP)] = None,
    default_sort: Annotated[str | None, typer.Option("--default-sort", help=_DEFAULT_SORT_HELP)] = None,
    description: Annotated[
        str | None, typer.Option("--description", help="Human-readable purpose of the experiment.")
    ] = None,
    insight_id: Annotated[
        str | None,
        typer.Option("--insight-id", help="Reference to an external insight that seeded this experiment, if any."),
    ] = None,
    is_favorite: Annotated[bool | None, typer.Option("--is-favorite", help=_IS_FAVORITE_HELP)] = None,
    metadata: Annotated[
        str | None, typer.Option("--metadata", help="Free-form producer metadata for the experiment. (JSON string)")
    ] = None,
    pareto: Annotated[str | None, typer.Option("--pareto", help=_PARETO_HELP)] = None,
    show_evaluations_over_time: Annotated[
        bool | None, typer.Option("--show-evaluations-over-time", help=_SHOW_OVER_TIME_HELP)
    ] = None,
    summary: Annotated[
        str | None, typer.Option("--summary", help="Human- or agent-authored summary of the experiment's findings.")
    ] = None,
    input_file: Annotated[
        str | None,
        typer.Option("--input-file", help="Path to JSON file (use '-' for stdin)", rich_help_panel="Input Options"),
    ] = None,
    input_data: Annotated[
        str | None,
        typer.Option("--input-data", help="Input data for the request (JSON or YAML)", rich_help_panel="Input Options"),
    ] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Update Experiment

    [bold red]Required fields:[/] body_name

    [green]Examples:[/]
    nemo experiments update <path_name> --input-file config.json
    nemo experiments update <path_name> --input-data '{"body_name": "value"}'
    echo '{"json": "data"}' | nemo experiments update <path_name> --input-file -
    nemo experiments update <path_name> --<option> "value"
    """
    # Read base input (optional if all fields provided via flags)
    if input_file or input_data:
        input_payload = read_data_input_with_flags(input_file=input_file, input_data=input_data)
    else:
        input_payload = {}

    # Apply CLI flag overrides (flags take precedence)
    if workspace is not None:
        input_payload["workspace"] = workspace
    if body_name is not None:
        input_payload["body_name"] = body_name
    if baseline_evaluation_name is not None:
        input_payload["baseline_evaluation_name"] = baseline_evaluation_name
    if column_layout is not None:
        input_payload["column_layout"] = read_payload("column_layout", column_layout)
    if default_sort is not None:
        input_payload["default_sort"] = default_sort
    if description is not None:
        input_payload["description"] = description
    if insight_id is not None:
        input_payload["insight_id"] = insight_id
    if is_favorite is not None:
        input_payload["is_favorite"] = is_favorite
    if metadata is not None:
        input_payload["metadata"] = read_payload("metadata", metadata)
    if pareto is not None:
        input_payload["pareto"] = read_payload("pareto", pareto)
    if show_evaluations_over_time is not None:
        input_payload["show_evaluations_over_time"] = show_evaluations_over_time
    if summary is not None:
        input_payload["summary"] = summary
    # Validate required fields are present after merging
    validate_required_fields(
        input_payload,
        ["body_name"],
        "experiments update",
        {
            "body_name": "Workspace-unique experiment name. (required)",
        },
    )

    # The flag wins over a "name" key in the input file, like every other flag above.
    body_payload: dict[str, Any] = {**input_payload, "name": input_payload["body_name"]}
    body = build_request_body(
        ExperimentUpdateRequest, body_payload, exclude={"workspace", "body_name"}, command_name="experiments update"
    )
    kwargs = build_kwargs(name=path_name, workspace=input_payload.get("workspace"), body=body)

    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    if handle_code_generation(IntakeClient, "update_experiment", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(IntakeClient).update_experiment(**kwargs)

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )
