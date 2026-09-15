# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, Literal, cast

import typer
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.response import NemoPaginatedResponse
from nemo_platform_plugin.jobs.client import JobsClient
from nemo_platform_plugin.jobs.schemas import PlatformJobResultCreateRequest
from nemo_platform_plugin.jobs.types import (
    CreatePlatformJobRequest,
    JobLogsQueryParams,
    JobStatusDetailsUpdate,
    ListJobResultsQueryParams,
    ListJobsQueryParams,
    ListStepsQueryParams,
    PlatformJobStatusUpdateRequest,
    PlatformJobTaskUpdate,
)

from nemo_platform_ext.cli.core.api import build_kwargs, merge_filter_dict
from nemo_platform_ext.cli.core.code_generator import format_code_output
from nemo_platform_ext.cli.core.context import CLIContext
from nemo_platform_ext.cli.core.errors import handle_errors
from nemo_platform_ext.cli.core.formatters import (
    Column,
    check_output_columns_with_format,
    format_output,
    validate_stream_output_format,
)
from nemo_platform_ext.cli.core.help_formatter import collect_warnings, create_typer_app
from nemo_platform_ext.cli.core.job_log_renderer import render_job_logs
from nemo_platform_ext.cli.core.job_watch_renderer import JobWatchRenderResult, render_job_watch_events
from nemo_platform_ext.cli.core.pagination import (
    AllCursorPagesResponse,
    AllPagesResponse,
    PaginationType,
    warn_if_more_pages,
)
from nemo_platform_ext.cli.core.stdin_utils import read_data_input_with_flags, read_payload, validate_required_fields
from nemo_platform_ext.cli.core.types import (
    EntityOutputFormatOption,
    ListOutputFormatOption,
    NoTruncateOption,
    OutputColumnsOption,
    StreamOutputOption,
)
from nemo_platform_ext.cli.core.waiters import wait_for_platform_job

JobStatusValue = Literal[
    "created",
    "pending",
    "active",
    "cancelled",
    "cancelling",
    "error",
    "completed",
    "paused",
    "pausing",
    "resuming",
]

app = create_typer_app(name="jobs", help="Manage jobs.")
results_app = create_typer_app(name="results", help="Manage results")
steps_app = create_typer_app(name="steps", help="Manage steps")
tasks_app = create_typer_app(name="tasks", help="Manage tasks")

app.add_typer(results_app, name="results")
app.add_typer(steps_app, name="steps")
app.add_typer(tasks_app, name="tasks")


class _OffsetPageResponse:
    def __init__(self, items: list[Any], metadata: dict[str, Any]) -> None:
        self.data = items
        self.sort = None
        self.pagination = SimpleNamespace(**metadata)

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        return {
            "data": [_model_dump_item(item, mode=mode) for item in self.data],
            "sort": self.sort,
            "pagination": vars(self.pagination),
        }


class _CursorPageResponse:
    def __init__(self, items: list[Any], metadata: dict[str, Any]) -> None:
        self.data = items
        self.total = metadata.get("total", len(items))
        self.next_page = metadata.get("next_page")
        self.prev_page = metadata.get("prev_page")

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        return {
            "data": [_model_dump_item(item, mode=mode) for item in self.data],
            "total": self.total,
            "next_page": self.next_page,
            "prev_page": self.prev_page,
        }


def _model_dump_item(item: Any, *, mode: str) -> Any:
    if hasattr(item, "model_dump"):
        return item.model_dump(mode=mode)
    if isinstance(item, list):
        return [_model_dump_item(child, mode=mode) for child in item]
    if isinstance(item, dict):
        return {key: _model_dump_item(value, mode=mode) for key, value in item.items()}
    return item


def _jobs_client_from_state(state: CLIContext) -> tuple[NeMoPlatform, JobsClient]:
    platform_client = state.get_client()
    return platform_client, client_from_platform(platform_client, JobsClient)


def _source_filter_query(value: str | dict[str, Any] | None) -> str | None:
    if isinstance(value, dict):
        return json.dumps(value)
    return value


def _list_jobs_query_params(
    *,
    filter_value: str | None,
    page: int | None,
    page_size: int | None,
    sort: str | None,
) -> ListJobsQueryParams | None:
    query_params: ListJobsQueryParams = {}
    if filter_value is not None:
        query_params["filter"] = filter_value
    if page is not None:
        query_params["page"] = page
    if page_size is not None:
        query_params["page_size"] = page_size
    if sort is not None:
        query_params["sort"] = sort
    return query_params or None


def _list_steps_query_params(
    *,
    filter_value: str | None,
    page: int | None,
    page_size: int | None,
    sort: str | None,
) -> ListStepsQueryParams | None:
    query_params: ListStepsQueryParams = {}
    if filter_value is not None:
        query_params["filter"] = filter_value
    if page is not None:
        query_params["page"] = page
    if page_size is not None:
        query_params["page_size"] = page_size
    if sort is not None:
        query_params["sort"] = sort
    return query_params or None


def _source_offset_page(
    response: NemoPaginatedResponse[Any, Any],
    *,
    all_pages: bool,
) -> _OffsetPageResponse | AllPagesResponse:
    if not all_pages:
        page = response.page()
        return _OffsetPageResponse(list(page.items), dict(page.metadata))

    items: list[Any] = []
    total_results = 0
    total_pages = 0
    page_size: int | None = None
    for page in response.pages():
        items.extend(page.items)
        metadata = dict(page.metadata)
        total_results = int(metadata.get("total_results") or total_results)
        total_pages = int(metadata.get("total_pages") or total_pages)
        page_size = cast(int | None, metadata.get("page_size") or page_size)

    return AllPagesResponse(
        data=items,
        total_items=total_results or len(items),
        total_pages=total_pages or 1,
        page_size=page_size,
    )


def _source_cursor_page(
    response: NemoPaginatedResponse[Any, Any],
    *,
    all_pages: bool,
    limit: int | None,
) -> _CursorPageResponse | AllCursorPagesResponse:
    if not all_pages:
        page = response.page()
        return _CursorPageResponse(list(page.items), dict(page.metadata))

    items = [item for page in response.pages() for item in page.items]
    return AllCursorPagesResponse(data=items, limit=limit)


def _without_keys(payload: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in keys}


def _format_python_literal(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    return repr(value)


def handle_code_generation(
    resource_path: list[str],
    method: str,
    sdk_kwargs: dict[str, Any],
    output_format: str | None,
    context: CLIContext,
    wait_config: dict[str, Any] | None = None,
    wait_options: dict[str, Any] | None = None,
    watch_config: dict[str, Any] | None = None,
    watch_options: dict[str, Any] | None = None,
) -> bool:
    if output_format != "code":
        return False

    code = _generate_jobs_python_code(
        resource_path=resource_path,
        method=method,
        args=sdk_kwargs,
        base_url=context.get_base_url("http://localhost:8080"),
        wait_config=wait_config,
        wait_options=wait_options,
        watch_config=watch_config,
        watch_options=watch_options,
    )
    print(format_code_output(code, language="python"))
    return True


def _generate_jobs_python_code(
    *,
    resource_path: list[str],
    method: str,
    args: dict[str, Any],
    base_url: str | None,
    wait_config: dict[str, Any] | None,
    wait_options: dict[str, Any] | None,
    watch_config: dict[str, Any] | None,
    watch_options: dict[str, Any] | None,
) -> str:
    lifecycle_config = watch_config or wait_config
    lifecycle_mode = "watch" if watch_config else "wait" if wait_config else None
    lines = [
        "from pathlib import Path",
        "",
        "from nemo_platform import NeMoPlatform",
        "from nemo_platform_plugin.client.adapter import client_from_platform",
        "from nemo_platform_plugin.jobs.client import JobsClient",
        "from nemo_platform_plugin.jobs.schemas import PlatformJobResultCreateRequest",
        (
            "from nemo_platform_plugin.jobs.types import CreatePlatformJobRequest, "
            "JobStatusDetailsUpdate, PlatformJobStatusUpdateRequest, PlatformJobTaskUpdate"
        ),
    ]
    if lifecycle_mode == "wait":
        lines.append("from nemo_platform_plugin.jobs.watch_types import JobStatusEvent, JobWatchTimeoutError")
    lines.extend(
        [
            "",
            f"platform_client = NeMoPlatform(base_url={_format_python_literal(base_url)})"
            if base_url
            else "platform_client = NeMoPlatform()",
            "jobs_client = client_from_platform(platform_client, JobsClient)",
            f"args = {_format_python_literal(args)}",
            "",
            *_render_source_jobs_call(resource_path, method),
        ]
    )

    if lifecycle_config:
        lines.extend(
            [
                "",
                *_render_lifecycle_code(
                    args,
                    wait_config=wait_config,
                    wait_options=wait_options,
                    watch_config=watch_config,
                    watch_options=watch_options,
                ),
            ]
        )
    return "\n".join(lines)


def _render_source_jobs_call(resource_path: list[str], method: str) -> list[str]:
    operation = (tuple(resource_path), method)
    if operation == (("jobs",), "create"):
        return [
            "request = dict(args)",
            'workspace = request.pop("workspace", None)',
            "response = jobs_client.create_job(",
            "    workspace=workspace,",
            "    body=CreatePlatformJobRequest.model_validate(request),",
            ")",
            "print(response.data())",
        ]
    if operation == (("jobs",), "list"):
        return [
            "query_params = dict(args)",
            'workspace = query_params.pop("workspace", None)',
            "response = jobs_client.list_jobs(workspace=workspace, query_params=query_params or None)",
            "for item in response.page().items:",
            "    print(item)",
        ]
    if operation == (("jobs",), "get_logs"):
        return [
            "query_params = dict(args)",
            'workspace = query_params.pop("workspace", None)',
            'name = query_params.pop("name")',
            "response = jobs_client.list_job_logs(workspace=workspace, name=name, query_params=query_params or None)",
            "for item in response.page().items:",
            "    print(item)",
        ]
    if operation == (("jobs",), "list_execution_profiles"):
        return ["response = jobs_client.get_execution_profiles()", "for item in response.data():", "    print(item)"]
    if operation == (("jobs",), "retrieve"):
        return [
            "response = jobs_client.get_job(name=args['name'], workspace=args.get('workspace'))",
            "print(response.data())",
        ]
    if operation == (("jobs",), "delete"):
        return ["jobs_client.delete_job(name=args['name'], workspace=args.get('workspace'))"]
    if operation == (("jobs",), "cancel"):
        return [
            "response = jobs_client.cancel_job(name=args['name'], workspace=args.get('workspace'))",
            "print(response.data())",
        ]
    if operation == (("jobs",), "pause"):
        return [
            "response = jobs_client.pause_job(name=args['name'], workspace=args.get('workspace'))",
            "print(response.data())",
        ]
    if operation == (("jobs",), "resume"):
        return [
            "response = jobs_client.resume_job(name=args['name'], workspace=args.get('workspace'))",
            "print(response.data())",
        ]
    if operation == (("jobs",), "get_status"):
        return [
            "response = jobs_client.get_job_status(name=args['name'], workspace=args.get('workspace'))",
            "print(response.data())",
        ]
    if operation == (("jobs",), "update_status_details"):
        return [
            "response = jobs_client.update_job_status_details(",
            "    name=args['name'],",
            "    workspace=args.get('workspace'),",
            "    body=JobStatusDetailsUpdate.model_validate(args['body']),",
            ")",
            "print(response.data())",
        ]
    if operation == (("jobs", "results"), "create"):
        return [
            "request = {key: value for key, value in args.items() if key not in {'workspace', 'job', 'name'}}",
            "response = jobs_client.create_job_result(",
            "    workspace=args.get('workspace'),",
            "    job=args['job'],",
            "    name=args['name'],",
            "    body=PlatformJobResultCreateRequest.model_validate(request),",
            ")",
            "print(response.data())",
        ]
    if operation == (("jobs", "results"), "list"):
        return [
            "query_params = {key: value for key, value in args.items() if key not in {'workspace', 'name'}}",
            "response = jobs_client.list_job_results(",
            "    workspace=args.get('workspace'),",
            "    name=args['name'],",
            "    query_params=query_params or None,",
            ")",
            "for item in response.data().data:",
            "    print(item)",
        ]
    if operation == (("jobs", "results"), "retrieve"):
        return [
            "response = jobs_client.get_job_result(",
            "    workspace=args.get('workspace'),",
            "    job=args['job'],",
            "    name=args['name'],",
            ")",
            "print(response.data())",
        ]
    if operation == (("jobs", "results"), "download"):
        return [
            "response = jobs_client.download_job_result(",
            "    workspace=args.get('workspace'),",
            "    job=args['job'],",
            "    name=args['name'],",
            ")",
            "with Path(args['output_file']).open('wb') as output:",
            "    with response.stream() as chunks:",
            "        for chunk in chunks:",
            "            output.write(chunk)",
        ]
    if operation == (("jobs", "steps"), "list"):
        return [
            "query_params = {key: value for key, value in args.items() if key not in {'workspace', 'name'}}",
            "response = jobs_client.list_steps(",
            "    workspace=args.get('workspace'),",
            "    name=args['name'],",
            "    query_params=query_params or None,",
            ")",
            "for item in response.page().items:",
            "    print(item)",
        ]
    if operation == (("jobs", "steps"), "retrieve"):
        return [
            "response = jobs_client.get_job_step(",
            "    workspace=args.get('workspace'),",
            "    job=args['job'],",
            "    name=args['name'],",
            ")",
            "print(response.data())",
        ]
    if operation == (("jobs", "steps"), "update_status"):
        return [
            "request = {key: value for key, value in args.items() if key not in {'workspace', 'job', 'name'}}",
            "response = jobs_client.update_job_step_status(",
            "    workspace=args.get('workspace'),",
            "    job=args['job'],",
            "    name=args['name'],",
            "    body=PlatformJobStatusUpdateRequest.model_validate(request),",
            ")",
            "print(response.data())",
        ]
    if operation == (("jobs", "tasks"), "create_or_update"):
        return [
            "request = {key: value for key, value in args.items() if key not in {'workspace', 'job', 'step', 'name'}}",
            "response = jobs_client.update_job_step_task(",
            "    workspace=args.get('workspace'),",
            "    job=args['job'],",
            "    step=args['step'],",
            "    name=args['name'],",
            "    body=PlatformJobTaskUpdate.model_validate(request),",
            ")",
            "print(response.data())",
        ]
    if operation == (("jobs", "tasks"), "list"):
        return [
            "response = jobs_client.list_job_step_tasks(",
            "    workspace=args.get('workspace'),",
            "    job=args['job'],",
            "    name=args['name'],",
            ")",
            "for item in response.data().data:",
            "    print(item)",
        ]
    if operation == (("jobs", "tasks"), "retrieve"):
        return [
            "response = jobs_client.get_job_step_task(",
            "    workspace=args.get('workspace'),",
            "    job=args['job'],",
            "    step=args['step'],",
            "    name=args['name'],",
            ")",
            "print(response.data())",
        ]
    raise ValueError(f"Unsupported source-owned jobs code generation for {resource_path!r}.{method}")


def _render_lifecycle_code(
    args: dict[str, Any],
    *,
    wait_config: dict[str, Any] | None,
    wait_options: dict[str, Any] | None,
    watch_config: dict[str, Any] | None,
    watch_options: dict[str, Any] | None,
) -> list[str]:
    mode = "watch" if watch_config else "wait" if wait_config else None
    options = (watch_options if watch_config else wait_options) or {}
    timeout = options.get("timeout")
    poll_interval = options.get("poll_interval", 3)
    workspace = _format_python_literal(args["workspace"]) if args.get("workspace") is not None else "None"
    if mode == "watch":
        return [
            'resource_name = getattr(response, "name", None) or args.get("name")',
            "if not resource_name:",
            '    raise RuntimeError("Unable to determine created resource name for --watch")',
            "for event in jobs_client.watch_job(",
            "    resource_name,",
            f"    workspace={workspace},",
            f"    timeout={timeout},",
            f"    poll_interval={poll_interval},",
            "):",
            "    print(event)",
        ]
    if mode == "wait":
        resource_label = (wait_config or {}).get("resource_label", "job")
        return [
            'resource_name = getattr(response, "name", None) or args.get("name")',
            "if not resource_name:",
            '    raise RuntimeError("Unable to determine created resource name for --wait")',
            f"resource_label = {_format_python_literal(resource_label)}",
            "try:",
            "    for event in jobs_client.watch_job(",
            "        resource_name,",
            f"        workspace={workspace},",
            f"        timeout={timeout},",
            f"        poll_interval={poll_interval},",
            "        include_logs=False,",
            "    ):",
            "        if not isinstance(event, JobStatusEvent):",
            "            continue",
            "        if not event.terminal:",
            "            continue",
            "        if event.successful:",
            "            break",
            "        raise RuntimeError(",
            '            f"{resource_label.title()} {resource_name!r} ended with status {event.status!r}"',
            "        )",
            "except JobWatchTimeoutError as exc:",
            '    raise TimeoutError(f"Timed out waiting for {resource_label} {resource_name!r} to complete") from exc',
        ]
    return []


@app.command("cancel")
@collect_warnings
@handle_errors
def cancel_jobs(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Cancel a platform job."""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = {"name": name, **build_kwargs(workspace=workspace)}
    if handle_code_generation(["jobs"], "cancel", kwargs, resolved_output_format, state):
        return

    _, jobs_client = _jobs_client_from_state(state)
    result = jobs_client.cancel_job(name=name, workspace=workspace).data()

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@app.command("create")
@collect_warnings
@handle_errors
def create_jobs(
    ctx: typer.Context,
    name: Annotated[str | None, typer.Argument()] = None,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    platform_spec: Annotated[
        str | None,
        typer.Option(
            "--platform-spec",
            help="Specification for a platform job, containing steps and secrets. (JSON string) (required)",
        ),
    ] = None,
    source: Annotated[str | None, typer.Option("--source", help="(required)")] = None,
    spec: Annotated[str | None, typer.Option("--spec", help="JSON string (required)")] = None,
    custom_fields: Annotated[str | None, typer.Option("--custom-fields", help="JSON string")] = None,
    description: Annotated[str | None, typer.Option("--description")] = None,
    output_location: Annotated[str | None, typer.Option("--output-location")] = None,
    ownership: Annotated[str | None, typer.Option("--ownership", help="JSON string")] = None,
    project: Annotated[str | None, typer.Option("--project")] = None,
    input_file: Annotated[
        str | None,
        typer.Option("--input-file", help="Path to JSON file (use '-' for stdin)", rich_help_panel="Input Options"),
    ] = None,
    input_data: Annotated[
        str | None,
        typer.Option("--input-data", help="Input data for the request (JSON or YAML)", rich_help_panel="Input Options"),
    ] = None,
    output_format: EntityOutputFormatOption = None,
    wait: Annotated[
        bool,
        typer.Option(
            "--wait",
            help="Wait for the created job to reach a terminal state without streaming logs",
            rich_help_panel="Lifecycle Options",
        ),
    ] = False,
    watch: Annotated[
        bool,
        typer.Option("--watch", help="Watch the created job to a terminal state", rich_help_panel="Lifecycle Options"),
    ] = False,
    timeout: Annotated[
        int | None,
        typer.Option(
            "--timeout", min=1, help="Maximum time to wait or watch in seconds", rich_help_panel="Lifecycle Options"
        ),
    ] = None,
    poll_interval: Annotated[
        int,
        typer.Option(
            "--poll-interval", min=1, help="Seconds between status checks", rich_help_panel="Lifecycle Options"
        ),
    ] = 3,
) -> None:
    """Create a new platform job.

    [bold red]Required fields:[/] platform_spec, source, spec

    [green]Examples:[/]
    nemo jobs create <name> --input-file config.json
    nemo jobs create <name> --input-data '{"platform_spec": {}, "source": "value", "spec": {}}'
    echo '{"json": "data"}' | nemo jobs create <name> --input-file -
    nemo jobs create <name> --<option> "value"
    """
    if input_file or input_data:
        input_payload = read_data_input_with_flags(input_file=input_file, input_data=input_data)
    else:
        input_payload = {}

    if workspace is not None:
        input_payload["workspace"] = workspace
    if platform_spec is not None:
        input_payload["platform_spec"] = read_payload("platform_spec", platform_spec)
    if source is not None:
        input_payload["source"] = source
    if spec is not None:
        input_payload["spec"] = read_payload("spec", spec)
    if custom_fields is not None:
        input_payload["custom_fields"] = read_payload("custom_fields", custom_fields)
    if description is not None:
        input_payload["description"] = description
    if name is not None:
        input_payload["name"] = name
    if output_location is not None:
        input_payload["output_location"] = output_location
    if ownership is not None:
        input_payload["ownership"] = read_payload("ownership", ownership)
    if project is not None:
        input_payload["project"] = project

    validate_required_fields(
        input_payload,
        ["platform_spec", "source", "spec"],
        "jobs create",
        {
            "platform_spec": "Specification for a platform job, containing steps and secrets. (JSON string) (required)",
            "source": "(required)",
            "spec": "JSON string (required)",
        },
    )

    all_kwargs = dict(input_payload)
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    if wait and watch:
        raise typer.BadParameter("Cannot combine --wait and --watch.")

    if handle_code_generation(
        ["jobs"],
        "create",
        all_kwargs,
        resolved_output_format,
        state,
        watch_config={"type": "platform_job", "resource_label": "job"} if watch else None,
        watch_options={"timeout": timeout, "poll_interval": poll_interval} if watch else None,
        wait_config={"type": "platform_job", "resource_label": "job"} if wait else None,
        wait_options={"timeout": timeout if timeout is not None else 1200, "poll_interval": poll_interval}
        if wait
        else None,
    ):
        return

    platform_client, jobs_client = _jobs_client_from_state(state)
    request_payload = dict(all_kwargs)
    request_workspace = cast(str | None, request_payload.pop("workspace", None))
    result = jobs_client.create_job(
        workspace=request_workspace,
        body=CreatePlatformJobRequest.model_validate(request_payload),
    ).data()
    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )
    if wait or watch:
        wait_name = getattr(result, "name", None) or all_kwargs.get("name")
        if not wait_name:
            raise RuntimeError("Unable to determine created resource name for --wait/--watch")
        wait_workspace = getattr(result, "workspace", None) or request_workspace
        if wait_workspace is None:
            wait_workspace = platform_client._get_workspace_path_param()
        if wait:
            if not wait_for_platform_job(
                jobs_client,
                wait_name,
                workspace=wait_workspace,
                resource_label="job",
                timeout=timeout if timeout is not None else 1200,
                poll_interval=poll_interval,
            ):
                raise typer.Exit(1)
            return
        events = jobs_client.watch_job(
            wait_name,
            workspace=wait_workspace,
            timeout=timeout,
            poll_interval=poll_interval,
        )
        watch_result = render_job_watch_events(events, resource_label="job")
        if watch_result is JobWatchRenderResult.INTERRUPTED:
            raise typer.Exit(130)
        if watch_result is not JobWatchRenderResult.SUCCEEDED:
            raise typer.Exit(1)
        return


@app.command("delete")
@collect_warnings
@handle_errors
def delete_jobs(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
) -> None:
    """Delete a platform job."""
    state: CLIContext = ctx.obj
    kwargs = {"name": name, **build_kwargs(workspace=workspace)}
    if handle_code_generation(["jobs"], "delete", kwargs, "json", state):
        return

    _, jobs_client = _jobs_client_from_state(state)
    jobs_client.delete_job(name=name, workspace=workspace).data()

    typer.echo("✓ Deleted successfully")


@app.command("get-logs")
@collect_warnings
@handle_errors
def get_logs_jobs(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    attempt_id: Annotated[int | None, typer.Option("--attempt-id", help="Filter logs by job attempt ID")] = None,
    limit: Annotated[int | None, typer.Option("--limit", help="Maximum number of logs to return")] = None,
    page_cursor: Annotated[str | None, typer.Option("--page-cursor", help="Page cursor")] = None,
    step_id: Annotated[str | None, typer.Option("--step-id", help="Filter logs by step name")] = None,
    tail: Annotated[int | None, typer.Option("--tail", help="Number of newest log lines to return")] = None,
    task_id: Annotated[str | None, typer.Option("--task-id", help="Filter logs by task ID")] = None,
    output_format: ListOutputFormatOption = None,
    no_truncate: NoTruncateOption = None,
    columns: OutputColumnsOption = None,
    stream: StreamOutputOption = False,
    all_pages: Annotated[bool, typer.Option("--all-pages", help="Fetch all pages")] = False,
) -> None:
    """Get paginated logs for a platform job."""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)
    validate_stream_output_format(resolved_output_format, stream)

    check_output_columns_with_format(columns, resolved_output_format)

    default_columns = [
        Column("timestamp", None),
        Column("message", None),
    ]
    output_columns: str | list[Column] | None = columns
    if columns is None or str(columns).strip() == "default":
        output_columns = default_columns

    kwargs = build_kwargs(
        workspace=workspace,
        attempt_id=attempt_id,
        limit=limit,
        page_cursor=page_cursor,
        step_id=step_id,
        tail=tail,
        task_id=task_id,
    )
    codegen_kwargs = {"name": name, **kwargs}
    if handle_code_generation(["jobs"], "get_logs", codegen_kwargs, resolved_output_format, state):
        return

    _, jobs_client = _jobs_client_from_state(state)
    query_params = cast(JobLogsQueryParams, _without_keys(kwargs, {"workspace"})) or None
    response = jobs_client.list_job_logs(workspace=workspace, name=name, query_params=query_params)
    pagination_type = PaginationType.CURSOR
    items = _source_cursor_page(response, all_pages=all_pages, limit=limit)

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


@app.command("get-status")
@collect_warnings
@handle_errors
def get_status_jobs(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Get the status of a platform job."""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = {"name": name, **build_kwargs(workspace=workspace)}
    if handle_code_generation(["jobs"], "get_status", kwargs, resolved_output_format, state):
        return

    _, jobs_client = _jobs_client_from_state(state)
    result = jobs_client.get_job_status(name=name, workspace=workspace).data()

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@app.command("list")
@collect_warnings
@handle_errors
def list_jobs(
    ctx: typer.Context,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    filter: Annotated[
        str | None,
        typer.Option(
            "--filter",
            metavar="FILTER_JSON",
            help="Use --filter with JSON for complex/nested queries, or --filter.FIELD options for simple fields. Both can be combined, with field options taking precedence.\nJSON-only fields:\n  created_at: {gte: str, lte: str}\n  updated_at: {gte: str, lte: str}\n\nFilter jobs by workspace, project, name, status, source, created_at, and updated_at.",
            rich_help_panel="Filter Options",
        ),
    ] = None,
    filter_name: Annotated[str | None, typer.Option("--filter.name", rich_help_panel="Filter Options")] = None,
    filter_project: Annotated[str | None, typer.Option("--filter.project", rich_help_panel="Filter Options")] = None,
    filter_source: Annotated[str | None, typer.Option("--filter.source", rich_help_panel="Filter Options")] = None,
    filter_status: Annotated[
        list[str] | None, typer.Option("--filter.status", rich_help_panel="Filter Options")
    ] = None,
    filter_workspace: Annotated[
        str | None, typer.Option("--filter.workspace", rich_help_panel="Filter Options")
    ] = None,
    page: Annotated[int | None, typer.Option("--page", help="Page number.")] = None,
    page_size: Annotated[int | None, typer.Option("--page-size", help="Page size.")] = None,
    sort: Annotated[
        Literal["created_at", "-created_at", "updated_at", "-updated_at", "source", "-source"] | None,
        typer.Option(
            "--sort", help="The field to sort by. To sort in decreasing order, use `-` in front of the field name."
        ),
    ] = None,
    output_format: ListOutputFormatOption = None,
    no_truncate: NoTruncateOption = None,
    columns: OutputColumnsOption = None,
    stream: StreamOutputOption = False,
    all_pages: Annotated[bool, typer.Option("--all-pages", help="Fetch all pages")] = False,
) -> None:
    """List platform jobs with filtering and pagination."""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)
    validate_stream_output_format(resolved_output_format, stream)

    check_output_columns_with_format(columns, resolved_output_format)

    default_columns = [
        Column("name", None),
        Column("description", None),
        Column("status", None),
        Column("created_at", None),
    ]
    output_columns: str | list[Column] | None = columns
    if columns is None or str(columns).strip() == "default":
        output_columns = default_columns

    filter_value = _source_filter_query(
        merge_filter_dict(
            filter,
            name=filter_name,
            project=filter_project,
            source=filter_source,
            status=filter_status,
            workspace=filter_workspace,
        )
    )
    query_params = _list_jobs_query_params(filter_value=filter_value, page=page, page_size=page_size, sort=sort)
    kwargs = build_kwargs(
        workspace=workspace,
        **(query_params or {}),
    )

    if handle_code_generation(["jobs"], "list", dict(kwargs), resolved_output_format, state):
        return

    _, jobs_client = _jobs_client_from_state(state)
    response = jobs_client.list_jobs(workspace=workspace, query_params=query_params)
    pagination_type = PaginationType.PAGE_NUMBER
    items = _source_offset_page(response, all_pages=all_pages)

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


@app.command("list-execution-profiles")
@collect_warnings
@handle_errors
def list_execution_profiles_jobs(
    ctx: typer.Context,
    output_format: ListOutputFormatOption = None,
    no_truncate: NoTruncateOption = None,
    columns: OutputColumnsOption = None,
    stream: StreamOutputOption = False,
) -> None:
    """Get all currently configured execution profiles.

    Returns the capability-filtered merge from jobs config. In local standalone the
    controller may prune the shared list further after registry boot; in split
    topologies the API advertises its own merge result (not controller process
    memory)."""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)
    validate_stream_output_format(resolved_output_format, stream)

    check_output_columns_with_format(columns, resolved_output_format)

    default_columns = [
        Column("profile", None),
        Column("backend", None),
        Column("provider", None),
        Column("config", None),
    ]
    output_columns: str | list[Column] | None = columns
    if columns is None or str(columns).strip() == "default":
        output_columns = default_columns

    if handle_code_generation(["jobs"], "list_execution_profiles", {}, resolved_output_format, state):
        return

    _, jobs_client = _jobs_client_from_state(state)
    items = jobs_client.get_execution_profiles().data()

    format_output(
        items,
        is_list=True,
        output_format=resolved_output_format,
        output_columns=output_columns,
        no_truncate=state.get_no_truncate(no_truncate),
        timestamp_format=state.get_timestamp_format(),
        stream=stream,
    )


@app.command("pause")
@collect_warnings
@handle_errors
def pause_jobs(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Pause a platform job."""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = {"name": name, **build_kwargs(workspace=workspace)}
    if handle_code_generation(["jobs"], "pause", kwargs, resolved_output_format, state):
        return

    _, jobs_client = _jobs_client_from_state(state)
    result = jobs_client.pause_job(name=name, workspace=workspace).data()

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@app.command("resume")
@collect_warnings
@handle_errors
def resume_jobs(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Resume a paused platform job."""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = {"name": name, **build_kwargs(workspace=workspace)}
    if handle_code_generation(["jobs"], "resume", kwargs, resolved_output_format, state):
        return

    _, jobs_client = _jobs_client_from_state(state)
    result = jobs_client.resume_job(name=name, workspace=workspace).data()

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@app.command("get")
@collect_warnings
@handle_errors
def retrieve_jobs(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Get a platform job by name."""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = {"name": name, **build_kwargs(workspace=workspace)}
    if handle_code_generation(["jobs"], "retrieve", kwargs, resolved_output_format, state):
        return

    _, jobs_client = _jobs_client_from_state(state)
    result = jobs_client.get_job(name=name, workspace=workspace).data()

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@app.command("update-status-details")
@collect_warnings
@handle_errors
def update_status_details_jobs(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    body: Annotated[str | None, typer.Option("--body", help="JSON string (required)")] = None,
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
    """Update the status details of a platform job.

    [bold red]Required fields:[/] body

    [green]Examples:[/]
    nemo jobs update-status-details <name> --input-file config.json
    nemo jobs update-status-details <name> --input-data '{"body": {}}'
    echo '{"json": "data"}' | nemo jobs update-status-details <name> --input-file -
    nemo jobs update-status-details <name> --<option> "value"
    """
    if input_file or input_data:
        input_payload = read_data_input_with_flags(input_file=input_file, input_data=input_data)
    else:
        input_payload = {}

    if workspace is not None:
        input_payload["workspace"] = workspace
    if body is not None:
        input_payload["body"] = read_payload("body", body)

    validate_required_fields(
        input_payload,
        ["body"],
        "jobs update-status-details",
        {
            "body": "JSON string (required)",
        },
    )

    all_kwargs = {"name": name, **input_payload}
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    if handle_code_generation(["jobs"], "update_status_details", all_kwargs, resolved_output_format, state):
        return

    _, jobs_client = _jobs_client_from_state(state)
    result = jobs_client.update_job_status_details(
        name=name,
        workspace=cast(str | None, input_payload.get("workspace")),
        body=JobStatusDetailsUpdate.model_validate(input_payload["body"]),
    ).data()

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@app.command("watch")
@collect_warnings
@handle_errors
def watch_platform_job(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Name of the platform job to watch")],
    workspace: Annotated[str | None, typer.Option("--workspace", help="Workspace containing the job")] = None,
    attempt_id: Annotated[int | None, typer.Option("--attempt-id", help="Filter logs to an attempt ID")] = None,
    step_id: Annotated[str | None, typer.Option("--step-id", help="Filter logs to a step ID")] = None,
    task_id: Annotated[str | None, typer.Option("--task-id", help="Filter logs to a task ID")] = None,
    limit: Annotated[int | None, typer.Option("--limit", min=1, help="Maximum logs to fetch per page")] = None,
    timeout: Annotated[int | None, typer.Option("--timeout", min=1, help="Maximum watch time in seconds")] = None,
    poll_interval: Annotated[
        int,
        typer.Option("--poll-interval", min=1, help="Seconds between status checks"),
    ] = 3,
    include_history: Annotated[
        bool,
        typer.Option("--history/--no-history", help="Include logs already present before watching"),
    ] = True,
) -> None:
    """Watch a platform job until it reaches a terminal status."""
    state: CLIContext = ctx.obj
    platform_client, jobs_client = _jobs_client_from_state(state)
    if workspace is None:
        workspace = platform_client._get_workspace_path_param()

    events = jobs_client.watch_job(
        name,
        workspace=workspace,
        attempt_id=attempt_id,
        step_id=step_id,
        task_id=task_id,
        limit=limit,
        timeout=timeout,
        poll_interval=poll_interval,
        include_history=include_history,
    )
    watch_result = render_job_watch_events(events, resource_label="job")
    if watch_result is JobWatchRenderResult.INTERRUPTED:
        raise typer.Exit(130)
    if watch_result is not JobWatchRenderResult.SUCCEEDED:
        raise typer.Exit(1)


@app.command("tail")
@collect_warnings
@handle_errors
def tail_platform_job(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Name of the platform job whose logs to tail")],
    lines: Annotated[int, typer.Option("-n", "--lines", min=1, max=10_000, help="Number of lines to show")] = 100,
    workspace: Annotated[str | None, typer.Option("--workspace", help="Workspace containing the job")] = None,
    attempt_id: Annotated[int | None, typer.Option("--attempt-id", help="Filter logs to an attempt ID")] = None,
    step_id: Annotated[str | None, typer.Option("--step-id", help="Filter logs to a step ID")] = None,
    task_id: Annotated[str | None, typer.Option("--task-id", help="Filter logs to a task ID")] = None,
) -> None:
    """Print the newest lines from a platform job log."""
    state: CLIContext = ctx.obj
    platform_client, jobs_client = _jobs_client_from_state(state)
    if workspace is None:
        workspace = platform_client._get_workspace_path_param()

    query_params: JobLogsQueryParams = {"tail": lines}
    if attempt_id is not None:
        query_params["attempt_id"] = attempt_id
    if step_id is not None:
        query_params["step_id"] = step_id
    if task_id is not None:
        query_params["task_id"] = task_id

    response = jobs_client.list_job_logs(workspace=workspace, name=name, query_params=query_params)
    render_job_logs(response.page().items)


@results_app.command("create")
@collect_warnings
@handle_errors
def create_results(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    job: Annotated[str | None, typer.Option("--job", help="(required)")] = None,
    artifact_storage_type: Annotated[
        Literal["fileset"] | None, typer.Option("--artifact-storage-type", help="(required)")
    ] = None,
    artifact_url: Annotated[str | None, typer.Option("--artifact-url", help="(required)")] = None,
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
    """Create a new result for a job.

    [bold red]Required fields:[/] job, artifact_storage_type, artifact_url

    [green]Examples:[/]
    nemo jobs results create <name> --input-file config.json
    nemo jobs results create <name> --input-data '{"job": "value", "artifact_storage_type": "value", "artifact_url": "value"}'
    echo '{"json": "data"}' | nemo jobs results create <name> --input-file -
    nemo jobs results create <name> --<option> "value"
    """
    if input_file or input_data:
        input_payload = read_data_input_with_flags(input_file=input_file, input_data=input_data)
    else:
        input_payload = {}

    if workspace is not None:
        input_payload["workspace"] = workspace
    if job is not None:
        input_payload["job"] = job
    if artifact_storage_type is not None:
        input_payload["artifact_storage_type"] = artifact_storage_type
    if artifact_url is not None:
        input_payload["artifact_url"] = artifact_url

    validate_required_fields(
        input_payload,
        ["job", "artifact_storage_type", "artifact_url"],
        "jobs results create",
        {
            "job": "(required)",
            "artifact_storage_type": "(required)",
            "artifact_url": "(required)",
        },
    )

    all_kwargs = {"name": name, **input_payload}
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    if handle_code_generation(["jobs", "results"], "create", all_kwargs, resolved_output_format, state):
        return

    _, jobs_client = _jobs_client_from_state(state)
    result = jobs_client.create_job_result(
        workspace=cast(str | None, input_payload.get("workspace")),
        job=cast(str, input_payload["job"]),
        name=name,
        body=PlatformJobResultCreateRequest.model_validate(
            _without_keys(input_payload, {"workspace", "job"}),
        ),
    ).data()

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@results_app.command("download")
@handle_errors
def download_results(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    job: Annotated[str | None, typer.Option("--job", help="(required)")] = None,
    output_file: Annotated[
        Path | None, typer.Option("--output-file", "-o", help="Output file path (required).")
    ] = None,
) -> None:
    """Download a job result file."""
    if job is None:
        raise typer.BadParameter("--job is required")
    if output_file is None:
        raise typer.BadParameter("--output-file is required")

    state: CLIContext = ctx.obj
    kwargs = {"name": name, **build_kwargs(workspace=workspace, job=job), "output_file": str(output_file)}
    if handle_code_generation(["jobs", "results"], "download", kwargs, "json", state):
        return

    _, jobs_client = _jobs_client_from_state(state)
    response = jobs_client.download_job_result(workspace=workspace, job=job, name=name)
    with output_file.open("wb") as handle:
        with response.stream() as chunks:
            for chunk in chunks:
                handle.write(chunk)
    typer.echo(f"✓ Downloaded to {output_file!r}")


@results_app.command("list")
@collect_warnings
@handle_errors
def list_results(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    sort: Annotated[
        Literal["created_at", "-created_at", "updated_at", "-updated_at"] | None,
        typer.Option("--sort", help="The field to sort by."),
    ] = None,
    output_format: ListOutputFormatOption = None,
    no_truncate: NoTruncateOption = None,
    columns: OutputColumnsOption = None,
    stream: StreamOutputOption = False,
) -> None:
    """List results for a job."""
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

    kwargs = build_kwargs(
        workspace=workspace,
        sort=sort,
    )
    codegen_kwargs = {"name": name, **kwargs}

    if handle_code_generation(["jobs", "results"], "list", codegen_kwargs, resolved_output_format, state):
        return

    _, jobs_client = _jobs_client_from_state(state)
    query_params = cast(ListJobResultsQueryParams, _without_keys(kwargs, {"workspace"})) or None
    items = jobs_client.list_job_results(workspace=workspace, name=name, query_params=query_params).data()

    format_output(
        items,
        is_list=True,
        output_format=resolved_output_format,
        output_columns=output_columns,
        no_truncate=state.get_no_truncate(no_truncate),
        timestamp_format=state.get_timestamp_format(),
        stream=stream,
    )


@results_app.command("get")
@collect_warnings
@handle_errors
def retrieve_results(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    job: Annotated[str | None, typer.Option("--job", help="(required)")] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Get a specific job result."""
    if job is None:
        raise typer.BadParameter("--job is required")

    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = {"name": name, **build_kwargs(workspace=workspace, job=job)}
    if handle_code_generation(["jobs", "results"], "retrieve", kwargs, resolved_output_format, state):
        return

    _, jobs_client = _jobs_client_from_state(state)
    result = jobs_client.get_job_result(workspace=workspace, job=job, name=name).data()

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@steps_app.command("list")
@collect_warnings
@handle_errors
def list_steps(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    filter: Annotated[
        str | None,
        typer.Option(
            "--filter",
            metavar="FILTER_JSON",
            help="Use --filter with JSON for complex/nested queries, or --filter.FIELD options for simple fields. Both can be combined, with field options taking precedence.\nJSON-only fields:\n  status: ['created' | 'pending' | 'active' | 'cancelled' | 'cancelling' | 'error' | 'completed' | 'paused' | 'pausing' | 'resuming']\n\nFilter steps by job, status, and source.",
            rich_help_panel="Filter Options",
        ),
    ] = None,
    filter_job: Annotated[str | None, typer.Option("--filter.job", rich_help_panel="Filter Options")] = None,
    filter_source: Annotated[str | None, typer.Option("--filter.source", rich_help_panel="Filter Options")] = None,
    page: Annotated[int | None, typer.Option("--page", help="Page number.")] = None,
    page_size: Annotated[int | None, typer.Option("--page-size", help="Page size.")] = None,
    sort: Annotated[
        Literal["created_at", "-created_at", "updated_at", "-updated_at"] | None,
        typer.Option(
            "--sort", help="The field to sort by. To sort in decreasing order, use `-` in front of the field name."
        ),
    ] = None,
    output_format: ListOutputFormatOption = None,
    no_truncate: NoTruncateOption = None,
    columns: OutputColumnsOption = None,
    stream: StreamOutputOption = False,
    all_pages: Annotated[bool, typer.Option("--all-pages", help="Fetch all pages")] = False,
) -> None:
    """List job steps with pagination and filtering."""
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

    filter_value = _source_filter_query(merge_filter_dict(filter, job=filter_job, source=filter_source))
    query_params = _list_steps_query_params(filter_value=filter_value, page=page, page_size=page_size, sort=sort)
    kwargs = build_kwargs(
        workspace=workspace,
        **(query_params or {}),
    )
    codegen_kwargs = {"name": name, **kwargs}

    if handle_code_generation(["jobs", "steps"], "list", codegen_kwargs, resolved_output_format, state):
        return

    _, jobs_client = _jobs_client_from_state(state)
    response = jobs_client.list_steps(workspace=workspace, name=name, query_params=query_params)
    pagination_type = PaginationType.PAGE_NUMBER
    items = _source_offset_page(response, all_pages=all_pages)

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


@steps_app.command("get")
@collect_warnings
@handle_errors
def retrieve_steps(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    job: Annotated[str | None, typer.Option("--job", help="(required)")] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Get a specific job step."""
    if job is None:
        raise typer.BadParameter("--job is required")

    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = {"name": name, **build_kwargs(workspace=workspace, job=job)}
    if handle_code_generation(["jobs", "steps"], "retrieve", kwargs, resolved_output_format, state):
        return

    _, jobs_client = _jobs_client_from_state(state)
    result = jobs_client.get_job_step(workspace=workspace, job=job, name=name).data()

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@steps_app.command("update-status")
@collect_warnings
@handle_errors
def update_status_steps(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    job: Annotated[str | None, typer.Option("--job", help="(required)")] = None,
    status: Annotated[
        JobStatusValue | None,
        typer.Option(
            "--status",
            help="Enumeration of possible job statuses.This enum represents the various states a job can be in during its lifecycle, from creation to a terminal state. (required)",
        ),
    ] = None,
    error_details: Annotated[
        str | None,
        typer.Option("--error-details", help="Optional error details related to the status update. (JSON string)"),
    ] = None,
    status_details: Annotated[
        str | None,
        typer.Option("--status-details", help="Optional status details related to the status update. (JSON string)"),
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
    """Update a job step status.

    [bold red]Required fields:[/] job, status

    [green]Examples:[/]
    nemo jobs steps update-status <name> --input-file config.json
    nemo jobs steps update-status <name> --input-data '{"job": "value", "status": "value"}'
    echo '{"json": "data"}' | nemo jobs steps update-status <name> --input-file -
    nemo jobs steps update-status <name> --<option> "value"
    """
    if input_file or input_data:
        input_payload = read_data_input_with_flags(input_file=input_file, input_data=input_data)
    else:
        input_payload = {}

    if workspace is not None:
        input_payload["workspace"] = workspace
    if job is not None:
        input_payload["job"] = job
    if status is not None:
        input_payload["status"] = status
    if error_details is not None:
        input_payload["error_details"] = read_payload("error_details", error_details)
    if status_details is not None:
        input_payload["status_details"] = read_payload("status_details", status_details)

    validate_required_fields(
        input_payload,
        ["job", "status"],
        "jobs steps update-status",
        {
            "job": "(required)",
            "status": "Enumeration of possible job statuses.This enum represents the various states a job can be in during its lifecycle, from creation to a terminal state. (required)",
        },
    )

    all_kwargs = {"name": name, **input_payload}
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    if handle_code_generation(["jobs", "steps"], "update_status", all_kwargs, resolved_output_format, state):
        return

    _, jobs_client = _jobs_client_from_state(state)
    result = jobs_client.update_job_step_status(
        workspace=cast(str | None, input_payload.get("workspace")),
        job=cast(str, input_payload["job"]),
        name=name,
        body=PlatformJobStatusUpdateRequest.model_validate(
            _without_keys(input_payload, {"workspace", "job"}),
        ),
    ).data()

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@tasks_app.command("create-or-update")
@collect_warnings
@handle_errors
def create_or_update_tasks(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    job: Annotated[str | None, typer.Option("--job", help="(required)")] = None,
    step: Annotated[str | None, typer.Option("--step", help="(required)")] = None,
    error_details: Annotated[str | None, typer.Option("--error-details", help="JSON string")] = None,
    error_stack: Annotated[str | None, typer.Option("--error-stack")] = None,
    status: Annotated[
        JobStatusValue | None,
        typer.Option(
            "--status",
            help="Enumeration of possible job statuses.This enum represents the various states a job can be in during its lifecycle, from creation to a terminal state.",
        ),
    ] = None,
    status_details: Annotated[str | None, typer.Option("--status-details", help="JSON string")] = None,
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
    """Update a job step task.

    [bold red]Required fields:[/] job, step

    [green]Examples:[/]
    nemo jobs tasks create-or-update <name> --input-file config.json
    nemo jobs tasks create-or-update <name> --input-data '{"job": "value", "step": "value"}'
    echo '{"json": "data"}' | nemo jobs tasks create-or-update <name> --input-file -
    nemo jobs tasks create-or-update <name> --<option> "value"
    """
    if input_file or input_data:
        input_payload = read_data_input_with_flags(input_file=input_file, input_data=input_data)
    else:
        input_payload = {}

    if workspace is not None:
        input_payload["workspace"] = workspace
    if job is not None:
        input_payload["job"] = job
    if step is not None:
        input_payload["step"] = step
    if error_details is not None:
        input_payload["error_details"] = read_payload("error_details", error_details)
    if error_stack is not None:
        input_payload["error_stack"] = error_stack
    if status is not None:
        input_payload["status"] = status
    if status_details is not None:
        input_payload["status_details"] = read_payload("status_details", status_details)

    validate_required_fields(
        input_payload,
        ["job", "step"],
        "jobs tasks create-or-update",
        {
            "job": "(required)",
            "step": "(required)",
        },
    )

    all_kwargs = {"name": name, **input_payload}
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    if handle_code_generation(["jobs", "tasks"], "create_or_update", all_kwargs, resolved_output_format, state):
        return

    _, jobs_client = _jobs_client_from_state(state)
    result = jobs_client.update_job_step_task(
        workspace=cast(str | None, input_payload.get("workspace")),
        job=cast(str, input_payload["job"]),
        step=cast(str, input_payload["step"]),
        name=name,
        body=PlatformJobTaskUpdate.model_validate(
            _without_keys(input_payload, {"workspace", "job", "step"}),
        ),
    ).data()

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@tasks_app.command("list")
@collect_warnings
@handle_errors
def list_tasks(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    job: Annotated[str | None, typer.Option("--job", help="(required)")] = None,
    output_format: ListOutputFormatOption = None,
    no_truncate: NoTruncateOption = None,
    columns: OutputColumnsOption = None,
    stream: StreamOutputOption = False,
) -> None:
    """List tasks for a job step."""
    if job is None:
        raise typer.BadParameter("--job is required")

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

    kwargs = build_kwargs(
        workspace=workspace,
        job=job,
    )
    codegen_kwargs = {"name": name, **kwargs}

    if handle_code_generation(["jobs", "tasks"], "list", codegen_kwargs, resolved_output_format, state):
        return

    _, jobs_client = _jobs_client_from_state(state)
    items = jobs_client.list_job_step_tasks(workspace=workspace, job=job, name=name).data()

    format_output(
        items,
        is_list=True,
        output_format=resolved_output_format,
        output_columns=output_columns,
        no_truncate=state.get_no_truncate(no_truncate),
        timestamp_format=state.get_timestamp_format(),
        stream=stream,
    )


@tasks_app.command("get")
@collect_warnings
@handle_errors
def retrieve_tasks(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    job: Annotated[str | None, typer.Option("--job", help="(required)")] = None,
    step: Annotated[str | None, typer.Option("--step", help="(required)")] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Get a specific job step task."""
    if job is None:
        raise typer.BadParameter("--job is required")
    if step is None:
        raise typer.BadParameter("--step is required")

    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = {"name": name, **build_kwargs(workspace=workspace, job=job, step=step)}
    if handle_code_generation(["jobs", "tasks"], "retrieve", kwargs, resolved_output_format, state):
        return

    _, jobs_client = _jobs_client_from_state(state)
    result = jobs_client.get_job_step_task(workspace=workspace, job=job, step=step, name=name).data()

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )
