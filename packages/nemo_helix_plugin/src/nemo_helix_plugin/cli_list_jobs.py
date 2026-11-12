# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""List jobs by job type: ``nemo <plugin> list-jobs`` and ``nemo jobs list --type``.

Every :class:`~nemo_helix_plugin.job.NemoJob` a plugin mounts with
:func:`~nemo_helix_plugin.jobs.routes.add_job_routes` gets its own list route, which returns only
jobs of that type. These commands call that route, not the core Jobs list route.

When several job types are selected, each type's route is queried and the results are merged in
sort order (newest first by default). Selecting multiple job types also affects pagination:
- ``--page-size`` effectively caps the length of the (merged) list
- ``--all-pages`` fetches everything
- ``--page`` is not supported

:func:`add_list_jobs_command` is called by :func:`~nemo_helix_plugin.commands.add_job_commands`
for every plugin with jobs. A plugin can customize its ``list-jobs`` command by registering its
own (in its `NemoCLI.get_cli` implementation); it should use :func:`run_list_jobs` and option aliases
in this module to maintain some consistency with the rest of the CLI.
"""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, Self

import click
import typer
from nemo_helix_plugin.cli_codegen import handle_code_generation
from nemo_helix_plugin.cli_error_handling import handle_errors
from nemo_helix_plugin.cli_options import (
    AllPagesOption,
    ListOutputFormatOption,
    NoTruncateOption,
    OutputColumnsOption,
    WorkspaceOption,
)
from nemo_helix_plugin.cli_output import Column, check_output_columns_with_format, format_output
from nemo_helix_plugin.cli_pagination import (
    AllPagesResponse,
    OffsetPageResponse,
    PaginationType,
    collect_offset_pages,
    warn_if_more_pages,
)
from nemo_helix_plugin.cli_state import cli_state, resolve_cli_workspace, resolve_output_format
from nemo_helix_plugin.cli_warnings import add_warning, collect_warnings
from nemo_helix_plugin.client.errors import NotFoundError, PermissionDeniedError
from nemo_helix_plugin.job import NemoJob, job_collection_path_for
from nemo_helix_plugin.plugin_jobs.client import PluginJobsClient
from nemo_helix_plugin.plugin_jobs.types import ListPluginJobsQueryParams, PluginJob

#: Name of the per-plugin command.
LIST_JOBS_COMMAND = "list-jobs"

JobsSort = Literal["created_at", "-created_at", "updated_at", "-updated_at"]

_DEFAULT_SORT: JobsSort = "-created_at"

#: Default table columns. ``type`` is added by the CLI so a merged list shows each job's type.
DEFAULT_JOB_COLUMNS = [Column("name"), Column("type"), Column("status"), Column("created_at")]

_FILTER_PANEL = "Filter Options"

JobFilterOption = Annotated[
    str | None,
    typer.Option(
        "--filter",
        metavar="FILTER_JSON",
        help='Filter as a JSON object (e.g. \'{"name": {"$like": "nightly"}}\') or a filter expression. '
        "The field options below are merged into a JSON filter and take precedence.",
        rich_help_panel=_FILTER_PANEL,
    ),
]
JobFilterNameOption = Annotated[str | None, typer.Option("--filter.name", rich_help_panel=_FILTER_PANEL)]
JobFilterProjectOption = Annotated[str | None, typer.Option("--filter.project", rich_help_panel=_FILTER_PANEL)]
JobFilterStatusOption = Annotated[
    list[str] | None,
    typer.Option(
        "--filter.status",
        help="Only jobs in this status (e.g. active, completed). Repeat for several.",
        rich_help_panel=_FILTER_PANEL,
    ),
]
JobSortOption = Annotated[
    JobsSort | None,
    typer.Option("--sort", help="Sort field. Prefix with '-' for descending. Default: -created_at."),
]
PageOption = Annotated[int | None, typer.Option("--page", help="Page number. Needs exactly one job type.", min=1)]
PageSizeOption = Annotated[int | None, typer.Option("--page-size", help="Number of jobs per page.", min=1)]


@dataclass(frozen=True, slots=True)
class JobCollection:
    """Where one job type's jobs are listed."""

    #: The ``{api}`` segment of the route: the plugin, taken from the entry-point key's prefix.
    api: str
    #: The job's :attr:`~nemo_helix_plugin.job.NemoJob.name`, which is also its submit command.
    name: str
    #: The collection path below ``/v2/workspaces/{workspace}/``, e.g. ``jobs/create``.
    path: str

    @classmethod
    def for_job(cls, key: str, job_cls: type[NemoJob]) -> Self:
        """The collection *job_cls* is mounted at, given its ``nemo.jobs`` entry-point *key*.

        Only the key's plugin prefix is used; the job's own ``name`` names the type, so the
        type always matches the submit command even if the key were registered differently.
        """
        api, separator, _ = key.partition(".")
        if not separator or not api:
            raise ValueError(f"Job entry-point key {key!r} is not '<plugin>.<job>'")
        return cls(api=api, name=job_cls.name, path=job_collection_path_for(job_cls).strip("/"))

    @property
    def job_type(self) -> str:
        """The job type, ``<plugin>.<name>`` (``data-designer.create``)."""
        return f"{self.api}.{self.name}"


def job_collections(jobs: Mapping[str, object]) -> list[JobCollection]:
    """Collections for the :class:`~nemo_helix_plugin.job.NemoJob` classes in *jobs*.

    *jobs* maps ``nemo.jobs`` entry-point keys to the loaded values; values that are not
    ``NemoJob`` subclasses (post-run actions, functions) have no job collection and are skipped.
    So are keys without a ``<plugin>.`` prefix, which don't say which API serves the job.
    """
    return [
        JobCollection.for_job(key, value)
        for key, value in jobs.items()
        if isinstance(value, type) and issubclass(value, NemoJob) and key.partition(".")[0] and "." in key
    ]


def select_job_types(collections: Sequence[JobCollection], names: Sequence[str] | None) -> list[JobCollection]:
    """Narrow *collections* to the job types *names* (a job name or a full ``<plugin>.<job>`` type).

    No *names* selects every collection. Raises :class:`click.BadParameter` for an unknown name.
    """
    if not names:
        return list(collections)
    by_name = {collection.name: collection for collection in collections}
    by_name.update({collection.job_type: collection for collection in collections})
    selected: dict[str, JobCollection] = {}
    for name in names:
        collection = by_name.get(name)
        if collection is None:
            raise click.BadParameter(
                f"Unknown job type {name!r}. Choose from: {', '.join(c.name for c in collections)}.",
                param_hint="'--type'",
            )
        selected[collection.job_type] = collection
    return list(selected.values())


def resolve_job_types(types: Sequence[str]) -> list[JobCollection]:
    """Resolve ``nemo jobs list --type`` values against the installed ``nemo.jobs`` entry points.

    A value is a plugin (``evals``, every job type that plugin registers) or one job type
    (``evals.evaluate``, ``<plugin>.<NemoJob.name>``). Only the named plugin's entry points are
    loaded; entries that are not jobs are skipped. When a plugin is named, an entry that fails to
    load is skipped with a warning; when a job type is named and isn't found, the error lists
    any that failed to load, since the type may be one of them.
    """
    from nemo_helix_plugin.discovery import discover_entry_points

    entry_points = discover_entry_points("nemo.jobs")
    collections: dict[str, JobCollection] = {}
    for value in types:
        plugin, separator, _ = value.partition(".")
        keys = [key for key in entry_points if key.startswith(f"{plugin}.")]
        if not keys:
            plugins = sorted({key.partition(".")[0] for key in entry_points})
            raise click.BadParameter(
                f"No installed plugin or job type {value!r}. Plugins with jobs: {', '.join(plugins) or 'none'}.",
                param_hint="'--type'",
            )
        found, failed = _load_job_collections({key: entry_points[key] for key in keys})
        if not separator:
            for key, exc in failed.items():
                add_warning(f"Skipping job type {key!r}: it failed to load ({exc}).")
            matches = found
        else:
            matches = [collection for collection in found if collection.job_type == value]
            if not matches:
                known = ", ".join(collection.job_type for collection in found) or "none"
                message = f"Unknown job type {value!r}. {plugin} job types: {known}."
                if failed:
                    message += f" Failed to load: {', '.join(f'{key} ({exc})' for key, exc in failed.items())}."
                raise click.BadParameter(message, param_hint="'--type'")
        for collection in matches:
            collections.setdefault(collection.job_type, collection)
    return list(collections.values())


def _load_job_collections(entry_points: Mapping[str, Any]) -> tuple[list[JobCollection], dict[str, Exception]]:
    """Load *entry_points*; return the job collections and the keys that failed to load."""
    loaded: dict[str, object] = {}
    failed: dict[str, Exception] = {}
    for key, entry_point in entry_points.items():
        try:
            loaded[key] = entry_point.load()
        except Exception as exc:
            failed[key] = exc
    return job_collections(loaded), failed


def _with_type(item: PluginJob, job_type: str) -> PluginJob:
    return PluginJob.model_validate({**item.model_dump(mode="json", exclude_unset=True), "type": job_type})


def _sort_key(sort: JobsSort) -> tuple[str, bool]:
    return sort.removeprefix("-"), sort.startswith("-")


def _merge(items: Sequence[PluginJob], sort: JobsSort) -> list[PluginJob]:
    """Order jobs from several collections like the server orders one; jobs missing the field go last.

    A timestamp without a timezone is taken as UTC, so collections that serialize differently
    still compare.
    """
    field, descending = _sort_key(sort)

    def _timestamp(item: PluginJob) -> datetime:
        value: datetime = getattr(item, field)
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)

    present = [item for item in items if getattr(item, field) is not None]
    missing = [item for item in items if getattr(item, field) is None]
    present.sort(key=_timestamp, reverse=descending)
    return present + missing


def _query_params(
    *, query_filter: str | None, sort: JobsSort | None, page: int | None, page_size: int | None
) -> ListPluginJobsQueryParams:
    """The list route's query, with only the values the user set."""
    query_params: ListPluginJobsQueryParams = {}
    if query_filter is not None:
        query_params["filter"] = query_filter
    if sort is not None:
        query_params["sort"] = sort
    if page is not None:
        query_params["page"] = page
    if page_size is not None:
        query_params["page_size"] = page_size
    return query_params


def list_jobs_in_collections(
    client: PluginJobsClient,
    collections: Sequence[JobCollection],
    *,
    workspace: str,
    query_filter: str | None = None,
    sort: JobsSort | None = None,
    page: int | None = None,
    page_size: int | None = None,
    all_pages: bool = False,
) -> OffsetPageResponse | AllPagesResponse:
    """List the jobs in *collections*, each tagged with its job ``type``.

    One collection is passed straight through, so its pagination is exactly the server's. Several
    are fetched one by one and merged in *sort* order: the first *page_size* jobs overall, or all of
    them with *all_pages*. A collection that answers 404 (the plugin's service may not be enabled)
    or 403 (job types can be authorized separately) is skipped with a warning, unless every one
    does. Any other error fails the listing, so a partial list is never mistaken for a full one.
    """
    if not collections:
        raise click.UsageError("No job types to list.")
    if page is not None and len(collections) > 1:
        raise click.BadParameter("--page needs exactly one job type; pass --type.", param_hint="'--page'")

    query_params = _query_params(query_filter=query_filter, sort=sort, page=page, page_size=page_size)

    if len(collections) == 1:
        (collection,) = collections
        response = client.list_plugin_jobs(
            api=collection.api, collection=collection.path, workspace=workspace, query_params=query_params or None
        )
        listing = collect_offset_pages(response, all_pages=all_pages)
        listing.data = [_with_type(item, collection.job_type) for item in listing.data]
        return listing

    items: list[PluginJob] = []
    total_results = 0
    server_page_size: int | None = None
    skipped: list[NotFoundError | PermissionDeniedError] = []
    for collection in collections:
        try:
            response = client.list_plugin_jobs(
                api=collection.api, collection=collection.path, workspace=workspace, query_params=query_params or None
            )
            listing = collect_offset_pages(response, all_pages=all_pages)
        except NotFoundError as exc:
            skipped.append(exc)
            add_warning(
                f"Skipping job type {collection.job_type!r}: the platform has no /{collection.path} route for it."
            )
            continue
        except PermissionDeniedError as exc:
            skipped.append(exc)
            add_warning(f"Skipping job type {collection.job_type!r}: you don't have permission to list its jobs.")
            continue
        items.extend(_with_type(item, collection.job_type) for item in listing.data)
        total_results += int(getattr(listing.pagination, "total_results", 0) or len(listing.data))
        server_page_size = server_page_size or getattr(listing.pagination, "page_size", None)
    if len(skipped) == len(collections):
        raise skipped[0]

    merged = _merge(items, sort or _DEFAULT_SORT)
    if all_pages:
        return AllPagesResponse(
            data=merged,
            total_items=len(merged),
            total_pages=1,
            page_size=len(merged) or None,
            envelope={"sort": sort or _DEFAULT_SORT},
        )
    limit = page_size or server_page_size or len(merged)
    shown = merged[:limit]
    return OffsetPageResponse(
        shown,
        {
            "page": 1,
            "page_size": limit,
            "current_page_size": len(shown),
            "total_pages": max(1, math.ceil(total_results / limit)) if limit else 1,
            "total_results": total_results,
        },
        {"sort": sort or _DEFAULT_SORT},
    )


def job_status_filter(statuses: Sequence[str] | None) -> str | dict[str, list[str]] | None:
    """The ``status`` filter value for repeated ``--filter.status``: one value as is, several as ``$in``."""
    if not statuses:
        return None
    if len(statuses) == 1:
        return statuses[0]
    return {"$in": list(statuses)}


def build_job_filter(
    filter_json: str | None,
    *,
    name: str | None = None,
    project: str | None = None,
    status: Sequence[str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str | None:
    """Combine ``--filter`` with ``--filter.FIELD`` options (and *extra* fields) into one query value."""
    from nemo_helix_plugin.cli_kwargs import merge_filter_dict

    merged = merge_filter_dict(filter_json, name=name, project=project, status=job_status_filter(status))
    if extra:
        if isinstance(merged, str):
            raise click.UsageError("A filter expression can't be combined with these options; pass --filter as JSON.")
        merged = {**(merged or {}), **extra}
    if merged is None:
        return None
    return merged if isinstance(merged, str) else json.dumps(merged)


def run_list_jobs(
    ctx: typer.Context,
    collections: Sequence[JobCollection],
    *,
    workspace: str | None = None,
    query_filter: str | None = None,
    sort: JobsSort | None = None,
    page: int | None = None,
    page_size: int | None = None,
    all_pages: bool = False,
    output_format: ListOutputFormatOption = None,
    no_truncate: bool | None = None,
    columns: str | None = None,
    stream: bool = False,
) -> None:
    """List *collections* and print them the way every ``nemo`` list command does."""
    state = cli_state(ctx)
    resolved_output_format = resolve_output_format(ctx, output_format)
    check_output_columns_with_format(columns, resolved_output_format)
    resolved_workspace = resolve_cli_workspace(ctx, workspace)

    if resolved_output_format == "code":
        if len(collections) != 1:
            raise click.UsageError("--output-format code lists one job type; pass a single --type.")
        query_params = _query_params(query_filter=query_filter, sort=sort, page=page, page_size=page_size)
        (collection,) = collections
        kwargs: dict[str, Any] = {"api": collection.api, "collection": collection.path, "workspace": resolved_workspace}
        if query_params:
            kwargs["query_params"] = query_params
        handle_code_generation(
            PluginJobsClient,
            "list_plugin_jobs",
            kwargs,
            resolved_output_format,
            state,
            result="all-pages" if all_pages else "list",
        )
        return

    listing = list_jobs_in_collections(
        state.typed_client(PluginJobsClient),
        collections,
        workspace=resolved_workspace,
        query_filter=query_filter,
        sort=sort,
        page=page,
        page_size=page_size,
        all_pages=all_pages,
    )
    output_columns: str | list[Column] = (
        DEFAULT_JOB_COLUMNS if columns is None or columns.strip() == "default" else columns
    )
    format_output(
        listing,
        is_list=True,
        output_format=resolved_output_format,
        output_columns=output_columns,
        no_truncate=state.get_no_truncate(no_truncate),
        timestamp_format=state.get_timestamp_format(),
        stream=stream,
    )
    if not all_pages:
        warn_if_more_pages(listing, PaginationType.PAGE_NUMBER)


def has_command(app: typer.Typer, name: str) -> bool:
    """Whether *app* already registers a command or group called *name*."""
    for info in app.registered_commands:
        command_name = info.name or getattr(info.callback, "__name__", "").replace("_", "-")
        if command_name == name:
            return True
    return any(
        group.name == name or (group.typer_instance is not None and group.typer_instance.info.name == name)
        for group in app.registered_groups
    )


def add_list_jobs_command(app: typer.Typer, jobs: Mapping[str, object], *, rich_help_panel: str = "Jobs") -> None:
    """Register ``list-jobs`` on a plugin's *app* for the jobs in *jobs* (entry-point key → class).

    Does nothing when *jobs* holds no job collections or *app* already has a ``list-jobs`` command,
    which is how a plugin replaces the generated one.
    """
    collections = job_collections(jobs)
    if not collections or has_command(app, LIST_JOBS_COMMAND):
        return
    type_names = ", ".join(collection.name for collection in collections)

    def list_jobs(
        ctx: typer.Context,
        job_type: Annotated[
            list[str] | None,
            typer.Option(
                "--type",
                "-t",
                metavar="TYPE",
                help=f"Only list jobs of this type, one of: {type_names}. Repeat for several. Default: all.",
            ),
        ] = None,
        workspace: WorkspaceOption = None,
        filter_json: JobFilterOption = None,
        filter_name: JobFilterNameOption = None,
        filter_project: JobFilterProjectOption = None,
        filter_status: JobFilterStatusOption = None,
        sort: JobSortOption = None,
        page: PageOption = None,
        page_size: PageSizeOption = None,
        all_pages: AllPagesOption = False,
        output_format: ListOutputFormatOption = None,
        no_truncate: NoTruncateOption = None,
        columns: OutputColumnsOption = None,
    ) -> None:
        run_list_jobs(
            ctx,
            select_job_types(collections, job_type),
            workspace=workspace,
            query_filter=build_job_filter(filter_json, name=filter_name, project=filter_project, status=filter_status),
            sort=sort,
            page=page,
            page_size=page_size,
            all_pages=all_pages,
            output_format=output_format,
            no_truncate=no_truncate,
            columns=columns,
        )

    app.command(
        LIST_JOBS_COMMAND,
        help=f"List this plugin's jobs, newest first. Job types: {type_names}.",
        rich_help_panel=rich_help_panel,
    )(collect_warnings(handle_errors(list_jobs)))


__all__ = [
    "DEFAULT_JOB_COLUMNS",
    "LIST_JOBS_COMMAND",
    "JobCollection",
    "JobFilterNameOption",
    "JobFilterOption",
    "JobFilterProjectOption",
    "JobFilterStatusOption",
    "JobSortOption",
    "JobsSort",
    "PageOption",
    "PageSizeOption",
    "add_list_jobs_command",
    "build_job_filter",
    "has_command",
    "job_collections",
    "job_status_filter",
    "list_jobs_in_collections",
    "resolve_job_types",
    "run_list_jobs",
    "select_job_types",
]
