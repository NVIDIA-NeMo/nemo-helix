# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo agents skills``: publish, read, tag and delete stored skills.

A skill is published from a directory in a fileset (``--fileset``). ``--from-dir`` uploads a local
directory into a fileset first, at a path named by a hash of its contents, so an unchanged
directory reuses the same upload and a changed one never mixes with stale files.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Annotated, Any, Optional

import click
import typer
from nemo_helix_plugin.agents.client import AgentsClient
from nemo_helix_plugin.agents.skills import (
    SKILL_NAME_MAX_LENGTH,
    ListSkillRevisionsQueryParams,
    ListSkillsQueryParams,
    SkillInput,
    SkillSource,
    SkillTagQueryParams,
)
from nemo_helix_plugin.cli import create_typer_app
from nemo_helix_plugin.cli_codegen import handle_code_generation
from nemo_helix_plugin.cli_confirm import confirm_destructive
from nemo_helix_plugin.cli_error_handling import handle_errors
from nemo_helix_plugin.cli_kwargs import merge_filter_dict
from nemo_helix_plugin.cli_options import (
    AllPagesOption,
    EntityOutputFormat,
    EntityOutputFormatOption,
    ListOutputFormatOption,
    NoTruncateOption,
    OutputColumnsOption,
    WorkspaceOption,
    YesOption,
)
from nemo_helix_plugin.cli_output import Column, check_output_columns_with_format, format_output
from nemo_helix_plugin.cli_pagination import PaginationType, collect_offset_pages, warn_if_more_pages
from nemo_helix_plugin.cli_state import CLIState, cli_state, resolve_cli_workspace, resolve_output_format
from nemo_helix_plugin.cli_warnings import collect_warnings
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.skill_archive import DEFAULT_LIMITS, EXCLUDED_TOP_LEVEL_DIRS

#: Fileset that ``--from-dir`` uploads into when ``--upload-fileset`` is not given.
DEFAULT_UPLOAD_FILESET = "skill-sources"

_LIST_COLUMNS = [
    Column("name"),
    Column("revision"),
    Column("frontmatter.description", header="description"),
    Column("frontmatter.tags", header="frontmatter tags"),
    Column("tags", header="revision tags"),
    Column("updated_at"),
]
_REVISION_COLUMNS = [
    Column("revision"),
    Column("content_hash"),
    Column("tags"),
    Column("published_at"),
]

NameArgument = Annotated[
    str,
    typer.Argument(
        help=f"Skill name (starts with a letter; lowercase, hyphenated; 2-{SKILL_NAME_MAX_LENGTH} characters)."
    ),
]
FromDirOption = Annotated[
    Optional[Path],
    typer.Option(
        "--from-dir",
        help="Local skill directory (holding SKILL.md) to upload and publish. Files are uploaded as regular "
        "files: the Files service does not keep file modes, so executable bits are not preserved.",
        file_okay=False,
        dir_okay=True,
        exists=True,
        resolve_path=True,
    ),
]
FilesetOption = Annotated[
    Optional[str],
    typer.Option(
        "--fileset",
        help="Skill directory already in a fileset you can read: workspace/fileset#path/to/skill.",
    ),
]
UploadFilesetOption = Annotated[
    Optional[str],
    typer.Option(
        "--upload-fileset",
        help=f"With --from-dir: the fileset in the skill's workspace to upload into (default "
        f"'{DEFAULT_UPLOAD_FILESET}', created if missing).",
    ),
]
TasksetOption = Annotated[
    Optional[str],
    typer.Option(
        "--taskset",
        help="Linked taskset revision, digest-pinned and fully qualified: workspace/taskset#<64-hex digest>.",
    ),
]
TagOption = Annotated[
    Optional[list[str]],
    typer.Option("--tag", "-t", help="Revision tag to point at the published revision; repeatable."),
]


def _upload_dir(files: FilesClient, directory: Path, *, workspace: str, fileset: str, skill: str) -> str:
    """Upload ``directory`` under ``<skill>/<content hash>/`` in ``fileset``; return its fileset reference.

    Applies the archive rules before anything is uploaded: a top-level ``evals/`` directory is left
    out (it is test material the archive excludes), symlinks are refused, and the archive limits on
    entries, path length, file size and total size are checked from ``stat``. Files are hashed in
    chunks and read one at a time for upload, so memory stays bounded by the largest file.
    """
    limits = DEFAULT_LIMITS
    members: list[tuple[str, Path]] = []
    entries = 0
    total = 0
    for root, dirnames, filenames in os.walk(directory, followlinks=False):
        root_path = Path(root)
        if root_path == directory:
            dirnames[:] = [d for d in dirnames if d not in EXCLUDED_TOP_LEVEL_DIRS]
        dirnames.sort()
        for name in [*dirnames, *sorted(filenames)]:
            path = root_path / name
            relative = path.relative_to(directory).as_posix()
            if path.is_symlink():
                raise click.BadParameter(
                    f"{relative!r} is a symlink; skills hold regular files and directories only",
                    param_hint="'--from-dir'",
                )
            entries += 1
            if entries > limits.max_entries:
                raise click.BadParameter(f"more than {limits.max_entries} entries", param_hint="'--from-dir'")
            if len(f"{skill}/{relative}".encode()) > limits.max_path_length:
                raise click.BadParameter(f"{relative!r} is too long a path", param_hint="'--from-dir'")
            if name in filenames:
                if not path.is_file():
                    raise click.BadParameter(f"{relative!r} is not a regular file", param_hint="'--from-dir'")
                size = path.stat().st_size
                total += size
                if size > limits.max_file_bytes or total > limits.max_total_payload_bytes:
                    raise click.BadParameter(f"{relative!r} exceeds the skill size limits", param_hint="'--from-dir'")
                members.append((relative, path))
    members.sort()
    if not any(relative == "SKILL.md" for relative, _ in members):
        raise click.BadParameter(f"{str(directory)!r} has no SKILL.md at its root", param_hint="'--from-dir'")

    digest = hashlib.sha256()
    for relative, path in members:
        file_digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                file_digest.update(chunk)
        digest.update(relative.encode() + b"\0" + file_digest.digest())
    prefix = f"{skill}/{digest.hexdigest()}"

    files.create_fileset(workspace=workspace, body=CreateFilesetRequest(name=fileset), exist_ok=True).data()
    for relative, path in members:
        files.upload_file(
            workspace=workspace, name=fileset, path=f"{prefix}/{relative}", content=path.read_bytes()
        ).data()
    return f"{workspace}/{fileset}#{prefix}"


def _skill_input(
    state: CLIState,
    *,
    name: str,
    workspace: str,
    from_dir: Path | None,
    fileset: str | None,
    upload_fileset: str | None,
    taskset: str | None,
    tags: list[str] | None,
    output_format: str | None,
) -> SkillInput:
    if from_dir is not None and fileset is not None:
        raise click.UsageError("Pass exactly one of --from-dir or --fileset.")
    if upload_fileset is not None and from_dir is None:
        raise click.UsageError("--upload-fileset applies only with --from-dir.")
    if fileset is None:
        if from_dir is None:
            raise click.UsageError("Pass exactly one of --from-dir or --fileset.")
        if output_format == "code":
            raise click.UsageError("--output-format code cannot express the --from-dir upload; use --fileset.")
        fileset = _upload_dir(
            state.typed_client(FilesClient),
            from_dir,
            workspace=workspace,
            fileset=upload_fileset or DEFAULT_UPLOAD_FILESET,
            skill=name,
        )
    return SkillInput(source=SkillSource(fileset=fileset), taskset=taskset, tags=tags or [])


def build_skills_app() -> typer.Typer:
    """The ``skills`` command group, mounted under ``nemo agents``."""
    app = create_typer_app(name="skills", help="Manage stored agent skills and their revisions.")

    def _publish(
        ctx: typer.Context,
        method: str,
        *,
        name: str,
        workspace: str | None,
        from_dir: Path | None,
        fileset: str | None,
        upload_fileset: str | None,
        taskset: str | None,
        tags: list[str] | None,
        output_format: EntityOutputFormat | None,
    ) -> None:
        state = cli_state(ctx)
        resolved_output_format = resolve_output_format(ctx, output_format)
        resolved_workspace = resolve_cli_workspace(ctx, workspace)
        body = _skill_input(
            state,
            name=name,
            workspace=resolved_workspace,
            from_dir=from_dir,
            fileset=fileset,
            upload_fileset=upload_fileset,
            taskset=taskset,
            tags=tags,
            output_format=resolved_output_format,
        )
        kwargs = {"workspace": resolved_workspace, "name": name, "body": body}
        if handle_code_generation(AgentsClient, method, kwargs, resolved_output_format, state):
            return
        format_output(getattr(state.typed_client(AgentsClient), method)(**kwargs), output_format=resolved_output_format)

    @app.command("create")
    @collect_warnings
    @handle_errors
    def create(
        ctx: typer.Context,
        name: NameArgument,
        from_dir: FromDirOption = None,
        fileset: FilesetOption = None,
        upload_fileset: UploadFilesetOption = None,
        taskset: TasksetOption = None,
        tag: TagOption = None,
        workspace: WorkspaceOption = None,
        output_format: EntityOutputFormatOption = None,
    ) -> None:
        """Create a skill and publish revision 1. Fails if the skill exists; use `publish` for later revisions.

        The SKILL.md frontmatter `name` must equal the skill name.
        """
        _publish(
            ctx,
            "create_skill",
            name=name,
            workspace=workspace,
            from_dir=from_dir,
            fileset=fileset,
            upload_fileset=upload_fileset,
            taskset=taskset,
            tags=tag,
            output_format=output_format,
        )

    @app.command("publish")
    @collect_warnings
    @handle_errors
    def publish(
        ctx: typer.Context,
        name: NameArgument,
        from_dir: FromDirOption = None,
        fileset: FilesetOption = None,
        upload_fileset: UploadFilesetOption = None,
        taskset: TasksetOption = None,
        tag: TagOption = None,
        workspace: WorkspaceOption = None,
        output_format: EntityOutputFormatOption = None,
    ) -> None:
        """Publish the skill's next revision, creating the skill if absent.

        Content identical to the latest revision cuts no new revision; the tags are still applied.
        """
        _publish(
            ctx,
            "publish_skill",
            name=name,
            workspace=workspace,
            from_dir=from_dir,
            fileset=fileset,
            upload_fileset=upload_fileset,
            taskset=taskset,
            tags=tag,
            output_format=output_format,
        )

    @app.command("get")
    @collect_warnings
    @handle_errors
    def get(
        ctx: typer.Context,
        name: NameArgument,
        revision: Annotated[
            Optional[str],
            typer.Option("--revision", "-r", help="A revision tag or content digest; default: latest."),
        ] = None,
        workspace: WorkspaceOption = None,
        output_format: EntityOutputFormatOption = None,
    ) -> None:
        """Get a skill at its latest revision, or at the revision a tag or digest names."""
        state = cli_state(ctx)
        resolved_output_format = resolve_output_format(ctx, output_format)
        kwargs: dict[str, Any] = {"workspace": resolve_cli_workspace(ctx, workspace), "name": name}
        method = "get_skill"
        if revision is not None:
            method = "get_skill_revision"
            kwargs["revision"] = revision
        if handle_code_generation(AgentsClient, method, kwargs, resolved_output_format, state):
            return
        format_output(getattr(state.typed_client(AgentsClient), method)(**kwargs), output_format=resolved_output_format)

    @app.command("list")
    @collect_warnings
    @handle_errors
    def list_skills(
        ctx: typer.Context,
        tag: Annotated[
            Optional[list[str]],
            typer.Option(
                "--tag",
                "-t",
                help="Only skills with a published revision whose frontmatter metadata.tags has this tag; repeatable.",
            ),
        ] = None,
        filter_expression: Annotated[
            Optional[str],
            typer.Option(
                "--filter",
                help='Filter as JSON, e.g. \'{"description": {"$like": "pdf"}}\'. Fields: name, description, tags.',
            ),
        ] = None,
        sort: Annotated[
            Optional[str], typer.Option("--sort", help="name, created_at or updated_at; prefix '-' to descend.")
        ] = None,
        page: Annotated[Optional[int], typer.Option("--page", help="Page number (1-indexed).")] = None,
        page_size: Annotated[Optional[int], typer.Option("--page-size", help="Items per page.")] = None,
        all_pages: AllPagesOption = False,
        workspace: WorkspaceOption = None,
        output_format: ListOutputFormatOption = None,
        no_truncate: NoTruncateOption = None,
        columns: OutputColumnsOption = None,
    ) -> None:
        """List skills, each at its latest revision. Filters match the frontmatter of any published revision."""
        state = cli_state(ctx)
        resolved_output_format = resolve_output_format(ctx, output_format)
        check_output_columns_with_format(columns, resolved_output_format)
        query_params: ListSkillsQueryParams = {}
        merged = merge_filter_dict(filter_expression)
        if tag:
            # Combine with $and rather than adding a sibling key, so a JSON filter with a top-level
            # logical operator ($or, $and, $not) stays valid.
            tag_filter = {"tags": {"$in": tag}}
            if merged is None:
                merged = tag_filter
            elif isinstance(merged, dict):
                merged = {"$and": [merged, tag_filter]}
            else:
                raise click.UsageError("--tag cannot be combined with a text filter; pass --filter as JSON.")
        if merged:
            query_params["filter"] = merged if isinstance(merged, str) else json.dumps(merged)
        if sort is not None:
            query_params["sort"] = sort
        if page is not None:
            query_params["page"] = page
        if page_size is not None:
            query_params["page_size"] = page_size
        kwargs = {"workspace": resolve_cli_workspace(ctx, workspace), "query_params": query_params or None}
        if handle_code_generation(
            AgentsClient,
            "list_skills",
            kwargs,
            resolved_output_format,
            state,
            result="all-pages" if all_pages else "list",
        ):
            return
        items = collect_offset_pages(state.typed_client(AgentsClient).list_skills(**kwargs), all_pages=all_pages)
        format_output(
            items,
            is_list=True,
            output_format=resolved_output_format,
            output_columns=columns if columns and columns.strip() != "default" else _LIST_COLUMNS,
            no_truncate=state.get_no_truncate(no_truncate),
            timestamp_format=state.get_timestamp_format(),
        )
        if not all_pages:
            warn_if_more_pages(items, PaginationType.PAGE_NUMBER)

    @app.command("revisions")
    @collect_warnings
    @handle_errors
    def revisions(
        ctx: typer.Context,
        name: NameArgument,
        page: Annotated[Optional[int], typer.Option("--page", help="Page number (1-indexed).")] = None,
        page_size: Annotated[Optional[int], typer.Option("--page-size", help="Items per page.")] = None,
        all_pages: AllPagesOption = False,
        workspace: WorkspaceOption = None,
        output_format: ListOutputFormatOption = None,
        no_truncate: NoTruncateOption = None,
        columns: OutputColumnsOption = None,
    ) -> None:
        """List a skill's published revisions, newest first."""
        state = cli_state(ctx)
        resolved_output_format = resolve_output_format(ctx, output_format)
        check_output_columns_with_format(columns, resolved_output_format)
        query_params: ListSkillRevisionsQueryParams = {}
        if page is not None:
            query_params["page"] = page
        if page_size is not None:
            query_params["page_size"] = page_size
        kwargs = {
            "workspace": resolve_cli_workspace(ctx, workspace),
            "name": name,
            "query_params": query_params or None,
        }
        if handle_code_generation(
            AgentsClient,
            "list_skill_revisions",
            kwargs,
            resolved_output_format,
            state,
            result="all-pages" if all_pages else "list",
        ):
            return
        items = collect_offset_pages(
            state.typed_client(AgentsClient).list_skill_revisions(**kwargs), all_pages=all_pages
        )
        format_output(
            items,
            is_list=True,
            output_format=resolved_output_format,
            output_columns=columns if columns and columns.strip() != "default" else _REVISION_COLUMNS,
            no_truncate=state.get_no_truncate(no_truncate),
            timestamp_format=state.get_timestamp_format(),
        )
        if not all_pages:
            warn_if_more_pages(items, PaginationType.PAGE_NUMBER)

    @app.command("tag")
    @collect_warnings
    @handle_errors
    def tag(
        ctx: typer.Context,
        name: NameArgument,
        tag: Annotated[str, typer.Argument(help="Tag to point; 'latest' is reserved.")],
        revision: Annotated[
            str, typer.Option("--revision", "-r", help="Revision to tag: a content digest or another tag.")
        ],
        workspace: WorkspaceOption = None,
        output_format: EntityOutputFormatOption = None,
    ) -> None:
        """Point a tag at an already-published revision."""
        state = cli_state(ctx)
        resolved_output_format = resolve_output_format(ctx, output_format)
        query_params: SkillTagQueryParams = {"revision": revision}
        kwargs = {
            "workspace": resolve_cli_workspace(ctx, workspace),
            "name": name,
            "tag": tag,
            "query_params": query_params,
        }
        if handle_code_generation(AgentsClient, "tag_skill_revision", kwargs, resolved_output_format, state):
            return
        format_output(
            state.typed_client(AgentsClient).tag_skill_revision(**kwargs), output_format=resolved_output_format
        )

    @app.command("delete")
    @collect_warnings
    @handle_errors
    def delete(
        ctx: typer.Context,
        name: NameArgument,
        workspace: WorkspaceOption = None,
        yes: YesOption = False,
    ) -> None:
        """Delete a skill and all its revisions."""
        state = cli_state(ctx)
        kwargs = {"workspace": resolve_cli_workspace(ctx, workspace), "name": name}
        confirm_destructive(f"Delete skill '{name}' and all its revisions?", yes=yes)
        state.typed_client(AgentsClient).delete_skill(**kwargs).data()
        typer.echo(f"Skill '{name}' deleted.")

    return app
