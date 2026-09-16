# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI integration for NemoJob and NemoFunction primitives.

Two parallel helpers, one per primitive:

- :func:`add_job_commands` — three-verb subgroups
  (``run``/``submit``/``explain``) per :class:`~nemo_platform_plugin.job.NemoJob`,
  routed through :class:`~nemo_platform_plugin.scheduler.NemoJobScheduler`.
- :func:`add_function_commands` — two-verb subgroups
  (``run``/``submit``) per :class:`~nemo_platform_plugin.function.NemoFunction`.
  Functions don't have an ``explain`` verb because their only schema
  is :attr:`~nemo_platform_plugin.function.NemoFunction.spec_schema` and that's
  introspected through ``--help`` directly.

:func:`add_job_commands` is the bridge between the ``nemo.jobs`` and
``nemo.cli`` surfaces. The platform calls it at startup for each plugin that
has registered both a CLI group and jobs, injecting a generated **sub-group**
for every job into the plugin's :class:`typer.Typer` group. Each sub-group
exposes three verbs — ``run``, ``submit``, ``explain`` — matching
:class:`~nemo_platform_plugin.scheduler.NemoJobScheduler`.

Plugin authors do **not** call this themselves — it is called automatically
by the platform's CLI loader. The result is that each job becomes available
as::

    nemo <plugin> <job-name> run      [--config ...] [--config-file ...]
    nemo <plugin> <job-name> submit   [--profile ...] [--cluster ...] [-o ...]
    nemo <plugin> <job-name> explain  [--profile ...] [--cluster ...]

The **bare form** ``nemo <plugin> <job-name>`` prints usage and exits with
status 1. No implicit default verb — the submitter's choice of execution
target is always explicit. This breaks the previous one-line form; the
fix is typing ``run`` (or ``submit``) explicitly.

Phase 1 MR 1.2c delivers the CLI shape. ``submit`` and ``explain`` delegate
to :class:`NemoJobScheduler` stubs that raise
:class:`NotImplementedError`; MR 1.3 and MR 1.4 wire them. ``run`` works
end-to-end today.

Generated command interface
---------------------------

``run``
    Execute the job in-process. Accepts ``--config <json>`` (default ``{}``)
    and ``--config-file <path>`` (takes precedence over ``--config``). The
    scheduler validates against :attr:`~nemo_platform_plugin.job.NemoJob.spec_schema`
    / :attr:`~nemo_platform_plugin.job.NemoJob.input_spec_schema` when declared.

``submit``
    Submit the job to a cluster. Phase 1 MR 1.3 wires this.

``explain``
    Print the job's spec / options schemas. Phase 1 MR 1.4 wires this.

Example
-------

Given::

    class SayHelloJob(NemoJob):
        name = "say-hello"
        description = "Greet a name."

        def run(self, config: dict) -> dict:
            return {"result": f"Hello, {config.get('name', 'world')}!"}

The platform generates::

    $ nemo example say-hello run --config '{"name": "Claude"}'
    {
      "result": "Hello, Claude!"
    }

    $ nemo example say-hello
    Usage: nemo example say-hello [OPTIONS] COMMAND [ARGS]...
    ...
    $ echo $?
    1
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, Optional, cast

import httpx
import typer
from nemo_platform_plugin._spec_flags import (
    UNSET,
    SpecLeafField,
    build_callback_signature,
    build_epilog,
    build_overlay,
    deep_merge,
    kw,
    walk_spec_leaves,
)
from nemo_platform_plugin.cli import NemoCLI
from nemo_platform_plugin.cli_errors import print_http_request_error, print_http_status_error
from nemo_platform_plugin.cli_renderer import CLIRenderer, RendererContext
from nemo_platform_plugin.cli_state import resolve_local_cli_sdks
from nemo_platform_plugin.function import NemoFunction, returns_async_iterator
from nemo_platform_plugin.function_context import FunctionContext
from nemo_platform_plugin.functions.routes import DEFAULT_FUNCTION_PATH, NDJSON_MEDIA_TYPE
from nemo_platform_plugin.job import NemoJob
from nemo_platform_plugin.jobs._cli_options import (
    load_options_file,
    load_spec_file,
    merge_options,
    parse_dotted_kv_list,
)
from nemo_platform_plugin.jobs.telemetry import merge_job_telemetry_custom_fields
from nemo_platform_plugin.run_dependencies import LocalRunError
from nemo_platform_plugin.scheduler import NemoJobScheduler
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

# Submit host resolution for both functions and jobs:
# ``--base-url`` > ``--cluster`` (resolved from CLI config) > active context
# base URL > ``$NMP_BASE_URL`` > localhost.
_DEFAULT_BASE_URL_ENV_VAR = "NMP_BASE_URL"
_DEFAULT_BASE_URL = "http://localhost:8080"

# Rich-help panels for the function and job CLIs. Mirror PR #160's
# panel names but scoped per primitive so jobs and functions sharing a
# plugin still group cleanly when both are documented in --help.
_PANEL_SPEC_SOURCE: str = "Spec Source"
_PANEL_FUNCTION_SPEC: str = "Function Spec"
_PANEL_JOB_SPEC: str = "Job Spec"
_PANEL_SUBMISSION: str = "Submission"

# Static flag names declared by each function verb. Spec fields whose
# names collide with these are dropped from the auto-generated flag
# set; they remain reachable via ``--spec`` / ``--spec-file``. ``run``
# is deliberately permissive — only the JSON-passing flags and the
# context-input ``workspace`` flag are reserved. ``submit`` layers on
# the URL-routing flags + the request-id surface.
_FN_RUN_RESERVED_FLAGS: frozenset[str] = frozenset({"spec", "spec_file", "workspace"})
_FN_SUBMIT_RESERVED_FLAGS: frozenset[str] = _FN_RUN_RESERVED_FLAGS | frozenset({"cluster", "base_url", "request_id"})

# Static flag names declared by each job verb. ``run`` reserves the
# spec-source flags plus the deprecated ``--config`` / ``--config-file``
# aliases (kept during the rename transition). ``submit`` layers on the
# submission-routing flags (``--profile`` / ``--cluster`` / ``--base-url``
# / ``--workspace``) and the options-passthrough flags (``-o`` /
# ``--options-file``). Reserved spec fields remain reachable via
# ``--spec`` / ``--spec-file`` JSON.
_JOB_RUN_RESERVED_FLAGS: frozenset[str] = frozenset({"spec", "spec_file", "config", "config_file"})
_JOB_SUBMIT_RESERVED_FLAGS: frozenset[str] = _JOB_RUN_RESERVED_FLAGS | frozenset(
    {"options", "options_file", "profile", "cluster", "base_url", "workspace"}
)


# ---------------------------------------------------------------------------
# Renderer plumbing
# ---------------------------------------------------------------------------


def _output_format_is_json(typer_ctx: typer.Context | None) -> bool:
    """Return True when the global ``--output-format json`` flag is set.

    Read from ``typer_ctx.obj.overrides`` — the ``CLIContext`` populated by
    the top-level :func:`nemo_platform_ext.cli.app.main` callback. Defensive
    against non-``CLIContext`` objects so plugins exercising the CLI directly
    in tests (with ``ctx.obj = None``) still work.
    """
    if typer_ctx is None:
        return False
    state = typer_ctx.obj
    if state is None:
        return False
    overrides = getattr(state, "overrides", None)
    if not isinstance(overrides, dict):
        return False
    return overrides.get("output_format") == "json"


def _make_renderer_context(
    *,
    cli_kwargs: Mapping[str, Any],
    verb: Literal["run", "submit"],
    is_local: bool,
) -> RendererContext:
    """Build the per-invocation :class:`RendererContext` passed to renderer methods."""
    from rich.console import Console

    return RendererContext(console=Console(), cli_kwargs=cli_kwargs, verb=verb, is_local=is_local)


async def _drive_async_renderer(
    stream: AsyncIterator[Any],
    renderer_cls: type[CLIRenderer],
    *,
    rctx: RendererContext,
) -> CLIRenderer:
    """Drive *renderer_cls*'s ``on_start`` + ``on_frame`` over an async stream.

    Returns the renderer instance so the sync caller (the one calling
    :func:`asyncio.run`) can fire :meth:`CLIRenderer.on_complete` *after*
    the loop tears down — important because renderers commonly use
    event-loop-bound libraries (e.g. ``prompt_toolkit.Application.run``)
    in ``on_complete``, and those use their own :func:`asyncio.run` which
    raises ``RuntimeError: asyncio.run() cannot be called from a running
    event loop`` when nested.

    :meth:`CLIRenderer.on_error` is fired in-loop because the exception
    is most usefully handled where it surfaced; the exception still
    propagates so the caller sees it.
    """
    renderer = renderer_cls()
    renderer.on_start(ctx=rctx)
    try:
        async for frame in stream:
            renderer.on_frame(frame, ctx=rctx)
    except BaseException as exc:
        renderer.on_error(exc, ctx=rctx)
        raise
    return renderer


def _drive_sync_renderer(
    line_iter: Iterable[str],
    renderer_cls: type[CLIRenderer],
    *,
    rctx: RendererContext,
) -> CLIRenderer:
    """Drive *renderer_cls*'s ``on_start`` + ``on_frame`` over a sync line iterator.

    Returns the renderer; the caller fires :meth:`CLIRenderer.on_complete`.
    Symmetric with :func:`_drive_async_renderer` so all three drivers obey
    the same contract: in-loop work happens here (``on_start`` / per-frame
    / ``on_error``), the post-stream summary happens at the call site after
    this returns. That contract matters even for sync drivers: it keeps the
    plugin-author mental model consistent and avoids re-introducing
    nested-event-loop hazards if a sync driver is ever wrapped in an async
    context.

    Each non-empty NDJSON line is parsed as JSON and passed to
    :meth:`CLIRenderer.on_frame`. Lines that fail to parse pass through as
    raw strings — renderers can decide how to handle non-JSON output.
    """
    renderer = renderer_cls()
    renderer.on_start(ctx=rctx)
    try:
        for line in line_iter:
            if not line:
                continue
            try:
                frame: object = json.loads(line)
            except json.JSONDecodeError:
                frame = line
            renderer.on_frame(frame, ctx=rctx)
    except BaseException as exc:
        renderer.on_error(exc, ctx=rctx)
        raise
    return renderer


def _drive_single_value_renderer(
    callable_: Callable[[], Any],
    renderer_cls: type[CLIRenderer],
    *,
    rctx: RendererContext,
) -> CLIRenderer:
    """Drive *renderer_cls* around a synchronous, non-streaming result.

    Used for jobs (which return a single dict). The result is passed to
    :meth:`CLIRenderer.on_frame` once. Returns the renderer so the caller
    can fire :meth:`CLIRenderer.on_complete` — same contract as the
    streaming drivers.
    """
    renderer = renderer_cls()
    renderer.on_start(ctx=rctx)
    try:
        result = callable_()
        renderer.on_frame(result, ctx=rctx)
    except BaseException as exc:
        renderer.on_error(exc, ctx=rctx)
        raise
    return renderer


def add_job_commands(
    cli_app: typer.Typer,
    jobs: dict[str, type[NemoJob]],
    *,
    cli: NemoCLI | None = None,
) -> None:
    """Inject three-verb subcommand groups for each job into *cli_app*.

    Each entry in *jobs* produces one :class:`typer.Typer` sub-group
    registered under :attr:`~nemo_platform_plugin.job.NemoJob.name` in a ``"Jobs"``
    rich help panel. The sub-group owns ``run`` / ``submit`` / ``explain``
    commands.

    Args:
        cli_app: The plugin's :class:`typer.Typer` group to inject
            sub-groups into. Typically the value returned by
            :meth:`~nemo_platform_plugin.cli.NemoCLI.get_cli`.
        jobs: Mapping of entry-point key → :class:`~nemo_platform_plugin.job.NemoJob`
            subclass, already filtered to the relevant plugin. Typically a
            subset of the dict returned by
            :func:`~nemo_platform_plugin.discovery.discover_jobs`.
        cli: Optional plugin :class:`~nemo_platform_plugin.cli.NemoCLI` instance. When
            supplied, :meth:`~nemo_platform_plugin.cli.NemoCLI.update_job_cli` is
            invoked once per job after default verb registration and before
            the sub-group is mounted, giving the plugin a chance to amend
            the auto-generated CLI surface.
    """
    scheduler = NemoJobScheduler()
    for job_cls in jobs.values():
        _register_job_subgroup(cli_app, job_cls, scheduler, cli=cli)


def _register_job_subgroup(
    cli_app: typer.Typer,
    job_cls: type[NemoJob],
    scheduler: NemoJobScheduler,
    *,
    cli: NemoCLI | None = None,
) -> None:
    """Register a ``<job-name>`` sub-group with run / submit / explain verbs."""
    if not job_cls.generate_legacy_verbs:
        _add_submit_command(cli_app, job_cls, scheduler, cli=cli, command_name=job_cls.name, rich_help_panel="Jobs")
        if cli is not None:
            cli.update_job_cli(job_cls, cli_app)
        return

    job_group = typer.Typer(
        name=job_cls.name,
        help=job_cls.description or f"Manage the {job_cls.name} job.",
        no_args_is_help=False,  # we override bare behavior to exit non-zero
    )

    @job_group.callback(invoke_without_command=True)
    def _root(ctx: typer.Context) -> None:  # pragma: no cover - trivial delegation
        """Stub callback — prints usage and exits 1 when no verb is given."""
        if ctx.invoked_subcommand is None:
            typer.echo(ctx.get_help())
            raise typer.Exit(code=1)

    _add_run_command(job_group, job_cls, scheduler, cli=cli)
    _add_submit_command(job_group, job_cls, scheduler, cli=cli)
    _add_explain_command(job_group, job_cls, scheduler)

    if cli is not None:
        cli.update_job_cli(job_cls, job_group)

    cli_app.add_typer(job_group, name=job_cls.name, rich_help_panel="Jobs")


# ---------------------------------------------------------------------------
# run — local execution, wired via NemoJobScheduler.run_local
# ---------------------------------------------------------------------------


def _job_input_schema(job_cls: type[NemoJob]) -> type[BaseModel] | None:
    """Pick the schema that the submitter's input is validated against.

    Mirrors the precedence used by
    :meth:`~nemo_platform_plugin.scheduler.NemoJobScheduler._validate_and_compile`:
    :attr:`~nemo_platform_plugin.job.NemoJob.input_spec_schema` when declared,
    else :attr:`~nemo_platform_plugin.job.NemoJob.spec_schema`. Both ``run`` and
    ``submit`` accept the same shape from the user, so we walk a single
    schema for both verbs. Returns ``None`` when neither attribute is
    declared — the auto-flag generator treats ``None`` as "no leaves",
    keeping legacy schema-less jobs unchanged.
    """
    return job_cls.input_spec_schema or job_cls.spec_schema


def _add_run_command(
    group: typer.Typer,
    job_cls: type[NemoJob],
    scheduler: NemoJobScheduler,
    *,
    cli: NemoCLI | None = None,
) -> None:
    """Register the ``run`` verb. Generates per-field flags from the input schema.

    Each scalar leaf in :func:`_job_input_schema` becomes a Typer option
    named after its dotted path, kebab-cased per segment (``--name``,
    ``--target.url``).  Precedence at runtime is ``--spec-file`` (base)
    → ``--spec`` JSON (overlay) → per-field flags (top overlay), with
    the deprecated ``--config`` / ``--config-file`` aliases routed
    through the same precedence as ``--spec`` / ``--spec-file``.

    When *cli* supplies a renderer via ``get_job_renderer(verb="run")`` (and
    ``--output-format json`` is not set), the renderer's lifecycle wraps the
    synchronous ``run_local`` call: ``on_start`` → ``on_frame(result)`` →
    ``on_complete``. The default-printer fallback echoes the dict result
    when no renderer is supplied.
    """
    schema = _job_input_schema(job_cls)
    unavailable: list[str] = []
    leaves = walk_spec_leaves(schema, reserved=_JOB_RUN_RESERVED_FLAGS, unavailable=unavailable)

    def _run(typer_ctx: typer.Context, **kwargs: object) -> None:
        original_kwargs = dict(kwargs)
        spec_str: str = cast(str, kwargs.pop("spec", "{}"))
        spec_file: Path | None = cast("Path | None", kwargs.pop("spec_file", None))
        config: str | None = cast("str | None", kwargs.pop("config", None))
        config_file: Path | None = cast("Path | None", kwargs.pop("config_file", None))

        effective_spec_str = config if config is not None else spec_str
        effective_spec_file = config_file if config_file is not None else spec_file
        base = _load_spec(effective_spec_str, effective_spec_file)
        overlay = build_overlay(leaves, kwargs, unset_sentinel=UNSET)
        data = deep_merge(base, overlay)
        logger.debug("Running job %r locally with spec %r", job_cls.name, data)
        sdk, async_sdk = resolve_local_cli_sdks(typer_ctx)

        renderer_cls: type[CLIRenderer] | None = None
        if cli is not None and not _output_format_is_json(typer_ctx):
            renderer_cls = cli.get_job_renderer(job_cls, verb="run")

        def _do_run() -> Any:
            return scheduler.run_local(job_cls, data, sdk=sdk, async_sdk=async_sdk)

        renderer: CLIRenderer | None = None
        rctx: RendererContext | None = None
        try:
            if renderer_cls is not None:
                rctx = _make_renderer_context(
                    cli_kwargs=original_kwargs,
                    verb="run",
                    is_local=True,
                )
                renderer = _drive_single_value_renderer(_do_run, renderer_cls, rctx=rctx)
            else:
                result = _do_run()
                typer.echo(json.dumps(result, indent=2))
        except LocalRunError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc

        # on_complete fires after the driver returns, in the same contract
        # the async renderer driver follows — see _drive_sync_renderer.
        if renderer is not None and rctx is not None:
            renderer.on_complete(ctx=rctx)

    help_text = "Run locally, in-process."
    epilog = build_epilog(schema=schema, leaves=leaves, kind="Job", unavailable=unavailable)
    setattr(_run, "__signature__", _build_job_run_signature(leaves))
    group.command(name="run", help=help_text, epilog=epilog)(_run)


def _build_job_run_signature(leaves: list[SpecLeafField]) -> inspect.Signature:
    """Compose the synthetic signature for the job ``run`` verb.

    Static flags (``--spec`` / ``--spec-file``) come first under the
    ``Spec Source`` panel, followed by the auto-generated per-field
    flags under ``Job Spec``, then the hidden ``--config`` /
    ``--config-file`` deprecated aliases.
    """
    static_params = [
        # ``typer.Context`` is auto-injected by Click via ``pass_context``,
        # which passes it as the first positional argument. Hand-built
        # rather than via ``kw()`` because ``kw()`` only emits
        # ``KEYWORD_ONLY`` params (which Click's positional injection
        # would reject).
        inspect.Parameter(
            "typer_ctx",
            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=typer.Context,
        ),
        kw(
            "spec",
            str,
            typer.Option(
                "{}",
                "--spec",
                help="Spec as a JSON string.",
                rich_help_panel=_PANEL_SPEC_SOURCE,
            ),
        ),
        kw(
            "spec_file",
            Optional[Path],
            typer.Option(
                None,
                "--spec-file",
                help="Path to a YAML or JSON spec file (used as base; per-flag values override).",
                rich_help_panel=_PANEL_SPEC_SOURCE,
            ),
        ),
    ]
    # Deprecated aliases — kept for the transition period (MR 1.3b).
    # Trailing so they sort below the auto-generated panel in --help,
    # though they're hidden anyway and only matter for backwards-compatible
    # invocations.
    trailing_params = [
        kw(
            "config",
            Optional[str],
            typer.Option(
                None,
                "--config",
                help="(Deprecated — use --spec.) Spec as a JSON string.",
                hidden=True,
            ),
        ),
        kw(
            "config_file",
            Optional[Path],
            typer.Option(
                None,
                "--config-file",
                help="(Deprecated — use --spec-file.) Path to a JSON spec file.",
                hidden=True,
            ),
        ),
    ]
    return build_callback_signature(
        static_params,
        leaves,
        rich_help_panel=_PANEL_JOB_SPEC,
        trailing_params=trailing_params,
    )


# ---------------------------------------------------------------------------
# submit — stub delegate to scheduler.submit_remote (wired in MR 1.3)
# ---------------------------------------------------------------------------


def _add_submit_command(
    group: typer.Typer,
    job_cls: type[NemoJob],
    scheduler: NemoJobScheduler,
    *,
    cli: NemoCLI | None = None,
    command_name: str = "submit",
    rich_help_panel: str | None = None,
) -> None:
    """Register the ``submit`` verb. Generates per-field flags + static submit flags.

    Static flags (``--spec`` / ``--spec-file``, ``-o`` / ``--options-file``,
    ``--profile``, ``--cluster``, ``--base-url``, ``--workspace``, plus the
    hidden ``--config`` / ``--config-file`` deprecated aliases) come first;
    spec fields whose names collide with any of those are dropped from
    the auto-generated set and remain reachable via ``--spec`` /
    ``--spec-file``.

    When *cli* supplies a renderer via ``get_job_renderer(verb="submit")``
    (and ``--output-format json`` is not set), the renderer's lifecycle
    wraps the ``submit_remote`` call: ``on_start`` → ``on_frame(result)`` →
    ``on_complete``.
    """
    schema = _job_input_schema(job_cls)
    unavailable: list[str] = []
    leaves = walk_spec_leaves(schema, reserved=_JOB_SUBMIT_RESERVED_FLAGS, unavailable=unavailable)

    def _submit(typer_ctx: typer.Context, **kwargs: object) -> None:
        original_kwargs = dict(kwargs)
        spec_str: str = cast(str, kwargs.pop("spec", "{}"))
        spec_file: Path | None = cast("Path | None", kwargs.pop("spec_file", None))
        options: list[str] = cast("list[str]", kwargs.pop("options", []))
        options_file: Path | None = cast("Path | None", kwargs.pop("options_file", None))
        profile: str | None = cast("str | None", kwargs.pop("profile", None))
        cluster: str | None = cast("str | None", kwargs.pop("cluster", None))
        base_url: str | None = cast("str | None", kwargs.pop("base_url", None))
        workspace: str = cast(str, kwargs.pop("workspace", "default"))
        config: str | None = cast("str | None", kwargs.pop("config", None))
        config_file: Path | None = cast("Path | None", kwargs.pop("config_file", None))

        effective_spec_str = config if config is not None else spec_str
        effective_spec_file = config_file if config_file is not None else spec_file
        base = _load_spec(effective_spec_str, effective_spec_file)
        overlay = build_overlay(leaves, kwargs, unset_sentinel=UNSET)
        spec_data = deep_merge(base, overlay)

        try:
            merged_options = _merge_options_inputs(options, options_file)
        except (OSError, ValueError) as exc:
            typer.echo(f"Error: invalid options — {exc}", err=True)
            raise typer.Exit(code=1) from exc

        renderer_cls: type[CLIRenderer] | None = None
        if cli is not None and not _output_format_is_json(typer_ctx):
            renderer_cls = cli.get_job_renderer(job_cls, verb="submit")

        def _do_submit() -> Any:
            submit_kwargs: dict[str, Any] = {
                "base_url": _resolve_submit_base_url(typer_ctx, base_url=base_url, cluster=cluster),
                "workspace": workspace,
                "profile": profile,
                "options": merged_options or None,
                "headers": _resolve_submit_auth_headers(typer_ctx) or None,
            }
            metadata = _resolve_submit_metadata(typer_ctx)
            if metadata is not None:
                submit_kwargs["metadata"] = metadata
            return scheduler.submit_remote(job_cls, spec_data, **submit_kwargs)

        renderer: CLIRenderer | None = None
        rctx: RendererContext | None = None
        try:
            if renderer_cls is not None:
                rctx = _make_renderer_context(
                    cli_kwargs=original_kwargs,
                    verb="submit",
                    is_local=False,
                )
                renderer = _drive_single_value_renderer(_do_submit, renderer_cls, rctx=rctx)
            else:
                result = _do_submit()
                typer.echo(json.dumps(result, indent=2))
        except (NotImplementedError, ValueError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=2) from exc
        except httpx.HTTPStatusError as exc:
            print_http_status_error(exc, action=f"submit {job_cls.name}")
            raise typer.Exit(code=2) from exc
        except httpx.RequestError as exc:
            print_http_request_error(exc, action=f"submit {job_cls.name}")
            raise typer.Exit(code=2) from exc
        except httpx.HTTPError as exc:
            typer.echo(f"Error: submit {job_cls.name} failed: {exc}", err=True)
            raise typer.Exit(code=2) from exc

        if renderer is not None and rctx is not None:
            renderer.on_complete(ctx=rctx)

    help_text = job_cls.description if command_name == job_cls.name and job_cls.description else "Submit to a cluster."
    epilog = build_epilog(schema=schema, leaves=leaves, kind="Job", unavailable=unavailable)
    setattr(_submit, "__signature__", _build_job_submit_signature(leaves))
    group.command(name=command_name, help=help_text, epilog=epilog, rich_help_panel=rich_help_panel)(_submit)


def _resolve_submit_metadata(typer_ctx: typer.Context) -> dict[str, object] | None:
    state = typer_ctx.obj
    if state is None:
        return None
    getter = getattr(state, "get_job_telemetry_custom_fields", None)
    if not callable(getter):
        return None
    try:
        telemetry_custom_fields = getter()
    except Exception:
        return None
    if not isinstance(telemetry_custom_fields, dict) or not telemetry_custom_fields:
        return None
    return {"custom_fields": merge_job_telemetry_custom_fields(None, telemetry_custom_fields)}


def _build_job_submit_signature(leaves: list[SpecLeafField]) -> inspect.Signature:
    """Compose the synthetic signature for the job ``submit`` verb."""
    static_params = [
        inspect.Parameter(
            "typer_ctx",
            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=typer.Context,
        ),
        kw(
            "spec",
            str,
            typer.Option(
                "{}",
                "--spec",
                help="Spec as a JSON string.",
                rich_help_panel=_PANEL_SPEC_SOURCE,
            ),
        ),
        kw(
            "spec_file",
            Optional[Path],
            typer.Option(
                None,
                "--spec-file",
                help="Path to a YAML or JSON spec file (used as base; per-flag values override).",
                rich_help_panel=_PANEL_SPEC_SOURCE,
            ),
        ),
        kw(
            "options",
            list[str],
            typer.Option(
                [],  # noqa: B006 — Typer evaluates the default lazily per invocation
                "-o",
                help="Backend option override, 'backend.key=value' (repeatable).",
                rich_help_panel=_PANEL_SUBMISSION,
            ),
        ),
        kw(
            "options_file",
            Optional[Path],
            typer.Option(
                None,
                "--options-file",
                help="Path to a YAML or JSON options file (nested by backend name).",
                rich_help_panel=_PANEL_SUBMISSION,
            ),
        ),
        kw(
            "profile",
            Optional[str],
            typer.Option(
                None,
                "--profile",
                help=(
                    "Execution profile (operator-configured). Required unless the active "
                    "cluster has a 'default' profile."
                ),
                rich_help_panel=_PANEL_SUBMISSION,
            ),
        ),
        kw(
            "cluster",
            Optional[str],
            typer.Option(
                None,
                "--cluster",
                help="Configured cluster name to resolve via the NeMo CLI config.",
                rich_help_panel=_PANEL_SUBMISSION,
            ),
        ),
        kw(
            "base_url",
            Optional[str],
            typer.Option(
                None,
                "--base-url",
                help=("Explicit plugin-service base URL (overrides --cluster and all other submit host resolution)."),
                rich_help_panel=_PANEL_SUBMISSION,
            ),
        ),
        kw(
            "workspace",
            str,
            typer.Option(
                "default",
                "--workspace",
                help="Workspace scope for the submission.",
                rich_help_panel=_PANEL_SUBMISSION,
            ),
        ),
    ]
    trailing_params = [
        kw(
            "config",
            Optional[str],
            typer.Option(
                None,
                "--config",
                help="(Deprecated — use --spec.) Spec as a JSON string.",
                hidden=True,
            ),
        ),
        kw(
            "config_file",
            Optional[Path],
            typer.Option(
                None,
                "--config-file",
                help="(Deprecated — use --spec-file.) Path to a JSON spec file.",
                hidden=True,
            ),
        ),
    ]
    return build_callback_signature(
        static_params,
        leaves,
        rich_help_panel=_PANEL_JOB_SPEC,
        trailing_params=trailing_params,
    )


# ---------------------------------------------------------------------------
# explain — stub delegate to scheduler.explain (wired in MR 1.4)
# ---------------------------------------------------------------------------


def _add_explain_command(
    group: typer.Typer,
    job_cls: type[NemoJob],
    scheduler: NemoJobScheduler,
) -> None:
    def _explain(
        profile: Optional[str] = typer.Option(
            None,
            "--profile",
            help="Annotate the bundle with this profile. Profile metadata lands in MR 1.4b.",
        ),
        cluster: Optional[str] = typer.Option(
            None,
            "--cluster",
            help="Accepted for forward compatibility; unused in MR 1.4a.",
        ),
    ) -> None:
        """Print schemas for the job (spec / input_spec / options)."""
        del cluster  # reserved for MR 1.4b when /execution-profiles fetch lands
        bundle = scheduler.explain(job_cls, profile=profile)
        typer.echo(json.dumps(bundle, indent=2))

    _explain.__doc__ = "Show input/output schemas."
    group.command(name="explain", help=_explain.__doc__)(_explain)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_spec(spec_str: str, spec_file: Path | None) -> dict:
    """Load the spec dict from ``--spec`` JSON or ``--spec-file`` path.

    ``--spec-file`` wins when both are provided. Supports YAML or JSON
    (extension-driven). Raises :class:`typer.Exit` with a helpful error
    message on malformed content **or** on top-level non-mapping
    payloads (e.g. ``--spec '[]'``); the downstream
    :func:`~nemo_platform_plugin._spec_flags.deep_merge` assumes a dict and
    would otherwise fail with a raw ``TypeError`` after a per-field
    overlay was applied.
    """
    try:
        if spec_file is not None:
            loaded = load_spec_file(spec_file)
        else:
            loaded = json.loads(spec_str)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        typer.echo(f"Error: invalid spec — {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if not isinstance(loaded, dict):
        typer.echo(
            f"Error: invalid spec — expected a JSON/YAML object, got {type(loaded).__name__}.",
            err=True,
        )
        raise typer.Exit(code=1)
    return loaded


def _merge_options_inputs(options: list[str], options_file: Path | None) -> dict:
    """Combine ``--options-file`` contents (base) with ``-o`` overrides (overlay).

    Precedence: individual ``-o`` flags win over values in ``--options-file``.
    Both inputs are optional; an empty result is fine (the submit body
    omits the ``options`` field when the merged dict is empty).
    """
    base: dict = load_options_file(options_file) if options_file is not None else {}
    overlay: dict = parse_dotted_kv_list(options)
    return merge_options(base, overlay)


# ---------------------------------------------------------------------------
# NemoFunction CLI — two verbs (run / submit), no `explain`
# ---------------------------------------------------------------------------


def add_function_commands(
    cli_app: typer.Typer,
    functions: dict[str, type[NemoFunction]],
    *,
    cli: NemoCLI | None = None,
) -> None:
    """Inject two-verb subcommand groups for each function into *cli_app*.

    Each entry produces one :class:`typer.Typer` sub-group registered
    under :attr:`~nemo_platform_plugin.function.NemoFunction.name` in a
    ``"Functions"`` rich help panel. The sub-group owns ``run`` and
    ``submit`` commands. There is no ``explain`` verb — functions
    have a single ``spec_schema`` and ``--help`` is the introspection
    surface.

    Args:
        cli_app: The plugin's :class:`typer.Typer` group to inject
            sub-groups into. Typically the value returned by
            :meth:`~nemo_platform_plugin.cli.NemoCLI.get_cli`.
        functions: Mapping of entry-point key →
            :class:`~nemo_platform_plugin.function.NemoFunction` subclass,
            already filtered to the relevant plugin. Typically a
            subset of the dict returned by
            :func:`~nemo_platform_plugin.discovery.discover_functions`.
        cli: Optional plugin :class:`~nemo_platform_plugin.cli.NemoCLI` instance. When
            supplied, :meth:`~nemo_platform_plugin.cli.NemoCLI.update_function_cli` is
            invoked once per function after default verb registration and
            before the sub-group is mounted, giving the plugin a chance to
            amend the auto-generated CLI surface.
    """
    for fn_cls in functions.values():
        _register_function_subgroup(cli_app, fn_cls, cli=cli)


def _register_function_subgroup(
    cli_app: typer.Typer,
    fn_cls: type[NemoFunction],
    *,
    cli: NemoCLI | None = None,
) -> None:
    """Register a ``<fn-name>`` sub-group with run / submit verbs."""
    if not fn_cls.generate_legacy_verbs:
        _add_function_submit_command(cli_app, fn_cls, cli=cli, command_name=fn_cls.name, rich_help_panel="Functions")
        if cli is not None:
            cli.update_function_cli(fn_cls, cli_app)
        return

    fn_group = typer.Typer(
        name=fn_cls.name,
        help=fn_cls.description or f"Manage the {fn_cls.name} function.",
        no_args_is_help=False,
    )

    @fn_group.callback(invoke_without_command=True)
    def _root(ctx: typer.Context) -> None:  # pragma: no cover - trivial delegation
        """Stub callback — prints usage and exits 1 when no verb is given."""
        if ctx.invoked_subcommand is None:
            typer.echo(ctx.get_help())
            raise typer.Exit(code=1)

    _add_function_run_command(fn_group, fn_cls, cli=cli)
    _add_function_submit_command(fn_group, fn_cls, cli=cli)

    if cli is not None:
        cli.update_function_cli(fn_cls, fn_group)

    cli_app.add_typer(fn_group, name=fn_cls.name, rich_help_panel="Functions")


# ---- run --------------------------------------------------------- #


def _add_function_run_command(
    group: typer.Typer,
    fn_cls: type[NemoFunction],
    *,
    cli: NemoCLI | None = None,
) -> None:
    """Register the ``run`` verb. Generates per-field flags from ``spec_schema``.

    Each scalar leaf in ``spec_schema`` becomes a Typer option named
    after its dotted path, kebab-cased per segment (``--name``,
    ``--target.url``).  Precedence at runtime is ``--spec-file`` (base)
    → ``--spec`` JSON (overlay) → per-field flags (top overlay).
    Validation happens after the merge — surfaces a single
    ``ValidationError`` for the merged spec rather than per-overlay.

    When *cli* supplies a renderer via ``get_function_renderer(verb="run")``
    (and ``--output-format json`` is not set), the renderer's lifecycle
    drives the streamed output instead of the default per-frame echo.
    """
    unavailable: list[str] = []
    leaves = walk_spec_leaves(fn_cls.spec_schema, reserved=_FN_RUN_RESERVED_FLAGS, unavailable=unavailable)

    def _run(typer_ctx: typer.Context, **kwargs: object) -> None:
        original_kwargs = dict(kwargs)
        spec_str: str = cast(str, kwargs.pop("spec", "{}"))
        spec_file: Path | None = cast("Path | None", kwargs.pop("spec_file", None))
        workspace: str = cast(str, kwargs.pop("workspace", "default"))

        base = _load_spec(spec_str, spec_file)
        overlay = build_overlay(leaves, kwargs, unset_sentinel=UNSET)
        merged = deep_merge(base, overlay)
        try:
            spec_obj = fn_cls.spec_schema.model_validate(merged, context={"is_local": True})
        except ValidationError as exc:
            typer.echo(f"Error: invalid spec for {fn_cls.name}: {exc}", err=True)
            raise typer.Exit(code=1) from exc

        sdk, async_sdk = resolve_local_cli_sdks(typer_ctx)
        ctx = FunctionContext(workspace=workspace)
        renderer_cls: type[CLIRenderer] | None = None
        if cli is not None and not _output_format_is_json(typer_ctx):
            renderer_cls = cli.get_function_renderer(fn_cls, verb="run")
        try:
            outcome = asyncio.run(
                _invoke_function_locally(
                    fn_cls,
                    spec_obj,
                    ctx,
                    sdk=sdk,
                    async_sdk=async_sdk,
                    renderer_cls=renderer_cls,
                    cli_kwargs=original_kwargs,
                )
            )
        except LocalRunError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        except KeyboardInterrupt as exc:  # pragma: no cover - terminal-only
            typer.echo("Interrupted.", err=True)
            raise typer.Exit(code=130) from exc

        # Fire on_complete *outside* asyncio.run so renderers using event-loop-bound
        # libraries (e.g. prompt_toolkit's Application.run for an interactive
        # record browser) don't collide with the loop we just tore down.
        if outcome is not None:
            renderer, rctx = outcome
            renderer.on_complete(ctx=rctx)

    help_text = f"Run {fn_cls.name} locally, in-process."
    epilog = build_epilog(schema=fn_cls.spec_schema, leaves=leaves, kind="Function", unavailable=unavailable)
    setattr(_run, "__signature__", _build_function_run_signature(leaves))
    group.command(name="run", help=help_text, epilog=epilog)(_run)


def _build_function_run_signature(leaves: list[SpecLeafField]) -> inspect.Signature:
    """Compose the synthetic signature for the function ``run`` verb."""
    static_params = [
        # ``typer.Context`` is auto-injected by Click via ``pass_context``,
        # which passes it as the first positional argument. Hand-built
        # rather than via ``kw()`` because ``kw()`` only emits
        # ``KEYWORD_ONLY`` params (which Click's positional injection
        # would reject).
        inspect.Parameter(
            "typer_ctx",
            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=typer.Context,
        ),
        kw(
            "spec",
            str,
            typer.Option(
                "{}",
                "--spec",
                help="Spec as a JSON string.",
                rich_help_panel=_PANEL_SPEC_SOURCE,
            ),
        ),
        kw(
            "spec_file",
            Optional[Path],
            typer.Option(
                None,
                "--spec-file",
                help="Path to a YAML or JSON spec file (used as base; per-flag values override).",
                rich_help_panel=_PANEL_SPEC_SOURCE,
            ),
        ),
        kw(
            "workspace",
            str,
            typer.Option(
                "default",
                "--workspace",
                help="Workspace identity passed to the function as ctx.workspace.",
                rich_help_panel=_PANEL_SPEC_SOURCE,
            ),
        ),
    ]
    return build_callback_signature(
        static_params,
        leaves,
        rich_help_panel=_PANEL_FUNCTION_SPEC,
    )


async def _invoke_function_locally(
    fn_cls: type[NemoFunction],
    spec_obj: BaseModel,
    ctx: FunctionContext,
    *,
    sdk: object | None,
    async_sdk: object | None,
    renderer_cls: type[CLIRenderer] | None = None,
    cli_kwargs: Mapping[str, Any] | None = None,
) -> tuple[CLIRenderer, RendererContext] | None:
    """Call ``fn_cls().run(spec, ...)`` and print result(s) to stdout.

    Mirrors :func:`~nemo_platform_plugin.run_dependencies.resolve_run_kwargs` for the SDK
    parameters: inject when the caller supplied an SDK, leave the parameter
    unbound (Python default applies) when the function's signature has a
    default to fall back on, and raise :class:`LocalRunError` only when
    the function declared a required SDK parameter that we can't satisfy.

    When *renderer_cls* is supplied, drive the renderer's ``on_start`` +
    ``on_frame`` around the streamed iterator and return ``(renderer, rctx)``
    so the caller can fire ``on_complete`` *after* :func:`asyncio.run`
    returns — see :func:`_drive_async_renderer`. Renderers don't apply to
    non-streaming returns — those still echo via
    :func:`_format_value_for_stdout` and this function returns ``None``.
    """
    instance = fn_cls()
    run_params = fn_cls.run_signature().parameters
    kwargs: dict[str, Any] = {}
    if "ctx" in run_params:
        kwargs["ctx"] = ctx
    if "is_local" in run_params:
        kwargs["is_local"] = True
    for param_name, value in (("sdk", sdk), ("async_sdk", async_sdk)):
        param = run_params.get(param_name)
        if param is None:
            continue
        if value is not None:
            kwargs[param_name] = value
            continue
        if param.default is inspect.Parameter.empty:
            raise LocalRunError(
                f"{fn_cls.__name__}.run requires a `{param_name}` argument; "
                f"configure your `nemo` CLI context (e.g. `nemo config use-context ...`) "
                f"or pass `{param_name}` to the local invoker."
            )

    result = instance.run(spec_obj, **kwargs)
    if returns_async_iterator(result):
        if renderer_cls is not None:
            rctx = _make_renderer_context(
                cli_kwargs=cli_kwargs or {},
                verb="run",
                is_local=True,
            )
            renderer = await _drive_async_renderer(result, renderer_cls, rctx=rctx)
            return renderer, rctx
        async for frame in result:
            typer.echo(_format_frame_for_stdout(frame))
        return
    awaited = await result
    typer.echo(_format_value_for_stdout(awaited))


def _resolve_submit_auth_headers(typer_ctx: typer.Context) -> dict[str, str]:
    """Bearer (and other) default headers from the active CLI context."""
    state = typer_ctx.obj
    if state is None or not hasattr(state, "get_sdk_context"):
        return {}
    try:
        ctx = state.get_sdk_context()
        client_config = ctx.user.get_client_config()
        headers = client_config.get("default_headers")
        if isinstance(headers, dict):
            return {str(k): str(v) for k, v in headers.items()}
    except Exception:
        return {}
    return {}


# ---- submit ------------------------------------------------------ #


def _add_function_submit_command(
    group: typer.Typer,
    fn_cls: type[NemoFunction],
    *,
    cli: NemoCLI | None = None,
    command_name: str = "submit",
    rich_help_panel: str | None = None,
) -> None:
    """Register the ``submit`` verb. Generates per-field flags + static submit flags.

    When *cli* supplies a renderer via ``get_function_renderer(verb="submit")``
    (and ``--output-format json`` is not set), the renderer's lifecycle drives
    the streamed NDJSON response instead of the default per-line echo.
    """
    unavailable: list[str] = []
    leaves = walk_spec_leaves(fn_cls.spec_schema, reserved=_FN_SUBMIT_RESERVED_FLAGS, unavailable=unavailable)

    def _submit(typer_ctx: typer.Context, **kwargs: object) -> None:
        original_kwargs = dict(kwargs)
        spec_str: str = cast(str, kwargs.pop("spec", "{}"))
        spec_file: Path | None = cast("Path | None", kwargs.pop("spec_file", None))
        cluster: str | None = cast("str | None", kwargs.pop("cluster", None))
        base_url: str | None = cast("str | None", kwargs.pop("base_url", None))
        workspace: str = cast(str, kwargs.pop("workspace", "default"))
        request_id: str | None = cast("str | None", kwargs.pop("request_id", None))

        base = _load_spec(spec_str, spec_file)
        overlay = build_overlay(leaves, kwargs, unset_sentinel=UNSET)
        spec_data = deep_merge(base, overlay)
        try:
            fn_cls.spec_schema.model_validate(spec_data)
        except ValidationError as exc:
            typer.echo(f"Error: invalid spec for {fn_cls.name}: {exc}", err=True)
            raise typer.Exit(code=1) from exc

        url = _build_function_submit_url(
            typer_ctx,
            fn_cls,
            base_url=base_url,
            cluster=cluster,
            workspace=workspace,
        )
        headers = _resolve_submit_auth_headers(typer_ctx)
        if request_id is not None:
            headers["X-Request-ID"] = request_id

        renderer_cls: type[CLIRenderer] | None = None
        if cli is not None and not _output_format_is_json(typer_ctx):
            renderer_cls = cli.get_function_renderer(fn_cls, verb="submit")

        try:
            _post_function_submit(
                url,
                spec_data,
                headers=headers,
                renderer_cls=renderer_cls,
                cli_kwargs=original_kwargs,
            )
        except httpx.HTTPStatusError as exc:
            print_http_status_error(exc, action=f"submit {fn_cls.name}")
            raise typer.Exit(code=2) from exc
        except httpx.RequestError as exc:
            print_http_request_error(exc, action=f"submit {fn_cls.name}")
            raise typer.Exit(code=2) from exc
        except httpx.HTTPError as exc:
            typer.echo(f"Error: submit {fn_cls.name} failed: {exc}", err=True)
            raise typer.Exit(code=2) from exc

    help_text = f"Submit {fn_cls.name} over HTTP."
    epilog = build_epilog(schema=fn_cls.spec_schema, leaves=leaves, kind="Function", unavailable=unavailable)
    setattr(_submit, "__signature__", _build_function_submit_signature(leaves))
    group.command(name=command_name, help=help_text, epilog=epilog, rich_help_panel=rich_help_panel)(_submit)


def _build_function_submit_signature(leaves: list[SpecLeafField]) -> inspect.Signature:
    """Compose the synthetic signature for the function ``submit`` verb."""
    static_params = [
        inspect.Parameter(
            "typer_ctx",
            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=typer.Context,
        ),
        kw(
            "spec",
            str,
            typer.Option(
                "{}",
                "--spec",
                help="Spec as a JSON string.",
                rich_help_panel=_PANEL_SPEC_SOURCE,
            ),
        ),
        kw(
            "spec_file",
            Optional[Path],
            typer.Option(
                None,
                "--spec-file",
                help="Path to a YAML or JSON spec file (used as base; per-flag values override).",
                rich_help_panel=_PANEL_SPEC_SOURCE,
            ),
        ),
        kw(
            "cluster",
            Optional[str],
            typer.Option(
                None,
                "--cluster",
                help="Configured cluster name to resolve via the NeMo CLI config.",
                rich_help_panel=_PANEL_SUBMISSION,
            ),
        ),
        kw(
            "base_url",
            Optional[str],
            typer.Option(
                None,
                "--base-url",
                help=("Explicit plugin-service base URL (overrides --cluster and all other submit host resolution)."),
                rich_help_panel=_PANEL_SUBMISSION,
            ),
        ),
        kw(
            "workspace",
            str,
            typer.Option(
                "default",
                "--workspace",
                help="Workspace path segment used in the submit URL.",
                rich_help_panel=_PANEL_SUBMISSION,
            ),
        ),
        kw(
            "request_id",
            Optional[str],
            typer.Option(
                None,
                "--request-id",
                help="Set the X-Request-ID header (echoed back in ctx.request_id).",
                rich_help_panel=_PANEL_SUBMISSION,
            ),
        ),
    ]
    return build_callback_signature(
        static_params,
        leaves,
        rich_help_panel=_PANEL_FUNCTION_SPEC,
    )


def _post_function_submit(
    url: str,
    body: dict,
    *,
    headers: dict[str, str],
    timeout: float = 30.0,
    renderer_cls: type[CLIRenderer] | None = None,
    cli_kwargs: Mapping[str, Any] | None = None,
) -> None:
    """POST *body* to *url* and stream/print the response.

    Streams NDJSON line by line when the server returns
    :data:`NDJSON_MEDIA_TYPE`; otherwise pretty-prints the JSON body.
    Surfaces 4xx / 5xx as ``httpx.HTTPStatusError`` for the caller to
    translate into a Typer exit.

    When *renderer_cls* is supplied, the renderer drives the per-line NDJSON
    iteration through its lifecycle. Each parsed JSON line is passed to
    :meth:`CLIRenderer.on_frame`. The non-NDJSON fallback path is unchanged.
    """
    logger.debug("submit %s", url)
    with httpx.Client(timeout=timeout) as client:
        with client.stream("POST", url, json=body, headers=headers) as response:
            if response.status_code >= 400:
                # Buffer the body so the caller's error formatter
                # can read ``exc.response.text`` without raising
                # ``ResponseNotRead`` — ``client.stream`` opens the
                # response unbuffered, and the ``with`` block closes
                # the stream before the caller sees the exception.
                response.read()
                response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if NDJSON_MEDIA_TYPE in content_type:
                if renderer_cls is not None:
                    rctx = _make_renderer_context(
                        cli_kwargs=cli_kwargs or {},
                        verb="submit",
                        is_local=False,
                    )
                    renderer = _drive_sync_renderer(response.iter_lines(), renderer_cls, rctx=rctx)
                    renderer.on_complete(ctx=rctx)
                    return
                for line in response.iter_lines():
                    if not line:
                        continue
                    typer.echo(_pretty_print_jsonl_line(line))
                return
            response.read()
            try:
                typer.echo(json.dumps(response.json(), indent=2))
            except ValueError:
                typer.echo(response.text)


# ---- helpers ----------------------------------------------------- #


def _resolve_cluster_name_to_base_url(cluster_name: str) -> str:
    """Resolve a configured cluster name to its base URL."""
    from nemo_platform_ext.config.config import Config

    config = Config.load()
    for cluster in config.get_config_file().clusters:
        if cluster.name == cluster_name:
            return str(cluster.base_url)
    raise ValueError(
        f"Unknown cluster '{cluster_name}'. Use `nemo config view --all-contexts` to inspect configured clusters or pass `--base-url`."
    )


def _resolve_submit_base_url(
    typer_ctx: typer.Context,
    *,
    base_url: str | None,
    cluster: str | None,
) -> str:
    """Resolve submit host precedence shared by function and job submit."""
    if base_url is not None:
        return base_url
    if cluster is not None:
        return _resolve_cluster_name_to_base_url(cluster)

    state = typer_ctx.obj
    if state is not None and hasattr(state, "get_base_url"):
        resolved = state.get_base_url(default=None)
        if resolved:
            return cast(str, resolved)

    return os.environ.get(_DEFAULT_BASE_URL_ENV_VAR) or _DEFAULT_BASE_URL


def _build_function_submit_url(
    typer_ctx: typer.Context,
    fn_cls: type[NemoFunction],
    *,
    base_url: str | None,
    cluster: str | None,
    workspace: str,
) -> str:
    """Resolve the full POST URL for *fn_cls*.

    Precedence for the host: explicit ``--base-url`` > configured
    ``--cluster`` > active context base URL > ``$NMP_BASE_URL`` >
    localhost. The path is the canonical
    ``/apis/<plugin>/v2/workspaces/<ws>/<name>``, with
    :attr:`NemoFunction.endpoint` substituting the trailing segment
    when set.
    """
    host = _resolve_submit_base_url(typer_ctx, base_url=base_url, cluster=cluster)
    api_segment = _api_segment_for_function(fn_cls)
    trailing = (fn_cls.endpoint or DEFAULT_FUNCTION_PATH).replace("{name}", fn_cls.name)
    if not trailing.startswith("/"):
        trailing = "/" + trailing
    return f"{host.rstrip('/')}/apis/{api_segment}/v2/workspaces/{workspace}{trailing}"


def _api_segment_for_function(fn_cls: type[NemoFunction]) -> str:
    """Plugin-name segment of the URL for *fn_cls*.

    Functions register under the ``nemo.functions`` entry-point group
    keyed as ``<plugin>.<function>``, and the platform mounts their
    routes under that ``<plugin>`` segment. The authoritative source of
    truth is therefore the entry-point key — not the Python module
    path. We resolve it via :func:`~nemo_platform_plugin.discovery.discover_functions`
    so a plugin registered as ``my-plugin.greet`` correctly maps to
    ``/apis/my-plugin/...``, even though its package directory is
    ``nemo_my_plugin/`` (which the old module-name heuristic would
    have collapsed to ``my`` and 404'd against).

    When the function class isn't installed as an entry point — unit
    tests with inline classes, ad-hoc invocations from a checkout —
    fall back to deriving the segment from the top-level package name
    with the ``nemo_`` prefix stripped and underscores converted to
    dashes. This mirrors :func:`nemo_platform_plugin.scheduler._api_segment_for`
    so the two CLIs stay consistent. Crucially, the fallback no longer
    strips a trailing ``_plugin`` from the module name: a plugin whose
    actual key happens to be ``my-plugin`` would be silently rewritten
    to ``my``.
    """
    from nemo_platform_plugin.discovery import discover_functions

    try:
        registered = discover_functions()
    except Exception:
        registered = {}
    for key, registered_cls in registered.items():
        if registered_cls is fn_cls and "." in key:
            return key.split(".", 1)[0]

    module = fn_cls.__module__.split(".")[0]
    if module.startswith("nemo_"):
        module = module[len("nemo_") :]
    return module.replace("_", "-")


def _format_frame_for_stdout(frame: Any) -> str:
    """Render a streaming frame as a single line of pretty JSON.

    Pydantic models go through ``model_dump_json``; dicts and lists
    through ``json.dumps``; anything else through ``str()``. The
    output is one line per frame so consumers can pipe to ``jq -c``
    or grep for ``kind`` discriminators without a parser.
    """
    if isinstance(frame, BaseModel):
        return frame.model_dump_json()
    if isinstance(frame, (dict, list)):
        return json.dumps(frame, default=str)
    return str(frame)


def _format_value_for_stdout(value: Any) -> str:
    """Render a non-streaming return value as multi-line pretty JSON."""
    if isinstance(value, BaseModel):
        return value.model_dump_json(indent=2)
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2, default=str)
    return str(value)


def _pretty_print_jsonl_line(line: str) -> str:
    """Best-effort pretty-print for an NDJSON line.

    Falls back to the raw line when the line isn't valid JSON so
    operators see exactly what the server emitted (useful for
    debugging non-conforming streams). Streams are kept one frame per
    line — pretty-printing within a frame would defeat the JSONL
    contract for downstream piping.
    """
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError:
        return line
    return json.dumps(parsed)
