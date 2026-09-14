# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agents CLI — ``nemo agents`` command group.

Registered under the ``nemo.cli`` entry-point group.  The platform
discovers this class and mounts it as ``nemo agents <command>``.

**Local commands (no platform required):**

These run against a local agent config and work without a running NeMo Platform
instance.

- ``invoke``   — single invocation
- ``run``      — start a persistent local FastAPI server

The ``evaluate`` command is auto-generated from the
``EvaluateAgentJob`` registered under the
``nemo.jobs`` entry-point group — the platform injects it into this CLI
group at startup. Numeric optimize is likewise auto-injected from
``agents.optimize`` (``OptimizeJob`` in ``nemo-optimization``).

**Agent Resources commands (require a running cluster):**

- ``create``       — register an agent config on the platform
- ``list``         — list agents
- ``get``          — get an agent by name
- ``delete``       — delete an agent
- ``deploy``       — create a deployment for an agent (waits for ``running`` by default)
- ``chat``         — open an interactive new or existing deployed-agent session
- ``sessions``     — sub-group: list / get / close persisted sessions
- ``undeploy``     — stop and remove a deployment
- ``logs``         — print or tail the local deployment log file
- ``deployments``  — sub-group: list / get / delete deployments
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import tempfile
import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from pkgutil import resolve_name
from types import SimpleNamespace
from typing import Any, ClassVar, Literal, Optional, cast
from urllib.parse import urlencode

import click
import httpx
import typer
import yaml
from nemo_agents_plugin.cli_context import (
    DEFAULT_BASE_URL as _DEFAULT_BASE_URL,
)
from nemo_agents_plugin.cli_context import (
    BaseUrlOption,
)
from nemo_agents_plugin.cli_context import (
    resolve_base_url as _resolve_base_url,
)
from nemo_agents_plugin.cli_context import (
    resolve_context_headers as _resolve_context_headers,
)
from nemo_agents_plugin.deployment_routing import is_deployment_routable
from nemo_agents_plugin.entities import (
    AGENT_SPEC_FILENAME,
    CONTAINER_DEPLOYMENT_MODES,
    ETHOS_FILENAME,
    ETHOS_LOCAL_ROOT,
    MAX_ETHOS_STAGED_BYTES,
    MAX_ETHOS_STAGED_FILES,
    NAT_WORKFLOW_CONFIG_FORMAT,
    NEMO_AGENTS_SPEC_CONFIG_FORMAT,
    AgentDeployment,
    AgentSession,
    SessionStatus,
    ethos_fileset_name,
)
from nemo_agents_plugin.leaderboard.cli import register_leaderboard_commands
from nemo_agents_plugin.session_lifecycle import session_expiration_is_due
from nemo_agents_plugin.session_protocol import SESSION_ID_HEADER
from nemo_agents_plugin.usage.cli import register_usage_commands
from nemo_platform import NeMoPlatform
from nemo_platform_ext.cli.chat_tui import ExitAction, StreamingResponse, run_chat_tui
from nemo_platform_ext.cli.core.api import is_tty
from nemo_platform_ext.cli.core.formatters import Column, format_output
from nemo_platform_ext.cli.core.help_formatter import NmpGroup
from nemo_platform_ext.ui.prompts import is_interactive
from nemo_platform_plugin.cli import NemoCLI
from nemo_platform_plugin.cli_errors import print_http_request_error, print_http_status_error
from nemo_platform_plugin.cli_progress import request_progress
from nemo_platform_plugin.discovery import AGENT_CLI_GROUP, discover_entry_points
from nemo_platform_plugin.job import NemoJob
from pydantic import ValidationError
from typer.main import get_command as _typer_get_command

logger = logging.getLogger(__name__)

_DEFAULT_WORKSPACE = "default"
_LIST_OUTPUT_FORMAT = Literal["table", "json", "yaml", "csv", "markdown", "raw"]
_DEPLOYMENT_RESOLUTION_PAGE_SIZE = 100

_AGENT_LIST_COLUMNS = [
    Column("name"),
    Column("workspace"),
    Column("description"),
    Column("config_format"),
    Column("created_at"),
]
_DEPLOYMENT_LIST_COLUMNS = [
    Column("name"),
    Column("agent"),
    Column("workspace"),
    Column("status"),
    Column("endpoint"),
    Column("created_at"),
]
_SESSION_LIST_COLUMNS = [
    Column("name"),
    Column("status"),
    Column("deployment_id"),
    Column("last_active_at"),
    Column("expires_at"),
    Column("created_at"),
]
_ENVIRONMENT_LIST_COLUMNS = [
    Column("name"),
    Column("workspace"),
    Column("environment_spec"),
    Column("compute_spec"),
    Column("description"),
    Column("created_at"),
]
_ENVIRONMENT_SPEC_LIST_COLUMNS = [
    Column("name"),
    Column("workspace"),
    Column("provider"),
    Column("description"),
    Column("created_at"),
]
_COMPUTE_SPEC_LIST_COLUMNS = [
    Column("name"),
    Column("workspace"),
    Column("description"),
    Column("created_at"),
]


_AGENT_CLI_PANEL = "Platform agents"


@dataclass(frozen=True)
class _LazyAgentCliEntry:
    """Metadata-only stand-in for a not-yet-imported agent CLI extension.

    ``help`` is a generic placeholder (entry-point metadata carries no
    description) — same trade-off the top-level ``nemo`` CLI already makes
    for lazily-loaded plugin commands (see ``functional_plugin_entry`` in
    ``nemo_platform_ext.cli.manifest``).
    """

    import_path: str
    help: str
    panel: str = _AGENT_CLI_PANEL
    hidden: bool = False


class _LazyAgentCliGroup(NmpGroup):
    """Group that defers importing plugin agent-CLI extensions until needed.

    ``discover_agent_cli()`` (``nemo_platform_plugin.discovery``) fully imports
    every ``nemo.cli.agents`` entry point up front, which makes plain ``nemo
    agents -h`` pay the import cost of every contributing plugin (e.g. the
    nemo-insights analyst stack) even though none of them is being invoked.
    This group instead lists subcommand names from cheap entry-point metadata
    and only imports/builds a given plugin's Typer app when that specific
    subcommand name is resolved by Click — including when rendering its own
    one-line help, which ``NmpGroup.format_commands`` reads from
    ``_lazy_entries`` instead of calling ``get_command`` for every row.
    Mirrors ``ManifestBackedNmpGroup`` (top-level lazy loading in
    ``nemo_platform_ext.cli.core.lazy_load``) one level down.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        callback = getattr(self, "callback", None)
        self._lazy_entries: dict[str, _LazyAgentCliEntry] = getattr(callback, "__nmp_lazy_agent_cli__", {})

    def list_commands(self, ctx: click.Context) -> list[str]:
        names = list(self.commands)
        for name in self._lazy_entries:
            if name not in self.commands:
                names.append(name)
        return names

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        command = super().get_command(ctx, cmd_name)
        if command is not None:
            return command

        entry = self._lazy_entries.get(cmd_name)
        if entry is None:
            return None

        try:
            cli_cls = resolve_name(entry.import_path)
            cli = cli_cls().get_cli()
        except Exception:
            logger.warning("Failed to load agent CLI extension %r; skipping", cmd_name, exc_info=True)
            return None

        loaded = self._patch_command(_typer_get_command(cli))
        loaded.name = cmd_name
        loaded.hidden = loaded.hidden or entry.hidden
        panel = getattr(loaded, "rich_help_panel", None)
        if panel is None or type(panel).__name__ == "DefaultPlaceholder":
            setattr(loaded, "rich_help_panel", entry.panel)
        self.commands[cmd_name] = loaded
        return loaded


class AgentsCLI(NemoCLI):
    """CLI commands for the Agents plugin."""

    name: ClassVar[str] = "agents"
    description: ClassVar[str] = "Agent lifecycle management — local execution and platform-managed deployments."

    def get_cli(self) -> typer.Typer:
        app = typer.Typer(
            name="agents",
            help=self.description,
            no_args_is_help=False,
            cls=_LazyAgentCliGroup,
        )

        @app.callback(invoke_without_command=True)
        def agents_callback(ctx: typer.Context) -> None:
            if ctx.invoked_subcommand is None:
                typer.echo(ctx.get_help())
                raise typer.Exit(0)

        # Metadata-only discovery: reads entry-point names without importing
        # the plugin modules they point at. Actual imports happen lazily in
        # ``_LazyAgentCliGroup.get_command`` only for the subcommand resolved.
        setattr(
            agents_callback,
            "__nmp_lazy_agent_cli__",
            {
                name: _LazyAgentCliEntry(
                    import_path=entry_point.value,
                    help=f"Agent CLI commands contributed by the {name!r} plugin.",
                )
                for name, entry_point in discover_entry_points(AGENT_CLI_GROUP).items()
            },
        )

        _register_local_commands(app)
        _register_package_command(app)
        _register_platform_commands(app)
        _register_environment_commands(app)
        register_leaderboard_commands(app)
        register_usage_commands(app)
        return app

    def update_job_cli(self, job_cls: type[NemoJob], group: typer.Typer) -> None:
        """Amend generated job commands with job-specific subcommands.

        ``optimize`` gains ``prepare-fileset``: the remote optimize command requires an
        already-staged bundle, so the staging step belongs next to the job command that
        consumes it.
        """
        if job_cls.name != "optimize":
            return
        try:
            from nemo_agents_plugin.jobs.optimize_cli import register_prepare_fileset_command
        except ImportError:
            logger.warning("nemo-optimization unavailable; skipping optimize prepare-fileset", exc_info=True)
            return

        if not job_cls.generate_legacy_verbs:
            _replace_flat_job_command_with_group(
                group,
                job_cls=job_cls,
                register_subcommands=register_prepare_fileset_command,
            )
            return
        register_prepare_fileset_command(group)


def _replace_flat_job_command_with_group(
    app: typer.Typer,
    *,
    job_cls: type[NemoJob],
    register_subcommands: Callable[[typer.Typer], None],
) -> None:
    """Keep ``nemo agents <job>`` as remote submit while allowing job-specific subcommands."""
    command_info = next((command for command in app.registered_commands if command.name == job_cls.name), None)
    if command_info is None or command_info.callback is None:
        if getattr(app.info, "name", None) == job_cls.name:
            register_subcommands(app)
            return
        logger.warning("Cannot extend generated job command %r because it was not registered", job_cls.name)
        return

    original = command_info.callback
    app.registered_commands = [command for command in app.registered_commands if command is not command_info]

    job_group = typer.Typer(
        name=job_cls.name,
        help=job_cls.description or f"Manage the {job_cls.name} job.",
        epilog=command_info.epilog,
        no_args_is_help=False,
    )

    def submit(typer_ctx: typer.Context, **kwargs: object) -> None:
        if typer_ctx.invoked_subcommand is not None:
            return
        original(typer_ctx, **kwargs)

    submit.__doc__ = getattr(original, "__doc__", None)
    signature = getattr(original, "__signature__", None)
    if signature is not None:
        setattr(submit, "__signature__", signature)
    job_group.callback(invoke_without_command=True)(submit)

    register_subcommands(job_group)
    app.add_typer(job_group, name=job_cls.name, rich_help_panel="Jobs")


# ---------------------------------------------------------------------------
# Local commands — no platform required
# ---------------------------------------------------------------------------


def _register_local_commands(app: typer.Typer) -> None:
    """Register local agent commands onto *app*."""

    @app.command(rich_help_panel="Local commands")
    def invoke(
        agent_config: Optional[Path] = typer.Option(
            None,
            "--agent-config",
            "-c",
            help="Path to an agent YAML config file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
        input: Optional[str] = typer.Option(
            None,
            "--input",
            "-i",
            help="Input query string for local invocation.",
        ),
        input_file: Optional[Path] = typer.Option(
            None,
            "--input-file",
            help="JSON file containing a list of input queries for batch invocation.",
            exists=True,
        ),
        agent: Optional[str] = typer.Option(
            None,
            "--agent",
            "-a",
            help="Name of a platform-deployed agent to invoke (platform required).",
        ),
        agent_deployment: Optional[str] = typer.Option(
            None,
            "--agent-deployment",
            "-d",
            help="Name of a specific deployment to invoke (platform required).",
        ),
        workspace: str = typer.Option(_DEFAULT_WORKSPACE, "--workspace", "-w"),
        base_url: BaseUrlOption = None,
        timeout: float = typer.Option(
            300,
            "--timeout",
            "-t",
            envvar="NEMO_AGENTS_INVOKE_TIMEOUT",
            help="Request timeout in seconds for platform invocation.",
        ),
        no_progress: bool = typer.Option(
            False,
            "--no-progress",
            help="Suppress the stderr spinner while waiting for the response.",
        ),
    ) -> None:
        """Invoke an agent — locally (with --agent-config) or via the platform (with --agent or --agent-deployment)."""
        base_url = _resolve_base_url(base_url)
        if agent_config:
            _local_invoke(agent_config, input, input_file, workspace=workspace, base_url=base_url)
        elif agent or agent_deployment:
            _platform_invoke(
                base_url,
                workspace,
                agent,
                agent_deployment,
                input,
                input_file,
                timeout=timeout,
                no_progress=no_progress,
            )
        else:
            typer.echo(
                "Error: provide --agent-config for local execution or --agent/--agent-deployment for platform invocation.",
                err=True,
            )
            raise typer.Exit(code=1)

    @app.command(rich_help_panel="Local commands")
    def run(
        agent_config: Path = typer.Option(
            ...,
            "--agent-config",
            "-c",
            help="Path to an agent YAML config file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
        host: str = typer.Option("0.0.0.0", "--host"),
        port: int = typer.Option(8080, "--port", "-p"),
    ) -> None:
        """Run an agent locally as a persistent FastAPI server."""
        import subprocess

        config = _load_yaml(agent_config)
        if not isinstance(config, dict):
            typer.echo(f"Error: agent config {agent_config} root must be a YAML mapping.", err=True)
            raise typer.Exit(code=1)

        config_format = config.get("config_format", NAT_WORKFLOW_CONFIG_FORMAT)
        if config_format == NEMO_AGENTS_SPEC_CONFIG_FORMAT:
            cmd = [
                sys.executable,
                "-m",
                "nemo_agents_plugin.fabric.server",
                "--agent-config",
                agent_config.name,
                "--host",
                host,
                "--port",
                str(port),
            ]
        elif config_format == NAT_WORKFLOW_CONFIG_FORMAT:
            cmd = ["nat", "start", "fastapi", "--config_file", agent_config.name, "--host", host, "--port", str(port)]
        else:
            typer.echo(f"Error: unsupported config_format {config_format!r}", err=True)
            raise typer.Exit(code=1)

        typer.echo(f"Starting agent server: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, cwd=agent_config.parent)
        except subprocess.CalledProcessError as exc:
            typer.echo(f"Agent server exited with code {exc.returncode}.", err=True)
            raise typer.Exit(code=exc.returncode)
        except FileNotFoundError:
            if config_format == NAT_WORKFLOW_CONFIG_FORMAT:
                typer.echo("Error: 'nat' command not found.  Install nvidia-nat-core.", err=True)
            else:
                typer.echo(f"Error: server command {cmd[0]!r} was not found.", err=True)
            raise typer.Exit(code=1)


# Note: job commands such as ``evaluate`` and ``optimize`` are auto-generated
# from ``nemo.jobs`` entry points.


# ---------------------------------------------------------------------------
# Packaging command — no platform required
# ---------------------------------------------------------------------------

_PACKAGE_PANEL = "Packaging (no platform required)"


def _register_package_command(app: typer.Typer) -> None:
    """Register the unified ``package`` command onto *app*.

    Single command whose flags select how far the render → validate → build
    → publish pipeline runs:

    * ``--no-build``               stop after render (Dockerfile + .dockerignore only)
    * default                      render → validate → build
    * ``--publish --registry ...`` render → validate → build → publish
    """

    @app.command(rich_help_panel=_PACKAGE_PANEL)
    def package(
        agent: Path = typer.Option(
            ...,
            "--agent",
            "-c",
            help="Path to a NAT workflow YAML config file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
        pyproject: Optional[Path] = typer.Option(
            None,
            "--pyproject",
            help="Path to pyproject.toml (enables project mode).",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
        no_build: bool = typer.Option(
            False,
            "--no-build",
            help="Stop after render — emit Dockerfile + .dockerignore only (no image built).",
        ),
        publish: bool = typer.Option(
            False,
            "--publish",
            help="After building, tag and push to --registry.",
        ),
        format: str = typer.Option(
            "docker",
            "--format",
            help="Packaging format: 'docker' (Jinja2 Dockerfile). 'whl' is reserved for future wheel-based builds and is currently rejected.",
        ),
        dockerfile: Optional[Path] = typer.Option(
            None,
            "--dockerfile",
            help="Use an existing Dockerfile instead of rendering (skips render stage).",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
        tag: Optional[str] = typer.Option(
            None,
            "--tag",
            "-t",
            help="Image tag.  Defaults to '<agent-name>-<agent-id>:<agent-version>'.",
        ),
        platform: Optional[list[str]] = typer.Option(
            None,
            "--platform",
            help=(
                "Target platform (e.g. 'linux/amd64' or 'linux/arm64'). "
                "When omitted, defaults to the local daemon's native "
                "platform. Multi-arch builds via buildx are not yet "
                "implemented; pass at most one value."
            ),
        ),
        registry: Optional[str] = typer.Option(
            None,
            "--registry",
            "-r",
            help="Remote registry URL (required when --publish is set).",
        ),
        push_tag: Optional[str] = typer.Option(
            None,
            "--push-tag",
            help="Fully-qualified remote tag.  Defaults to '<registry>/<tag>'.",
        ),
        output: Optional[Path] = typer.Option(
            None,
            "--output",
            "-o",
            help="Output path for rendered Dockerfile (only used with --no-build). "
            "Defaults to 'Dockerfile' next to --pyproject when given (project root, "
            "so COPY statements resolve), otherwise next to the agent config.",
        ),
        base_image_url: Optional[str] = typer.Option(None, "--base-image-url", envvar="NEMO_AGENTS_BASE_IMAGE_URL"),
        base_image_tag: Optional[str] = typer.Option(None, "--base-image-tag", envvar="NEMO_AGENTS_BASE_IMAGE_TAG"),
        python_version: Optional[str] = typer.Option(None, "--python-version", envvar="NEMO_AGENTS_PYTHON_VERSION"),
        nat_version: Optional[str] = typer.Option(
            None,
            "--nat-version",
            help=(
                "NAT release to install (e.g. '1.7.0').  Strongly recommended: "
                "pin explicitly so image tags/labels/deps are reproducible.  "
                "When omitted, a baked-in default is used and a warning is printed."
            ),
        ),
        uv_version: Optional[str] = typer.Option(None, "--uv-version", envvar="NEMO_AGENTS_UV_VERSION"),
        allow_root: bool = typer.Option(
            False, "--allow-root", help="Disable non-root USER hardening in the rendered Dockerfile."
        ),
        sandbox_runtime: Optional[str] = typer.Option(
            None,
            "--sandbox-runtime",
            help=(
                "Render an image compatible with a sandbox runtime (e.g. 'openshell'). "
                "Discovers the runtime's image profile and bakes in its required apt "
                "packages + users so the image can run inside that sandbox supervisor."
            ),
        ),
        generate_ignore: bool = typer.Option(
            True, "--ignore/--no-ignore", help="Generate a .dockerignore file alongside the Dockerfile."
        ),
        skip_validation: bool = typer.Option(
            False, "--skip-validation", help="Bypass agent config validation before build."
        ),
        agent_version: Optional[str] = typer.Option(None, "--agent-version", help="Override agent version OCI label."),
        agent_author: Optional[str] = typer.Option(None, "--agent-author", help="Override agent author OCI label."),
        template: Optional[str] = typer.Option(
            None, "--template", help="Path to an external Jinja2 Dockerfile template."
        ),
    ) -> None:
        """Package a NAT agent -- render -> validate -> build -> publish.

        \b
        Progressive pipeline controlled by flags:
          --no-build                    emit Dockerfile + .dockerignore (no image)
          (default)                     render + validate + build
          --publish --registry ...      render + validate + build + push

        \b
        Platform behavior:
          - no --platform     image built for the local daemon's native platform
          - one --platform    image built for that platform (cross-arch via buildx)
          - multi --platform  rejected -- multi-arch builds via buildx are not
                              yet wired up; build per-arch and combine with
                              ``docker buildx imagetools create`` until then.
        """
        _validate_package_flags(
            no_build=no_build,
            publish=publish,
            registry=registry,
            format=format,
            template=template,
            platform=platform,
        )

        from nemo_agents_plugin.container.builder import detect_agent_config_format

        try:
            config_format = detect_agent_config_format(agent)
        except ValueError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1)

        if config_format == NAT_WORKFLOW_CONFIG_FORMAT:
            _warn_if_nat_version_unpinned(nat_version)
        elif config_format == NEMO_AGENTS_SPEC_CONFIG_FORMAT and nat_version is not None:
            typer.echo(
                "Error: --nat-version is only valid for NAT workflow packaging; "
                "Fabric agent packaging does not use NAT_VERSION.",
                err=True,
            )
            raise typer.Exit(code=1)

        if no_build:
            _package_render_only(
                agent_config=agent,
                config_format=config_format,
                pyproject=pyproject,
                output=output,
                format=format,
                template=template,
                allow_root=allow_root,
                sandbox_runtime=sandbox_runtime,
                agent_version=agent_version,
                agent_author=agent_author,
                generate_ignore=generate_ignore,
                base_image_url=base_image_url,
                base_image_tag=base_image_tag,
                python_version=python_version,
                nat_version=nat_version,
                uv_version=uv_version,
            )
            return

        from nemo_agents_plugin.container.builder import build_fabric_agent_image, build_nat_agent_image
        from nemo_agents_plugin.container.errors import AgentPackagingError

        try:
            if config_format == NEMO_AGENTS_SPEC_CONFIG_FORMAT:
                result_tag = build_fabric_agent_image(
                    agent,
                    pyproject=pyproject,
                    dockerfile=dockerfile,
                    tag=tag,
                    base_image_url=base_image_url,
                    base_image_tag=base_image_tag,
                    python_version=python_version,
                    uv_version=uv_version,
                    allow_root=allow_root,
                    sandbox_runtime=sandbox_runtime,
                    agent_version=agent_version,
                    agent_author=agent_author,
                    template_path=template,
                    skip_validation=skip_validation,
                    generate_ignore=generate_ignore,
                    platforms=platform,
                    on_progress=typer.echo,
                )
            else:
                result_tag = build_nat_agent_image(
                    agent,
                    pyproject=pyproject,
                    dockerfile=dockerfile,
                    tag=tag,
                    nat_version=nat_version,
                    base_image_url=base_image_url,
                    base_image_tag=base_image_tag,
                    python_version=python_version,
                    uv_version=uv_version,
                    allow_root=allow_root,
                    sandbox_runtime=sandbox_runtime,
                    agent_version=agent_version,
                    agent_author=agent_author,
                    template_path=template,
                    skip_validation=skip_validation,
                    generate_ignore=generate_ignore,
                    platforms=platform,
                    on_progress=typer.echo,
                )
        except (ValueError, AgentPackagingError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1)

        typer.echo(f"Image ready: {result_tag}")

        if not publish:
            return

        from nemo_agents_plugin.container.publisher import docker_push

        assert registry is not None  # guaranteed by _validate_package_flags
        try:
            remote = docker_push(local_tag=result_tag, registry=registry, push_tag=push_tag, on_progress=typer.echo)
        except AgentPackagingError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1)
        typer.echo(f"Published: {remote}")


def _validate_package_flags(
    *,
    no_build: bool,
    publish: bool,
    registry: Optional[str],
    format: str,
    template: Optional[str],
    platform: Optional[list[str]] = None,
) -> None:
    """Fail fast on flag combinations that cannot be satisfied."""
    if no_build and publish:
        typer.echo(
            "Error: --no-build and --publish are mutually exclusive.  "
            "--no-build emits a Dockerfile without building, so there is nothing to publish.",
            err=True,
        )
        raise typer.Exit(code=1)

    if publish and not registry:
        typer.echo(
            "Error: --publish requires --registry (e.g. --registry nvcr.io/my-org).",
            err=True,
        )
        raise typer.Exit(code=1)

    if format not in {"docker", "whl"}:
        typer.echo(f"Error: --format must be 'docker' or 'whl' (got '{format}').", err=True)
        raise typer.Exit(code=1)

    # ``whl`` was scaffolded in the original CLI surface but never wired
    # into the build path — reject up front so we don't silently ignore
    # the flag in a build invocation. ``--agent-whl`` was removed entirely;
    # when wheel packaging actually lands, re-add the flag together with
    # the validator branch that checks for it.
    if format == "whl":
        typer.echo(
            "Error: --format whl is not yet implemented. "
            "Use --format docker (the default) until wheel packaging lands.",
            err=True,
        )
        raise typer.Exit(code=1)

    if template is not None and not Path(template).is_file():
        typer.echo(f"Error: --template file not found: {template}", err=True)
        raise typer.Exit(code=1)

    # Multi-arch builds require a buildx-backed pipeline that this PR does
    # not implement.  Rejecting the flag prevents the earlier behavior of
    # printing a fake "Multi-arch manifest pushed via buildx" success while
    # actually building (and pushing) only a single-arch image.
    if platform and len(platform) > 1:
        typer.echo(
            "Error: multi-arch --platform is not yet implemented. "
            "Pass at most one --platform; for multi-arch images, build each "
            "platform separately and combine with `docker buildx imagetools create`.",
            err=True,
        )
        raise typer.Exit(code=1)


def _warn_if_nat_version_unpinned(nat_version: Optional[str]) -> None:
    """Emit a soft warning when ``--nat-version`` falls through to the default.

    Reproducibility hinges on callers pinning ``nvidia-nat`` explicitly (via
    ``--nat-version`` or the ``NAT_VERSION`` env var) — otherwise the OCI
    labels, image tags, and installed plugin set are implicitly tied to
    whatever default happens to be baked into the plugin.  The warning goes
    to stderr so it does not corrupt piped Dockerfile output in ``--no-build``
    renders.
    """
    from nemo_agents_plugin.container.template import resolve_value_with_source

    resolved, source = resolve_value_with_source("nat_version", nat_version)
    if source == "default":
        typer.echo(
            f"warning: --nat-version not provided; defaulting to '{resolved}'. "
            "Pass --nat-version or set NAT_VERSION to pin explicitly.",
            err=True,
        )


def _package_render_only(
    *,
    agent_config: Path,
    config_format: str,
    pyproject: Optional[Path],
    output: Optional[Path],
    format: str,
    template: Optional[str],
    allow_root: bool,
    sandbox_runtime: Optional[str],
    agent_version: Optional[str],
    agent_author: Optional[str],
    generate_ignore: bool,
    base_image_url: Optional[str],
    base_image_tag: Optional[str],
    python_version: Optional[str],
    nat_version: Optional[str],
    uv_version: Optional[str],
) -> None:
    """Implements the ``--no-build`` path: render files and exit."""
    # ``--format whl`` is rejected globally by ``_validate_package_flags``
    # before we get here; assert for the developer who deletes that guard.
    assert format == "docker", f"unreachable: format={format!r}"

    from nemo_agents_plugin.container.template import (
        render_dockerignore,
        render_fabric_dockerfile,
        render_nat_dockerfile,
    )

    try:
        if config_format == NEMO_AGENTS_SPEC_CONFIG_FORMAT:
            content = render_fabric_dockerfile(
                agent_config,
                pyproject,
                base_image_url=base_image_url,
                base_image_tag=base_image_tag,
                python_version=python_version,
                uv_version=uv_version,
                allow_root=allow_root,
                sandbox_runtime=sandbox_runtime,
                agent_version=agent_version,
                agent_author=agent_author,
                template_path=template,
            )
        else:
            content = render_nat_dockerfile(
                agent_config,
                pyproject,
                base_image_url=base_image_url,
                base_image_tag=base_image_tag,
                python_version=python_version,
                nat_version=nat_version,
                uv_version=uv_version,
                allow_root=allow_root,
                sandbox_runtime=sandbox_runtime,
                agent_version=agent_version,
                agent_author=agent_author,
                template_path=template,
            )
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    user_chose_output = output is not None
    if output is None:
        # In --pyproject (project) mode the Dockerfile MUST live at the project
        # root so ``COPY pyproject.toml .``, ``COPY uv.lock* .`` and ``COPY . .``
        # resolve against the correct build context.  In config-only mode there
        # is no project root, so fall back to the config's directory.
        if pyproject is not None:
            output = pyproject.parent / "Dockerfile"
        else:
            output = agent_config.parent / "Dockerfile"

    # Refuse to clobber a pre-existing Dockerfile when we picked the path
    # ourselves — silently overwriting a hand-tuned Dockerfile is the kind
    # of data loss CI runs are too coarse to catch.  A file we wrote on a
    # previous run (identified by the plugin's sentinel header) is safe to
    # regenerate.  When the user passes ``--output`` explicitly we treat
    # that as informed consent and overwrite unconditionally.
    from nemo_agents_plugin.container.template import is_plugin_managed

    if not user_chose_output and output.exists() and not is_plugin_managed(output):
        typer.echo(
            f"Error: refusing to overwrite existing file {output}. "
            "Pass --output to choose a different path (or to overwrite "
            "explicitly).",
            err=True,
        )
        raise typer.Exit(code=1)

    # Filesystem writes can fail for reasons completely unrelated to the
    # render logic (read-only mount, missing parent dir, disk full,
    # ENOSPC, EACCES).  Convert those into the same ``Error: ...`` +
    # ``typer.Exit(1)`` shape as the ``ValueError`` branch above so the
    # operator sees a clean CLI error instead of a Python traceback, and
    # so success-path stdout is never partially printed before a crash.
    try:
        output.write_text(content, encoding="utf-8")
    except OSError as exc:
        typer.echo(f"Error: failed to write Dockerfile to {output}: {exc}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Dockerfile written to {output}")

    if generate_ignore:
        # ``render_dockerignore`` returns ``None`` when a user-owned
        # ``.dockerignore`` is preserved (first-line sentinel check).  Be
        # explicit about which outcome happened so the user knows whether
        # their file was touched.
        try:
            ignore_path = render_dockerignore(output.parent)
        except OSError as exc:
            typer.echo(
                f"Error: failed to write .dockerignore to {output.parent / '.dockerignore'}: {exc}",
                err=True,
            )
            raise typer.Exit(code=1)
        if ignore_path is None:
            typer.echo(
                f"Preserved existing .dockerignore at {output.parent / '.dockerignore'} (not generated by this plugin)."
            )
        else:
            typer.echo(f".dockerignore written to {ignore_path}")


# ---------------------------------------------------------------------------
# Agent Resources commands — require a running cluster
# ---------------------------------------------------------------------------


def _register_platform_commands(app: typer.Typer) -> None:
    """Register Agent Resources commands (require a running cluster) onto *app*."""

    @app.command(rich_help_panel="Deployed agent interaction (requires running cluster)")
    def chat(
        input: Optional[str] = typer.Option(
            None,
            "--input",
            "-i",
            help="Optional first message to send before prompting for the next turn.",
        ),
        agent_deployment: Optional[str] = typer.Option(
            None,
            "--agent-deployment",
            "-d",
            help="Deployment to start a new persisted session against.",
        ),
        session: Optional[str] = typer.Option(
            None,
            "--session",
            "-s",
            help="Name of an existing persisted session to resume.",
        ),
        session_name: Optional[str] = typer.Option(
            None,
            "--session-name",
            help="Name for a new session; valid only with --agent-deployment.",
        ),
        workspace: str = typer.Option(_DEFAULT_WORKSPACE, "--workspace", "-w"),
        base_url: BaseUrlOption = None,
        timeout: float = typer.Option(
            300,
            "--timeout",
            "-t",
            min=0.001,
            envvar="NEMO_AGENTS_INVOKE_TIMEOUT",
            help="Request timeout in seconds for each streamed agent turn.",
        ),
    ) -> None:
        """Chat interactively with a new or existing deployed-agent session."""
        _validate_session_chat_options(
            agent_deployment=agent_deployment,
            session=session,
            session_name=session_name,
            input=input,
        )
        if not _is_interactive_session_chat():
            typer.echo(
                "Error: Agent chat requires an interactive terminal. "
                "Use `nemo agents invoke` for one-shot or scripted input.",
                err=True,
            )
            raise typer.Exit(code=2)

        _platform_session_chat(
            base_url=_resolve_base_url(base_url),
            workspace=workspace,
            agent_deployment=agent_deployment,
            session=session,
            session_name=session_name,
            input=input,
            timeout=timeout,
        )

    sessions_app = typer.Typer(
        name="sessions",
        help="Discover and manage persisted deployed-agent sessions.",
        no_args_is_help=True,
    )
    app.add_typer(sessions_app, rich_help_panel="Deployed agent interaction (requires running cluster)")

    @sessions_app.command(name="list")
    def sessions_list(
        ctx: typer.Context,
        agent_deployment: Optional[str] = typer.Option(
            None,
            "--agent-deployment",
            "-d",
            help="Limit results to sessions bound to this deployment.",
        ),
        workspace: str = typer.Option(_DEFAULT_WORKSPACE, "--workspace", "-w"),
        base_url: BaseUrlOption = None,
        output_format: Optional[_LIST_OUTPUT_FORMAT] = typer.Option(
            None,
            "--format",
            "--output-format",
            "-o",
            "-f",
            help="Output format for the list of sessions.",
            rich_help_panel="Output Options",
        ),
        no_truncate: Optional[bool] = typer.Option(
            None,
            "--no-truncate",
            help="Don't truncate long values in table/markdown/csv output.",
            rich_help_panel="Output Options",
        ),
    ) -> None:
        """List persisted sessions, newest first."""
        base_url = _resolve_base_url(base_url)
        path = f"/apis/agents/v2/workspaces/{workspace}/sessions"
        if agent_deployment is not None:
            if not agent_deployment.strip():
                typer.echo("Error: --agent-deployment must not be empty.", err=True)
                raise typer.Exit(code=2)
            deployment = _api_request(
                "GET",
                base_url,
                f"/apis/agents/v2/workspaces/{workspace}/deployments/{agent_deployment}",
            )
            deployment_id = _require_deployment_id(deployment, agent_deployment)
            path += f"?{urlencode({'filter[deployment_id]': deployment_id})}"

        resp = _api_request("GET", base_url, path)
        _print_list_response(
            ctx,
            resp,
            default_columns=_SESSION_LIST_COLUMNS,
            output_format=output_format,
            no_truncate=no_truncate,
        )

    @sessions_app.command(name="get")
    def sessions_get(
        name: str = typer.Argument(..., help="Session name."),
        workspace: str = typer.Option(_DEFAULT_WORKSPACE, "--workspace", "-w"),
        base_url: BaseUrlOption = None,
    ) -> None:
        """Get a persisted session by name."""
        base_url = _resolve_base_url(base_url)
        resp = _api_request("GET", base_url, f"/apis/agents/v2/workspaces/{workspace}/sessions/{name}")
        typer.echo(json.dumps(resp, indent=2))

    @sessions_app.command(name="close")
    def sessions_close(
        name: str = typer.Argument(..., help="Session name."),
        workspace: str = typer.Option(_DEFAULT_WORKSPACE, "--workspace", "-w"),
        base_url: BaseUrlOption = None,
        yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
    ) -> None:
        """Close a session and release its deployed runtime."""
        base_url = _resolve_base_url(base_url)
        if not yes:
            typer.confirm(f"Close session '{name}'? It cannot be resumed after it is closed.", abort=True)
        _api_request("POST", base_url, f"/apis/agents/v2/workspaces/{workspace}/sessions/{name}/close")
        typer.echo(f"Session '{name}' closed.")

    @app.command(rich_help_panel="Agent Resources (requires running cluster)")
    def create(
        name: str = typer.Option(..., "--name", "-n", help="Agent name."),
        agent_config: Path = typer.Option(
            ...,
            "--agent-config",
            "-c",
            help="Path to an agent YAML config file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
        description: str = typer.Option("", "--description"),
        workspace: str = typer.Option(_DEFAULT_WORKSPACE, "--workspace", "-w"),
        base_url: BaseUrlOption = None,
    ) -> None:
        """Register an agent on the platform."""
        base_url = _resolve_base_url(base_url)
        from nemo_agents_plugin.utils import inject_default_model

        config_dict = _load_yaml(agent_config)
        config_format = config_dict.get("config_format", NAT_WORKFLOW_CONFIG_FORMAT)
        if config_format in {NEMO_AGENTS_SPEC_CONFIG_FORMAT, NAT_WORKFLOW_CONFIG_FORMAT}:
            # Resolve ${NEMO_DEFAULT_MODEL} client-side — the agents service has
            # no user context at deploy time.
            config_dict = inject_default_model(config_dict)
            if _contains_default_model_placeholder(config_dict):
                typer.echo(
                    "Error: agent config references ${NEMO_DEFAULT_MODEL} but no "
                    "default model is selected. Run `nemo setup` to pick one, or "
                    "replace the placeholder in the config with an explicit model name.",
                    err=True,
                )
                raise typer.Exit(code=1)
        if config_format == NEMO_AGENTS_SPEC_CONFIG_FORMAT:
            config_dict = _validate_platform_agent_config_for_cli(config_dict, base_dir=agent_config.parent)
            for line in _spec_package_warning(name, agent_config):
                typer.echo(line, err=True)
        elif config_format != NAT_WORKFLOW_CONFIG_FORMAT:
            typer.echo(f"Error: unsupported config_format {config_format!r}", err=True)
            raise typer.Exit(code=1)
        payload = {
            "name": name,
            "description": description,
            "config": config_dict,
            "config_format": config_format,
        }
        resp = _api_request("POST", base_url, f"/apis/agents/v2/workspaces/{workspace}/agents", json_body=payload)
        if config_format == NEMO_AGENTS_SPEC_CONFIG_FORMAT:
            try:
                _upload_ethos_fileset(
                    agent_name=name,
                    workspace=workspace,
                    agent_root=agent_config.parent,
                    base_url=base_url,
                )
            except Exception as exc:
                typer.echo(
                    f"Error: failed to upload Ethos fileset for {name!r}: {exc}",
                    err=True,
                )
                try:
                    _delete_agent_entity(
                        agent_name=name,
                        workspace=workspace,
                        base_url=base_url,
                    )
                # ``typer.Exit`` subclasses ``Exception``, so this also covers the
                # exit raised by ``_api_request`` on an HTTP error.
                except Exception:
                    logger.exception(
                        "Failed to roll back agent %r after fileset upload failure",
                        name,
                    )
                    typer.echo(
                        f"Error: failed to roll back agent {name!r}; it may still exist on the "
                        f"platform. Remove it with `nemo agents delete {name}`.",
                        err=True,
                    )
                raise typer.Exit(code=1) from exc
        typer.echo(json.dumps(resp, indent=2))

    @app.command(name="list", rich_help_panel="Agent Resources (requires running cluster)")
    def list_agents(
        ctx: typer.Context,
        workspace: str = typer.Option(_DEFAULT_WORKSPACE, "--workspace", "-w"),
        base_url: BaseUrlOption = None,
        output_format: Optional[_LIST_OUTPUT_FORMAT] = typer.Option(
            None,
            "--format",
            "--output-format",
            "-o",
            "-f",
            help="Output format for the list of agents.",
            rich_help_panel="Output Options",
        ),
        no_truncate: Optional[bool] = typer.Option(
            None,
            "--no-truncate",
            help="Don't truncate long values in table/markdown/csv output.",
            rich_help_panel="Output Options",
        ),
    ) -> None:
        """List agents on the platform."""
        base_url = _resolve_base_url(base_url)
        resp = _api_request("GET", base_url, f"/apis/agents/v2/workspaces/{workspace}/agents")
        _print_list_response(
            ctx,
            resp,
            default_columns=_AGENT_LIST_COLUMNS,
            output_format=output_format,
            no_truncate=no_truncate,
        )

    @app.command(rich_help_panel="Agent Resources (requires running cluster)")
    def get(
        name: str = typer.Argument(..., help="Agent name."),
        workspace: str = typer.Option(_DEFAULT_WORKSPACE, "--workspace", "-w"),
        base_url: BaseUrlOption = None,
    ) -> None:
        """Get an agent by name."""
        base_url = _resolve_base_url(base_url)
        resp = _api_request("GET", base_url, f"/apis/agents/v2/workspaces/{workspace}/agents/{name}")
        typer.echo(json.dumps(resp, indent=2))

    @app.command(rich_help_panel="Agent Resources (requires running cluster)")
    def delete(
        name: str = typer.Argument(..., help="Agent name."),
        workspace: str = typer.Option(_DEFAULT_WORKSPACE, "--workspace", "-w"),
        base_url: BaseUrlOption = None,
        yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
    ) -> None:
        """Delete an agent from the platform."""
        base_url = _resolve_base_url(base_url)
        if not yes:
            typer.confirm(f"Delete agent '{name}'?", abort=True)
        _delete_agent_entity(agent_name=name, workspace=workspace, base_url=base_url)
        typer.echo(f"Agent '{name}' deleted.")

    @app.command(rich_help_panel="Agent Resources (requires running cluster)")
    def deploy(
        agent: str = typer.Option(..., "--agent", "-a", help="Name of the agent to deploy."),
        name: Optional[str] = typer.Option(None, "--name", "-n", help="Deployment name (auto-generated if omitted)."),
        mode: str = typer.Option(
            "subprocess",
            "--mode",
            help="Runtime backend: subprocess (default), docker, or k8s.",
        ),
        image: Optional[str] = typer.Option(
            None,
            "--image",
            "-i",
            help="Container image for docker/k8s modes (falls back to deployments.default_image).",
        ),
        use_image_entrypoint: bool = typer.Option(
            False,
            "--use-image-entrypoint",
            help=(
                "For docker/k8s modes, preserve the image ENTRYPOINT/CMD instead of "
                "injecting the platform-owned agent server command."
            ),
        ),
        environment: Optional[str] = typer.Option(
            None,
            "--environment",
            "-e",
            help=(
                "AgentEnvironment to deploy under, as a 'workspace/name' ref "
                "(e.g. 'default/repo-research-ben'). Its EnvironmentSpec is merged "
                "into the agent config and its ComputeSpec/secret refs are "
                "snapshotted onto the deployment at create time."
            ),
        ),
        wait: bool = typer.Option(
            True,
            "--wait/--no-wait",
            help=(
                "Wait for the deployment to reach a terminal status (running or failed) "
                "before returning.  Exits 0 only on running; exits 1 with the failure "
                "reason if runtime startup fails or readiness times out. "
                "Pass --no-wait for fire-and-forget behaviour (the original "
                "default — returns the pending deployment immediately as JSON)."
            ),
        ),
        timeout: int = typer.Option(
            300,
            "--timeout",
            "-t",
            help="Maximum seconds to wait for a terminal status (only with --wait).",
        ),
        workspace: str = typer.Option(_DEFAULT_WORKSPACE, "--workspace", "-w"),
        base_url: BaseUrlOption = None,
    ) -> None:
        """Deploy an agent on the platform.

        Blocks until the deployment is ``running`` (exit 0) or ``failed`` /
        timed out (exit 1) by default, so the exit code reflects the actual
        outcome of runtime startup instead of merely the API call.  Use
        ``--no-wait`` to keep the previous fire-and-forget behaviour for
        scripted pipelines that prefer to poll separately via ``nemo agents
        deployments wait``.

        Container modes (``--mode docker|k8s``) compile to the nemo-deployments
        plugin. Requires a configured deployments executor (``deployments.executors``
        / ``agents.deployments.docker_executor`` or ``k8s_executor``). Container
        endpoint gateway routing and the full k8s runtime contract (in-cluster
        inference gateway, wheel staging) are still evolving — docker mode is the
        supported local path today.
        """
        valid_modes: tuple[str, ...] = ("subprocess", *sorted(CONTAINER_DEPLOYMENT_MODES))
        if mode not in valid_modes:
            typer.echo(f"Invalid --mode {mode!r}; expected {', '.join(valid_modes)}.", err=True)
            raise typer.Exit(code=2)
        if image and mode == "subprocess":
            typer.echo("--image requires --mode docker or k8s.", err=True)
            raise typer.Exit(code=2)
        if use_image_entrypoint and mode == "subprocess":
            typer.echo("--use-image-entrypoint requires --mode docker or k8s.", err=True)
            raise typer.Exit(code=2)

        base_url = _resolve_base_url(base_url)
        if environment is not None and not environment.strip():
            typer.echo("--environment must not be empty.", err=True)
            raise typer.Exit(code=2)
        payload: dict = {"agent": agent, "deployment_mode": mode}
        if name:
            payload["name"] = name
        if image:
            payload["image"] = image
        if use_image_entrypoint:
            payload["use_image_entrypoint"] = True
        if environment is not None:
            payload["environment"] = environment
        resp = _api_request("POST", base_url, f"/apis/agents/v2/workspaces/{workspace}/deployments", json_body=payload)
        if not wait:
            typer.echo(json.dumps(resp, indent=2))
            return

        # The API returns the pending entity; wait for it to settle before exiting.
        deployment_name = resp.get("name") if isinstance(resp, dict) else None
        if not deployment_name:
            # Defensive: should never happen if the API contract holds.
            typer.echo(json.dumps(resp, indent=2))
            typer.echo(
                "Warning: deployment created but its name was missing from the response; "
                "skipping --wait. Use `nemo agents deployments list` to find it.",
                err=True,
            )
            return

        success = _wait_for_deployment(base_url, workspace, deployment_name, timeout=timeout)
        raise typer.Exit(code=0 if success else 1)

    @app.command(rich_help_panel="Agent Resources (requires running cluster)")
    def logs(
        name: Optional[str] = typer.Argument(
            None,
            help=(
                "Deployment name to print logs for. If omitted, pass --agent to look up "
                "the most recent deployment for that agent."
            ),
        ),
        agent: Optional[str] = typer.Option(
            None,
            "--agent",
            "-a",
            help=(
                "Resolve the most recent deployment for this agent (by ``created_at``), "
                "including failed ones — handy for post-mortem on a deploy that just died."
            ),
        ),
        follow: bool = typer.Option(
            False, "--follow", "-f", help="Tail the log file and stream new output as it is written."
        ),
        tail: Optional[int] = typer.Option(
            None,
            "--tail",
            "-n",
            help="Print only the last N lines before exiting (or before following). Default: print full log.",
        ),
        path_only: bool = typer.Option(
            False,
            "--path",
            help="Print only the absolute log file path and exit (useful for scripting).",
        ),
        workspace: str = typer.Option(_DEFAULT_WORKSPACE, "--workspace", "-w"),
        base_url: BaseUrlOption = None,
    ) -> None:
        """Show logs for an agent deployment.

        Reads the log file written by the local in-memory runner backend.
        NAT subprocess deployments write process output there; Fabric-backed
        deployments write validation/preparation entries there. The log file
        location is the same convention the backend uses internally:
        ``nmp_user_data_dir() / 'agents' / 'system' / <deployment-name>.log``
        by default. This command is therefore only meaningful when the CLI runs
        on the same host as the platform — once a remote backend lands, log
        retrieval should move to a server-side endpoint.

        With ``--follow`` (``-f``), this command behaves like ``tail -f`` and
        streams new output until interrupted with Ctrl-C.
        """
        if tail is not None and tail <= 0:
            typer.echo("Error: --tail must be a positive integer.", err=True)
            raise typer.Exit(code=1)

        if not name and not agent:
            typer.echo("Error: provide a deployment name or --agent.", err=True)
            raise typer.Exit(code=1)

        if agent and not name:
            base_url = _resolve_base_url(base_url)
            candidates = [
                d
                for d in _unwrap_list(
                    _api_request("GET", base_url, f"/apis/agents/v2/workspaces/{workspace}/deployments")
                )
                if d.get("agent") == agent and d.get("status") not in ("deleting",)
            ]
            if not candidates:
                typer.echo(f"Error: no deployment found for agent '{agent}'.", err=True)
                raise typer.Exit(code=1)
            # Pick the most recent by creation time so failed-and-immediately-
            # superseded deployments don't shadow the user's intent.  The
            # API serialises ``created_at`` as an ISO-8601 string; parse it
            # explicitly so the ordering is chronological even if Pydantic's
            # serialiser ever stops emitting zero-padded components.
            candidates.sort(key=_deployment_created_at_key)
            name = candidates[-1]["name"]

        # ``name`` is guaranteed non-None by the checks above; cast() narrows
        # the type without runtime overhead and survives ``python -O``.
        log_path = _agent_log_path_for(workspace, cast(str, name))
        if path_only:
            typer.echo(str(log_path))
            return

        if not log_path.exists():
            typer.echo(
                f"Error: log file does not exist on disk: {log_path}\n"
                "(The deployment may not have been spawned yet, the platform may be "
                "running on a different host, or the file was cleaned up.)",
                err=True,
            )
            raise typer.Exit(code=1)

        _print_log(log_path, tail=tail, follow=follow)

    @app.command(rich_help_panel="Agent Resources (requires running cluster)")
    def undeploy(
        name: Optional[str] = typer.Argument(None, help="Deployment name to remove."),
        agent: Optional[str] = typer.Option(
            None, "--agent", "--all", "-a", help="Remove all deployments for this agent."
        ),
        workspace: str = typer.Option(_DEFAULT_WORKSPACE, "--workspace", "-w"),
        base_url: BaseUrlOption = None,
        yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
    ) -> None:
        """Stop and remove a deployment (or all deployments for an agent)."""
        base_url = _resolve_base_url(base_url)
        if name:
            if not yes:
                typer.confirm(f"Undeploy '{name}'?", abort=True)
            _api_request("DELETE", base_url, f"/apis/agents/v2/workspaces/{workspace}/deployments/{name}")
            typer.echo(f"Deployment '{name}' marked for deletion.")
        elif agent:
            deps = _unwrap_list(_api_request("GET", base_url, f"/apis/agents/v2/workspaces/{workspace}/deployments"))
            removed = [d for d in deps if d.get("agent") == agent]
            if not yes:
                typer.confirm(f"Undeploy {len(removed)} deployment(s) for agent '{agent}'?", abort=True)
            for d in removed:
                _api_request("DELETE", base_url, f"/apis/agents/v2/workspaces/{workspace}/deployments/{d['name']}")
            typer.echo(f"Marked {len(removed)} deployment(s) for agent '{agent}' for deletion.")
        else:
            typer.echo("Error: provide a deployment name or --agent.", err=True)
            raise typer.Exit(code=1)

    # deployments sub-group
    deps_app = typer.Typer(name="deployments", help="Manage agent deployments.", no_args_is_help=True)
    app.add_typer(deps_app, rich_help_panel="Agent Resources (requires running cluster)")

    @deps_app.command(name="list")
    def deployments_list(
        ctx: typer.Context,
        workspace: str = typer.Option(_DEFAULT_WORKSPACE, "--workspace", "-w"),
        base_url: BaseUrlOption = None,
        output_format: Optional[_LIST_OUTPUT_FORMAT] = typer.Option(
            None,
            "--format",
            "--output-format",
            "-o",
            "-f",
            help="Output format for the list of deployments.",
            rich_help_panel="Output Options",
        ),
        no_truncate: Optional[bool] = typer.Option(
            None,
            "--no-truncate",
            help="Don't truncate long values in table/markdown/csv output.",
            rich_help_panel="Output Options",
        ),
    ) -> None:
        """List deployments."""
        base_url = _resolve_base_url(base_url)
        resp = _api_request("GET", base_url, f"/apis/agents/v2/workspaces/{workspace}/deployments")
        _print_list_response(
            ctx,
            resp,
            default_columns=_DEPLOYMENT_LIST_COLUMNS,
            output_format=output_format,
            no_truncate=no_truncate,
        )

    @deps_app.command(name="get")
    def deployments_get(
        name: str = typer.Argument(..., help="Deployment name."),
        workspace: str = typer.Option(_DEFAULT_WORKSPACE, "--workspace", "-w"),
        base_url: BaseUrlOption = None,
    ) -> None:
        """Get a deployment by name."""
        base_url = _resolve_base_url(base_url)
        resp = _api_request("GET", base_url, f"/apis/agents/v2/workspaces/{workspace}/deployments/{name}")
        typer.echo(json.dumps(resp, indent=2))

    @deps_app.command(name="delete")
    def deployments_delete(
        name: str = typer.Argument(..., help="Deployment name."),
        workspace: str = typer.Option(_DEFAULT_WORKSPACE, "--workspace", "-w"),
        base_url: BaseUrlOption = None,
        yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
    ) -> None:
        """Delete a deployment by name."""
        base_url = _resolve_base_url(base_url)
        if not yes:
            typer.confirm(f"Delete deployment '{name}'?", abort=True)
        _api_request("DELETE", base_url, f"/apis/agents/v2/workspaces/{workspace}/deployments/{name}")
        typer.echo(f"Deployment '{name}' marked for deletion.")

    @deps_app.command(name="wait")
    def deployments_wait(
        name: Optional[str] = typer.Argument(None, help="Deployment name to wait for."),
        agent: Optional[str] = typer.Option(
            None,
            "--agent",
            "-a",
            help="Wait for the latest active deployment of this agent (alternative to passing a name directly).",
        ),
        timeout: int = typer.Option(300, "--timeout", "-t", help="Maximum seconds to wait."),
        interval: float = typer.Option(2.0, "--interval", help="Poll interval in seconds."),
        workspace: str = typer.Option(_DEFAULT_WORKSPACE, "--workspace", "-w"),
        base_url: BaseUrlOption = None,
    ) -> None:
        """Wait for a deployment to reach 'running' or 'failed' status.

        Polls the deployment until it is running (exit 0) or failed / timed out (exit 1).
        Prints a status line each time the status changes.

        Provide either a deployment name directly or --agent to resolve the
        latest active deployment for that agent automatically.
        """
        base_url = _resolve_base_url(base_url)
        if not name and not agent:
            typer.echo("Error: provide a deployment name or --agent.", err=True)
            raise typer.Exit(code=1)

        if agent and not name:
            active = [
                d
                for d in _unwrap_list(
                    _api_request("GET", base_url, f"/apis/agents/v2/workspaces/{workspace}/deployments")
                )
                if d.get("agent") == agent and d.get("status") not in ("failed", "deleting")
            ]
            if not active:
                typer.echo(f"Error: no active deployment found for agent '{agent}'.", err=True)
                raise typer.Exit(code=1)
            name = active[-1]["name"]

        assert name  # guaranteed by the checks above
        success = _wait_for_deployment(base_url, workspace, name, timeout=timeout, interval=interval)
        raise typer.Exit(code=0 if success else 1)


# ---------------------------------------------------------------------------
# Environment / EnvironmentSpec / ComputeSpec commands
# ---------------------------------------------------------------------------


def _spec_body_from_inputs(
    *,
    name: str,
    spec_file: Optional[Path],
    spec_json: Optional[str],
) -> dict:
    """Build a create-request body from a ``--spec-file`` or ``--spec`` input.

    Exactly one of *spec_file* / *spec_json* may be given. The resulting mapping
    is merged under ``{"name": name, ...}`` (the caller's ``name`` wins). Returns
    just ``{"name": name}`` when neither is provided (an empty inline spec).
    """
    if spec_file is not None and spec_json is not None:
        typer.echo("Error: pass only one of --spec-file or --spec.", err=True)
        raise typer.Exit(code=2)

    body: dict = {}
    if spec_file is not None:
        try:
            loaded = _load_yaml(spec_file)  # YAML is a superset of JSON
        except (OSError, yaml.YAMLError) as exc:
            typer.echo(f"Error: could not read spec file {spec_file}: {exc}", err=True)
            raise typer.Exit(code=2) from exc
        if not isinstance(loaded, dict):
            typer.echo(f"Error: spec file {spec_file} must contain a mapping.", err=True)
            raise typer.Exit(code=2)
        body = loaded
    elif spec_json is not None:
        try:
            loaded = json.loads(spec_json)
        except json.JSONDecodeError as exc:
            typer.echo(f"Error: --spec is not valid JSON: {exc}", err=True)
            raise typer.Exit(code=2) from exc
        if not isinstance(loaded, dict):
            typer.echo("Error: --spec must be a JSON object.", err=True)
            raise typer.Exit(code=2)
        body = loaded

    return {**body, "name": name}


def _register_environment_commands(app: typer.Typer) -> None:
    """Register ``environment-specs``, ``environments``, and ``compute-specs`` sub-groups.

    These are the request/fulfill entities an AgentDeployment or ``agents.execute``
    job references via its ``environment`` field: an EnvironmentSpec fulfills the
    dependencies an Agent declares (endpoints, plaintext env, secret refs); a
    ComputeSpec supplies k8s-style resources; an AgentEnvironment composes the two.
    Bodies are the ``*Inline`` shapes (see ``entities.py``); create takes a
    ``--spec-file`` (JSON/YAML) or an inline ``--spec`` JSON string.
    """
    _PANEL = "Agent Resources (requires running cluster)"

    # -- environment-specs ---------------------------------------------------
    espec_app = typer.Typer(name="environment-specs", help="Manage agent environment specs.", no_args_is_help=True)
    app.add_typer(espec_app, rich_help_panel=_PANEL)

    @espec_app.command(name="create")
    def environment_spec_create(
        name: str = typer.Argument(..., help="Unique environment-spec name."),
        spec_file: Optional[Path] = typer.Option(
            None, "--spec-file", "-f", help="Path to a JSON/YAML EnvironmentSpec body (without 'name')."
        ),
        spec: Optional[str] = typer.Option(None, "--spec", help="Inline EnvironmentSpec body as a JSON object string."),
        workspace: str = typer.Option(_DEFAULT_WORKSPACE, "--workspace", "-w"),
        base_url: BaseUrlOption = None,
    ) -> None:
        """Create an environment spec from a file or inline JSON."""
        base_url = _resolve_base_url(base_url)
        body = _spec_body_from_inputs(name=name, spec_file=spec_file, spec_json=spec)
        sdk = _agents_sdk(base_url, workspace)
        resp = _run_sdk(
            "POST agent API",
            lambda: sdk.environment_specs.create(name=body.pop("name"), workspace=workspace, spec=body),
        )
        typer.echo(json.dumps(resp, indent=2))

    @espec_app.command(name="list")
    def environment_spec_list(
        ctx: typer.Context,
        workspace: str = typer.Option(_DEFAULT_WORKSPACE, "--workspace", "-w"),
        base_url: BaseUrlOption = None,
        output_format: Optional[_LIST_OUTPUT_FORMAT] = typer.Option(
            None,
            "--format",
            "--output-format",
            "-o",
            "-f",
            help="Output format.",
            rich_help_panel="Output Options",
        ),
        no_truncate: Optional[bool] = typer.Option(
            None,
            "--no-truncate",
            help="Don't truncate long values.",
            rich_help_panel="Output Options",
        ),
    ) -> None:
        """List environment specs."""
        base_url = _resolve_base_url(base_url)
        resp = _api_request("GET", base_url, f"/apis/agents/v2/workspaces/{workspace}/environment-specs")
        _print_list_response(
            ctx,
            resp,
            default_columns=_ENVIRONMENT_SPEC_LIST_COLUMNS,
            output_format=output_format,
            no_truncate=no_truncate,
        )

    @espec_app.command(name="get")
    def environment_spec_get(
        name: str = typer.Argument(..., help="Environment-spec name."),
        workspace: str = typer.Option(_DEFAULT_WORKSPACE, "--workspace", "-w"),
        base_url: BaseUrlOption = None,
    ) -> None:
        """Get an environment spec by name."""
        base_url = _resolve_base_url(base_url)
        resp = _api_request("GET", base_url, f"/apis/agents/v2/workspaces/{workspace}/environment-specs/{name}")
        typer.echo(json.dumps(resp, indent=2))

    @espec_app.command(name="delete")
    def environment_spec_delete(
        name: str = typer.Argument(..., help="Environment-spec name."),
        workspace: str = typer.Option(_DEFAULT_WORKSPACE, "--workspace", "-w"),
        base_url: BaseUrlOption = None,
        yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
    ) -> None:
        """Delete an environment spec by name."""
        base_url = _resolve_base_url(base_url)
        if not yes:
            typer.confirm(f"Delete environment-spec '{name}'?", abort=True)
        _api_request("DELETE", base_url, f"/apis/agents/v2/workspaces/{workspace}/environment-specs/{name}")
        typer.echo(f"Environment-spec '{name}' deleted.")

    # -- environments --------------------------------------------------------
    env_app = typer.Typer(name="environments", help="Manage agent environments.", no_args_is_help=True)
    app.add_typer(env_app, rich_help_panel=_PANEL)

    @env_app.command(name="create")
    def environment_create(
        name: str = typer.Argument(..., help="Unique environment name."),
        environment_spec: Optional[str] = typer.Option(
            None, "--environment-spec", help="'workspace/name' ref to a stored AgentEnvironmentSpec."
        ),
        compute_spec: Optional[str] = typer.Option(
            None, "--compute-spec", help="'workspace/name' ref to a stored AgentComputeSpec."
        ),
        spec_file: Optional[Path] = typer.Option(
            None, "--spec-file", "-f", help="Path to a JSON/YAML AgentEnvironment body (without 'name')."
        ),
        spec: Optional[str] = typer.Option(
            None, "--spec", help="Inline AgentEnvironment body as a JSON object string."
        ),
        workspace: str = typer.Option(_DEFAULT_WORKSPACE, "--workspace", "-w"),
        base_url: BaseUrlOption = None,
    ) -> None:
        """Create an AgentEnvironment.

        Use --environment-spec / --compute-spec for the common ref case, or
        --spec-file / --spec for a fully inline body. The ref flags override the
        matching keys from a file/inline body.
        """
        base_url = _resolve_base_url(base_url)
        body = _spec_body_from_inputs(name=name, spec_file=spec_file, spec_json=spec)
        if environment_spec is not None:
            body["environment_spec"] = environment_spec
        if compute_spec is not None:
            body["compute_spec"] = compute_spec
        sdk = _agents_sdk(base_url, workspace)
        resp = _run_sdk(
            "POST agent API",
            lambda: sdk.environments.create(name=body.pop("name"), workspace=workspace, spec=body),
        )
        typer.echo(json.dumps(resp, indent=2))

    @env_app.command(name="list")
    def environment_list(
        ctx: typer.Context,
        workspace: str = typer.Option(_DEFAULT_WORKSPACE, "--workspace", "-w"),
        base_url: BaseUrlOption = None,
        output_format: Optional[_LIST_OUTPUT_FORMAT] = typer.Option(
            None,
            "--format",
            "--output-format",
            "-o",
            "-f",
            help="Output format.",
            rich_help_panel="Output Options",
        ),
        no_truncate: Optional[bool] = typer.Option(
            None,
            "--no-truncate",
            help="Don't truncate long values.",
            rich_help_panel="Output Options",
        ),
    ) -> None:
        """List environments."""
        base_url = _resolve_base_url(base_url)
        resp = _api_request("GET", base_url, f"/apis/agents/v2/workspaces/{workspace}/environments")
        _print_list_response(
            ctx,
            resp,
            default_columns=_ENVIRONMENT_LIST_COLUMNS,
            output_format=output_format,
            no_truncate=no_truncate,
        )

    @env_app.command(name="get")
    def environment_get(
        name: str = typer.Argument(..., help="Environment name."),
        workspace: str = typer.Option(_DEFAULT_WORKSPACE, "--workspace", "-w"),
        base_url: BaseUrlOption = None,
    ) -> None:
        """Get an environment by name."""
        base_url = _resolve_base_url(base_url)
        resp = _api_request("GET", base_url, f"/apis/agents/v2/workspaces/{workspace}/environments/{name}")
        typer.echo(json.dumps(resp, indent=2))

    @env_app.command(name="delete")
    def environment_delete(
        name: str = typer.Argument(..., help="Environment name."),
        workspace: str = typer.Option(_DEFAULT_WORKSPACE, "--workspace", "-w"),
        base_url: BaseUrlOption = None,
        yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
    ) -> None:
        """Delete an environment by name."""
        base_url = _resolve_base_url(base_url)
        if not yes:
            typer.confirm(f"Delete environment '{name}'?", abort=True)
        _api_request("DELETE", base_url, f"/apis/agents/v2/workspaces/{workspace}/environments/{name}")
        typer.echo(f"Environment '{name}' deleted.")

    # -- compute-specs -------------------------------------------------------
    cspec_app = typer.Typer(name="compute-specs", help="Manage agent compute specs.", no_args_is_help=True)
    app.add_typer(cspec_app, rich_help_panel=_PANEL)

    @cspec_app.command(name="create")
    def compute_spec_create(
        name: str = typer.Argument(..., help="Unique compute-spec name."),
        spec_file: Optional[Path] = typer.Option(
            None, "--spec-file", "-f", help="Path to a JSON/YAML ComputeSpec body (without 'name')."
        ),
        spec: Optional[str] = typer.Option(None, "--spec", help="Inline ComputeSpec body as a JSON object string."),
        workspace: str = typer.Option(_DEFAULT_WORKSPACE, "--workspace", "-w"),
        base_url: BaseUrlOption = None,
    ) -> None:
        """Create a compute spec from a file or inline JSON."""
        base_url = _resolve_base_url(base_url)
        body = _spec_body_from_inputs(name=name, spec_file=spec_file, spec_json=spec)
        sdk = _agents_sdk(base_url, workspace)
        resp = _run_sdk(
            "POST agent API",
            lambda: sdk.compute_specs.create(name=body.pop("name"), workspace=workspace, spec=body),
        )
        typer.echo(json.dumps(resp, indent=2))

    @cspec_app.command(name="list")
    def compute_spec_list(
        ctx: typer.Context,
        workspace: str = typer.Option(_DEFAULT_WORKSPACE, "--workspace", "-w"),
        base_url: BaseUrlOption = None,
        output_format: Optional[_LIST_OUTPUT_FORMAT] = typer.Option(
            None,
            "--format",
            "--output-format",
            "-o",
            "-f",
            help="Output format.",
            rich_help_panel="Output Options",
        ),
        no_truncate: Optional[bool] = typer.Option(
            None,
            "--no-truncate",
            help="Don't truncate long values.",
            rich_help_panel="Output Options",
        ),
    ) -> None:
        """List compute specs."""
        base_url = _resolve_base_url(base_url)
        resp = _api_request("GET", base_url, f"/apis/agents/v2/workspaces/{workspace}/compute-specs")
        _print_list_response(
            ctx,
            resp,
            default_columns=_COMPUTE_SPEC_LIST_COLUMNS,
            output_format=output_format,
            no_truncate=no_truncate,
        )

    @cspec_app.command(name="get")
    def compute_spec_get(
        name: str = typer.Argument(..., help="Compute-spec name."),
        workspace: str = typer.Option(_DEFAULT_WORKSPACE, "--workspace", "-w"),
        base_url: BaseUrlOption = None,
    ) -> None:
        """Get a compute spec by name."""
        base_url = _resolve_base_url(base_url)
        resp = _api_request("GET", base_url, f"/apis/agents/v2/workspaces/{workspace}/compute-specs/{name}")
        typer.echo(json.dumps(resp, indent=2))

    @cspec_app.command(name="delete")
    def compute_spec_delete(
        name: str = typer.Argument(..., help="Compute-spec name."),
        workspace: str = typer.Option(_DEFAULT_WORKSPACE, "--workspace", "-w"),
        base_url: BaseUrlOption = None,
        yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
    ) -> None:
        """Delete a compute spec by name."""
        base_url = _resolve_base_url(base_url)
        if not yes:
            typer.confirm(f"Delete compute-spec '{name}'?", abort=True)
        _api_request("DELETE", base_url, f"/apis/agents/v2/workspaces/{workspace}/compute-specs/{name}")
        typer.echo(f"Compute-spec '{name}' deleted.")


# ---------------------------------------------------------------------------
# Deployment wait helper
# ---------------------------------------------------------------------------

_TERMINAL_STATUSES = {"running", "failed"}


def _deployment_address(dep: dict[str, Any]) -> str:
    """Best-effort address for CLI output (loopback endpoint or first projected URL)."""
    endpoint = dep.get("endpoint")
    if isinstance(endpoint, str) and endpoint:
        return endpoint
    for ep in dep.get("endpoints") or []:
        if isinstance(ep, dict) and ep.get("url"):
            return str(ep["url"])
    return ""


def _wait_for_deployment(
    base_url: str,
    workspace: str,
    name: str,
    *,
    timeout: int = 300,
    interval: float = 2.0,
) -> bool:
    """Poll a deployment until it reaches a terminal status.

    Args:
        base_url: Platform base URL.
        workspace: Workspace the deployment belongs to.
        name: Deployment name.
        timeout: Maximum seconds to wait before giving up.
        interval: Seconds between polls.

    Returns:
        ``True`` if the deployment reached ``running``, ``False`` if it
        reached ``failed`` or the timeout expired.
    """
    path = f"/apis/agents/v2/workspaces/{workspace}/deployments/{name}"
    start = time.monotonic()
    last_status = ""

    typer.echo(f"Waiting for deployment '{name}' (timeout={timeout}s)...")

    while time.monotonic() - start < timeout:
        dep = _api_request("GET", base_url, path)
        status = dep.get("status", "")
        elapsed = int(time.monotonic() - start)

        if status != last_status:
            line = f"  [{elapsed:>4}s] status: {status}"
            if status == "failed" and dep.get("error"):
                line += f" — {dep['error']}"
            typer.echo(line)
            last_status = status

        if status == "running":
            typer.echo(f"Deployment '{name}' is running at {_deployment_address(dep) or '?'}")
            return True

        if status == "failed":
            typer.echo(f"Deployment '{name}' failed.", err=True)
            return False

        time.sleep(interval)

    elapsed = int(time.monotonic() - start)
    typer.echo(f"Timeout after {elapsed}s. Last status: {last_status}", err=True)
    return False


# ---------------------------------------------------------------------------
# Log printing / tailing helper
# ---------------------------------------------------------------------------


def _agent_log_path_for(workspace: str, deployment_name: str) -> Path:
    """Return the absolute log-file path the runner backend uses for a deployment.

    Imports the convention from the runner module so the CLI and the running
    platform agree on layout without round-tripping a host-bound path
    through the public API surface.  Correct only for the in-memory backend
    on the same host as the CLI invoker.

    The path is workspace-namespaced (``<system_dir>/<workspace>/<name>.log``)
    so two workspaces with same-named deployments don't share a file.
    """
    from nemo_agents_plugin.runner.in_memory import log_path_for_deployment

    return log_path_for_deployment(workspace, deployment_name)


def _deployment_created_at_key(dep: dict[str, Any]) -> datetime:
    """Sort key for deployments — parses the API's ISO-8601 ``created_at``.

    Falls back to ``datetime.min`` when the field is missing or unparseable
    so a malformed entry sorts to the start (and the most recent valid
    deployment wins ``[-1]``).  ``datetime.fromisoformat`` accepts the
    Pydantic default serialisation (``2026-05-18T17:36:26.639200``).
    """
    raw = dep.get("created_at")
    if not isinstance(raw, str):
        return datetime.min
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return datetime.min


def _print_log(log_path: Path, *, tail: Optional[int] = None, follow: bool = False) -> None:
    """Print *log_path* to stdout, optionally tailing the last N lines or following.

    Implemented in pure Python (rather than shelling out to ``tail``) so the
    behaviour is identical on every host the platform may run on.  In
    ``follow`` mode the same file handle is reused across the read and the
    poll loop, so lines written between the two would not be lost.  The
    poll interval is 0.5s — plenty responsive for log review.
    """
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            if tail is not None and tail > 0:
                # Read whole file and emit only the trailing N lines.  Files
                # are typically small (single-process subprocess logs); a
                # streaming "real tail" would be premature optimisation.
                lines = fh.readlines()
                for line in lines[-tail:]:
                    typer.echo(line, nl=False)
            else:
                for line in fh:
                    typer.echo(line, nl=False)

            if not follow:
                return

            # Continue from EOF on the same handle so writes between the
            # initial read and the poll loop aren't dropped.
            fh.seek(0, os.SEEK_END)
            try:
                while True:
                    chunk = fh.read()
                    if chunk:
                        typer.echo(chunk, nl=False)
                    else:
                        time.sleep(0.5)
            except KeyboardInterrupt:
                return
    except BrokenPipeError:
        # Consumer (e.g. ``| head -20``) closed the pipe early.  Exit
        # quietly instead of raising a traceback at the user.
        return


# ---------------------------------------------------------------------------
# Local execution helpers
# ---------------------------------------------------------------------------


def _local_invoke(
    agent_config: Path,
    input: Optional[str],
    input_file: Optional[Path],
    workspace: str = _DEFAULT_WORKSPACE,
    base_url: str = _DEFAULT_BASE_URL,
) -> None:
    """Invoke a local agent config once and print the result.

    NAT workflow configs delegate to ``nat run``. Platform-owned
    ``nemo-agents-spec-v1`` configs translate to an in-memory ``FabricConfig``
    and use Fabric's one-shot runtime lifecycle.
    """
    import subprocess

    from nemo_agents_plugin.utils import temp_injected_config

    if input_file:
        queries = json.loads(input_file.read_text(encoding="utf-8"))
        if not isinstance(queries, list):
            queries = [queries]
    elif input:
        queries = [input]
    else:
        typer.echo("Error: provide --input or --input-file.", err=True)
        raise typer.Exit(code=1)

    config_dict = _load_yaml(agent_config)
    config_format = config_dict.get("config_format", NAT_WORKFLOW_CONFIG_FORMAT)
    if config_format == NEMO_AGENTS_SPEC_CONFIG_FORMAT:
        _local_fabric_invoke(config_dict, queries, base_dir=agent_config.parent)
        return
    if config_format != NAT_WORKFLOW_CONFIG_FORMAT:
        typer.echo(f"Error: unsupported config_format {config_format!r}", err=True)
        raise typer.Exit(code=1)

    with temp_injected_config(agent_config, workspace, base_url=base_url) as injected_path:
        for query in queries:
            cmd = ["nat", "run", "--config_file", injected_path.name, "--input", query]
            try:
                subprocess.run(cmd, check=True, cwd=injected_path.parent)
            except subprocess.CalledProcessError as exc:
                typer.echo(f"Error: nat run exited with code {exc.returncode}.", err=True)
                raise typer.Exit(code=exc.returncode)
            except FileNotFoundError:
                typer.echo("Error: 'nat' command not found.  Install nvidia-nat-core.", err=True)
                raise typer.Exit(code=1)


def _local_fabric_invoke(config: dict[str, Any], inputs: list[Any], *, base_dir: Path) -> None:
    """Invoke a Platform-owned agent config through Fabric and print results."""
    from nemo_agents_plugin.agent_config import AgentConfig
    from nemo_agents_plugin.fabric.invocation import invoke_agent_config_once
    from nemo_agents_plugin.fabric.runtime import FabricRuntimeExecutionError
    from nemo_agents_plugin.fabric.translator import FabricTranslationError

    try:
        agent_config = AgentConfig.model_validate(config)
        results = asyncio.run(invoke_agent_config_once(agent_config, inputs, base_dir=base_dir))
    except (FabricRuntimeExecutionError, FabricTranslationError, ValidationError) as error:
        typer.echo(f"Error: Fabric invocation failed: {error}", err=True)
        raise typer.Exit(code=1) from error

    failed = False
    for result in results:
        typer.echo(json.dumps(asdict(result), indent=2))
        if result.status != "succeeded":
            failed = True
    if failed:
        raise typer.Exit(code=1)


def _platform_invoke(
    base_url: str,
    workspace: str,
    agent: Optional[str],
    deployment: Optional[str],
    input: Optional[str],
    input_file: Optional[Path],
    *,
    timeout: float = 300,
    no_progress: bool = False,
) -> None:
    """Invoke an agent through the platform gateway."""
    if input_file:
        queries = json.loads(input_file.read_text(encoding="utf-8"))
        if not isinstance(queries, list):
            queries = [queries]
    elif input:
        queries = [input]
    else:
        typer.echo("Error: provide --input or --input-file.", err=True)
        raise typer.Exit(code=1)

    if agent:
        path = f"/apis/agents/v2/workspaces/{workspace}/agents/{agent}/-/v1/chat/completions"
    else:
        path = f"/apis/agents/v2/workspaces/{workspace}/deployments/{deployment}/-/v1/chat/completions"

    url = base_url.rstrip("/") + path
    headers = _resolve_context_headers()
    target_label = agent or deployment
    for query in queries:
        payload = {"messages": [{"role": "user", "content": query}], "stream": False}
        try:
            with request_progress(f"Waiting for agent '{target_label}'...", disabled=no_progress):
                with httpx.Client(timeout=timeout) as client:
                    resp = client.post(url, json=payload, headers=headers or None)
                    resp.raise_for_status()
                    body = resp.json()
                typer.echo(json.dumps(body, indent=2))
        except httpx.TimeoutException as exc:
            typer.echo(
                f"Error: invoke agent timed out after {timeout:.0f}s. "
                "Use --timeout to increase or set NEMO_AGENTS_INVOKE_TIMEOUT.",
                err=True,
            )
            raise typer.Exit(code=1) from exc
        except httpx.HTTPStatusError as exc:
            print_http_status_error(exc, action="invoke agent")
            raise typer.Exit(code=1)
        except httpx.RequestError as exc:
            print_http_request_error(exc, action="invoke agent")
            raise typer.Exit(code=1)


def _is_interactive_session_chat() -> bool:
    """Return whether the process can safely run the agent chat TUI."""
    return is_interactive() and is_tty()


def _validate_session_chat_options(
    *,
    agent_deployment: str | None,
    session: str | None,
    session_name: str | None,
    input: str | None,
) -> None:
    """Validate the mutually exclusive new-session and resume selectors."""
    if agent_deployment is not None and not agent_deployment.strip():
        typer.echo("Error: --agent-deployment must not be empty.", err=True)
        raise typer.Exit(code=2)
    if session is not None and not session.strip():
        typer.echo("Error: --session must not be empty.", err=True)
        raise typer.Exit(code=2)
    if session_name is not None and not session_name.strip():
        typer.echo("Error: --session-name must not be empty.", err=True)
        raise typer.Exit(code=2)
    if input is not None and not input.strip():
        typer.echo("Error: --input must not be empty.", err=True)
        raise typer.Exit(code=2)

    if (agent_deployment is None) == (session is None):
        typer.echo("Error: Provide exactly one of --agent-deployment or --session.", err=True)
        raise typer.Exit(code=2)
    if session_name is not None and agent_deployment is None:
        typer.echo("Error: --session-name can only be used with --agent-deployment.", err=True)
        raise typer.Exit(code=2)


def _platform_session_chat(
    *,
    base_url: str,
    workspace: str,
    agent_deployment: str | None,
    session: str | None,
    session_name: str | None,
    input: str | None,
    timeout: float,
) -> None:
    """Resolve a new or existing persisted session, then enter its chat transport."""
    if agent_deployment is not None:
        deployment, created_session, session_id = _create_session_for_deployment(
            base_url=base_url,
            workspace=workspace,
            deployment_name=agent_deployment,
            session_name=session_name,
        )
        typer.echo(
            f"Session '{created_session.name}' created. Resume with:\n"
            f"  nemo agents chat --session {created_session.name} --workspace {workspace} --base-url {base_url}"
        )
        _run_resolved_session_chat(
            base_url=base_url,
            workspace=workspace,
            deployment=deployment,
            session=created_session,
            session_id=session_id,
            input=input,
            timeout=timeout,
        )
        return

    assert session is not None
    deployment, resolved_session, session_id = _resolve_existing_session(
        base_url=base_url,
        workspace=workspace,
        session_name=session,
    )
    typer.echo("Resuming runtime context; prior messages are not redisplayed.")
    _run_resolved_session_chat(
        base_url=base_url,
        workspace=workspace,
        deployment=deployment,
        session=resolved_session,
        session_id=session_id,
        input=input,
        timeout=timeout,
    )


def _run_resolved_session_chat(
    *,
    base_url: str,
    workspace: str,
    deployment: AgentDeployment,
    session: AgentSession,
    session_id: str,
    input: str | None,
    timeout: float,
) -> None:
    """Run the shared TUI for an already-resolved session."""
    url = (
        f"{base_url.rstrip('/')}/apis/agents/v2/workspaces/{workspace}"
        f"/deployments/{deployment.name}/-/v1/chat/completions"
    )
    headers = {**_resolve_context_headers(), SESSION_ID_HEADER: session_id}
    display_info = {
        "Deployment": deployment.name,
        "Session": session.name,
        "Status": session.status.value,
    }
    if session.expires_at is not None:
        display_info["Expires"] = session.expires_at.isoformat()

    try:
        with httpx.Client(timeout=timeout) as client:

            def send_turn(user_input: str) -> StreamingResponse:
                request = client.build_request(
                    "POST",
                    url,
                    headers=headers,
                    json={"messages": [{"role": "user", "content": user_input}], "stream": True},
                )
                response = client.send(request, stream=True)
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError:
                    response.read()
                    response.close()
                    raise
                return cast(StreamingResponse, closing(response))

            run_chat_tui(
                send_turn=send_turn,
                display_info=display_info,
                initial_message=input,
                exit_action=ExitAction.DETACH,
            )
    except httpx.TimeoutException as exc:
        typer.echo(
            f"Error: session chat request timed out after {timeout:.0f}s. Use --timeout to increase it.",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    except httpx.HTTPStatusError as exc:
        print_http_status_error(exc, action="chat with deployed agent")
        raise typer.Exit(code=1)
    except httpx.RequestError as exc:
        print_http_request_error(exc, action="chat with deployed agent")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Platform API helpers
# ---------------------------------------------------------------------------


def _require_deployment_id(response: Any, deployment_name: str) -> str:
    """Return a valid deployment ID or exit with the standard response error."""
    try:
        deployment_id = response["id"]
        if not isinstance(deployment_id, str) or not deployment_id:
            raise ValueError
        return deployment_id
    except (KeyError, TypeError, ValueError):
        typer.echo(f"Error: Deployment '{deployment_name}' returned an invalid response.", err=True)
        raise typer.Exit(code=1)


def _create_session_for_deployment(
    *,
    base_url: str,
    workspace: str,
    deployment_name: str,
    session_name: str | None,
) -> tuple[AgentDeployment, AgentSession, str]:
    """Validate a Fabric deployment and create its persisted Platform session."""
    deployment_response = _api_request(
        "GET",
        base_url,
        f"/apis/agents/v2/workspaces/{workspace}/deployments/{deployment_name}",
    )
    deployment_id = _require_deployment_id(deployment_response, deployment_name)
    try:
        deployment = AgentDeployment.model_validate(deployment_response)
    except (KeyError, TypeError, ValueError):
        typer.echo(f"Error: Deployment '{deployment_name}' returned an invalid response.", err=True)
        raise typer.Exit(code=1)

    config_format = deployment.config.get("config_format")
    if config_format != NEMO_AGENTS_SPEC_CONFIG_FORMAT:
        typer.echo(
            f"Error: Deployment '{deployment.name}' is not Fabric-backed (config_format={config_format!r}).",
            err=True,
        )
        raise typer.Exit(code=1)
    if not is_deployment_routable(deployment):
        typer.echo(
            f"Error: Deployment '{deployment.name}' is not routable "
            f"(mode='{deployment.deployment_mode}', status='{deployment.status}').",
            err=True,
        )
        raise typer.Exit(code=1)

    payload = {"deployment_id": deployment_id}
    if session_name is not None:
        payload["name"] = session_name
    response = _api_request(
        "POST",
        base_url,
        f"/apis/agents/v2/workspaces/{workspace}/sessions",
        json_body=payload,
    )
    try:
        session_id = response["id"]
        if not isinstance(session_id, str) or not session_id:
            raise ValueError
        created_session = AgentSession.model_validate(response)
    except (KeyError, TypeError, ValueError):
        typer.echo("Error: Session creation returned an invalid response.", err=True)
        raise typer.Exit(code=1)

    return deployment, created_session, session_id


def _resolve_existing_session(
    *,
    base_url: str,
    workspace: str,
    session_name: str,
) -> tuple[AgentDeployment, AgentSession, str]:
    """Resolve a resumable persisted session and its bound deployment."""
    response = _api_request(
        "GET",
        base_url,
        f"/apis/agents/v2/workspaces/{workspace}/sessions/{session_name}",
    )
    try:
        session_id = response["id"]
        if not isinstance(session_id, str) or not session_id:
            raise ValueError
        resolved_session = AgentSession.model_validate(response)
    except (KeyError, TypeError, ValueError):
        typer.echo(f"Error: Session '{session_name}' returned an invalid response.", err=True)
        raise typer.Exit(code=1)
    if resolved_session.status is not SessionStatus.ACTIVE:
        typer.echo(
            f"Error: Session '{session_name}' is {resolved_session.status.value} and cannot be resumed.",
            err=True,
        )
        raise typer.Exit(code=1)
    if session_expiration_is_due(resolved_session, at=datetime.now(UTC)):
        typer.echo(f"Error: Session '{session_name}' is expired and cannot be resumed.", err=True)
        raise typer.Exit(code=1)

    deployment = _resolve_deployment_by_id(
        base_url=base_url,
        workspace=workspace,
        deployment_id=resolved_session.deployment_id,
        session_name=session_name,
    )
    return deployment, resolved_session, session_id


def _resolve_deployment_by_id(
    *,
    base_url: str,
    workspace: str,
    deployment_id: str,
    session_name: str,
) -> AgentDeployment:
    """Find a session's bound deployment through the paginated list API."""
    page = 1
    while True:
        query = urlencode({"page": page, "page_size": _DEPLOYMENT_RESOLUTION_PAGE_SIZE})
        response = _api_request(
            "GET",
            base_url,
            f"/apis/agents/v2/workspaces/{workspace}/deployments?{query}",
        )
        for candidate in _unwrap_list(response):
            if candidate.get("id") != deployment_id:
                continue
            try:
                deployment = AgentDeployment.model_validate(candidate)
            except ValidationError:
                typer.echo(
                    f"Error: Deployment ID '{deployment_id}' for session '{session_name}' "
                    "returned an invalid response.",
                    err=True,
                )
                raise typer.Exit(code=1)
            return deployment

        pagination = response.get("pagination") if isinstance(response, dict) else None
        total_pages = pagination.get("total_pages") if isinstance(pagination, dict) else None
        if not isinstance(total_pages, int) or page >= total_pages:
            break
        page += 1

    typer.echo(
        f"Error: Session '{session_name}' references deployment ID '{deployment_id}', "
        f"but no matching deployment exists in workspace '{workspace}'.",
        err=True,
    )
    raise typer.Exit(code=1)


def _unwrap_list(resp: Any) -> list[dict[str, Any]]:
    """Extract the item list from a paginated or raw API response."""
    items = resp.get("data", resp) if isinstance(resp, dict) else resp
    return [d for d in items if isinstance(d, dict)]


def _print_list_response(
    ctx: typer.Context,
    response: Any,
    *,
    default_columns: list[Column],
    output_format: _LIST_OUTPUT_FORMAT | None,
    no_truncate: bool | None,
) -> None:
    """Print a list response with table output by default and JSON opt-in."""
    format_output(
        response,
        is_list=True,
        output_format=_resolve_list_output_format(ctx, output_format),
        output_columns=default_columns,
        no_truncate=_resolve_no_truncate(ctx, no_truncate),
        timestamp_format=_resolve_timestamp_format(ctx),
    )


def _resolve_list_output_format(ctx: typer.Context, output_format: _LIST_OUTPUT_FORMAT | None) -> str:
    """Resolve command-level format, then global CLI preference, then table."""
    if output_format is not None:
        return output_format

    state = ctx.obj
    if state is not None and hasattr(state, "get_output_format"):
        try:
            return state.get_output_format(apply_non_tty_default=False)
        except Exception:
            logger.debug("Failed to resolve global output format for agents list", exc_info=True)
    return "table"


def _resolve_no_truncate(ctx: typer.Context, no_truncate: bool | None) -> bool | None:
    """Resolve command-level truncation, falling back to the global CLI preference."""
    if no_truncate is not None:
        return no_truncate

    state = ctx.obj
    if state is not None and hasattr(state, "get_no_truncate"):
        try:
            return state.get_no_truncate()
        except Exception:
            logger.debug("Failed to resolve global truncation preference for agents list", exc_info=True)
    return None


def _resolve_timestamp_format(ctx: typer.Context) -> str | None:
    """Resolve the global timestamp preference when the plugin is mounted under ``nemo``."""
    state = ctx.obj
    if state is not None and hasattr(state, "get_timestamp_format"):
        try:
            return state.get_timestamp_format()
        except Exception:
            logger.debug("Failed to resolve global timestamp format for agents list", exc_info=True)
    return None


def _api_request(method: str, base_url: str, path: str, *, json_body: dict[str, Any] | None = None) -> Any:
    url = base_url.rstrip("/") + path
    request_kwargs: dict[str, Any] = {}
    if json_body is not None:
        request_kwargs["json"] = json_body
    headers = _resolve_context_headers()
    if headers:
        request_kwargs["headers"] = headers
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.request(method, url, **request_kwargs)
            resp.raise_for_status()
            if resp.status_code == 204 or not resp.content:
                return None
            return resp.json()
    except httpx.HTTPStatusError as exc:
        print_http_status_error(exc, action=f"{method} agent API")
        raise typer.Exit(code=1)
    except httpx.RequestError as exc:
        print_http_request_error(exc, action=f"{method} agent API")
        raise typer.Exit(code=1)


def _platform_sdk(base_url: str) -> Any:
    """Return an auth-aware platform SDK client for fileset upload/delete."""
    headers = _resolve_context_headers()
    if headers:
        return NeMoPlatform(base_url=base_url, default_headers=headers)
    return NeMoPlatform(base_url=base_url)


def _agents_sdk(base_url: str, workspace: str) -> Any:
    """Return the agents-plugin SDK resource bound to *base_url* / *workspace*.

    The CLI reuses the plugin SDK (:class:`~nemo_agents_plugin.sdk.AgentsResource`)
    so request paths and payload shapes are defined once. The SDK reads
    ``base_url``, ``workspace``, and ``default_headers`` off the object it is
    handed — mirroring the auth-header threading ``_api_request`` does.
    """
    from nemo_agents_plugin.sdk import AgentsResource

    platform = SimpleNamespace(
        base_url=base_url,
        workspace=workspace,
        default_headers=_resolve_context_headers() or None,
    )
    return AgentsResource(platform)


def _run_sdk(action: str, call: Callable[[], Any]) -> Any:
    """Invoke an SDK *call*, translating HTTP errors to CLI-friendly output.

    The plugin SDK raises raw ``httpx`` errors; the CLI wants the same rich
    messages ``_api_request`` prints. This keeps error UX identical whether a
    command talks to the API directly or through the SDK.
    """
    try:
        return call()
    except httpx.HTTPStatusError as exc:
        print_http_status_error(exc, action=action)
        raise typer.Exit(code=1)
    except httpx.RequestError as exc:
        print_http_request_error(exc, action=action)
        raise typer.Exit(code=1)


def _collect_text_agent_artifacts(
    agent_root: Path,
    *,
    excluded_paths: set[Path] | None = None,
    warn_on_binary: bool = False,
) -> list[tuple[Path, bytes]]:
    """Return the UTF-8 files that can be delivered as Fabric config files.

    Container deployments carry agent artifacts as text-only ``ConfigFile``
    values. Binary neighbors such as ``__pycache__/*.pyc`` and wheels are not
    agent configuration, so omit them from the fileset instead of making agent
    creation fail later during container staging.

    Symlinks are rejected rather than skipped: the upload enumerates them and
    would otherwise ship target content from outside *agent_root*.
    """
    excluded_paths = excluded_paths or set()
    total_bytes = 0
    file_count = 0
    artifacts: list[tuple[Path, bytes]] = []
    for path in sorted(agent_root.rglob("*")):
        relative_path = path.relative_to(agent_root)
        if path.is_symlink():
            raise ValueError(
                f"agent directory {str(agent_root)!r} contains symlink "
                f"{relative_path.as_posix()!r}; the fileset upload follows "
                "symlinks, which would stage content from outside the agent directory. "
                "Replace it with a regular file or move the target inside the agent directory"
            )
        if not path.is_file() or relative_path in excluded_paths:
            continue
        try:
            content = path.read_bytes()
            content.decode("utf-8")
        except UnicodeDecodeError:
            if warn_on_binary:
                typer.echo(
                    f"Warning: skipping non-UTF-8 agent artifact {relative_path.as_posix()!r}; "
                    "Fabric fileset staging supports text files only.",
                    err=True,
                )
            continue
        file_count += 1
        total_bytes += len(content)
        if file_count > MAX_ETHOS_STAGED_FILES:
            raise ValueError(
                f"agent directory {str(agent_root)!r} holds more than "
                f"{MAX_ETHOS_STAGED_FILES} files; point --agent-config at a "
                "directory containing only the agent's own artifacts"
            )
        if total_bytes > MAX_ETHOS_STAGED_BYTES:
            raise ValueError(
                f"agent directory {str(agent_root)!r} exceeds the "
                f"{MAX_ETHOS_STAGED_BYTES} byte limit for container config delivery; "
                "point --agent-config at a directory containing only the agent's own artifacts"
            )
        artifacts.append((relative_path, content))
    return artifacts


def _clear_existing_ethos_artifacts(
    *,
    sdk: NeMoPlatform,
    fileset: str,
    workspace: str,
) -> None:
    """Remove the previous executable snapshot while preserving durable Ethos."""
    from nemo_platform import NotFoundError as PlatformNotFoundError
    from nemo_platform_plugin.client.errors import NotFoundError as PluginNotFoundError

    preserved = {ETHOS_FILENAME}

    try:
        existing = sdk.files.list(fileset=fileset, workspace=workspace).data
    except (FileNotFoundError, PlatformNotFoundError, PluginNotFoundError):
        return

    for artifact in existing:
        remote_path = artifact.path
        if remote_path in preserved:
            continue
        try:
            sdk.files.delete(remote_path=remote_path, fileset=fileset, workspace=workspace)
        except (FileNotFoundError, PlatformNotFoundError, PluginNotFoundError):
            # Another client may have removed the same stale file after the list.
            continue


def _spec_package_warning(agent: str, agent_config: Path) -> tuple[str, ...]:
    """Return skill guidance when *agent_config* lives in a spec package."""
    if not agent or agent in {".", ".."} or "\0" in agent:
        return ()
    if "/" in agent or "\\" in agent or Path(agent).is_absolute() or Path(agent).name != agent:
        return ()
    package = Path(ETHOS_LOCAL_ROOT) / f"{agent}-spec"
    if agent_config.parent.resolve() != package.resolve():
        return ()
    if not (package / AGENT_SPEC_FILENAME).is_file():
        return ()
    return (
        f"Warning: This package uses {AGENT_SPEC_FILENAME}.",
        f"Run the nemo-ethos skill to write {ETHOS_FILENAME}, then delete the {agent}-spec package.",
    )


def _upload_ethos_fileset(
    *,
    agent_name: str,
    workspace: str,
    agent_root: Path,
    base_url: str,
) -> None:
    """Replace the executable snapshot in the conventional Ethos fileset.

    *agent_root* is ``agent.yaml``'s parent directory (Fabric ``base_dir``).
    UTF-8 sibling artifacts such as skills and prompts are uploaded; binary
    neighbors are skipped because deployment ``ConfigFile`` values are text.
    Existing runtime artifacts are removed first so recreating an agent cannot
    reuse a stale bundle. Durable ``ETHOS.md`` is preserved, while
    ``AGENT-SPEC.md`` is omitted so a spec package does not leave its legacy
    contract in the Ethos fileset.
    """
    from nemo_agents_plugin.jobs.fileset_io import upload_to_fileset

    excluded_paths = {Path(ETHOS_FILENAME), Path(AGENT_SPEC_FILENAME)}
    artifacts = _collect_text_agent_artifacts(
        agent_root,
        excluded_paths=excluded_paths,
        warn_on_binary=True,
    )
    fileset = ethos_fileset_name(agent_name)
    sdk = _platform_sdk(base_url)
    with tempfile.TemporaryDirectory(prefix=f".{agent_name}-ethos-upload-") as directory:
        staged = Path(directory) / agent_root.name
        staged.mkdir()
        for relative_path, content in artifacts:
            destination = staged / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)

        # Validate and stage the complete replacement before touching the remote
        # fileset. The durable Ethos contract survives even when it is absent
        # from this executable snapshot.
        _clear_existing_ethos_artifacts(
            sdk=sdk,
            fileset=fileset,
            workspace=workspace,
        )
        upload_to_fileset(
            staged,
            fileset=fileset,
            workspace=workspace,
            sdk=sdk,
        )


def _delete_agent_entity(*, agent_name: str, workspace: str, base_url: str) -> None:
    """Delete the agent entity, leaving the ``{agent}-ethos`` fileset in place.

    The fileset outlives the agent on purpose: it is the canonical home of
    ``ETHOS.md`` (see ``ethos_file_ref``), which ``nemo-ethos`` writes
    before the agent exists and ``nemo-build-agent`` reads on every rebuild.
    Deleting the fileset here would destroy that durable contract, so the
    executable artifacts it also carries are left behind instead.
    """
    _api_request("DELETE", base_url, f"/apis/agents/v2/workspaces/{workspace}/agents/{agent_name}")


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# Mirrors ``nemo_agents_plugin.utils._ENV_VAR_PATTERN`` semantics: matches
# ``${NEMO_DEFAULT_MODEL}`` and bare ``$NEMO_DEFAULT_MODEL`` (with an
# identifier-boundary lookahead so ``$NEMO_DEFAULT_MODELX`` is not matched).
_DEFAULT_MODEL_PLACEHOLDER = re.compile(r"\$(?:\{NEMO_DEFAULT_MODEL\}|NEMO_DEFAULT_MODEL(?![A-Za-z0-9_]))")


def _contains_default_model_placeholder(value: Any) -> bool:
    """Return True if *value* still contains an unresolved ``NEMO_DEFAULT_MODEL`` reference."""
    if isinstance(value, str):
        # Honor ``$$`` escape the same way ``expand_env_vars`` does.
        protected = value.replace("$$", "\0DOLLAR\0")
        return _DEFAULT_MODEL_PLACEHOLDER.search(protected) is not None
    if isinstance(value, dict):
        return any(_contains_default_model_placeholder(v) for v in value.values())
    if isinstance(value, list):
        return any(_contains_default_model_placeholder(v) for v in value)
    return False


def _validate_platform_agent_config_for_cli(config: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    from nemo_agents_plugin.fabric.validation import FabricValidationError, validate_platform_agent_config

    try:
        validation_result = asyncio.run(validate_platform_agent_config(config, base_dir=base_dir))
    except FabricValidationError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    return validation_result.agent_config.model_dump(exclude_none=True)
