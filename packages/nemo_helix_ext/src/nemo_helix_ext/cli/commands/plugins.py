# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI commands for plugin discovery and external plugin registration."""

from __future__ import annotations

from typing import Annotated

import click
import typer
from nemo_helix_plugin.cli_confirm import confirm_destructive
from nemo_helix_plugin.cli_options import YesOption
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.community_plugins.client import CommunityPluginsClient
from nemo_helix_plugin.community_plugins.types import (
    EnableRequest,
    InstallRequest,
    ManifestSource,
    PreviewResponse,
)

from nemo_helix_ext.cli.core.code_generator import handle_code_generation
from nemo_helix_ext.cli.core.context import CLIContext
from nemo_helix_ext.cli.core.errors import handle_errors
from nemo_helix_ext.cli.core.formatters import (
    Column,
    check_output_columns_with_format,
    format_output,
    validate_stream_output_format,
)
from nemo_helix_ext.cli.core.help_formatter import collect_warnings, create_typer_app
from nemo_helix_ext.cli.core.types import (
    EntityOutputFormatOption,
    ListOutputFormatOption,
    NoTruncateOption,
    OutputColumnsOption,
    StreamOutputOption,
)

app = create_typer_app(
    name="plugins",
    help="""\
Commands for plugin discovery.

External plugins served by their own HTTP adapter can be previewed, registered,
and removed with `preview`, `install`, `installed`, and `uninstall`.

Examples:
# List installed plugins.
nemo plugins list
# Review an external plugin's manifest without installing it.
nemo plugins preview http://127.0.0.1:6007/plugin.yaml
# Register it (shows the manifest and asks for consent).
nemo plugins install http://127.0.0.1:6007/plugin.yaml
# List and remove registered external plugins.
nemo plugins installed
nemo plugins uninstall phoenix""",
)


def _plugins_client(state: CLIContext) -> CommunityPluginsClient:
    return client_from_platform(state.get_client(), CommunityPluginsClient)


def _render_preview(preview: PreviewResponse) -> str:
    """Human-readable summary of everything an install would approve."""
    plugin = preview.manifest.plugin
    lines = [f"Plugin:   {plugin.name} ({plugin.title})", "Endpoints:"]
    lines += [
        f"  {e.method:<6} {e.access:<5} /apis/plugins/{plugin.name}{e.path} -> {e.upstream_url}"
        for e in plugin.endpoints
    ]
    if plugin.credential is not None:
        lines.append(f"Credential: workspace secret '{plugin.credential.secret_name}' ({plugin.credential.scheme})")
    if plugin.bundle is not None:
        lines.append(f"UI bundle:  {plugin.bundle.url}")
        lines.append(f"  sha256:   {plugin.bundle.sha256}")
    lines += [f"Nav:        {n.section} / {n.label} -> {n.path}" for n in plugin.navigation]
    lines += [f"Manifest sha256: {preview.manifest_sha256}", "", f"WARNING: {preview.warning}"]
    return "\n".join(lines)


@app.callback(invoke_without_command=True)
def plugins_callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command("list")
@collect_warnings
@handle_errors
def list_plugins(
    ctx: typer.Context,
    output_format: ListOutputFormatOption = None,
    no_truncate: NoTruncateOption = None,
    columns: OutputColumnsOption = None,
    stream: StreamOutputOption = False,
) -> None:
    """List installed plugins.

    Discovers installed plugins from registered NeMo plugin entry points.

    Examples:
      nemo plugins list
      nemo plugins list -f json
    """
    state: CLIContext = ctx.obj
    columns_explicit = columns is not None and str(columns).strip() != "default"
    output_format = state.get_output_format(output_format, apply_non_tty_default=not columns_explicit)
    validate_stream_output_format(output_format, stream)

    check_output_columns_with_format(columns, output_format)

    default_columns = [
        Column("name", None),
        Column("version", None),
        Column("description", None),
    ]
    output_columns: str | list[Column] | None = columns
    if not columns_explicit:
        output_columns = default_columns

    try:
        from nemo_helix_plugin.discovery import discover_manifests

        manifests = discover_manifests()
    except ImportError:
        manifests = {}

    items = [
        {
            "name": manifest.name,
            "version": manifest.version,
            "description": manifest.description,
        }
        for manifest in sorted(manifests.values(), key=lambda manifest: manifest.name)
    ]

    format_output(
        items,
        is_list=True,
        output_format=output_format,
        output_columns=output_columns,
        no_truncate=state.get_no_truncate(no_truncate),
        timestamp_format=state.get_timestamp_format(),
        stream=stream,
    )


@app.command("preview")
@collect_warnings
@handle_errors
def preview_plugin(
    ctx: typer.Context,
    url: Annotated[str, typer.Argument(help="URL of the external plugin's plugin.yaml manifest.")],
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Show what installing an external plugin would approve. Changes nothing.

    Fetches the manifest through the platform and prints its endpoints, credential
    reference, UI bundle hash, and the trust warning, plus the manifest digest that
    `install --manifest-sha256` pins.

    Examples:
      nemo plugins preview http://127.0.0.1:6007/plugin.yaml
      nemo plugins preview http://127.0.0.1:6007/plugin.yaml -f json
    """
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)
    body = ManifestSource(url=url)
    if handle_code_generation(CommunityPluginsClient, "preview_plugin", {"body": body}, resolved_output_format, state):
        return

    preview = _plugins_client(state).preview_plugin(body=body).data()
    if resolved_output_format in ("json", "yaml"):
        format_output(
            preview.model_dump(mode="json", by_alias=True), is_list=False, output_format=resolved_output_format
        )
    else:
        typer.echo(_render_preview(preview))


@app.command("install")
@collect_warnings
@handle_errors
def install_plugin(
    ctx: typer.Context,
    url: Annotated[str, typer.Argument(help="URL of the external plugin's plugin.yaml manifest.")],
    manifest_sha256: Annotated[
        str | None,
        typer.Option(
            "--manifest-sha256",
            help="Digest of the manifest you reviewed (from `plugins preview`). Install fails if it differs.",
        ),
    ] = None,
    yes: YesOption = False,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Register an external plugin after reviewing and consenting to its manifest.

    Previews the manifest, prints what you are approving, and asks for confirmation.
    The install is pinned to the previewed digest, so a manifest that changes in
    between is rejected. A UI bundle runs inside Studio with Studio-origin
    privileges; only install plugins you trust.

    --yes skips the prompt but requires --manifest-sha256, so a non-interactive
    install can only approve a snapshot someone already reviewed.

    Examples:
      nemo plugins install http://127.0.0.1:6007/plugin.yaml
      nemo plugins install http://127.0.0.1:6007/plugin.yaml --manifest-sha256 <digest> --yes
    """
    if yes and manifest_sha256 is None:
        raise click.UsageError("--yes requires --manifest-sha256 so only a reviewed manifest is approved.")

    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)
    source = ManifestSource(url=url)
    client = _plugins_client(state)
    preview = client.preview_plugin(body=source).data()
    if manifest_sha256 is not None and manifest_sha256 != preview.manifest_sha256:
        raise click.ClickException(
            f"Manifest digest mismatch: expected {manifest_sha256}, got {preview.manifest_sha256}. "
            "Run `nemo plugins preview` and review the manifest again."
        )

    typer.echo(_render_preview(preview), err=True)
    name = preview.manifest.plugin.name
    confirm_destructive(f"Install plugin '{name}' with the access shown above?", yes=yes)

    installed = client.install_plugin(
        body=InstallRequest(url=url, manifest_sha256=preview.manifest_sha256, consent=True)
    ).data()
    if resolved_output_format in ("json", "yaml"):
        format_output(
            installed.model_dump(mode="json", by_alias=True), is_list=False, output_format=resolved_output_format
        )
    else:
        typer.echo(f"✓ Installed '{installed.name}' (manifest {installed.manifest_sha256})")


@app.command("installed")
@collect_warnings
@handle_errors
def list_installed_plugins(
    ctx: typer.Context,
    output_format: ListOutputFormatOption = None,
    no_truncate: NoTruncateOption = None,
    columns: OutputColumnsOption = None,
) -> None:
    """List external plugins registered with the platform.

    Examples:
      nemo plugins installed
      nemo plugins installed -f json
    """
    state: CLIContext = ctx.obj
    columns_explicit = columns is not None and str(columns).strip() != "default"
    resolved_output_format = state.get_output_format(output_format, apply_non_tty_default=not columns_explicit)
    check_output_columns_with_format(columns, resolved_output_format)
    if handle_code_generation(CommunityPluginsClient, "list_installed_plugins", {}, resolved_output_format, state):
        return

    registry = _plugins_client(state).list_installed_plugins().data()
    items = [
        {
            "name": plugin.name,
            "title": plugin.title,
            "enabled": plugin.enabled,
            "endpoints": len(plugin.endpoints),
            "bundle_url": plugin.bundle_url,
            "manifest_sha256": plugin.manifest_sha256,
        }
        for plugin in registry.plugins
    ]
    output_columns: str | list[Column] | None = columns
    if not columns_explicit:
        output_columns = [
            Column("name", None),
            Column("title", None),
            Column("enabled", None),
            Column("endpoints", None),
        ]
    format_output(
        items,
        is_list=True,
        output_format=resolved_output_format,
        output_columns=output_columns,
        no_truncate=state.get_no_truncate(no_truncate),
        timestamp_format=state.get_timestamp_format(),
    )


@app.command("uninstall")
@collect_warnings
@handle_errors
def uninstall_plugin(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Name of the registered external plugin.")],
    yes: YesOption = False,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Remove an external plugin's registration.

    Removes its routes and Studio navigation entry. It does not stop the plugin's
    own server, delete data it stored, or delete the workspace secret it used.

    Examples:
      nemo plugins uninstall phoenix
    """
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)
    if handle_code_generation(
        CommunityPluginsClient, "uninstall_plugin", {"name": name}, resolved_output_format, state, result="none"
    ):
        return
    confirm_destructive(f"Uninstall plugin '{name}'?", yes=yes)
    _plugins_client(state).uninstall_plugin(name=name).data()
    typer.echo(f"✓ Uninstalled '{name}'")


def _set_enabled(ctx: typer.Context, name: str, enabled: bool) -> None:
    state: CLIContext = ctx.obj
    _plugins_client(state).set_plugin_enabled(name=name, body=EnableRequest(enabled=enabled)).data()
    typer.echo(f"✓ {'Enabled' if enabled else 'Disabled'} '{name}'")


@app.command("enable")
@collect_warnings
@handle_errors
def enable_plugin(
    ctx: typer.Context, name: Annotated[str, typer.Argument(help="Name of the registered external plugin.")]
) -> None:
    """Re-enable a disabled external plugin.

    Examples:
      nemo plugins enable phoenix
    """
    _set_enabled(ctx, name, True)


@app.command("disable")
@collect_warnings
@handle_errors
def disable_plugin(
    ctx: typer.Context, name: Annotated[str, typer.Argument(help="Name of the registered external plugin.")]
) -> None:
    """Disable an external plugin without removing its registration.

    Examples:
      nemo plugins disable phoenix
    """
    _set_enabled(ctx, name, False)
