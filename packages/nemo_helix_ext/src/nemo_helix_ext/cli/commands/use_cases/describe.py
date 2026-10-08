# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""nemo describe command - describe the installed CLI or any command in it."""

from __future__ import annotations

import json
import logging
from importlib.metadata import EntryPoint
from typing import Annotated, Any, Literal

import typer
from nemo_helix_plugin.cli_options import OUTPUT_FORMAT_FLAGS

from nemo_helix_ext.cli.core.command_description import describe_command, resolve_command_path
from nemo_helix_ext.cli.manifest import build_top_level_entries

DescribeOutputFormat = Literal["markdown", "json"]

logger = logging.getLogger(__name__)

_SURFACE_GROUPS: tuple[tuple[str, str], ...] = (
    ("nemo.cli", "CLI"),
    ("nemo.controllers", "Controllers"),
    ("nemo.docs", "Docs"),
    ("nemo.executors", "Executors"),
    ("nemo.inference_middleware", "InferenceMiddleware"),
    ("nemo.jobs", "Tasks"),
    ("nemo.mcp", "MCP"),
    ("nemo.sdk", "SDK"),
    ("nemo.seed", "Seed"),
    ("nemo.services", "Services"),
    ("nemo.skills", "Skills"),
    ("nemo.studio", "Studio"),
)


def _normalize_cell(value: object) -> str:
    if value is None:
        return ""
    return str(value).replace("|", r"\|").replace("\n", " ").replace("\r", "")


def _plugin_name_for_entry_point(entry_point_name: str, entry_point_group: str) -> str:
    if entry_point_group == "nemo.jobs":
        return entry_point_name.split(".", 1)[0]
    return entry_point_name


def _build_plugin_surfaces() -> dict[str, list[str]]:
    """Map plugin name → list of surface labels it provides (metadata-only)."""
    try:
        from nemo_helix_plugin.discovery import discover_entry_points
    except ImportError:
        return {}

    plugin_surfaces: dict[str, list[str]] = {}
    for entry_point_group, label in _SURFACE_GROUPS:
        try:
            for entry_point_name in discover_entry_points(entry_point_group):
                plugin_name = _plugin_name_for_entry_point(entry_point_name, entry_point_group)
                plugin_surfaces.setdefault(plugin_name, [])
                if label not in plugin_surfaces[plugin_name]:
                    plugin_surfaces[plugin_name].append(label)
        except Exception:
            logger.warning(
                "Failed to discover entry points for group %r",
                entry_point_group,
                exc_info=True,
            )

    return plugin_surfaces


def _all_top_level_entries() -> list[tuple[str, str, str]]:
    """Return (name, panel, help_first_line) for every registered top-level command."""
    from nemo_helix_ext.cli.app import module_entries_without_plugin_overrides
    from nemo_helix_ext.cli.commands.manifest_registry import TOP_LEVEL_ENTRIES

    plugin_entry_points: dict[str, EntryPoint] = {}

    try:
        from nemo_helix_plugin.discovery import discover_entry_points

        plugin_entry_points = discover_entry_points("nemo.cli")
    except ImportError:
        pass
    except Exception:  # noqa: BLE001
        logger.warning("Failed to discover CLI plugin entry points", exc_info=True)

    entries = build_top_level_entries(
        module_entries_without_plugin_overrides(TOP_LEVEL_ENTRIES, plugin_entry_points),
        plugin_entry_points,
        include_hidden=False,
    )

    return [(entry.name, entry.panel, entry.help.splitlines()[0] if entry.help else "") for entry in entries]


def describe_cli_command(
    ctx: typer.Context,
    path: Annotated[
        list[str] | None,
        typer.Argument(
            help="Command path to describe, such as 'models create'. Tokens after the path are ignored.",
            metavar="COMMAND...",
            show_default=False,
        ),
    ] = None,
    output_format: Annotated[
        DescribeOutputFormat,
        typer.Option(*OUTPUT_FORMAT_FLAGS, help="Output format. Put it before the command path."),
    ] = "markdown",
) -> None:
    """Describe the NeMo Helix CLI or any command in it.

    Without a command path, prints an overview of installed plugins,
    top-level commands, the plugin entry-point catalog, agent skills, and
    quick-reference patterns. With a command path, prints that command's
    usage, arguments, options, and subcommands.

    Tokens after the command path, such as the command's own arguments and
    options, are ignored, so 'describe' can go in front of a full command
    line. Put describe's own options before the path. Reads local metadata
    only and does not contact the platform.

    Examples:
    # Overview of the installed CLI.
    nemo describe
    # Describe one command.
    nemo describe models create
    # Describe a full command line as JSON.
    nemo describe -f json models create my-model --spec-file model.yaml
    """
    tokens = path or []
    if not tokens and output_format == "markdown":
        typer.echo(render_cli_description())
        return

    resolved = resolve_command_path(ctx.find_root().command, tokens)
    description = describe_command(
        resolved,
        # The root group lists its children from the manifest so describing
        # `nemo` itself does not import every plugin.
        subcommand_summaries=_top_level_summaries if resolved.context.parent is None else None,
    )
    if output_format == "json":
        typer.echo(json.dumps(description, indent=2))
    else:
        typer.echo(render_command_description(description))


# Stop parsing describe's options at the command path, so the described
# command's own options pass through as path tokens instead of being rejected.
describe_cli_command.__nhx_context_settings__ = {"allow_interspersed_args": False}  # type: ignore[attr-defined]


def _top_level_summaries() -> list[tuple[str, str]]:
    return [(name, description) for name, _panel, description in _all_top_level_entries()]


def render_command_description(description: dict[str, Any]) -> str:
    """Render one command's description as Markdown."""
    lines: list[str] = [f"# {description['command']}\n"]
    if description["deprecated"]:
        lines.append("_Deprecated._\n")
    if description["help"]:
        lines.append(f"{description['help']}\n")
    lines.append(f"Usage: `{description['usage']}`")

    if description["arguments"]:
        lines.append("\n## Arguments\n")
        lines.append("| Argument | Type | Required | Description |")
        lines.append("|----------|------|----------|-------------|")
        for argument in description["arguments"]:
            lines.append(
                f"| {_normalize_cell(argument['name'])} | {_normalize_cell(_type_label(argument))} "
                f"| {'yes' if argument['required'] else 'no'} | {_normalize_cell(argument['help'])} |"
            )

    if description["options"]:
        lines.append("\n## Options\n")
        lines.append("| Option | Type | Default | Description |")
        lines.append("|--------|------|---------|-------------|")
        for option in description["options"]:
            flags = ", ".join(option["flags"])
            if option["required"]:
                flags += " (required)"
            default = "" if option["default"] in (None, False, []) else f"`{option['default']}`"
            lines.append(
                f"| {_normalize_cell(flags)} | {_normalize_cell(_type_label(option))} "
                f"| {_normalize_cell(default)} | {_normalize_cell(option['help'])} |"
            )

    if description["subcommands"]:
        lines.append("\n## Subcommands\n")
        lines.append("| Command | Description |")
        lines.append("|---------|-------------|")
        for subcommand in description["subcommands"]:
            lines.append(
                f"| {_normalize_cell(description['command'])} {_normalize_cell(subcommand['name'])} "
                f"| {_normalize_cell(subcommand['help'])} |"
            )

    if description["ignored_args"]:
        lines.append(f"\n_Ignored arguments: `{' '.join(description['ignored_args'])}`_")

    return "\n".join(lines)


def _type_label(param: dict[str, Any]) -> str:
    if param.get("is_flag"):
        return "flag"
    if "choices" in param:
        return " | ".join(str(choice) for choice in param["choices"])
    return str(param["type"])


def render_cli_description() -> str:
    """Render the Markdown description of the installed CLI."""
    lines: list[str] = []

    lines.append("# NeMo Helix CLI\n")
    lines.append("## Installed Plugins\n")

    plugin_surfaces = _build_plugin_surfaces()

    try:
        from nemo_helix_plugin.discovery import discover_manifests

        manifests = discover_manifests()
    except Exception:
        logger.warning("Failed to discover plugin manifests", exc_info=True)
        manifests = {}

    all_plugin_names = sorted(manifests.keys() | plugin_surfaces.keys())
    if all_plugin_names:
        lines.append("| Plugin | Version | Description | Surfaces |")
        lines.append("|--------|---------|-------------|----------|")
        for name in all_plugin_names:
            manifest = manifests.get(name)
            version = _normalize_cell(getattr(manifest, "version", None))
            desc = _normalize_cell(getattr(manifest, "description", None))
            surfaces = _normalize_cell(", ".join(plugin_surfaces.get(name, [])))
            lines.append(f"| {_normalize_cell(name)} | {version} | {desc} | {surfaces} |")
    else:
        lines.append("_No plugins installed._")

    lines.append("\n## Available CLI Commands\n")
    lines.append(render_commands_table())

    lines.append("\n## Entry-Point Catalog\n")

    try:
        from nemo_helix_plugin.discovery import discover_entry_points

        for entry_point_group, label in _SURFACE_GROUPS:
            try:
                entry_points = discover_entry_points(entry_point_group)
                if not entry_points:
                    continue
                lines.append(f"### {label} (`{entry_point_group}`)\n")
                for entry_point_name in sorted(entry_points):
                    lines.append(f"- `{entry_point_name}`")
                lines.append("")
            except Exception:
                logger.warning(
                    "Failed to discover entry points for group %r",
                    entry_point_group,
                    exc_info=True,
                )
                lines.append(f"_Warning: {label} discovery failed._\n")
    except ImportError:
        lines.append("_nemo-helix-plugin not available; entry-point catalog unavailable._")

    lines.append("\n## Agent Skills\n")

    try:
        from nemo_helix_ext.cli.commands.skills.registry import list_agent_names, load_skills

        skills = load_skills()
        agent_names = list_agent_names()

        if skills:
            lines.append("| Skill | Version | Description |")
            lines.append("|-------|---------|-------------|")
            for skill in skills.values():
                lines.append(
                    f"| {_normalize_cell(skill.name)} | {_normalize_cell(skill.version)} | {_normalize_cell(skill.description)} |"
                )

        agents_str = ", ".join(agent_names)
        lines.append(f"\nInstall with: `nemo skills install --agent <agent>` (supported: {agents_str})")
        lines.append("List skills: `nemo skills list`")
    except Exception:
        logger.warning("Failed to load agent skills", exc_info=True)
        lines.append("_Skills unavailable._")

    lines.append("\n## Quick Reference\n")
    lines.append("- Read docs: `nemo docs <path>`")
    lines.append("- List doc topics: `nemo docs --list`")
    lines.append("- Describe a command: `nemo describe <command> [<subcommand>...]`")
    lines.append("- Describe a command as JSON: `nemo describe -f json <command> [<subcommand>...]`")
    lines.append("- Explore an API resource: `nemo <resource> --help`")
    lines.append("- List resources: `nemo <resource> list`")
    lines.append("- Get a resource: `nemo <resource> get <name-or-id>`")

    return "\n".join(lines)


def render_commands_table() -> str:
    """Render the Markdown table of visible top-level CLI commands."""
    lines: list[str] = ["| Command | Panel | Description |", "|---------|-------|-------------|"]
    for cmd_name, panel, description in _all_top_level_entries():
        lines.append(
            f"| nemo {_normalize_cell(cmd_name)} | {_normalize_cell(panel)} | {_normalize_cell(description)} |"
        )
    return "\n".join(lines)
