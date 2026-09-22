# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI commands for AI agent context and capability discovery."""

from __future__ import annotations

import logging
from importlib.metadata import EntryPoint

import typer

from nemo_platform_ext.cli.core.help_formatter import create_typer_app
from nemo_platform_ext.cli.manifest import build_top_level_entries

logger = logging.getLogger(__name__)

app = create_typer_app(
    name="agent",
    no_args_is_help=False,
    help="""\
Commands for AI agent context and capability discovery.

Examples:
# Dump full agent context (plugins, commands, skills).
nemo agent context
# List all available commands.
nemo agent commands""",
)

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
        from nemo_platform_plugin.discovery import discover_entry_points
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
    from nemo_platform_ext.cli.app import module_entries_without_plugin_overrides
    from nemo_platform_ext.cli.commands.manifest_registry import TOP_LEVEL_ENTRIES

    plugin_entry_points: dict[str, EntryPoint] = {}

    try:
        from nemo_platform_plugin.discovery import discover_entry_points

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


@app.callback(invoke_without_command=True)
def agent_callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(0)


@app.command("context")
def context_command() -> None:
    """Dump everything an agent needs in one call.

    Outputs installed plugins, CLI commands, entry-point catalog,
    available skills, and quick-reference patterns. Runs without a
    connected cluster (metadata-only).

    Examples:
      nemo agent context
    """
    lines: list[str] = []

    lines.append("# NeMo Platform Agent Context\n")
    lines.append("## Installed Plugins\n")

    plugin_surfaces = _build_plugin_surfaces()

    try:
        from nemo_platform_plugin.discovery import discover_manifests

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
    lines.append("| Command | Panel | Description |")
    lines.append("|---------|-------|-------------|")
    for cmd_name, panel, description in _all_top_level_entries():
        lines.append(
            f"| nemo {_normalize_cell(cmd_name)} | {_normalize_cell(panel)} | {_normalize_cell(description)} |"
        )

    lines.append("\n## Entry-Point Catalog\n")

    try:
        from nemo_platform_plugin.discovery import discover_entry_points

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
        lines.append("_nemo-platform-plugin not available; entry-point catalog unavailable._")

    lines.append("\n## Agent Skills\n")

    try:
        from nemo_platform_ext.cli.commands.skills.registry import list_agent_names, load_skills

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
    lines.append("- List all commands: `nemo agent commands`")
    lines.append("- Re-run this dump: `nemo agent context`")
    lines.append("- Explore an API resource: `nemo <resource> --help`")
    lines.append("- List resources: `nemo <resource> list`")
    lines.append("- Get a resource: `nemo <resource> get <name-or-id>`")

    typer.echo("\n".join(lines))


@app.command("commands")
def commands_command() -> None:
    """List all available top-level CLI commands.

    Outputs a flat list of commands with descriptions, useful for
    agent capability discovery.

    Examples:
      nemo agent commands
    """
    lines: list[str] = ["# NeMo CLI Commands\n"]
    lines.append("| Command | Panel | Description |")
    lines.append("|---------|-------|-------------|")
    for cmd_name, panel, description in _all_top_level_entries():
        lines.append(
            f"| nemo {_normalize_cell(cmd_name)} | {_normalize_cell(panel)} | {_normalize_cell(description)} |"
        )
    typer.echo("\n".join(lines))


# Typer registers commands in definition order, but help output follows registered_commands.
# Sort explicitly so "context" always appears before "commands" regardless of future additions.
_COMMAND_ORDER = {"context": 0, "commands": 1}
app.registered_commands.sort(key=lambda c: _COMMAND_ORDER.get(c.name or "", len(_COMMAND_ORDER)))
