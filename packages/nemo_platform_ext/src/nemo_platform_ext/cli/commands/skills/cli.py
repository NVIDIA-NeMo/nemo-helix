# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI commands for installing AI agent skill files."""

from pathlib import Path
from typing import Annotated

import typer
from nemo_platform_ext.cli.commands.skills.agents.custom import CustomPathInstaller
from nemo_platform_ext.cli.commands.skills.base import Scope, Skill
from nemo_platform_ext.cli.commands.skills.registry import (
    DuplicateSkillError,
    UnsupportedAgentError,
    get_installer,
    list_agent_names,
    load_skills,
)
from nemo_platform_ext.cli.core.context import CLIContext
from nemo_platform_ext.cli.core.formatters import (
    Column,
    check_output_columns_with_format,
    format_output,
    validate_stream_output_format,
)
from nemo_platform_ext.cli.core.help_formatter import create_typer_app
from nemo_platform_ext.cli.core.types import (
    ListOutputFormatOption,
    NoTruncateOption,
    OutputColumnsOption,
    StreamOutputOption,
)

_AGENT_NAMES = ", ".join(list_agent_names())

app = create_typer_app(
    name="skills",
    help=f"""\
Install AI agent skill files for Nemo.

Supported agents: {_AGENT_NAMES}

Examples:
# List available skills.
nemo skills list
# Show a skill's content.
nemo skills show inference
# Install all skills for Claude Code.
nemo skills install --agent claude
# Install specific skills only.
nemo skills install --agent claude --skill inference""",
)


@app.callback(invoke_without_command=True)
def skills_callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


def _find_project_root(project_dir: Path | None = None) -> Path:
    """Resolve the project install directory without inspecting parent directories."""
    return project_dir if project_dir is not None else Path.cwd()


# Distributions that ship the platform's own bundled skills. Both names show
# up depending on whether the user installed the source workspace package
# (``nemo-platform-ext``) or the vendored SDK wheel (``nemo-platform-sdk``);
# we collapse both into the friendly ``nemo-platform`` label so the column
# matches what a user typically thinks of as "the platform".
_PLATFORM_DISTS = frozenset({"nemo-platform-ext", "nemo-platform-sdk"})


def _format_skill_source(skill: Skill) -> str:
    """Build the user-facing ``Source`` label for a skill.

    Prefers the distribution (PyPI / wheel) name — this matches what users see
    in ``pip list`` / what they ``uv add``'d. Falls back to the entry-point name
    when the distribution isn't known (e.g. tests that fabricate ``Skill``
    instances without going through entry-point discovery).
    """
    if skill.source_dist is not None:
        if skill.source_dist in _PLATFORM_DISTS:
            return "nemo-platform"
        return skill.source_dist
    if skill.source_plugin is not None:
        return skill.source_plugin
    return "-"


def _filter_skills_by_source(skills: dict[str, Skill], sources: list[str]) -> dict[str, Skill]:
    """Filter ``skills`` to the union of those whose source matches any value in ``sources``.

    Matching is case-insensitive against the same user-facing label that
    ``_format_skill_source`` produces. Unknown source values are a hard error
    so users get a clear message listing what's actually available rather than
    a silently empty table.
    """
    requested = {s.strip().casefold() for s in sources if s.strip()}
    if not requested:
        return skills

    available_labels = {_format_skill_source(skill).casefold() for skill in skills.values()}
    unknown = sorted({s for s in requested if s not in available_labels})
    if unknown:
        available = ", ".join(sorted({_format_skill_source(s) for s in skills.values()}))
        typer.echo(
            f"Error: Unknown source(s): {', '.join(unknown)}. Available: {available}",
            err=True,
        )
        raise typer.Exit(code=1)

    return {name: skill for name, skill in skills.items() if _format_skill_source(skill).casefold() in requested}


def _resolve_skills(skill_names: list[str] | None) -> dict[str, Skill]:
    """Load skills, optionally filtering by name. Errors on unknown names."""
    try:
        all_skills = load_skills()
    except DuplicateSkillError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e
    if not skill_names:
        return all_skills
    unknown = set(skill_names) - set(all_skills.keys())
    if unknown:
        available = ", ".join(sorted(all_skills.keys()))
        typer.echo(f"Error: Unknown skill(s): {', '.join(sorted(unknown))}. Available: {available}", err=True)
        raise typer.Exit(code=1)
    return {name: all_skills[name] for name in skill_names}


@app.command("list")
def list_skills(
    ctx: typer.Context,
    output_format: ListOutputFormatOption = None,
    no_truncate: NoTruncateOption = None,
    columns: OutputColumnsOption = None,
    stream: StreamOutputOption = False,
    source: Annotated[
        list[str] | None,
        typer.Option(
            "--source",
            help=(
                "Filter to skills from a specific source (distribution / plugin name as shown in the "
                "`Source` column, e.g. `nemo-platform`, `nemo-agents-plugin`). Can be repeated to "
                "include multiple sources. Matching is case-insensitive."
            ),
        ),
    ] = None,
) -> None:
    """List available skills.

    The default table word-wraps long descriptions; use `--no-truncate` to let
    descriptions fill the full terminal width. For structured output use
    `-f json|yaml|csv|markdown`. When stdout is not a TTY (pipe/redirect),
    JSON is the default so callers get parseable output.

    Examples:
      nemo skills list
      nemo skills list --no-truncate
      nemo skills list -f json
      nemo skills list --source nemo-platform
      nemo skills list --source nemo-platform --source nemo-agents-plugin
    """
    state: CLIContext = ctx.obj
    columns_explicit = columns is not None and str(columns).strip() != "default"
    output_format = state.get_output_format(output_format, apply_non_tty_default=not columns_explicit)
    validate_stream_output_format(output_format, stream)
    check_output_columns_with_format(columns, output_format)

    default_columns = [
        Column("name", "Name"),
        Column("version", "Version"),
        Column("source", "Source"),
        Column("description", "Description"),
    ]
    output_columns: str | list[Column] | None = columns
    if not columns_explicit:
        output_columns = default_columns

    try:
        skills = load_skills()
    except DuplicateSkillError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e

    if source:
        skills = _filter_skills_by_source(skills, source)

    items = [
        {
            "name": skill.name,
            "version": skill.version,
            "description": skill.description,
            "preconditions": skill.preconditions,
            # `source` is the human-friendly column shown in `list` output:
            # the distribution name that registered the skill's entry point,
            # collapsed to `nemo-platform` for the platform's own packages.
            "source": _format_skill_source(skill),
            # Raw entry-point name and distribution name are preserved for
            # programmatic consumers (JSON / YAML / CSV / Markdown output).
            "source_plugin": skill.source_plugin,
            "source_dist": skill.source_dist,
        }
        for skill in skills.values()
    ]

    format_output(
        items,
        is_list=True,
        output_format=output_format,
        output_columns=output_columns,
        no_truncate=state.get_no_truncate(no_truncate),
        timestamp_format=state.get_timestamp_format(),
        wrap=True,
        stream=stream,
    )


@app.command("show")
def show(
    name: Annotated[
        str,
        typer.Argument(help="Skill name to show (use 'nemo skills list' to see available skills)"),
    ],
    agent: Annotated[
        str | None,
        typer.Option("--agent", "-a", help=f"Agent to format for. Supported: {_AGENT_NAMES}"),
    ] = None,
) -> None:
    """Print skill content to stdout.

    Without --agent, prints the raw skill content.
    With --agent, prints the agent-specific formatted version.

    Examples:
      nemo skills show inference
      nemo skills show --agent claude inference
      nemo skills show inference | pbcopy
    """
    try:
        skills = load_skills()
    except DuplicateSkillError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e
    if name not in skills:
        available = ", ".join(sorted(skills.keys()))
        typer.echo(f"Error: Unknown skill '{name}'. Available: {available}", err=True)
        raise typer.Exit(code=1)

    skill = skills[name]

    if agent is None:
        typer.echo(skill.raw, nl=False)
        return

    try:
        installer = get_installer(agent)
    except UnsupportedAgentError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo(installer.format_content(skill), nl=False)


@app.command("install")
def install(
    agent: Annotated[
        str | None,
        typer.Option("--agent", "-a", help=f"Agent to install for. Supported: {_AGENT_NAMES}"),
    ] = None,
    skill: Annotated[
        list[str] | None,
        typer.Option("--skill", "-s", help="Install specific skill(s) only. Can be repeated."),
    ] = None,
    user: Annotated[
        bool,
        typer.Option("--user", help="Install to user scope (default: project scope)"),
    ] = False,
    project_dir: Annotated[
        Path | None,
        typer.Option(
            "--project-dir",
            "--project-root",
            help="Project directory to install into (default: current working directory)",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    path: Annotated[
        Path | None,
        typer.Option(
            "--path",
            help="Install to a custom skills directory (for example ~/.my-agent/skills)",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
) -> None:
    """Install Nemo skill files for an AI coding agent.

    Use --agent for a known coding agent and --path for any other agent.
    Use --skill to select specific skills, --user for user scope, or
    --project-dir to explicitly select a known agent's project directory.

    Examples:
      nemo skills install --agent claude
      nemo skills install --agent claude --user
      nemo skills install --agent claude --skill inference
      nemo skills install --agent claude --project-dir /path/to/project
      nemo skills install --path ~/.my-agent/skills
    """
    if agent is None and path is None:
        typer.echo("Error: provide --agent or --path.", err=True)
        raise typer.Exit(code=1)
    if agent is not None and path is not None:
        typer.echo("Error: --agent and --path cannot be used together.", err=True)
        raise typer.Exit(code=1)

    if path is not None:
        installer = CustomPathInstaller()
    else:
        assert agent is not None
        try:
            installer = get_installer(agent)
        except UnsupportedAgentError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)

    scope = Scope.USER if user else Scope.PROJECT

    if path is not None and project_dir is not None:
        typer.echo("Error: --path and --project-dir cannot be used together.", err=True)
        raise typer.Exit(code=1)

    if scope not in installer.supported_scopes:
        supported = ", ".join(s.value for s in installer.supported_scopes)
        typer.echo(
            f"Error: {installer.display_name} does not support {scope.value} scope. Supported scopes: {supported}",
            err=True,
        )
        raise typer.Exit(code=1)

    skills = _resolve_skills(skill)
    project_root = path if path is not None else _find_project_root(project_dir)
    result_paths = installer.install(scope, project_root, skills)
    typer.echo(f"Installed {len(skills)} skill(s) for {installer.display_name}:")
    for path in result_paths:
        typer.echo(f"  {path}")


_SKILLS_COMMAND_ORDER = {
    "list": 0,
    "show": 1,
    "install": 2,
}
app.registered_commands.sort(
    key=lambda command: _SKILLS_COMMAND_ORDER.get(command.name or "", len(_SKILLS_COMMAND_ORDER))
)
