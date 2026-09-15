# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Generate markdown documentation for the NeMo Platform CLI.

This module provides a custom docs generator that:
- Groups commands and options by rich_help_panel
- Outputs clean markdown (no HTML entities or spans)
- Properly handles Rich markup in help text
"""

from __future__ import annotations

import re
from datetime import datetime
from functools import cache
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

import click
import typer

_CURRENT_YEAR = datetime.now().year
_SPDX_FRONTMATTER = (
    f"# SPDX-FileCopyrightText: Copyright (c) 2025-{_CURRENT_YEAR} "
    "NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
    "# SPDX-License-Identifier: Apache-2.0\n"
)


def strip_rich_markup(text: str) -> str:
    """Remove Rich markup tags from text, converting to plain text or markdown.

    Examples:
        [green]Examples:[/] -> **Examples:**
        [bold]text[/bold] -> **text**
        [bold red]text[/] -> **text**
        [dim]text[/] -> text
        :books: -> 📚
    """
    if not text:
        return text

    # Strip ANSI escape codes (e.g., \x1b[36m...\x1b[0m from terminal coloring)
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    # Strip bracket-style ANSI codes without escape prefix (e.g., [33m<NAME>[0m)
    text = re.sub(r"\[\d+(?:;\d+)*m\]?", "", text)

    # Convert emoji shortcodes to actual emoji
    emoji_map = {
        ":books:": "📚",
        ":book:": "📖",
        ":warning:": "⚠️",
        ":check:": "✓",
        ":x:": "✗",
        ":info:": "ℹ️",
        ":rocket:": "🚀",
        ":sparkles:": "✨",
    }
    for shortcode, emoji in emoji_map.items():
        text = text.replace(shortcode, emoji)

    # Convert compound Rich tags like [bold red]text[/] to markdown bold
    text = re.sub(r"\[bold (?:red|green|blue|cyan|yellow|magenta)\](.*?)\[/\]", r"**\1**", text)

    # Convert [green]text[/] or [green]text[/green] style tags
    # For colored headers like [green]Examples:[/], make them bold in markdown
    text = re.sub(r"\[(?:green|blue|cyan|yellow|red|magenta)\](.*?)\[/(?:\w+)?\]", r"**\1**", text)

    # Convert [bold]text[/bold] or [bold]text[/] to markdown bold
    text = re.sub(r"\[bold\](.*?)\[/(?:bold)?\]", r"**\1**", text)

    # Convert [italic]text[/italic] or [italic]text[/] to markdown italic
    text = re.sub(r"\[italic\](.*?)\[/(?:italic)?\]", r"*\1*", text)

    # Convert [code]text[/code] to markdown code
    text = re.sub(r"\[code\](.*?)\[/(?:code)?\]", r"`\1`", text)

    # Remove [dim]text[/dim] styling (just keep text)
    text = re.sub(r"\[dim\](.*?)\[/(?:dim)?\]", r"\1", text)

    # Remove any remaining Rich tags like [/], [red], [bold red], etc.
    text = re.sub(r"\[/?[a-zA-Z_ ]+\]", "", text)

    # Remove documentation link lines (we're generating the docs, so no need to link to them)
    text = re.sub(r"^.*Documentation:.*https?://.*$", "", text, flags=re.MULTILINE)

    # Remove stray ** that might be left over from Rich tag conversion
    text = re.sub(r"^\*\*\s*$", "", text, flags=re.MULTILINE)

    # Collapse multiple consecutive blank lines into at most two
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Escape $ in operator names like $eq, $like to prevent LaTeX math mode
    # Do this BEFORE JSON wrapping so $operators inside JSON stay clean
    # Match $operators followed by comma, space, period, closing bracket, or end
    text = re.sub(r"\$(\w+)(?=[,\s\.\]\)]|$)", r"`$\1`", text)

    # Wrap inline JSON examples in backticks to prevent LaTeX interpretation
    # Matches patterns like {"key":"value"} or {"key":{"nested":"value"}}
    # Use a more complete pattern to capture full JSON objects
    text = re.sub(r'(?<!`)\{"[^}]+\}+', lambda m: f"`{m.group(0)}`" if "`" not in m.group(0) else m.group(0), text)

    # Fix inline "Examples:" that are glued to previous text (e.g., "criteria.Examples:")
    # But don't break **Examples:** which is already properly formatted
    text = re.sub(r"([^\s*])(Examples?:)", r"\1\n\n\2", text)

    # Fix run-on sentences: period then capital letter with no space (e.g. "API.A" -> "API. A")
    text = re.sub(r"\.([A-Z])", r". \1", text)

    # Normalize indentation and handle example blocks
    # Process line by line to:
    # 1. Remove problematic leading whitespace (4+ spaces that would become code blocks)
    # 2. Format example sections appropriately (code blocks for nemo commands)
    lines = text.split("\n")
    normalized_lines = []
    in_example_block = False
    example_lines: list[str] = []

    def flush_examples() -> None:
        """Flush collected example lines, formatting as code block if appropriate."""
        nonlocal example_lines
        if not example_lines:
            return

        # Check if examples look like nemo commands (start with nemo or # comment)
        has_nemo_commands = any(re.match(r"^\s*(nemo\s|#\s)", line) for line in example_lines if line.strip())

        if has_nemo_commands:
            # Render as shell code block
            normalized_lines.append("```shell")
            for line in example_lines:
                # Strip leading whitespace for code block, unescape #
                clean_line = line.lstrip()
                if clean_line.startswith("\\#"):
                    clean_line = clean_line[1:]  # Remove escape
                # Remove backticks around JSON (not needed inside code blocks)
                clean_line = re.sub(r"`(\{[^}]+\})`", r"\1", clean_line)
                normalized_lines.append(clean_line)
            normalized_lines.append("```")
        else:
            # Render with trailing spaces for line breaks
            for line in example_lines:
                if line.strip() and not line.endswith("  "):
                    line = line + "  "
                normalized_lines.append(line)

        example_lines = []

    in_input_data_block = False
    input_data_lines: list[str] = []

    def flush_input_data() -> None:
        """Flush collected input data lines, formatting as shell code block."""
        nonlocal input_data_lines
        if not input_data_lines:
            return

        # Render as shell code block with list items as comments and commands
        normalized_lines.append("```shell")
        for line in input_data_lines:
            clean_line = line.lstrip()
            # Remove backticks around JSON (not needed inside code blocks)
            clean_line = re.sub(r"`(\{[^}]+\})`", r"\1", clean_line)
            # Convert "- label: command" to "# label:\ncommand"
            if match := re.match(r"^[-*]\s*(\w+):\s*(.+)$", clean_line):
                label, command = match.groups()
                normalized_lines.append(f"# {label}:")
                normalized_lines.append(command)
            else:
                normalized_lines.append(clean_line)
        normalized_lines.append("```")
        input_data_lines = []

    for line in lines:
        # Strip excessive leading whitespace (4+ spaces) that would create code blocks
        # List items get their indentation reduced to 2 spaces to maintain list formatting
        if re.match(r"^    +[-*]\s", line):
            # Convert 4+ space indented list items to 2-space indent
            line = re.sub(r"^    +", "  ", line)
        elif re.match(r"^    +\S", line):
            # Strip all indentation from non-list items
            line = line.lstrip()

        # Detect end of input data block (empty line or non-list line)
        if in_input_data_block and (not line.strip() or not re.match(r"^\s*[-*]\s", line)):
            flush_input_data()
            in_input_data_block = False

        if in_input_data_block:
            input_data_lines.append(line)
            continue

        # Detect start of example block (standalone "Examples:" line)
        if re.match(r"^\*?\*?Examples?:?\*?\*?\s*$", line, re.IGNORECASE):
            in_example_block = True
            # Always make Examples: bold
            normalized_lines.append("**Examples:**\n")
            continue

        # Detect end of example block:
        # - empty line
        # - new section starting with capital letter
        # - bold text (like **Required fields:**)
        # - section headers
        if in_example_block and (not line.strip() or re.match(r"^[A-Z]", line) or re.match(r"^\*\*[A-Z]", line)):
            flush_examples()
            in_example_block = False

        if in_example_block:
            # Collect example lines for later processing
            example_lines.append(line)
        else:
            # Escape # at start of line to prevent markdown header interpretation
            if re.match(r"^#+\s", line):
                line = "\\" + line
            normalized_lines.append(line)

    # Flush any remaining blocks at end of text
    if example_lines:
        flush_examples()
    if input_data_lines:
        flush_input_data()

    text = "\n".join(normalized_lines)

    return text.strip()


def get_panel_name(obj: click.Command | click.Option) -> str | None:
    """Get the rich_help_panel value from a command or option."""
    panel = getattr(obj, "rich_help_panel", None)
    if panel is not None:
        # Filter out DefaultPlaceholder objects from Typer
        if type(panel).__name__ == "DefaultPlaceholder":
            return None
    return panel


def _lazy_entry(obj: click.Group, cmd_name: str) -> Any | None:
    """Return lazy metadata for a command without loading it."""
    lazy_entries = getattr(obj, "_lazy_entries", {})
    get_entry = getattr(lazy_entries, "get", None)
    if not callable(get_entry):
        return None
    entry = get_entry(cmd_name)
    if entry is None or not hasattr(entry, "hidden") or not hasattr(entry, "help"):
        return None
    return entry


def cli_module(module_path: str) -> ModuleType:
    """Import a CLI module from source or from the vendored SDK package."""
    source_name = f"nemo_platform_ext.cli.{module_path}"
    try:
        return import_module(source_name)
    except ModuleNotFoundError as exc:
        if not is_missing_source_cli_module(exc.name, source_name):
            raise
        return import_module(f"nemo_platform.cli.{module_path}")


def is_missing_source_cli_module(missing_name: str | None, source_name: str) -> bool:
    """Return whether import failed because the source CLI module is unavailable."""
    if missing_name is None:
        return False
    return source_name == missing_name or source_name.startswith(f"{missing_name}.")


@cache
def cli_docs_metadata() -> tuple[list[str], dict[str, str], tuple[str, ...]]:
    """Return CLI metadata shared by generated docs."""
    manifest = cli_module("manifest")
    help_formatter = cli_module("core.help_formatter")
    return (
        list(manifest.PANEL_ORDER),
        dict(manifest.PANEL_DESCRIPTIONS),
        help_formatter.HELP_OPTION_NAMES,
    )


def format_option_name(param: click.Option) -> str:
    """Format option names with metavar."""
    if param.name == "help":
        parts = cli_docs_metadata()[2]
    else:
        parts = [*param.opts, *(param.secondary_opts or [])]

    opt_str = ", ".join(parts)

    # Add metavar if present
    if param.metavar:
        opt_str += f" {param.metavar}"
    elif param.type and not param.is_flag:
        metavar = param.type.name.upper()
        if metavar not in ("TEXT", "BOOLEAN"):
            opt_str += f" <{metavar}>"

    return opt_str


def generate_command_docs(
    obj: click.Command,
    ctx: click.Context,
    name: str = "",
    call_prefix: str = "",
    indent: int = 1,
    is_root: bool = False,
    help_override: str | None = None,
) -> str:
    """Generate markdown documentation for a single command."""
    docs = ""

    # Build full command name
    command_name = name or obj.name or ""
    if call_prefix:
        full_name = f"{call_prefix} {command_name}".strip()
    else:
        full_name = command_name

    # Header - use Fern MDX frontmatter for root, plain name for others
    # Markdown only supports up to 6 levels of headers, use bold for deeper nesting
    if is_root:
        docs += f'---\n{_SPDX_FRONTMATTER}title: "Full CLI Reference"\ndescription: ""\n---\n'
    elif indent <= 6:
        docs += "#" * indent + f" {full_name}\n\n"
    else:
        docs += f"**{full_name}**\n\n"

    # Help text (description)
    help_text = help_override if help_override is not None else obj.help
    if help_text:
        help_text = strip_rich_markup(help_text)
        docs += f"{help_text}\n\n"

    # Usage
    usage_pieces = obj.collect_usage_pieces(ctx)
    if usage_pieces or full_name:
        docs += "**Usage:**\n\n```shell\n"
        docs += f"{full_name}"
        if usage_pieces:
            docs += " " + " ".join(usage_pieces)
        docs += "\n```\n\n"

    # Collect parameters
    arguments: list[tuple[str, str]] = []
    options_by_panel: dict[str | None, list[tuple[str, str]]] = {}

    for param in obj.get_params(ctx):
        rv = param.get_help_record(ctx)
        if rv is None:
            continue

        param_name, param_help = rv
        param_name = strip_rich_markup(param_name) if param_name else ""
        param_help = strip_rich_markup(param_help) if param_help else ""

        if isinstance(param, click.Argument):
            arguments.append((param_name, param_help))
        elif isinstance(param, click.Option):
            panel = get_panel_name(param)
            # Put help/completion options in a "Help" panel
            if param.name in ("help", "install_completion", "show_completion"):
                panel = "Help"
            if panel not in options_by_panel:
                options_by_panel[panel] = []
            options_by_panel[panel].append((format_option_name(param), param_help))

    # Arguments section
    if arguments:
        docs += "**Arguments:**\n\n"
        for arg_name, arg_help in arguments:
            docs += f"* `{arg_name}`"
            if arg_help:
                docs += f": {arg_help}"
            docs += "\n"
        docs += "\n"

    # Options sections (grouped by panel)
    if options_by_panel:
        # First, ungrouped options (None panel)
        if None in options_by_panel:
            docs += "**Options:**\n\n"
            for opt_name, opt_help in options_by_panel[None]:
                docs += f"* `{opt_name}`"
                if opt_help:
                    docs += f": {opt_help}"
                docs += "\n"
            docs += "\n"

        # Then grouped options
        for panel_name in sorted(p for p in options_by_panel if p is not None):
            docs += f"**{panel_name}:**\n\n"
            for opt_name, opt_help in options_by_panel[panel_name]:
                docs += f"* `{opt_name}`"
                if opt_help:
                    docs += f": {opt_help}"
                docs += "\n"
            docs += "\n"

    # Epilog
    if obj.epilog:
        docs += f"{strip_rich_markup(obj.epilog)}\n\n"

    return docs


def generate_group_docs(
    obj: click.Group,
    ctx: click.Context,
    name: str = "",
    call_prefix: str = "",
    indent: int = 1,
    is_root: bool = False,
    help_override: str | None = None,
) -> str:
    """Generate markdown documentation for a command group."""
    docs = ""

    # Build full command name
    command_name = name or obj.name or ""
    if call_prefix:
        full_name = f"{call_prefix} {command_name}".strip()
    else:
        full_name = command_name

    # Generate docs for this group itself
    docs += generate_command_docs(
        obj,
        ctx,
        name=name,
        call_prefix=call_prefix,
        indent=indent,
        is_root=is_root,
        help_override=help_override,
    )

    # Group subcommands by rich_help_panel
    commands_by_panel: dict[str | None, list[tuple[str, click.Command, str | None]]] = {}

    for cmd_name in obj.list_commands(ctx):
        lazy_entry = _lazy_entry(obj, cmd_name)
        if lazy_entry is not None and lazy_entry.hidden:
            continue
        cmd = obj.get_command(ctx, cmd_name)
        if cmd is None or cmd.hidden:
            continue
        panel = get_panel_name(cmd)
        if panel not in commands_by_panel:
            commands_by_panel[panel] = []
        entry_help_override = lazy_entry.help if lazy_entry is not None else None
        commands_by_panel[panel].append((cmd_name, cmd, entry_help_override))

    if not commands_by_panel:
        return docs

    # Determine panel order: specific order for root, then alphabetically for others
    preferred_order = cli_docs_metadata()[0]
    panels_in_order: list[str | None] = []

    # Add None panel first if exists
    if None in commands_by_panel:
        panels_in_order.append(None)

    # Add panels in preferred order
    for panel in preferred_order:
        if panel in commands_by_panel:
            panels_in_order.append(panel)

    # Add any remaining panels alphabetically
    remaining = sorted(p for p in commands_by_panel if p is not None and p not in preferred_order)
    panels_in_order.extend(remaining)

    # For root command, create section structure with headers
    # For nested commands, use the simpler list format
    if is_root:
        # Root: create sections with headers for each panel
        for panel in panels_in_order:
            if panel is not None:
                # Section header for the panel
                docs += f"## {panel}\n\n"

            for cmd_name, cmd, root_help in commands_by_panel[panel]:
                if isinstance(cmd, click.Group):
                    docs += generate_group_docs(
                        cmd,
                        ctx,
                        name=cmd_name,
                        call_prefix=full_name,
                        indent=3,  # ### for commands under ## section
                        help_override=root_help,
                    )
                else:
                    docs += generate_command_docs(
                        cmd,
                        ctx,
                        name=cmd_name,
                        call_prefix=full_name,
                        indent=3,  # ### for commands under ## section
                        help_override=root_help,
                    )
    else:
        # Non-root: add commands list then document each
        docs += "**Commands:**\n\n"

        for panel in panels_in_order:
            if panel is not None:
                docs += f"*{panel}:*\n\n"

            for cmd_name, cmd, _ in commands_by_panel[panel]:
                short_help = cmd.get_short_help_str(limit=60)
                short_help = strip_rich_markup(short_help) if short_help else ""
                docs += f"* `{cmd_name}`"
                if short_help:
                    docs += f": {short_help}"
                docs += "\n"
            docs += "\n"

        # Recursively document subcommands
        for panel in panels_in_order:
            for cmd_name, cmd, _ in commands_by_panel[panel]:
                if isinstance(cmd, click.Group):
                    docs += generate_group_docs(
                        cmd,
                        ctx,
                        name=cmd_name,
                        call_prefix=full_name,
                        indent=indent + 1,
                    )
                else:
                    docs += generate_command_docs(
                        cmd,
                        ctx,
                        name=cmd_name,
                        call_prefix=full_name,
                        indent=indent + 1,
                    )

    return docs


def generate_index_snippet(app: typer.Typer, name: str = "nemo") -> str:
    """Generate a summary snippet for the CLI index page.

    Produces two markdown tables extracted from the Typer app:
    1. Global options (from the root command's callback parameters)
    2. Commands grouped by category (from rich_help_panel metadata)
    """
    click_obj = typer.main.get_command(app)
    ctx = click.Context(click_obj, info_name=name)

    lines: list[str] = []

    # Fern MDX frontmatter (snippet is included into other pages).
    lines.append("---")
    lines.extend(_SPDX_FRONTMATTER.rstrip().splitlines())
    lines.append('title: ""')
    lines.append('description: ""')
    lines.append("---")

    # --- Global options table ---
    lines.append("**Global options** apply to all commands:")
    lines.append("")
    lines.append("| Option | Description |")
    lines.append("|--------|-------------|")

    for param in click_obj.get_params(ctx):
        if not isinstance(param, click.Option):
            continue
        if param.hidden:
            continue
        panel = get_panel_name(param)
        if panel != "Global Options":
            continue
        rv = param.get_help_record(ctx)
        if rv is None:
            continue
        opt_name = format_option_name(param)
        _, help_text = rv
        help_text = strip_rich_markup(help_text).strip() if help_text else ""
        help_text = " ".join(help_text.split())
        lines.append(f"| `{opt_name}` | {help_text} |")

    lines.append("")

    # --- Commands table ---
    if isinstance(click_obj, click.Group):
        preferred_order, category_descriptions, _ = cli_docs_metadata()

        commands_by_panel: dict[str | None, list[tuple[str, click.Command]]] = {}
        for cmd_name in click_obj.list_commands(ctx):
            lazy_entry = _lazy_entry(click_obj, cmd_name)
            if lazy_entry is not None and lazy_entry.hidden:
                continue
            cmd = click_obj.get_command(ctx, cmd_name)
            if cmd is None or cmd.hidden:
                continue
            panel = get_panel_name(cmd)
            if panel not in commands_by_panel:
                commands_by_panel[panel] = []
            commands_by_panel[panel].append((cmd_name, cmd))

        panels_in_order: list[str | None] = []
        if None in commands_by_panel:
            panels_in_order.append(None)
        for panel in preferred_order:
            if panel in commands_by_panel:
                panels_in_order.append(panel)
        remaining = sorted(p for p in commands_by_panel if p is not None and p not in preferred_order)
        panels_in_order.extend(remaining)

        lines.append("**Commands** are organized into categories:")
        lines.append("")
        lines.append("| Category | Commands | Description |")
        lines.append("|----------|----------|-------------|")

        for panel in panels_in_order:
            category = panel or "Other"
            cmds = commands_by_panel[panel]
            cmd_list = ", ".join(f"`{cmd_name}`" for cmd_name, _ in cmds)
            description = category_descriptions.get(category, category)
            lines.append(f"| {category} | {cmd_list} | {description} |")

        lines.append("")

    return "\n".join(lines)


def generate_docs(app: typer.Typer, name: str = "nemo") -> str:
    """Generate complete markdown documentation for a Typer app.

    Args:
        app: The Typer application to document
        name: The CLI program name (used in examples)

    Returns:
        Complete markdown documentation as a string
    """
    # Convert Typer app to Click command
    click_obj = typer.main.get_command(app)

    # Create a context for introspection
    ctx = click.Context(click_obj, info_name=name)
    if isinstance(click_obj, click.Group):
        docs = generate_group_docs(click_obj, ctx, name=name, indent=1, is_root=True)
    else:
        docs = generate_command_docs(click_obj, ctx, name=name, indent=1, is_root=True)

    docs = docs.replace(
        "(docs/get-started/concepts/entity-references.md)",
        "{ref}`Entity references <entity-references>`",
    )

    docs = _escape_mdx(docs)

    return docs.strip() + "\n"


def _escape_mdx(text: str) -> str:
    """Escape `{` and `<` outside fenced code blocks so MDX does not parse them as JSX."""
    lines = text.split("\n")
    in_code_fence = False
    escaped: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            escaped.append(line)
            continue
        if in_code_fence:
            escaped.append(line)
            continue
        escaped.append(_escape_mdx_line(line))
    return "\n".join(escaped)


def _escape_mdx_line(line: str) -> str:
    """Escape `{` and `<` in `line`, preserving inline backtick spans."""
    out: list[str] = []
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if ch == "`":
            end = line.find("`", i + 1)
            if end == -1:
                out.append(line[i:])
                break
            out.append(line[i : end + 1])
            i = end + 1
            continue
        if ch == "{":
            out.append("\\{")
        elif ch == "<":
            out.append("\\<")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


_DOCUMENTED_PLUGIN_CLIS = (
    "agents",
    "anonymizer",
    "auditor",
    "customization",
    "data-designer",
    "evaluator",
    "insights",
    "safe-synthesizer",
)

_PLUGIN_DOCS_DISCOVERY_ENV = {
    # Nested CLI extensions do not have a surface-specific allowlist, so they
    # inherit the global value.
    "NEMO_PLUGIN_ALLOWLIST": "*",
    "NEMO_PLUGIN_CLI_ALLOWLIST": ",".join(_DOCUMENTED_PLUGIN_CLIS),
    # Plugin CLIs compose their help from these dependent discovery surfaces.
    "NEMO_PLUGIN_JOBS_ALLOWLIST": "*",
    "NEMO_PLUGIN_FUNCTIONS_ALLOWLIST": "*",
    "NEMO_PLUGIN_CUSTOMIZATION_CONTRIBUTORS_ALLOWLIST": "*",
}


_REPO_ROOT = Path(__file__).resolve().parents[3]
_REFERENCE_DOCS_PATH = _REPO_ROOT / "docs/cli/reference.mdx"
_SUMMARY_DOCS_PATH = _REPO_ROOT / "docs/fern/snippets/_snippets/cli-summary.mdx"


def _enable_plugin_cli_docs() -> None:
    """Include supported plugin commands in generated CLI documentation."""
    import os

    os.environ.update(_PLUGIN_DOCS_DISCOVERY_ENV)


def _with_trailing_newline(output: str) -> str:
    """Return output with exactly the trailing newline expected in generated files."""
    return output if output.endswith("\n") else output + "\n"


def write_docs_files(app: typer.Typer, reference_path: Path, summary_path: Path, name: str = "nemo") -> None:
    """Write generated CLI reference and summary docs from one imported CLI app."""
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.write_text(_with_trailing_newline(generate_docs(app, name=name)), encoding="utf-8")
    summary_path.write_text(_with_trailing_newline(generate_index_snippet(app, name=name)), encoding="utf-8")


def main() -> None:
    """Generate CLI documentation and print to stdout.

    Usage:
        docs_generator.py reference   # Full CLI reference
        docs_generator.py summary     # Index page summary snippet
        docs_generator.py all         # Write both generated docs files
    """
    import sys

    if len(sys.argv) != 2 or sys.argv[1] not in ("reference", "summary", "all"):
        print("Usage: docs_generator.py {reference|summary|all}", file=sys.stderr)
        sys.exit(1)

    _enable_plugin_cli_docs()

    from nemo_platform_ext.cli.app import app

    mode = sys.argv[1]
    if mode == "all":
        write_docs_files(app, _REFERENCE_DOCS_PATH, _SUMMARY_DOCS_PATH, name="nemo")
        return
    if mode == "summary":
        output = generate_index_snippet(app, name="nemo")
    else:
        output = generate_docs(app, name="nemo")
    sys.stdout.write(_with_trailing_newline(output))


if __name__ == "__main__":
    main()
