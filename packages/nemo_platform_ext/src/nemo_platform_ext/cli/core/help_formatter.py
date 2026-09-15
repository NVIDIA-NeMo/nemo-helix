# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NeMo Platform help formatter for Typer CLI.

Clean help output with uv-inspired style:
- Uses terminal width
- Colored output without frames
- Grouped options by section
- Clean, readable formatting
"""

from __future__ import annotations

import shutil
import sys
import time
from contextvars import ContextVar
from functools import wraps
from io import StringIO
from typing import Any, Callable, Mapping, ParamSpec, Sequence, TypeVar

import click
from click import Command
from rich.console import Console
from typer import Typer
from typer.core import TyperArgument, TyperCommand, TyperGroup, TyperOption

_REQUIRED_SUFFIX = " (required)"  # embedded by the generator in required body-param help text
HELP_OPTION_NAMES = ("--help", "-h")


def _strip_required_suffix(help_text: str) -> tuple[str, bool]:
    """Strip the generator-injected required suffix and report whether it was present."""
    if help_text.endswith(_REQUIRED_SUFFIX):
        return help_text[: -len(_REQUIRED_SUFFIX)], True
    return help_text, False


class NmpErrorHandlingMixin:
    """Mixin that provides custom error handling for Click commands."""

    _nmp_emit_command_invoked = False

    def main(
        self,
        args: list[str] | None = None,
        prog_name: str | None = None,
        complete_var: str | None = None,
        standalone_mode: bool = True,
        **extra: Any,
    ) -> Any:
        """Override main to use custom error handling and emit one telemetry event.

        This is the single path every command type flows through (hand-registered,
        OpenAPI-generated, plugin entry points), and Click only calls ``main()`` on the
        root command object, so this is the natural choke point for a per-invocation
        ``command_invoked`` event. The outcome->status mapping and the emit both live in
        a ``finally`` so telemetry fires even when a command fails, and the emit body is
        best-effort (never raises) so a telemetry bug can never change a command's exit
        code or output.
        """
        from nemo_platform_ext.cli.core.errors import handle_exception
        from nemo_platform_ext.cli.telemetry import runtime
        from nemo_platform_ext.cli.telemetry.events import TaskStatusEnum

        # Only the root group drives telemetry. Nested groups never have main()
        # called during dispatch, but this guard keeps the single-emit contract explicit.
        is_telemetry_root = bool(getattr(self, "_nmp_emit_command_invoked", False))
        help_requested = False
        if is_telemetry_root:
            runtime.reset()
            # Reading help (e.g. ``nemo docs --help``) is not usage, so it must not emit
            # a command_invoked event. Detect a help request from the resolved args; the
            # help option names are the same ones every command registers via
            # ``_context_settings_with_help``.
            resolved_args = args if args is not None else sys.argv[1:]
            help_requested = any(arg in HELP_OPTION_NAMES for arg in resolved_args)

        start = time.monotonic()
        status: TaskStatusEnum | None = None
        try:
            try:
                result = super().main(  # type: ignore[misc]  # ty: ignore[unresolved-attribute]
                    args=args,
                    prog_name=prog_name,
                    complete_var=complete_var,
                    standalone_mode=False,
                    **extra,
                )
                # When standalone_mode=False, Click returns the exit code instead of
                # raising. A non-zero code is an error regardless of how the outer caller
                # ultimately surfaces it.
                status = TaskStatusEnum.ERROR if isinstance(result, int) and result != 0 else TaskStatusEnum.COMPLETED
                # Convert non-zero exit codes back to SystemExit when the original caller
                # expected standalone_mode=True behavior.
                if standalone_mode and isinstance(result, int) and result != 0:
                    raise SystemExit(result)
                return result
            except click.UsageError as e:
                status = TaskStatusEnum.ERROR
                if standalone_mode:
                    handle_exception(e, e.ctx)
                raise
            except click.exceptions.Exit as e:
                status = TaskStatusEnum.COMPLETED if e.exit_code == 0 else TaskStatusEnum.ERROR
                if standalone_mode:
                    raise SystemExit(e.exit_code)
                raise
            except click.Abort:
                status = TaskStatusEnum.CANCELED
                if standalone_mode:
                    click.echo("Aborted!", err=True)
                    raise SystemExit(1)
                raise
            except click.ClickException as e:
                status = TaskStatusEnum.ERROR
                if standalone_mode:
                    handle_exception(e, getattr(e, "ctx", None))
                raise
        except SystemExit as e:
            if status is None:
                code = e.code
                status = TaskStatusEnum.COMPLETED if code in (0, None) else TaskStatusEnum.ERROR
            raise
        except KeyboardInterrupt:
            if status is None:
                status = TaskStatusEnum.CANCELED
            raise
        except BaseException:
            if status is None:
                status = TaskStatusEnum.ERROR
            raise
        finally:
            if is_telemetry_root and not help_requested:
                runtime.emit_command_invoked(status or TaskStatusEnum.COMPLETED, time.monotonic() - start)


def _get_terminal_width() -> int:
    """Get terminal width with reasonable bounds."""
    width = shutil.get_terminal_size(fallback=(120, 24)).columns
    return min(width - 2, 120)


def _context_settings_with_help(context_settings: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return context_settings with help_option_names defaulting to --help/-h."""
    settings = dict(context_settings or {})
    settings.setdefault("help_option_names", list(HELP_OPTION_NAMES))
    return settings


def _option_display_names(
    param: click.Option,
    help_option_names: Sequence[str] | None = None,
) -> list[str]:
    opts = list(param.opts)
    if param.secondary_opts:
        opts.extend(param.secondary_opts)
    configured_help_names = tuple(help_option_names or HELP_OPTION_NAMES)
    if len(opts) == len(configured_help_names) and set(opts) == set(configured_help_names):
        return list(configured_help_names)
    return opts


def _wrap_preserving_newlines(
    text: str,
    width: int,
    initial_indent: str = "",
    subsequent_indent: str = "",
) -> str:
    """Wrap text while preserving single newlines.

    Unlike click.wrap_text which only preserves double newlines (paragraphs),
    this preserves single newlines as explicit line breaks.
    """
    import textwrap

    lines = text.split("\n")
    wrapped_lines: list[str] = []

    for i, line in enumerate(lines):
        if not line.strip():
            # Empty line - preserve it
            wrapped_lines.append("")
        else:
            # Wrap this line individually
            indent = initial_indent if i == 0 else subsequent_indent
            wrapped = textwrap.fill(
                line,
                width=width,
                initial_indent=indent,
                subsequent_indent=subsequent_indent,
            )
            wrapped_lines.append(wrapped)

    return "\n".join(wrapped_lines)


class NmpHelpFormatter(click.HelpFormatter):
    """Help formatter with clean uv-inspired style."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize with terminal width."""
        width = _get_terminal_width()
        kwargs.setdefault("width", width)
        kwargs.setdefault("max_width", width)
        super().__init__(*args, **kwargs)

    def write_usage(self, prog: str, args: str = "", prefix: str | None = None) -> None:
        """Write usage line: Usage: prog [OPTIONS] <COMMAND>"""
        if prefix is None:
            prefix = "Usage:"
        # Color the prefix in bright green
        colored_prefix = click.style(prefix, fg="bright_green", bold=True)

        if "[GLOBAL OPTIONS]" not in prog:
            prog = prog.replace("nemo ", "nemo [GLOBAL OPTIONS] ")

        usage = f"{colored_prefix} {prog}"
        if args:
            usage += f" {args}"
        self.write(f"{usage}\n")

    def write_heading(self, heading: str) -> None:
        """Write section heading with bright green color."""
        # Add blank line before heading (except first)
        if self.buffer and not self.buffer[-1].isspace():
            self.write("\n")
        # Color the heading in bright green
        colored_heading = click.style(f"{heading}:", fg="bright_green", bold=True)
        self.write(f"{colored_heading}\n")

    def write_dl(self, rows: Sequence[tuple[str, str]], col_max: int = 45, col_spacing: int = 2) -> None:
        """Write definition list with proper indentation.

        Format (consistent within group):
          Either ALL on one line:
            --short <VAL>  Description
            --other <VAL>  Another description

          Or ALL wrapped (if any option needs wrapping):
            --short <VAL>
                Description
            --other <VAL>
                Another description
        """
        if not rows:
            return

        # Fixed indentation for wrapped descriptions
        desc_indent = 6
        terminal_width = self.width

        rows_list = list(rows)

        # Calculate optimal alignment column based on actual options
        visible_lengths = [len(click.unstyle(first)) for first, _ in rows_list]
        reasonable_lens = [length for length in visible_lengths if length <= col_max]

        if reasonable_lens:
            align_col = max(reasonable_lens) + col_spacing
        else:
            align_col = 30

        # FIRST PASS: Check if ALL options in this group can fit on one line
        all_fit = True
        for first, second in rows_list:
            if not second:
                continue

            first_visible_len = len(click.unstyle(first))
            second_visible_len = len(click.unstyle(second))

            # Calculate one-line length
            if first_visible_len <= col_max:
                one_line_length = 2 + align_col + second_visible_len
            else:
                one_line_length = 2 + first_visible_len + col_spacing + second_visible_len

            if one_line_length > terminal_width:
                all_fit = False
                break

        # SECOND PASS: Format all options consistently
        if all_fit:
            # All fit on one line - use single-line format for all
            for first, second in rows_list:
                if not second:
                    self.write(f"  {first}\n")
                    continue

                first_visible_len = len(click.unstyle(first))

                if first_visible_len <= col_max:
                    padding_size = align_col - first_visible_len
                else:
                    padding_size = col_spacing

                padding = " " * padding_size
                self.write(f"  {first}{padding}{second}\n")
        else:
            # At least one doesn't fit - use wrapped format for ALL
            for first, second in rows_list:
                self.write(f"  {first}\n")

                if second:
                    # Wrap the description text while preserving single newlines
                    available_width = terminal_width - desc_indent
                    self.write(
                        _wrap_preserving_newlines(second, available_width, " " * desc_indent, " " * desc_indent) + "\n"
                    )


class NmpOption(TyperOption):
    """Option formatter with uv-style coloring."""

    def get_help_record(self, ctx: click.Context) -> tuple[str, str] | None:
        """Format option with colors."""
        if self.hidden:
            return None

        # Build option flags part
        all_opts = _option_display_names(self, getattr(ctx, "help_option_names", None))

        # Color each flag
        colored_opts = [click.style(opt, fg="cyan", bold=True) for opt in all_opts]
        opts_str = ", ".join(colored_opts)

        # Add value placeholder if needed
        if not self.is_flag and not self.count:
            metavar = self.metavar or (self.name or "").upper()
            placeholder = click.style(f"<{metavar}>", fg="yellow")
            opts_str = f"{opts_str} {placeholder}"

        # Build description
        help_text = self.help or ""

        help_text, injected_required = _strip_required_suffix(help_text)
        metadata_parts = [click.style("[required]", fg="red", bold=True)] if injected_required else []

        # Check if default is actually a real value (not a placeholder)
        has_real_default = False
        if self.show_default and self.default is not None and not self.is_flag:
            # Check if it's not a Typer DefaultPlaceholder
            default_type = type(self.default).__name__
            if default_type != "DefaultPlaceholder":
                has_real_default = True

        # Possible values (choices) and default - combine if both present
        type_choices = getattr(self.type, "choices", None)
        if type_choices:
            choices = ", ".join(str(c) for c in type_choices)

            # Check if we also have a default to combine
            if has_real_default:
                if isinstance(self.default, (list, tuple)):
                    default_str = ", ".join(str(d) for d in self.default)
                else:
                    default_str = str(self.default)
                # Combine in one bracket with different colors for each part
                choices_part = click.style(f"[possible values: {choices}; ", fg="cyan")
                default_part = click.style(f"default: {default_str}", fg="yellow")
                closing_bracket = click.style("]", fg="cyan")
                metadata_parts.append(choices_part + default_part + closing_bracket)
            else:
                # Just choices
                metadata_parts.append(click.style(f"[possible values: {choices}]", fg="cyan"))
        else:
            # Default value only (no choices)
            if has_real_default:
                if isinstance(self.default, (list, tuple)):
                    default_str = ", ".join(str(d) for d in self.default)
                else:
                    default_str = str(self.default)
                metadata_parts.append(click.style(f"[default: {default_str}]", fg="yellow"))

        # Environment variable
        if self.envvar:
            env = self.envvar[0] if isinstance(self.envvar, (list, tuple)) else self.envvar
            metadata_parts.append(f"[env: {env}=]")

        # Required marker
        if self.required:
            metadata_parts.append(click.style("[required]", fg="red", bold=True))

        # Combine description with metadata
        if metadata_parts:
            metadata = " ".join(metadata_parts)
            if help_text:
                help_text = f"{help_text} {metadata}"
            else:
                help_text = metadata

        return opts_str, help_text


class NmpArgument(TyperArgument):
    """Argument formatter with uv-style coloring."""

    def get_help_record(self, ctx: click.Context) -> tuple[str, str] | None:
        """Format argument with colors."""
        if self.hidden:
            return None

        # Get the argument name (metavar) and strip any surrounding brackets
        metavar = self.make_metavar(ctx)
        # Remove square brackets that Click adds for optional arguments
        metavar = metavar.strip("[]")

        # Wrap in angle brackets like options do
        formatted_arg = f"<{metavar}>"

        # Color the argument name in yellow
        arg_str = click.style(formatted_arg, fg="yellow")

        # Build description
        help_text = self.help or ""

        help_text, injected_required = _strip_required_suffix(help_text)
        if self.required or injected_required:
            required_marker = click.style("[required]", fg="red", bold=True)
            help_text = f"{help_text} {required_marker}" if help_text else required_marker

        return arg_str, help_text


class NmpContext(click.Context):
    """Context that uses NeMo Platform formatter."""

    def __init__(self, *args, **kwargs):
        """Initialize with proper terminal width."""
        # Get actual terminal width
        term_width = _get_terminal_width()

        # Override terminal_width and max_content_width if not set
        if "terminal_width" not in kwargs or kwargs.get("terminal_width") is None:
            kwargs["terminal_width"] = term_width
        if "max_content_width" not in kwargs or kwargs.get("max_content_width") is None:
            kwargs["max_content_width"] = term_width

        super().__init__(*args, **kwargs)

    def make_formatter(self) -> NmpHelpFormatter:
        """Create NeMo Platform-style formatter."""
        return NmpHelpFormatter(
            width=self.terminal_width,
            max_width=self.max_content_width,
        )


class NmpCommand(NmpErrorHandlingMixin, TyperCommand):
    """Command with NeMo Platform-style help formatting."""

    context_class = NmpContext

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize and patch options and arguments."""
        super().__init__(*args, **kwargs)
        for param in self.params:
            if isinstance(param, TyperOption) and type(param) is not NmpOption:
                param.__class__ = NmpOption
            elif isinstance(param, TyperArgument) and type(param) is not NmpArgument:
                param.__class__ = NmpArgument

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Format complete help output."""
        self.format_usage(ctx, formatter)
        self.format_help_text(ctx, formatter)
        self.format_options(ctx, formatter)
        self.format_epilog(ctx, formatter)
        _maybe_format_agent_helpers(ctx, formatter)

    def format_help_text(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Write the help text (description)."""
        if self.help:
            trimmed_lines = (line.strip() for line in self.help.strip().split("\n"))
            _write_with_formatting(formatter, "\n" + "\n".join(trimmed_lines))

    def format_options(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Write options grouped by help panel."""
        # Separate arguments and options
        args: list[tuple[str, str]] = []
        groups: dict[str | None, list[tuple[str, str]]] = {}

        for param in self.get_params(ctx):
            rv = param.get_help_record(ctx)
            if rv is None:
                continue

            if isinstance(param, click.Argument):
                args.append(rv)
            elif isinstance(param, click.Option):
                # Color built-in options (help, install-completion, etc.) if not already colored
                if not isinstance(param, NmpOption) and rv:
                    opt_str, help_text = rv
                    # Check if already colored (contains ANSI codes)
                    if "\x1b[" not in opt_str:
                        # Color it like other options
                        colored_opts = [
                            click.style(option_name, fg="cyan", bold=True)
                            for option_name in _option_display_names(param, getattr(ctx, "help_option_names", None))
                        ]
                        opt_str = ", ".join(colored_opts)
                        rv = (opt_str, help_text)

                # Put --help, --install-completion, --show-completion in Help section
                if param.name in ("help", "install_completion", "show_completion"):
                    panel = "Help"
                else:
                    panel = getattr(param, "rich_help_panel", None)
                if panel not in groups:
                    groups[panel] = []
                groups[panel].append(rv)

        # Write arguments first
        if args:
            with formatter.section("Arguments"):
                formatter.write_dl(args)

        # Write ungrouped options
        if None in groups:
            with formatter.section("Options"):
                formatter.write_dl(groups[None])

        # Write grouped options (Help section last)
        sorted_panels = sorted(g for g in groups.keys() if g is not None and g != "Help")
        for panel_name in sorted_panels:
            with formatter.section(panel_name):
                formatter.write_dl(groups[panel_name])

        # Write Help section last
        if "Help" in groups:
            with formatter.section("Help"):
                formatter.write_dl(groups["Help"])

    def format_epilog(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Write epilog if any."""
        if self.epilog:
            formatter.write("\n")
            formatter.write(self.epilog)
            formatter.write("\n")


class NmpGroup(NmpErrorHandlingMixin, TyperGroup):
    """Group with NeMo Platform-style help formatting."""

    context_class = NmpContext

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize and patch options and arguments."""
        super().__init__(*args, **kwargs)
        for param in self.params:
            if isinstance(param, TyperOption) and type(param) is not NmpOption:
                param.__class__ = NmpOption
            elif isinstance(param, TyperArgument) and type(param) is not NmpArgument:
                param.__class__ = NmpArgument

    def list_commands(self, ctx: click.Context) -> list[str]:
        """Return commands in registration order instead of alphabetical."""
        return list(self.commands)

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """Show group help successfully when a command group is invoked bare."""
        if not args and self.no_args_is_help and not ctx.resilient_parsing:
            click.echo(ctx.get_help(), color=ctx.color)
            raise click.exceptions.Exit(0)
        return super().parse_args(ctx, args)

    def _format_active_context_line(self) -> str | None:
        """Format the current context/workspace for root help."""
        from nemo_platform_ext.config.config import get_context

        try:
            display_context = get_context()
        except Exception:
            return None

        return f"Active context: {display_context.context_name} (workspace: {display_context.workspace})"

    def command(self, *args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Command] | Command:  # pyright: ignore [reportIncompatibleMethodOverride]  # ty: ignore[invalid-method-override]
        """Override command decorator to use NmpCommand."""
        if "cls" not in kwargs:
            kwargs["cls"] = NmpCommand
        return super().command(*args, **kwargs)

    def resolve_command(self, ctx: click.Context, args: list[str]) -> tuple[str | None, Command | None, list[str]]:
        """Record the resolved subcommand name for telemetry, then resolve as usual."""
        cmd_name, cmd, remaining = super().resolve_command(ctx, args)
        from nemo_platform_ext.cli.telemetry import runtime

        runtime.note_subcommand(cmd_name)
        return cmd_name, cmd, remaining

    def get_command(self, ctx: click.Context, cmd_name: str) -> Command | None:
        """Override to patch command classes to use NeMo Platform formatting."""
        command = super().get_command(ctx, cmd_name)
        return self._patch_command(command) if command else None

    def _patch_command(self, command: Command) -> Command:
        """Patch Typer commands/groups to use NeMo Platform formatting."""
        if command.__class__.__name__ == "TyperCommand":
            command.__class__ = NmpCommand
            command.context_settings = _context_settings_with_help(command.context_settings)
            for param in command.params:
                if isinstance(param, TyperOption) and type(param) is not NmpOption:
                    param.__class__ = NmpOption
                elif isinstance(param, TyperArgument) and type(param) is not NmpArgument:
                    param.__class__ = NmpArgument
        elif command.__class__.__name__ == "TyperGroup":
            command.__class__ = NmpGroup
            command.context_settings = _context_settings_with_help(command.context_settings)
            for param in command.params:
                if isinstance(param, TyperOption) and type(param) is not NmpOption:
                    param.__class__ = NmpOption
                elif isinstance(param, TyperArgument) and type(param) is not NmpArgument:
                    param.__class__ = NmpArgument
        return command

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Format complete help output."""
        if ctx.parent is None:
            active_context_line = self._format_active_context_line()
            if active_context_line is not None:
                formatter.write(f"{active_context_line}\n\n")
        self.format_usage(ctx, formatter)
        self.format_help_text(ctx, formatter)
        self.format_commands(ctx, formatter)
        self.format_options(ctx, formatter)
        self.format_epilog(ctx, formatter)
        _maybe_format_agent_helpers(ctx, formatter)

    def format_help_text(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """
        Write the help text (description) and apply any rich markup.
        """
        if self.help:
            # Write description at top, before sections
            _write_with_formatting(formatter, self.help.strip())

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Write commands section, grouped by rich_help_panel."""
        # Group commands by rich_help_panel
        command_groups: dict[str | None, list[tuple[str, str]]] = {}

        # Compute how many characters are available for the help text on a single
        # line.  Each row is formatted as:
        #
        #   "  {name}{padding}{help_text}\n"
        #
        # The prefix occupies at most 29 characters:
        #   2  — leading indent
        #   25 — max command name (capped in the alignment pass below)
        #   2  — minimum column spacing
        #
        # get_short_help_str() truncates with "..." when help exceeds `limit`,
        # so passing the available width keeps long descriptions intact while
        # still fitting within the terminal.  The floor of 40 prevents absurdly
        # small limits on very narrow terminals.
        _help_limit = max(40, _get_terminal_width() - 29)

        for name in self.list_commands(ctx):
            lazy_entry = None
            if hasattr(self, "_lazy_entries"):
                lazy_entry = getattr(self, "_lazy_entries").get(name)
                if lazy_entry is not None and lazy_entry.hidden:
                    continue

            if name in self.commands or lazy_entry is None:
                cmd = self.get_command(ctx, name)
                if cmd is None or cmd.hidden:
                    continue
                help_text = cmd.get_short_help_str(limit=_help_limit)
                panel = getattr(cmd, "rich_help_panel", None)
            else:
                help_text = click.utils.make_default_short_help(lazy_entry.help or "", max_length=_help_limit)
                panel = lazy_entry.panel

            # Filter out DefaultPlaceholder objects
            if panel is not None:
                panel_type = type(panel).__name__
                if panel_type == "DefaultPlaceholder":
                    panel = None

            if panel not in command_groups:
                command_groups[panel] = []
            command_groups[panel].append((name, help_text))

        # Calculate max command name length across all commands for consistent alignment
        all_commands = [item for group in command_groups.values() for item in group]
        if all_commands:
            max_len = max(len(name) for name, _ in all_commands)
            max_len = min(max_len, 25)  # Cap at reasonable width
        else:
            return

        # Write ungrouped commands first
        if None in command_groups:
            with formatter.section("Commands"):
                for name, help_text in command_groups[None]:
                    colored_name = click.style(name, fg="cyan", bold=True)
                    visible_len = len(name)
                    padding = " " * (max_len - visible_len + 2)
                    formatter.write(f"  {colored_name}{padding}{help_text}\n")

        # Write grouped commands
        for panel_name in [g for g in command_groups.keys() if g is not None]:
            with formatter.section(panel_name):
                for name, help_text in command_groups[panel_name]:
                    colored_name = click.style(name, fg="cyan", bold=True)
                    visible_len = len(name)
                    padding = " " * (max_len - visible_len + 2)
                    formatter.write(f"  {colored_name}{padding}{help_text}\n")

    def format_options(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Write options grouped by help panel."""
        # Group options by rich_help_panel
        groups: dict[str | None, list[tuple[str, str]]] = {}

        for param in self.get_params(ctx):
            if not isinstance(param, click.Option):
                continue
            rv = param.get_help_record(ctx)
            if rv is None:
                continue

            # Color built-in options (help, install-completion, etc.) if not already colored
            if not isinstance(param, NmpOption) and rv:
                opt_str, help_text = rv
                # Check if already colored (contains ANSI codes)
                if "\x1b[" not in opt_str:
                    # Color it like other options
                    colored_opts = [
                        click.style(option_name, fg="cyan", bold=True)
                        for option_name in _option_display_names(param, getattr(ctx, "help_option_names", None))
                    ]
                    opt_str = ", ".join(colored_opts)
                    rv = (opt_str, help_text)

            # Put --help, --install-completion, --show-completion in Help section
            if param.name in ("help", "install_completion", "show_completion"):
                panel = "Help"
            else:
                panel = getattr(param, "rich_help_panel", None)
            if panel not in groups:
                groups[panel] = []
            groups[panel].append(rv)

        # Write ungrouped options first (if any)
        if None in groups:
            with formatter.section("Options"):
                formatter.write_dl(groups[None])

        # Write grouped options (Help section last)
        sorted_panels = sorted(g for g in groups.keys() if g is not None and g != "Help")
        for panel_name in sorted_panels:
            with formatter.section(panel_name):
                formatter.write_dl(groups[panel_name])

        # Write Help section last
        if "Help" in groups:
            with formatter.section("Help"):
                formatter.write_dl(groups["Help"])

    def format_epilog(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Write epilog if any."""
        if self.epilog:
            formatter.write("\n")
            formatter.write(self.epilog)
            formatter.write("\n")


def _style_examples(text: str) -> str:
    """Auto-style example blocks: dim comment lines, cyan command lines, consistent 2-space indent."""
    lines = text.split("\n")
    result = []
    in_examples = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("examples:") or stripped.lower().startswith("[green]examples:[/]"):
            in_examples = True
            result.append(line)
        elif in_examples and stripped:
            if stripped.startswith("#"):
                result.append(f"  [dim]{stripped}[/]")
            elif stripped.startswith("nemo "):
                result.append(f"  [cyan]{stripped}[/]")
            else:
                result.append(f"  {stripped}")
        else:
            if in_examples and not stripped:
                in_examples = False
            result.append(line)
    return "\n".join(result)


def _write_with_formatting(formatter: click.HelpFormatter, text: str) -> None:
    text = _style_examples(text)
    console = Console(file=StringIO(), record=True)
    console.print(text)
    formatter.write(console.export_text(styles=True))


def create_typer_app(**kwargs) -> Typer:
    """Create a Typer app with NeMo Platform-style formatting.

    This is a convenience wrapper around typer.Typer() that automatically
    applies NeMo Platform-style formatting to all commands and enables -h for help.

    Args:
        **kwargs: Arguments to pass to typer.Typer()

    Returns:
        A typer.Typer instance configured with NeMo Platform formatting
    """
    import typer

    kwargs.setdefault("cls", NmpGroup)
    kwargs.setdefault("no_args_is_help", True)
    # Shell completion is owned by the root ``nemo`` app; command groups (including
    # plugin-hosted roots mounted by the lazy loader) must not advertise it again.
    kwargs.setdefault("add_completion", False)
    kwargs["context_settings"] = _context_settings_with_help(kwargs.get("context_settings"))
    return typer.Typer(**kwargs)


def print_warnings(warnings: list[str | None] | None = None) -> None:
    """
    Print warnings as a bullet list to stderr.

    Args:
        warnings: List of warning messages to display (None values are filtered out)
    """
    if not warnings:
        return

    # Filter out None values
    warnings = [w for w in warnings if w]
    if not warnings:
        return

    error_console = Console(stderr=True)
    error_console.print()
    error_console.print("[bold yellow]Warnings:[/]")
    for warning in warnings:
        error_console.print(f"  • {warning}", style="yellow")


_warnings_context: ContextVar[list[str | None]] = ContextVar("warnings")

_P = ParamSpec("_P")
_R = TypeVar("_R")


def collect_warnings(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """
    Decorator that collects warnings and prints them at the end.

    Usage:
        @collect_warnings
        def my_command():
            add_warning("some warning")
            # ... warnings are automatically printed when the function returns
    """

    @wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        warnings: list[str | None] = []
        token = _warnings_context.set(warnings)
        try:
            return func(*args, **kwargs)
        finally:
            agent_hints = _get_agent_hints(args, kwargs)
            _warnings_context.reset(token)
            print_warnings(warnings)
            _print_agent_hints(agent_hints)

    return wrapper


def _get_agent_hints(args: tuple, kwargs: dict) -> list[str]:
    """Get agent hints if agent mode is active, extracting ctx from command args."""
    import typer as _typer

    ctx = None
    for arg in args:
        if isinstance(arg, _typer.Context):
            ctx = arg
            break
    if ctx is None:
        ctx = kwargs.get("ctx")
    if ctx is not None and hasattr(ctx, "obj") and getattr(ctx.obj, "agent_mode", False):
        from nemo_platform_ext.cli.core.agent_helpers import get_agent_helpers

        command_path = ctx.command_path
        parts = command_path.split(None, 1)
        command_path = parts[1] if len(parts) > 1 else ""
        return get_agent_helpers(command_path)
    return []


def _print_agent_hints(hints: list[str]) -> None:
    """Print agent hints to stderr under their own heading."""
    if not hints:
        return
    error_console = Console(stderr=True)
    error_console.print()
    error_console.print("[bold bright_green]AGENT HINTS:[/]")
    for hint in hints:
        error_console.print(f"  {hint}")


def add_warning(warning: str | list[str | None] | None) -> None:
    """
    Add one or more warnings to the current warnings collection.

    Must be called within a function decorated with `@collect_warnings`.
    If called outside such a function, the warning(s) are silently ignored.

    Args:
        warning: A single warning message, a list of warning messages, or None.
                 None values are allowed and filtered later when printing.
    """
    try:
        warnings = _warnings_context.get()
        if isinstance(warning, list):
            warnings.extend(warning)
        else:
            warnings.append(warning)
    except LookupError:
        # Not inside a collect_warnings context, ignore
        pass


def _maybe_format_agent_helpers(ctx: click.Context, formatter: click.HelpFormatter) -> None:
    """Append agent helper hints to help output when agent mode is active."""
    cli_ctx = getattr(ctx, "obj", None)
    if cli_ctx is None or not getattr(cli_ctx, "agent_mode", False):
        return

    from nemo_platform_ext.cli.core.agent_helpers import get_agent_helpers

    # Build command path without program name
    command_path = ctx.command_path
    parts = command_path.split(None, 1)
    command_path = parts[1] if len(parts) > 1 else ""

    helpers = get_agent_helpers(command_path)
    if helpers:
        formatter.write("\n")
        formatter.write(click.style("AGENT HINTS:", fg="bright_green", bold=True) + "\n")
        for helper in helpers:
            formatter.write(f"  {helper}\n")
