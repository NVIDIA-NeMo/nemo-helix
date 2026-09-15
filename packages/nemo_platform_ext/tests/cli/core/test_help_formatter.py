# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the NeMo Platform help formatter."""

import os
from contextlib import contextmanager
from typing import Annotated
from unittest.mock import patch

import pytest
import typer
from nemo_platform_ext.cli.core.help_formatter import (
    NmpHelpFormatter,
    _get_terminal_width,
    _wrap_preserving_newlines,
    add_warning,
    collect_warnings,
    create_typer_app,
    print_warnings,
)
from typer.testing import CliRunner


@contextmanager
def mock_terminal_size(columns: int, lines: int = 24):
    """Mock terminal size for testing."""
    with patch("nemo_platform_ext.cli.core.help_formatter.shutil.get_terminal_size") as mock_size:
        mock_size.return_value = os.terminal_size((columns, lines))
        yield mock_size


class TestGetTerminalWidth:
    """Tests for _get_terminal_width function."""

    def test_returns_bounded_width(self):
        """Test that width is bounded to max 120."""
        with mock_terminal_size(200):
            assert _get_terminal_width() == 120  # Max width

    def test_returns_terminal_width_minus_padding(self):
        """Test that width accounts for padding."""
        with mock_terminal_size(80):
            assert _get_terminal_width() == 78  # 80 - 2

    def test_uses_fallback_for_small_terminal(self):
        """Test fallback for small terminals."""
        with mock_terminal_size(40):
            assert _get_terminal_width() == 38  # 40 - 2


class TestWrapPreservingNewlines:
    """Tests for _wrap_preserving_newlines function."""

    def test_preserves_single_newlines(self):
        """Test that single newlines are preserved."""
        text = "Line one\nLine two\nLine three"
        result = _wrap_preserving_newlines(text, width=80)
        assert result == "Line one\nLine two\nLine three"

    def test_preserves_empty_lines(self):
        """Test that empty lines are preserved."""
        text = "Line one\n\nLine three"
        result = _wrap_preserving_newlines(text, width=80)
        assert result == "Line one\n\nLine three"

    def test_wraps_long_lines(self):
        """Test that long lines are wrapped."""
        text = "This is a very long line that should be wrapped at some point"
        result = _wrap_preserving_newlines(text, width=30)
        lines = result.split("\n")
        assert len(lines) > 1
        for line in lines:
            assert len(line) <= 30

    def test_applies_initial_indent(self):
        """Test that initial indent is applied."""
        text = "Line one\nLine two"
        result = _wrap_preserving_newlines(text, width=80, initial_indent="  ")
        lines = result.split("\n")
        assert lines[0].startswith("  ")

    def test_applies_subsequent_indent(self):
        """Test that subsequent indent is applied to wrapped lines."""
        text = "Short\nThis is a longer line that needs wrapping"
        result = _wrap_preserving_newlines(text, width=30, subsequent_indent="    ")
        lines = result.split("\n")
        # Check that continuation lines have indent
        assert any(line.startswith("    ") for line in lines[1:])


class TestNmpHelpFormatter:
    """Tests for NmpHelpFormatter class."""

    def test_write_usage_adds_global_options(self):
        """Test that usage line includes global options placeholder."""
        formatter = NmpHelpFormatter(width=80)
        formatter.write_usage("nemo workspaces list", "[OPTIONS]")
        output = formatter.getvalue()
        assert "[GLOBAL OPTIONS]" in output
        assert "nemo" in output

    def test_write_usage_preserves_existing_global_options(self):
        """Test that existing global options aren't duplicated."""
        formatter = NmpHelpFormatter(width=80)
        formatter.write_usage("nemo [GLOBAL OPTIONS] workspaces list", "[OPTIONS]")
        output = formatter.getvalue()
        # Should only have one occurrence
        assert output.count("[GLOBAL OPTIONS]") == 1

    def test_write_heading_adds_color(self):
        """Test that headings are colored."""
        formatter = NmpHelpFormatter(width=80)
        formatter.write_heading("Options")
        output = formatter.getvalue()
        # Should contain ANSI color codes
        assert "\x1b[" in output
        assert "Options" in output

    def test_write_dl_single_line_format(self):
        """Test definition list with short options fits on single line."""
        formatter = NmpHelpFormatter(width=120)
        rows = [
            ("--help", "Show help"),
            ("--version", "Show version"),
        ]
        formatter.write_dl(rows)
        output = formatter.getvalue()
        # Each row should be on a single line
        lines = [line for line in output.split("\n") if line.strip()]
        assert len(lines) == 2

    def test_write_dl_wrapped_format(self):
        """Test definition list wraps when options are long."""
        formatter = NmpHelpFormatter(width=40)
        rows = [
            ("--very-long-option-name", "This is a very long description that should wrap"),
        ]
        formatter.write_dl(rows)
        output = formatter.getvalue()
        lines = [line for line in output.split("\n") if line.strip()]
        # Should have multiple lines due to wrapping
        assert len(lines) > 1

    def test_write_dl_empty_rows(self):
        """Test definition list handles empty rows."""
        formatter = NmpHelpFormatter(width=80)
        formatter.write_dl([])
        output = formatter.getvalue()
        assert output == ""


class TestNmpOption:
    """Tests for NmpOption formatting via help output."""

    def test_basic_option_in_help(self):
        """Test that basic options are formatted correctly in help."""
        app = create_typer_app()

        @app.command()
        def cmd(name: str = typer.Option(None, "--name", "-n", help="The name")):
            pass

        runner = CliRunner()
        result = runner.invoke(app, ["cmd", "--help"])

        assert result.exit_code == 0
        assert "--name" in result.stdout
        assert "-n" in result.stdout
        assert "The name" in result.stdout

    def test_hidden_option_not_in_help(self):
        """Test that hidden options don't appear in help."""
        app = create_typer_app()

        @app.command()
        def cmd(secret: str = typer.Option(None, "--secret", help="Secret", hidden=True)):
            pass

        runner = CliRunner()
        result = runner.invoke(app, ["cmd", "--help"])

        assert result.exit_code == 0
        assert "--secret" not in result.stdout

    def test_option_with_metavar_in_help(self):
        """Test that custom metavar appears in help."""
        app = create_typer_app()

        @app.command()
        def cmd(file: str = typer.Option(None, "--file", metavar="PATH", help="Input file")):
            pass

        runner = CliRunner()
        result = runner.invoke(app, ["cmd", "--help"])

        assert result.exit_code == 0
        assert "--file" in result.stdout
        assert "PATH" in result.stdout
        assert "Input file" in result.stdout

    def test_required_option_shows_marker(self):
        """Test that required options show [required] marker."""
        app = create_typer_app()

        @app.command()
        def cmd(name: str = typer.Option(..., "--name", help="Required name")):
            pass

        runner = CliRunner()
        result = runner.invoke(app, ["cmd", "--help"])

        assert result.exit_code == 0
        assert "--name" in result.stdout
        assert "[required]" in result.stdout

    def test_flag_option_no_metavar(self):
        """Test that flag options don't show value placeholder."""
        app = create_typer_app()

        @app.command()
        def cmd(verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")):
            pass

        runner = CliRunner()
        result = runner.invoke(app, ["cmd", "--help"])

        assert result.exit_code == 0
        assert "--verbose" in result.stdout
        assert "-v" in result.stdout
        # Flag options show without value placeholder - just check it's there
        assert "Verbose output" in result.stdout

    @pytest.mark.parametrize("help_flag", ["-h", "--help"], ids=["short", "long"])
    def test_help_option_short_and_long(self, help_flag):
        """Test that -h and --help both show help for commands."""
        app = create_typer_app()

        @app.command()
        def cmd(name: str = typer.Option(None, "--name", "-n", help="The name")):
            pass

        runner = CliRunner()
        result = runner.invoke(app, ["cmd", help_flag])

        assert result.exit_code == 0
        assert "--name" in result.stdout
        assert "The name" in result.stdout

    def test_help_option_shows_short_and_long_in_help(self):
        """Test that help output lists -h and --help for the help option."""
        app = create_typer_app()

        @app.command()
        def cmd():
            """A command."""
            pass

        runner = CliRunner()
        result = runner.invoke(app, ["cmd", "--help"])

        assert result.exit_code == 0
        assert "--help" in result.stdout
        assert "-h" in result.stdout

    def test_option_named_help_uses_declared_option_names(self):
        """A user option named help should not be treated as Click's help option."""
        app = create_typer_app()

        @app.command()
        def cmd(
            help: Annotated[  # noqa: A002
                str | None,
                typer.Option("--help-topic", help="Help topic to display."),
            ] = None,
        ):
            """A command."""
            _ = help

        runner = CliRunner()
        result = runner.invoke(app, ["cmd", "--help"])

        assert result.exit_code == 0
        assert "--help-topic" in result.stdout


class TestNmpArgument:
    """Tests for NmpArgument formatting via help output."""

    def test_basic_argument_in_help(self):
        """Test that basic arguments are formatted correctly in help."""
        app = create_typer_app()

        @app.command()
        def cmd(name: str = typer.Argument(help="The name argument")):
            pass

        runner = CliRunner()
        result = runner.invoke(app, ["cmd", "--help"])

        assert result.exit_code == 0
        assert "NAME" in result.stdout
        assert "The name argument" in result.stdout

    def test_hidden_argument_not_in_help(self):
        """Test that hidden arguments don't appear in help."""
        app = create_typer_app()

        @app.command()
        def cmd(name: str = typer.Argument(help="Hidden arg", hidden=True)):
            pass

        runner = CliRunner()
        result = runner.invoke(app, ["cmd", "--help"])

        assert result.exit_code == 0
        # Hidden argument should not appear in Arguments section
        assert "Hidden arg" not in result.stdout

    def test_required_argument_shows_marker(self):
        """Test that required arguments show [required] marker."""
        app = create_typer_app()

        @app.command()
        def cmd(name: str = typer.Argument(help="Required argument")):
            pass

        runner = CliRunner()
        result = runner.invoke(app, ["cmd", "--help"])

        assert result.exit_code == 0
        assert "NAME" in result.stdout
        assert "[required]" in result.stdout


class TestCreateTyperApp:
    """Tests for create_typer_app function."""

    def test_creates_typer_app(self):
        """Test that function creates a Typer app."""
        app = create_typer_app()
        assert isinstance(app, typer.Typer)

    def test_applies_custom_kwargs(self):
        """Test that custom kwargs are passed through."""
        app = create_typer_app(name="test-app", help="Test help")
        assert app.info.name == "test-app"
        assert app.info.help == "Test help"

    def test_default_no_args_is_help(self):
        """Test that apps created with create_typer_app show help when invoked with no args."""
        app = create_typer_app()

        @app.command()
        def foo():
            pass

        @app.command()
        def bar():
            pass

        runner = CliRunner()
        result = runner.invoke(app, [])

        # With no_args_is_help=True (default), no args shows full help successfully.
        assert result.exit_code == 0
        assert "Usage:" in result.stdout
        assert "foo" in result.stdout
        assert "bar" in result.stdout

    def test_no_args_is_help_can_be_overridden(self):
        """Test that no_args_is_help can be overridden to False."""
        app = create_typer_app(no_args_is_help=False)

        @app.command()
        def foo():
            pass

        @app.command()
        def bar():
            pass

        runner = CliRunner()
        result = runner.invoke(app, [])

        # With no_args_is_help=False, no args yields "Missing command" error, not full help
        assert result.exit_code == 2
        assert "Missing command" in result.output


class TestPrintWarnings:
    """Tests for print_warnings function."""

    @pytest.mark.parametrize(
        "input_warnings",
        [None, [], [None, None, None]],
        ids=["none", "empty_list", "only_none_values"],
    )
    def test_does_nothing_for_empty_or_none_input(self, capsys, input_warnings):
        """Test that print_warnings produces no output for None, empty, or None-only lists."""
        print_warnings(input_warnings)
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_prints_warnings_and_filters_none_values(self, capsys):
        """Test that print_warnings prints warnings and filters out None values."""
        print_warnings(["Warning 1", None, "Warning 2", None, "Warning 3"])
        captured = capsys.readouterr()
        assert "Warning 1" in captured.err
        assert "Warning 2" in captured.err
        assert "Warning 3" in captured.err
        assert "Warnings:" in captured.err


class TestCollectWarnings:
    """Tests for collect_warnings decorator."""

    def test_prints_warnings_on_function_exit(self, capsys):
        """Test that collect_warnings prints warnings when decorated function exits."""

        @collect_warnings
        def my_func():
            add_warning("Appended warning")
            add_warning("Extended warning 1")
            add_warning("Extended warning 2")

        my_func()

        captured = capsys.readouterr()
        assert "Appended warning" in captured.err
        assert "Extended warning 1" in captured.err
        assert "Extended warning 2" in captured.err
        assert "Warnings:" in captured.err

    def test_filters_none_values(self, capsys):
        """Test that None values are filtered when printing."""

        @collect_warnings
        def my_func():
            add_warning("Real warning")
            add_warning(None)
            add_warning("Another warning")

        my_func()
        captured = capsys.readouterr()
        assert "Real warning" in captured.err
        assert "Another warning" in captured.err

    def test_empty_or_none_only_warnings_not_printed(self, capsys):
        """Test that empty or None-only warnings don't print anything."""

        @collect_warnings
        def empty_func():
            pass  # Empty - no warnings added

        empty_func()
        captured = capsys.readouterr()
        assert captured.err == ""

        @collect_warnings
        def none_only_func():
            add_warning(None)  # Only None values

        none_only_func()
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_nested_decorators_work_independently(self, capsys):
        """Test that nested collect_warnings decorated functions work independently."""

        @collect_warnings
        def inner_func():
            add_warning("Inner warning")

        @collect_warnings
        def outer_func():
            add_warning("Outer warning")
            inner_func()
            # Inner function should have printed by now
            captured_inner = capsys.readouterr()
            assert "Inner warning" in captured_inner.err
            assert "Outer warning" not in captured_inner.err

        outer_func()
        # Now outer function exits
        captured_outer = capsys.readouterr()
        assert "Outer warning" in captured_outer.err

    def test_preserves_function_return_value(self):
        """Test that the decorator preserves function return value."""

        @collect_warnings
        def my_func():
            add_warning("Some warning")
            return "expected_result"

        result = my_func()
        assert result == "expected_result"

    def test_preserves_function_metadata(self):
        """Test that the decorator preserves function metadata."""

        @collect_warnings
        def my_func():
            """My docstring."""
            pass

        assert my_func.__name__ == "my_func"
        assert my_func.__doc__ == "My docstring."


class TestAddWarning:
    """Tests for add_warning function."""

    def test_adds_warnings_inside_decorated_function(self, capsys):
        """Test that add_warning adds warnings when inside collect_warnings decorated function."""

        @collect_warnings
        def my_func():
            add_warning("First warning")
            add_warning(None)  # Should be accepted but filtered on print
            add_warning("Second warning")

        my_func()

        captured = capsys.readouterr()
        assert "First warning" in captured.err
        assert "Second warning" in captured.err

    def test_silently_ignores_outside_decorated_function(self, capsys):
        """Test that add_warning silently ignores when outside decorated function."""
        add_warning("This should be ignored")
        captured = capsys.readouterr()
        assert captured.err == ""


class TestAddWarningWithList:
    """Tests for add_warning function with list input."""

    def test_adds_multiple_warnings_inside_decorated_function(self, capsys):
        """Test that add_warning adds multiple warnings when given a list."""

        @collect_warnings
        def my_func():
            add_warning(["First warning", "Second warning", "Third warning"])

        my_func()

        captured = capsys.readouterr()
        assert "First warning" in captured.err
        assert "Second warning" in captured.err
        assert "Third warning" in captured.err

    def test_filters_none_values_in_list(self, capsys):
        """Test that None values in list are filtered when printing."""

        @collect_warnings
        def my_func():
            add_warning(["Real warning", None, "Another warning", None])

        my_func()

        captured = capsys.readouterr()
        assert "Real warning" in captured.err
        assert "Another warning" in captured.err

    def test_list_silently_ignores_outside_decorated_function(self, capsys):
        """Test that add_warning with list silently ignores when outside decorated function."""
        add_warning(["This should be ignored", "This too"])
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_empty_list(self, capsys):
        """Test that empty list doesn't cause issues."""

        @collect_warnings
        def my_func():
            add_warning([])

        my_func()
        captured = capsys.readouterr()
        assert captured.err == ""


def test_create_typer_app_does_not_add_completion_options():
    """Groups (including plugin-hosted roots) leave shell completion to the root app."""
    import typer.main
    from nemo_platform_ext.cli.core.help_formatter import create_typer_app

    app = create_typer_app(name="group", help="A group")

    @app.command("noop")
    def _noop() -> None:
        pass

    command = typer.main.get_command(app)
    assert not any(param.name in {"install_completion", "show_completion"} for param in command.params)
