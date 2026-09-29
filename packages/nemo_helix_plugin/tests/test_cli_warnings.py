# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for collecting and printing CLI warnings."""

import pytest
import typer
from nemo_helix_plugin.cli_warnings import add_warning, collect_warnings, print_warnings
from typer.testing import CliRunner


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


class _HintState:
    """Stand-in CLI state that records which command asked for hints."""

    def __init__(self, hints: list[str]) -> None:
        self.hints = hints
        self.command_paths: list[str] = []

    def get_agent_hints(self, command_path: str) -> list[str]:
        self.command_paths.append(command_path)
        return self.hints


def _app_with_warning() -> typer.Typer:
    app = typer.Typer()
    group = typer.Typer()
    app.add_typer(group, name="widgets")

    @group.command("list")
    @collect_warnings
    def list_widgets(ctx: typer.Context) -> None:
        add_warning("Use --no-truncate to see full values.")

    return app


class TestAgentHints:
    """Agent hints come from the CLI state and print after the command's warnings."""

    def test_asks_the_state_for_the_command_path_without_the_program_name(self) -> None:
        state = _HintState(["Run: nemo docs widgets"])
        result = CliRunner().invoke(_app_with_warning(), ["widgets", "list"], obj=state, prog_name="nemo")
        assert result.exit_code == 0, result.output
        assert state.command_paths == ["widgets list"]
        assert result.stderr.index("Warnings:") < result.stderr.index("AGENT HINTS:")
        assert "Run: nemo docs widgets" in result.stderr

    def test_no_heading_when_the_state_has_no_hints(self) -> None:
        result = CliRunner().invoke(_app_with_warning(), ["widgets", "list"], obj=_HintState([]))
        assert result.exit_code == 0, result.output
        assert "AGENT HINTS:" not in result.stderr

    @pytest.mark.parametrize("obj", [None, object()], ids=["no_state", "state_without_hints"])
    def test_states_without_hints_still_print_warnings(self, obj: object | None) -> None:
        result = CliRunner().invoke(_app_with_warning(), ["widgets", "list"], obj=obj)
        assert result.exit_code == 0, result.output
        assert "Use --no-truncate to see full values." in result.stderr
        assert "AGENT HINTS:" not in result.stderr
