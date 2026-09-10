# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import typer
from nemo_platform_ext.cli.app import app
from typer.testing import CliRunner

from ..utils import assert_exit_code

# Basic runner for help/format tests that don't need API
basic_runner = CliRunner(env={"NMP_BASE_URL": "http://127.0.0.1"})


def _get_all_help_commands(typer_app: typer.Typer, prefix: list[str] | None = None) -> list[list[str]]:
    """Recursively generate all help command paths from a typer app.

    Args:
        typer_app: The typer application to traverse
        prefix: Current command path prefix

    Returns:
        List of command argument lists that should work with --help
    """
    if prefix is None:
        prefix = []

    commands: list[list[str]] = []

    # Add help for current level (the app/group itself)
    commands.append([*prefix, "--help"])

    # Traverse registered sub-apps (typer groups)
    for group in typer_app.registered_groups:
        group_name = group.name or (group.typer_instance.info.name if group.typer_instance else None)
        if group_name and group.typer_instance:
            commands.extend(_get_all_help_commands(group.typer_instance, [*prefix, group_name]))

    # Include registered commands (leaf commands)
    for cmd in typer_app.registered_commands:
        cmd_name = cmd.name or (cmd.callback.__name__ if cmd.callback else None)
        if cmd_name:
            commands.append([*prefix, cmd_name, "--help"])

    return commands


def test_global_options():
    """Test that the main CLI help shows all global options and content correctly."""
    result = basic_runner.invoke(app, ["--help"])
    assert_exit_code(result, 0)
    assert "Command-line interface for NeMo Platform" in result.stdout
    assert "--base-url" in result.stdout
    assert "--output-format" in result.stdout
    assert "--no-truncate" in result.stdout
    assert "--timestamp-format" in result.stdout
    assert "--verbose" in result.stdout
    assert "Output format" in result.stdout

    result = basic_runner.invoke(app, ["--output-format", "raw", "--help"])
    assert_exit_code(result, 0)


@pytest.mark.parametrize("help_flag", ["-h", "--help"], ids=["short", "long"])
def test_help_flag_short_and_long(help_flag):
    """Test that -h and --help both work at root and for subcommands."""
    result = basic_runner.invoke(app, [help_flag])
    assert_exit_code(result, 0)
    assert "Command-line interface for NeMo Platform" in result.stdout

    result = basic_runner.invoke(app, ["models", help_flag])
    assert_exit_code(result, 0)
    assert "Manage models" in result.stdout
    assert "create" in result.stdout
    assert "list" in result.stdout


@pytest.mark.skip(reason="Skipping exhaustive help command tests for now to speed up CI.")
@pytest.mark.parametrize(
    "command_args",
    _get_all_help_commands(app),
    ids=lambda args: " ".join(args),
)
def test_all_help_commands_parametrized(command_args):
    """Parametrized test for all commands with --help flag."""
    result = basic_runner.invoke(app, command_args)
    command_path = " ".join(command_args)

    assert_exit_code(result, 0)
    assert len(result.stdout) > 0, f"Command '{command_path}' produced no output"


def test_projects_list_pagination(runner: CliRunner, random_workspace):
    """Test the workspaces list command with pagination parameters."""
    # Create 5 projects to test pagination
    for i in range(5):
        result = runner.invoke(
            app,
            f"projects create --workspace {random_workspace} project-{i:02d} --description 'Project {i}'",
        )
        assert_exit_code(result, 0)

    # Test with small page size
    result = runner.invoke(
        app,
        f"projects list --workspace {random_workspace} --page-size 2 --output-format json",
    )

    assert_exit_code(result, 0)

    output = json.loads(result.stdout)
    assert "data" in output
    assert "pagination" in output
    # First page with page_size=2 should have 2 workspaces
    assert len(output["data"]) == 2
    assert output["pagination"]["page_size"] == 2
    assert output["pagination"]["total_results"] == 5
    assert output["pagination"]["total_pages"] == 3


def test_list_with_all_pages(runner: CliRunner, random_workspace):
    """Test list commands with --all-pages flag fetches all pages."""
    # Create multiple projects to test pagination
    for i in range(15):
        result = runner.invoke(
            app,
            f"projects create --workspace {random_workspace} project-{i:02d} --description 'Project {i}'",
        )
        assert_exit_code(result, 0)

    # Test with --all-pages and small page size to force multiple pages
    result = runner.invoke(
        app,
        f"projects list --workspace {random_workspace} --all-pages --page-size 5 --output-format json",
    )

    assert_exit_code(result, 0)

    output = json.loads(result.stdout)
    assert "data" in output
    # Should have all 15 projects despite page_size=5
    assert len(output["data"]) == 15
    assert output["pagination"]["total_results"] == 15
    assert output["pagination"]["total_pages"] == 1
    assert "More pages" not in result.stderr


def test_global_format_flag():
    """Test that global --output-format flag works."""
    result = basic_runner.invoke(app, ["--output-format", "raw", "--help"])
    assert_exit_code(result, 0)

    # Test that format flag is visible in help
    result = basic_runner.invoke(app, ["--help"])
    assert_exit_code(result, 0)
    assert "--output-format" in result.stdout
    assert "Output format" in result.stdout


@pytest.fixture
def test_project(runner: CliRunner, random_workspace) -> dict:
    """Create a test project for use in tests."""
    result = runner.invoke(
        app, f"projects create --workspace {random_workspace} 'test-project' --description 'Test Project'"
    )
    assert_exit_code(result, 0)
    return {"name": "test-project", "workspace": random_workspace}


@pytest.mark.parametrize(
    "format_type,should_show_note",
    [
        ("json", True),
        ("yaml", True),
        ("code", True),
        ("table", False),
        ("csv", False),
        ("markdown", False),
    ],
)
def test_projects_list_output_columns_with_format(format_type, should_show_note, runner: CliRunner, test_project):
    """Test that projects list shows/hides note based on format compatibility with --output-columns."""
    result = runner.invoke(
        app,
        [
            "projects",
            "list",
            "--workspace",
            test_project["workspace"],
            "--output-columns",
            "name,workspace",
            "--output-format",
            format_type,
        ],
    )

    assert_exit_code(result, 0)

    if should_show_note:
        assert f"Note: --output-columns is not used with `--output-format {format_type}`" in result.stderr
    else:
        assert "Note: --output-columns is not used" not in result.stderr
