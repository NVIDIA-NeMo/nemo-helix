# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for resolving and describing commands in a click tree."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

import click
import pytest
from nemo_helix_ext.cli.core.command_description import describe_command, resolve_command_path


class _Color(Enum):
    RED = "red"
    BLUE = "blue"


@click.group()
def _root() -> None:
    """Root group."""


@_root.group("jobs")
@click.option("--profile", help="Execution profile.")
def _jobs() -> None:
    """[bold]Manage[/] jobs."""


@_jobs.command("run")
@click.argument("name")
@click.option("--color", type=click.Choice(_Color), default=_Color.RED, help="Pick a color.")
@click.option("--tag", multiple=True, default=("a", "b"))
@click.option("--out", type=click.Path(), default=Path("out.json"))
@click.option("--stamp", default=lambda: "computed")
@click.option("--dry-run", is_flag=True)
@click.option("--secret", hidden=True)
def _run(**_kwargs: object) -> None:
    """Run a job."""


@_root.command("broken")
def _broken() -> None:
    """Hidden broken command."""


_broken.hidden = True


def test_resolves_nested_path():
    resolved = resolve_command_path(_root, ["jobs", "run"])
    assert resolved.command is _run
    assert resolved.context.command_path == "nemo jobs run"
    assert resolved.ignored_args == []


def test_stops_at_first_option_and_after_a_leaf():
    at_option = resolve_command_path(_root, ["jobs", "--profile", "x", "run"])
    assert at_option.command is _jobs
    assert at_option.ignored_args == ["--profile", "x", "run"]

    after_leaf = resolve_command_path(_root, ["jobs", "run", "job-1", "--dry-run"])
    assert after_leaf.command is _run
    assert after_leaf.ignored_args == ["job-1", "--dry-run"]


def test_unknown_subcommand_raises_usage_error():
    with pytest.raises(click.UsageError, match=r"No such command 'nope' in 'nemo'\. Run 'nemo describe' to"):
        resolve_command_path(_root, ["nope"])


def test_describes_parameters_as_json_data():
    description = describe_command(resolve_command_path(_root, ["jobs", "run"]))

    json.dumps(description)
    assert description["kind"] == "command"
    assert description["arguments"] == [
        {
            "name": "NAME",
            "type": "text",
            "required": True,
            "multiple": False,
            "nargs": 1,
            "default": None,
            "help": "",
        }
    ]
    options = {option["name"]: option for option in description["options"]}
    assert set(options) == {"color", "tag", "out", "stamp", "dry_run"}
    assert options["color"]["choices"] == ["red", "blue"]
    assert options["color"]["default"] == "red"
    assert options["tag"]["default"] == ["a", "b"]
    assert options["tag"]["multiple"] is True
    assert options["out"]["default"] == "out.json"
    assert options["stamp"]["default"] is None
    assert options["dry_run"]["is_flag"] is True


def test_describes_group_subcommands_and_plain_help():
    description = describe_command(resolve_command_path(_root, ["jobs"]))
    assert description["kind"] == "group"
    assert description["help"] == "Manage jobs."
    assert description["subcommands"] == [{"name": "run", "help": "Run a job."}]
    assert [option["name"] for option in description["options"]] == ["profile"]


def test_hidden_subcommands_are_not_listed():
    description = describe_command(resolve_command_path(_root, []))
    assert [subcommand["name"] for subcommand in description["subcommands"]] == ["jobs"]


def test_subcommand_summaries_override_loading_children():
    description = describe_command(
        resolve_command_path(_root, []),
        subcommand_summaries=lambda: [("from-manifest", "Listed without loading.")],
    )
    assert description["subcommands"] == [{"name": "from-manifest", "help": "Listed without loading."}]
