# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared CLI option definitions.

The point of a shared ``--workspace`` option is that it cannot carry a literal
default -- that literal was the original bug, because it makes an omitted flag
indistinguishable from an explicit ``--workspace default``. These tests pin
that property, and that the two construction shapes behave identically.

They assert on *behavior* through a real Typer app rather than on
``OptionInfo`` internals. Typer stores the two shapes differently -- in the
``Annotated`` form the first string lands in ``OptionInfo.default`` and is
folded back into the flag names later -- so reading those attributes directly
reports something that looks broken but is not.
"""

from __future__ import annotations

import inspect

import pytest
import typer
from nemo_platform_plugin._spec_flags import kw
from nemo_platform_plugin.cli_options import (
    WORKSPACE_HELP,
    WORKSPACE_RESOLUTION,
    WorkspaceOption,
    workspace_help,
    workspace_option,
)
from typer.testing import CliRunner

runner = CliRunner()


def _app_using_alias() -> typer.Typer:
    """A hand-written command, the way plugin CLIs declare the flag."""
    app = typer.Typer()

    @app.command()
    def show(workspace: WorkspaceOption = None) -> None:
        typer.echo(repr(workspace))

    return app


def _app_using_factory() -> typer.Typer:
    """A programmatically-built command, the way the generated verbs do it."""
    app = typer.Typer()

    def show(**kwargs: object) -> None:
        typer.echo(repr(kwargs["workspace"]))

    setattr(
        show,
        "__signature__",
        inspect.Signature([kw("workspace", str | None, workspace_option())]),
    )
    app.command()(show)
    return app


@pytest.fixture(params=["alias", "factory"])
def app(request: pytest.FixtureRequest) -> typer.Typer:
    return _app_using_alias() if request.param == "alias" else _app_using_factory()


class TestWorkspaceOptionBehavior:
    """Both shapes must produce the same optional, default-free flag."""

    def test_omitted_flag_is_none_not_a_literal(self, app: typer.Typer) -> None:
        """The whole point: an omitted flag is distinguishable from an explicit one."""
        result = runner.invoke(app, [])
        assert result.exit_code == 0, result.output
        assert result.output.strip() == "None"

    def test_long_flag(self, app: typer.Typer) -> None:
        result = runner.invoke(app, ["--workspace", "team-a"])
        assert result.exit_code == 0, result.output
        assert result.output.strip() == "'team-a'"

    def test_short_flag(self, app: typer.Typer) -> None:
        result = runner.invoke(app, ["-w", "team-b"])
        assert result.exit_code == 0, result.output
        assert result.output.strip() == "'team-b'"

    def test_explicit_default_is_still_distinguishable(self, app: typer.Typer) -> None:
        """`--workspace default` must arrive as a value, not as "unset"."""
        result = runner.invoke(app, ["--workspace", "default"])
        assert result.exit_code == 0, result.output
        assert result.output.strip() == "'default'"

    def test_help_does_not_advertise_a_literal_default(self, app: typer.Typer) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0, result.output
        assert "[default: default]" not in result.output


class TestWorkspaceHelpText:
    def test_help_documents_the_resolution_order(self) -> None:
        assert WORKSPACE_RESOLUTION in WORKSPACE_HELP

    def test_env_outranks_the_configured_workspace(self) -> None:
        """Guard the precedence that has been documented backwards before.

        $NMP_WORKSPACE beats the context's configured workspace, because the
        SDK ``Config`` treats the env var as an override of the configured
        value (see ``Config.resolve``).
        """
        env_at = WORKSPACE_RESOLUTION.index("NMP_WORKSPACE")
        context_at = WORKSPACE_RESOLUTION.index("active CLI context")
        assert env_at < context_at, WORKSPACE_RESOLUTION

    def test_per_command_help_keeps_one_resolution_order(self) -> None:
        composed = workspace_help("Workspace path segment used in the submit URL.")
        assert composed.startswith("Workspace path segment used in the submit URL.")
        assert composed.endswith(WORKSPACE_RESOLUTION)

    def test_factory_help_override_reaches_the_cli(self) -> None:
        app = typer.Typer()

        def show(**kwargs: object) -> None:  # pragma: no cover - help only
            pass

        setattr(
            show,
            "__signature__",
            inspect.Signature([kw("workspace", str | None, workspace_option(help=workspace_help("Custom lead.")))]),
        )
        app.command()(show)

        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0, result.output
        assert "Custom lead." in result.output
