# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typer
from nemo_agent_hardener_plugin.jobs.run import AgentHardenerRunJob
from nemo_agent_hardener_plugin.jobs.synth_benign import AgentHardenerSynthBenignJob
from nemo_helix_plugin.commands import add_job_commands
from typer.testing import CliRunner


def _app() -> typer.Typer:
    app = typer.Typer()

    @app.callback()
    def _noop() -> None:
        pass

    add_job_commands(
        app,
        {
            "agent-hardener.war-game": AgentHardenerRunJob,
            "agent-hardener.synth": AgentHardenerSynthBenignJob,
        },
    )
    return app


def test_agent_hardener_jobs_use_flat_generated_cli() -> None:
    runner = CliRunner()
    app = _app()

    root_help = runner.invoke(app, ["--help"])
    assert root_help.exit_code == 0
    assert "war-game" in root_help.output
    assert "synth" in root_help.output

    for command in ("war-game", "synth"):
        help_result = runner.invoke(app, [command, "--help"])
        assert help_result.exit_code == 0
        assert "explain" in help_result.output

        legacy_submit = runner.invoke(app, [command, "submit", "--help"])
        assert legacy_submit.exit_code != 0
