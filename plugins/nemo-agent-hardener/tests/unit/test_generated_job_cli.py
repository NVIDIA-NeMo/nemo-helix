# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import httpx
import pytest
import typer
from fastapi import FastAPI
from nemo_agent_hardener_plugin.jobs.run import AgentHardenerRunJob
from nemo_agent_hardener_plugin.jobs.synth_benign import AgentHardenerSynthBenignJob
from nemo_agent_hardener_plugin.service import AgentHardenerPluginService
from nemo_helix_plugin.commands import add_job_commands
from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.scheduler import NemoJobScheduler
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


@pytest.mark.parametrize(
    ("job", "collection"),
    [(AgentHardenerRunJob, "/jobs"), (AgentHardenerSynthBenignJob, "/synth-benign/jobs")],
)
def test_scheduler_posts_to_registered_job_collection(job: type[NemoJob], collection: str) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, json={"name": "job-1"})

    expected_path = f"/apis/agent-hardener/v2/workspaces/default{collection}"
    app = FastAPI()
    for spec in AgentHardenerPluginService().get_routers():
        app.include_router(spec.router, prefix=f"/apis/agent-hardener{spec.prefix}")
    schema_path = expected_path.replace("/default", "/{workspace}")
    assert "post" in app.openapi()["paths"][schema_path]
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = NemoJobScheduler().submit_remote(
            job, {"manifest_id": "m1"}, base_url="http://platform.test", http_client=client
        )
    assert result["name"] == "job-1"
    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert requests[0].url.path == expected_path
