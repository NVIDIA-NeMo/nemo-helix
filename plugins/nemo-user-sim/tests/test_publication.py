# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import httpx
from nemo_helix_plugin.client.errors import ConflictError
from nemo_user_sim_plugin.engine.atif import row_to_atif
from nemo_user_sim_plugin.intake import publish_conversation, register_run
from nhx.intake.spans.ingest.atif import AtifIngestRequest
from nhx.intake.spans.ingest.atif_mapping import trajectory_to_spans

ROW = json.loads((Path(__file__).parent / "fixtures/finished_row.json").read_text())


def _payload(agent_name: str = "bank-support", run: str | None = None) -> dict:
    return row_to_atif(ROW, agent_name=agent_name, agent_version="1", model_name="m", run=run)


def _spans(payload: dict) -> list:
    """The spans Intake makes of an ATIF payload."""
    trajectory = AtifIngestRequest.model_validate(payload).to_trajectory()
    return trajectory_to_spans(workspace="default", trajectory=trajectory, ingested_at=datetime.now(UTC))


def test_conversation_becomes_one_intake_session_with_its_tool_calls():
    spans = _spans(_payload())

    assert {span.session_id for span in spans} == {ROW["trajectory_id"]}
    assert sorted(span.name for span in spans if str(span.kind) == "TOOL") == sorted(
        call["function"]["name"]
        for message in json.loads(ROW["conversation_messages"])
        for call in message.get("tool_calls") or []
    )


async def test_publication_failure_is_reported_not_raised():
    class Unreachable:
        async def create_atif(self, *, workspace, body):
            raise ConnectionError("intake unreachable")

    result = await publish_conversation(Unreachable(), "default", _payload())

    assert result == {"status": "failed", "session_id": ROW["trajectory_id"], "reason": "intake unreachable"}


def test_a_simulated_conversation_is_tagged_with_its_run_and_persona():
    spans = _spans(_payload(run="user-sim.simulate-abc"))
    root = next(span for span in spans if not span.external_parent_span_id)

    assert root.attributes_string["nemo.evaluation.name"] == "user-sim.simulate-abc"
    assert root.attributes_string["nemo.test_case.name"] == f"{ROW['persona_name']} · {ROW['probe_type']}"


def test_simulated_traces_never_show_under_the_real_agents_name():
    payload = _payload("compute-ops")
    spans = _spans(payload)

    assert payload["agent"]["name"] == "compute-ops (user-sim)"
    assert {span.attributes_string.get("gen_ai.agent.name") for span in spans} - {None} == {"compute-ops (user-sim)"}


async def test_a_run_registers_as_an_evaluation_once_and_survives_intake_trouble():
    class Intake:
        def __init__(self, error=None):
            self.error, self.created = error, []

        async def create_experiment(self, *, workspace, body, exist_ok):
            return SimpleNamespace(data=lambda: SimpleNamespace(id=f"exp-{body.name}"))

        async def create_evaluation(self, *, workspace, body):
            self.created.append(body.name)
            if self.error:
                raise self.error

    fresh, retried, down = (
        Intake(),
        Intake(ConflictError(httpx.Response(409, request=httpx.Request("POST", "http://intake")))),
        Intake(ConnectionError("down")),
    )

    assert await register_run(fresh, "default", "user-sim.simulate-abc", locale="en_US", target="bank-support")
    assert fresh.created == ["user-sim.simulate-abc"]
    assert await register_run(retried, "default", "user-sim.simulate-abc", locale="en_US", target="bank-support")
    assert not await register_run(down, "default", "user-sim.simulate-abc", locale="en_US", target="bank-support")
