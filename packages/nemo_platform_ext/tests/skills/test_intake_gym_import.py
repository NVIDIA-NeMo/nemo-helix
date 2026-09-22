# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gym evidence projection and compatibility with the existing Intake ingest contract."""

import importlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import pytest
from nmp.intake.spans.ingest.spans import DirectSpansIngestRequest, direct_span_to_domain

SCRIPTS = (
    Path(__file__).resolve().parents[4] / "packages/nemo_platform_ext/src/nemo_platform_ext/skills/nemo-intake/scripts"
)


@pytest.fixture
def importer(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("import_gym")


@pytest.fixture
def rollout() -> dict:
    return {
        "reward": 0.0,
        "ng_agent_observations": {"source": "test-agent", "records": [{"kind": "context_compaction"}]},
        "ng_trajectory": {
            "schema_version": "1.0",
            "task_id": "task-1",
            "rollout_id": "0-0",
            "invocations": [
                {
                    "invocation_id": "root",
                    "model_calls": [{"model_call_id": "call-1"}],
                    "status": "completed",
                    "duration_ms": 100,
                    "conversation": [
                        {"type": "function_call", "call_id": "tool-1", "name": "lookup", "arguments": '{"q":"x"}'}
                    ],
                },
                {"invocation_id": "child", "parent_invocation_id": "root", "status": "failed"},
            ],
            "model_calls": [
                {
                    "model_call_id": "call-1",
                    "started_at": 1790000000,
                    "completed_at": 1790000001,
                    "request": {"input": "Hello"},
                    "response": {"output": []},
                    "response_metadata": {"model": "test-model", "response_status": "completed"},
                    "token_stats": {"prompt_tokens": 12, "completion_tokens": 0},
                }
            ],
            "tool_calls": [
                {
                    "invocation_id": "root",
                    "tool_call_id": "tool-1",
                    "tool_name": "lookup",
                    "output": "42",
                    "duration_ms": 20,
                    "status": "completed",
                }
            ],
            "turns": [],
            "gaps": [{"code": "turns_unavailable"}],
        },
    }


def test_evidence_maps_to_existing_intake_contract(importer: ModuleType, rollout: dict) -> None:
    bundle = importer.map_gym_rollouts([rollout], run_id="run-1", agent_name="test-agent")
    request = DirectSpansIngestRequest.model_validate({"source": bundle.source, "spans": bundle.spans})
    root, agent, child, model, tool = request.spans
    assert model.kind.value == "LLM"
    assert model.parent_span_id == agent.span_id
    assert child.parent_span_id == agent.span_id
    assert tool.parent_span_id == agent.span_id
    assert agent.parent_span_id == root.span_id
    assert child.status.value == "error"
    # A successful top-level invocation may have recovered from a child failure.
    assert root.status.value == "success"
    assert root.attributes["gym.status_source"] == "top_level_invocations"
    assert root.attributes["gym.timing"] == "observed_child_window"
    assert root.started_at == model.started_at
    assert root.ended_at == model.ended_at
    assert model.input == {"input": "Hello"}
    assert tool.input == '{"q":"x"}'
    assert tool.output == "42"
    assert tool.ended_at is None
    assert agent.ended_at is None
    assert tool.attributes["gym.timing"] == "anchor_only"
    assert tool.attributes["gym.observed_duration_ms"] == 20
    assert root.attributes["gym.raw"] == rollout
    assert bundle.evaluator_results[0]["value"] == 0
    assert bundle.evaluator_results[0]["span_id"] == root.span_id
    stored, semantic = direct_span_to_domain(
        workspace="default",
        source="gym",
        span=model,
        ingested_at=datetime.now(timezone.utc),
    )
    assert stored.source_format == "gym"
    assert semantic.agent_name == "test-agent"
    assert stored.attributes_number["llm.token_count.prompt"] == 12
    assert stored.attributes_number["llm.token_count.completion"] == 0


def test_replay_and_run_isolation(importer: ModuleType, rollout: dict) -> None:
    first = importer.map_gym_rollouts([rollout], run_id="one", agent_name="agent")
    assert first.as_json() == importer.map_gym_rollouts([rollout], run_id="one", agent_name="agent").as_json()
    second = importer.map_gym_rollouts([rollout], run_id="two", agent_name="agent")
    assert {s["span_id"] for s in first.spans}.isdisjoint(s["span_id"] for s in second.spans)
    assert first.spans[0]["session_id"] != second.spans[0]["session_id"]


@pytest.mark.parametrize(
    "statuses,expected",
    [
        (["completed", "completed"], "success"),
        (["completed", "failed"], "error"),
        (["completed", "cancelled"], "cancelled"),
        (["completed", "incomplete"], "unknown"),
        (["unknown"], "unknown"),
        ([], "unknown"),
    ],
)
def test_rollout_status_summarizes_top_level_invocations(
    importer: ModuleType, rollout: dict, statuses: list[str], expected: str
) -> None:
    rollout["ng_trajectory"]["invocations"] = [
        {"invocation_id": str(index), "status": status} for index, status in enumerate(statuses)
    ]
    root = importer.map_gym_rollouts([rollout], run_id="one", agent_name="agent").spans[0]
    assert root["status"] == expected


def test_explicit_rollout_status_overrides_invocations(importer: ModuleType, rollout: dict) -> None:
    rollout["status"] = "failed"
    root = importer.map_gym_rollouts([rollout], run_id="one", agent_name="agent").spans[0]
    assert root["status"] == "error"
    assert root["attributes"]["gym.status_source"] == "rollout"


def test_missing_invocation_parent_does_not_imply_rollout_success(importer: ModuleType, rollout: dict) -> None:
    rollout["ng_trajectory"]["invocations"][1]["parent_invocation_id"] = "missing"
    root = importer.map_gym_rollouts([rollout], run_id="one", agent_name="agent").spans[0]
    assert root["status"] == "unknown"


def test_parallel_calls_use_elapsed_window_not_sum(importer: ModuleType, rollout: dict) -> None:
    first = rollout["ng_trajectory"]["model_calls"][0]
    second = deepcopy(first)
    second.update(model_call_id="call-2", started_at=1790000000.5, completed_at=1790000002)
    rollout["ng_trajectory"]["model_calls"].append(second)
    rollout["ng_perf"] = {"total_latency_ms": 2200}
    root = importer.map_gym_rollouts([rollout], run_id="one", agent_name="agent").spans[0]
    elapsed = datetime.fromisoformat(root["ended_at"]) - datetime.fromisoformat(root["started_at"])
    assert elapsed.total_seconds() == 2
    assert root["attributes"]["gym.observed_duration_ms"] == 2200
    assert root["attributes"]["gym.duration_source"] == "ng_perf.total_latency_ms"


def test_rollup_preserves_earlier_turn_anchor_for_replay(importer: ModuleType, rollout: dict) -> None:
    rollout["ng_trajectory"]["turns"] = [{"timestamp": 1789999999}]
    root = importer.map_gym_rollouts([rollout], run_id="one", agent_name="agent").spans[0]
    assert datetime.fromisoformat(root["started_at"]).timestamp() == 1789999999
    assert datetime.fromisoformat(root["ended_at"]).timestamp() == 1790000001


def test_elapsed_duration_does_not_fabricate_timestamps(importer: ModuleType, rollout: dict) -> None:
    rollout["ng_trajectory"]["model_calls"] = []
    rollout["ng_perf"] = {"total_latency_ms": 2200}
    root = importer.map_gym_rollouts(
        [rollout], run_id="one", agent_name="agent", started_at=datetime(2026, 9, 21, tzinfo=timezone.utc)
    ).spans[0]
    assert root["ended_at"] is None
    assert root["attributes"]["gym.timing"] == "anchor_only"
    assert root["attributes"]["gym.observed_duration_ms"] == 2200


def test_incomplete_observed_interval_does_not_imply_rollout_end(importer: ModuleType, rollout: dict) -> None:
    del rollout["ng_trajectory"]["model_calls"][0]["completed_at"]
    root = importer.map_gym_rollouts([rollout], run_id="one", agent_name="agent").spans[0]
    assert root["ended_at"] is None


def test_explicit_rollout_bounds_are_preserved(importer: ModuleType, rollout: dict) -> None:
    rollout.update(started_at=1789999999, completed_at=1790000003)
    root = importer.map_gym_rollouts([rollout], run_id="one", agent_name="agent").spans[0]
    elapsed = datetime.fromisoformat(root["ended_at"]) - datetime.fromisoformat(root["started_at"])
    assert elapsed.total_seconds() == 4
    assert root["attributes"]["gym.timing"] == "observed"


@pytest.mark.parametrize("duration", [-1, float("nan"), float("inf"), True, "100"])
def test_invalid_rollout_duration_rejected(importer: ModuleType, rollout: dict, duration: object) -> None:
    rollout["ng_perf"] = {"total_latency_ms": duration}
    with pytest.raises(ValueError, match="finite nonnegative"):
        importer.map_gym_rollouts([rollout], run_id="one", agent_name="agent")


def test_untimed_rollout_requires_explicit_anchor(importer: ModuleType, rollout: dict) -> None:
    call = rollout["ng_trajectory"]["model_calls"][0]
    del call["started_at"]
    del call["completed_at"]
    with pytest.raises(ValueError, match="no absolute timestamps"):
        importer.map_gym_rollouts([rollout], run_id="one", agent_name="agent")
    bundle = importer.map_gym_rollouts(
        [rollout],
        run_id="one",
        agent_name="agent",
        started_at=datetime(2026, 9, 21, tzinfo=timezone.utc),
    )
    assert all(s["ended_at"] is None for s in bundle.spans)
    assert all(s["attributes"]["gym.timing"] == "anchor_only" for s in bundle.spans)


def test_ambiguous_call_reference_does_not_invent_ownership(importer: ModuleType, rollout: dict) -> None:
    trajectory = rollout["ng_trajectory"]
    ref = {"model_ref": {"name": "model"}, "response_id": "response"}
    trajectory["invocations"][0]["model_calls"] = [ref]
    trajectory["model_calls"][0]["response_metadata"].update(ref)
    other = deepcopy(trajectory["model_calls"][0])
    other["model_call_id"] = "call-2"
    trajectory["model_calls"].append(other)
    bundle = importer.map_gym_rollouts([rollout], run_id="one", agent_name="agent")
    models = [s for s in bundle.spans if s["kind"] == "LLM"]
    assert len(models) == 2
    assert all(s["parent_span_id"] == bundle.spans[0]["span_id"] for s in models)
    assert all(s["attributes"]["gym.ownership"] == "unavailable_or_ambiguous" for s in models)


@pytest.mark.parametrize(
    "change,match",
    [
        ("version", "schema_version"),
        ("duplicate", "duplicate"),
        ("cycle", "cycle"),
        ("time", "precedes"),
    ],
)
def test_invalid_evidence_rejected(importer: ModuleType, rollout: dict, change: str, match: str) -> None:
    trajectory = rollout["ng_trajectory"]
    if change == "version":
        trajectory["schema_version"] = "2.0"
    elif change == "duplicate":
        trajectory["model_calls"].append(deepcopy(trajectory["model_calls"][0]))
    elif change == "cycle":
        trajectory["invocations"][0]["parent_invocation_id"] = "child"
    else:
        trajectory["model_calls"][0]["completed_at"] = 1
    with pytest.raises(ValueError, match=match):
        importer.map_gym_rollouts([rollout], run_id="one", agent_name="agent")


def test_jsonl_dry_run_never_opens_writer(
    importer: ModuleType, rollout: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    path = tmp_path / "rollouts.jsonl"
    path.write_text(json.dumps(rollout) + "\n")
    monkeypatch.setattr(
        "sys.argv", ["import_gym.py", "--input", str(path), "--run-id", "one", "--agent-name", "agent", "--dry-run"]
    )
    common = importlib.import_module("_import_common")

    def forbidden_writer(**kwargs: object) -> None:
        pytest.fail("dry run opened an Intake writer")

    monkeypatch.setattr(common, "IntakeWriter", forbidden_writer)
    assert importer.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["source"] == "gym"
    assert len(result["spans"]) == 5


def test_feedback_can_be_suppressed_and_malformed_jsonl_has_line(
    importer: ModuleType, rollout: dict, tmp_path: Path
) -> None:
    bundle = importer.map_gym_rollouts([rollout], run_id="one", agent_name="agent", include_feedback=False)
    assert bundle.evaluator_results == []
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(rollout) + "\nnope\n")
    with pytest.raises(ValueError, match="bad.jsonl:2"):
        importer.load_rollouts(path)
