# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for scoring the saved Evals skill trials."""

import importlib
import io
import json
import tarfile
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, create_autospec

import pytest
from nemo_helix_plugin.evals.client import EvaluatorClient
from nhx_evals_sdk.agent_eval.evaluator import AgentEvaluator
from nhx_evals_sdk.agent_eval.persistence import persist_run
from nhx_evals_sdk.agent_eval.tasks import AgentEvalTask
from nhx_evals_sdk.agent_eval.trials import (
    AgentEvalTrial,
    AgentOutput,
    TrialMeasurements,
    standard_evidence_descriptors,
)
from nhx_evals_sdk.metrics.runner_rewards import HarborRewardMetric
from nhx_evals_sdk.metrics.skills import SkillActivationMetric, TokenEfficiencyMetric, expected_behavior_metric
from nhx_evals_sdk.structured_output import StructuredOutputMode
from nhx_evals_sdk.values.evidence import CandidateEvidence
from nhx_evals_sdk.values.models import Model
from nhx_evals_sdk.values.skills import SkillSnapshot

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/harbor_evals_skill"
SKILL = "nemo-evals-plugin"
REGISTERED = f"/home/harbor-agent/.agents/skills/{SKILL}/SKILL.md"
DOCUMENT = "---\nname: nemo-evals-plugin\ndescription: Run evaluations.\n---\n# Run an evaluation\n"


def _load_example(monkeypatch: pytest.MonkeyPatch, name: str) -> ModuleType:
    assert (EXAMPLE / f"{name}.py").is_file(), f"example must implement {name}.py"
    monkeypatch.syspath_prepend(str(EXAMPLE))
    return importlib.import_module(name)


def _event(kind: str, **payload: object) -> dict:
    return {"type": kind, "payload": payload}


def _catalog(body: str | None, *, listing: bool = True, surface: str = "host_skills") -> dict:
    return _event("world_state", full=True, state={surface: {"includeInstructions": listing, "body": body}})


def _list_entry(description: str) -> str:
    return f"\n## Skills\nSkill catalog.\n### Available skills\n{description}\n"


@pytest.mark.parametrize(
    ("body", "surface"),
    [
        (_list_entry(f"- {SKILL}: Use Evals (file: {REGISTERED})"), "host_skills"),
        (
            "\n## Skills\nSkill catalog.\n### Skill roots\n"
            "- `r0` = `/home/harbor-agent/.agents/skills`\n### Available skills\n"
            f"- {SKILL}: Use Evals (file: r0/{SKILL}/SKILL.md)\n",
            "host_skills",
        ),
        (
            "\n## Skills\nSkill catalog.\n### Skill roots\n"
            "- `e0` = `skill://root-id/home/harbor-agent/.agents/skills`\n### Available skills\n"
            f"- {SKILL}: Use Evals (executor package: e0/{SKILL})\n",
            "skills",
        ),
    ],
)
def test_initial_catalog_binds_the_injected_skill_to_its_registered_path(
    monkeypatch: pytest.MonkeyPatch, body: str, surface: str
) -> None:
    evidence = _load_example(monkeypatch, "codex_evidence")
    observed = evidence.observe_codex([_catalog(body, surface=surface)], SKILL, DOCUMENT)
    assert observed.skill_listed is True
    assert observed.activated is None


def test_later_catalog_cannot_rewrite_the_explicitly_empty_initial_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    evidence = _load_example(monkeypatch, "codex_evidence")
    observed = evidence.observe_codex(
        [_catalog(None), _catalog(_list_entry(f"- {SKILL}: (file: {REGISTERED})"))], SKILL, DOCUMENT
    )
    assert observed.skill_listed is False


@pytest.mark.parametrize("with_skill", [False, True])
def test_pinned_rollout_omits_empty_executor_catalog_body(monkeypatch: pytest.MonkeyPatch, with_skill: bool) -> None:
    evidence = _load_example(monkeypatch, "codex_evidence")
    entries = "- skill-creator: Create documented skills (file: /opt/codex/skill-creator/SKILL.md)"
    if with_skill:
        entries += f"\n- {SKILL}: Use Evals (file: {REGISTERED})"
    catalog = _event(
        "world_state",
        full=True,
        state={
            "host_skills": {"body": _list_entry(entries), "includeInstructions": True},
            "skills": {"includeInstructions": True},
        },
    )
    count = _event(
        "event_msg",
        type="token_count",
        info={"total_token_usage": {"input_tokens": 120, "output_tokens": 13, "cached_input_tokens": 0}},
    )

    observed = evidence.observe_codex([catalog, count], SKILL, DOCUMENT)

    assert observed.skill_listed is with_skill
    assert observed.measurements == TrialMeasurements(prompt_tokens=120, completion_tokens=13, cache_read_tokens=0)


@pytest.mark.parametrize(
    "events",
    [
        [_catalog(None, listing=False)],
        [_catalog(_list_entry("- 1 additional skill omitted from this bounded skills list."))],
        [_catalog(_list_entry(f"- {SKILL}: (file: /workspace/.agents/skills/{SKILL}/SKILL.md)"))],
        [_event("response_item", type="function_call"), _catalog(None)],
    ],
)
def test_inconclusive_or_unregistered_launch_catalog_fails_closed(
    monkeypatch: pytest.MonkeyPatch, events: list[dict]
) -> None:
    evidence = _load_example(monkeypatch, "codex_evidence")
    with pytest.raises(ValueError, match="catalog"):
        evidence.observe_codex(events, SKILL, DOCUMENT)


def test_latest_cumulative_usage_keeps_zero_and_never_backfills_missing_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _load_example(monkeypatch, "codex_evidence")
    events = [
        _catalog(None),
        _event(
            "event_msg",
            type="token_count",
            info={"total_token_usage": {"input_tokens": 30, "output_tokens": 1, "cached_input_tokens": 0}},
        ),
        _event(
            "event_msg",
            type="token_count",
            info={"total_token_usage": {"input_tokens": 50, "output_tokens": 0}},
        ),
        _event("event_msg", type="token_count", info=None),
    ]
    usage = evidence.observe_codex(events, SKILL, DOCUMENT).measurements
    assert usage == TrialMeasurements(prompt_tokens=50, completion_tokens=0)
    assert usage.cache_read_tokens is None
    assert usage.total_tokens == 50
    usage = evidence.observe_codex(events[:2], SKILL, DOCUMENT).measurements
    assert usage.cache_read_tokens == 0
    assert usage.total_tokens == 31


def test_activation_requires_a_successful_read_of_the_registered_file(monkeypatch: pytest.MonkeyPatch) -> None:
    evidence = _load_example(monkeypatch, "codex_evidence")
    events = [
        _catalog(_list_entry(f"- {SKILL}: (file: {REGISTERED})")),
        _event(
            "response_item",
            type="function_call",
            namespace="skills",
            name="read",
            call_id="read1",
            arguments=json.dumps({"package": REGISTERED}),
        ),
        _event(
            "response_item",
            type="function_call_output",
            call_id="read1",
            output=json.dumps({"resource": REGISTERED, "contents": DOCUMENT}),
        ),
    ]
    assert evidence.observe_codex(events[:2], SKILL, DOCUMENT).activated is None
    assert evidence.observe_codex(events, SKILL, DOCUMENT).activated is True


def test_shell_reference_counts_only_when_the_read_returns_the_skill_document(monkeypatch: pytest.MonkeyPatch) -> None:
    evidence = _load_example(monkeypatch, "codex_evidence")
    launch = _catalog(_list_entry(f"- {SKILL}: (file: {REGISTERED})"))
    completion = _event(
        "event_msg",
        type="item_completed",
        item={
            "type": "CommandExecution",
            "command": ["/bin/bash", "-lc", f"sed -n '1,80p' \"{REGISTERED}\""],
            "exit_code": 0,
            "status": "completed",
            "aggregated_output": DOCUMENT,
        },
    )
    assert evidence.observe_codex([launch, completion], SKILL, DOCUMENT).activated is True
    completion["payload"]["item"]["command"] = ["echo", REGISTERED]
    assert evidence.observe_codex([launch, completion], SKILL, DOCUMENT).activated is None


def test_replay_payload_preserves_zero_reward_and_omits_inaccessible_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    scoring = _load_example(monkeypatch, "score")
    trial = AgentEvalTrial(
        id="trial-id",
        task_id="harbor/demo",
        status="completed",
        output=AgentOutput(output_text="  "),
        metadata={
            "harbor_primary_reward_key": "reward",
            "reward": 0.0,
            "reward_details": {"reward": 0.0},
            "harbor_trial_dir": "/remote/evidence/harbor/trial-id",
        },
        measurements=TrialMeasurements(prompt_tokens=123, completion_tokens=1, cache_read_tokens=0),
    )
    replay = scoring.replay_trial(trial, 0, TrialMeasurements(prompt_tokens=0, completion_tokens=0), None)
    assert (replay.id, replay.task_id, replay.output, replay.evidence) == (
        "trial-id",
        "harbor/demo",
        AgentOutput(),
        None,
    )
    assert replay.metadata == {
        "harbor_primary_reward_key": "reward",
        "reward": 0.0,
        "reward_details": {"reward": 0.0},
    }
    assert replay.measurements.total_tokens == 0
    assert replay.measurements.cache_read_tokens is None
    with pytest.raises(ValueError, match="reward"):
        scoring.replay_trial(trial, 1, TrialMeasurements(), True)


@pytest.mark.parametrize(
    ("answer", "inference_error", "view_mean", "reported_errors"),
    [
        ("It is complete", None, 1.0, []),
        (None, None, None, []),
        ("It is complete", RuntimeError("fixture failed"), None, ["llm-judge"]),
    ],
)
async def test_replay_reads_native_views_and_keeps_metric_failure_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    answer: str | None,
    inference_error: Exception | None,
    view_mean: float | None,
    reported_errors: list[str],
) -> None:
    scoring = _load_example(monkeypatch, "score")
    judge, views = expected_behavior_metric(Model(name="fixture", url="http://judge.invalid/v1"), ["Is complete?"])
    inference = AsyncMock(
        return_value={"choices": [{"message": {"content": '{"assertion_1":"met"}'}}]},
        side_effect=inference_error,
    )
    judge.set_inference_fn(inference)
    monkeypatch.setattr(
        "nhx_evals_sdk.metrics.llm_judge.detect_structured_output_mode",
        AsyncMock(return_value=StructuredOutputMode.UNSUPPORTED),
    )
    task = AgentEvalTask(
        id="harbor/demo",
        intent="Run an evaluation",
        inputs={"instruction": "Run an evaluation"},
        metrics=[
            HarborRewardMetric(output_name="reward"),
            SkillActivationMetric(skill_name=SKILL, should_activate=True),
            TokenEfficiencyMetric(),
            judge,
        ],
        views=views,
    )
    trial = AgentEvalTrial(
        id="trial",
        task_id=task.id,
        status="completed",
        output=AgentOutput(output_text=answer),
        metadata={"harbor_primary_reward_key": "reward", "reward": 1.0, "reward_details": {"reward": 1.0}},
    )
    result = await AgentEvaluator().run(tasks=[task], trials=[trial])
    source = tmp_path / "fixture"
    persist_run(result, source, write_html_dashboard=False)
    compressed = io.BytesIO()
    with tarfile.open(fileobj=compressed, mode="w:gz") as archive:
        archive.add(source, arcname="result")

    @contextmanager
    def stream():
        yield [compressed.getvalue()]

    client = create_autospec(EvaluatorClient, instance=True)
    client.download_agent_eval_job_result.return_value.stream.side_effect = stream
    archive_path = tmp_path / "scoring-results.tar.gz"
    replay, errors = scoring._replay_results(client, "fixture-job", trial, archive_path)

    assert archive_path.read_bytes() == compressed.getvalue()
    assert replay["bundle"] == archive_path.name
    assert errors == reported_errors
    aggregates = {entry["name"]: entry["mean"] for entry in replay["summary"]["scores"]["scores"]}
    assert aggregates["view.expected_behavior"] == aggregates["view.all_assertions_met"] == view_mean
    judge_score = next(score for score in replay["scores"] if score["metric_type"] == "llm-judge")
    if answer is None:
        inference.assert_not_awaited()
        assert judge_score["diagnostics"][0]["details"]["exception_type"] == "UndefinedError"
    if inference_error is not None:
        assert judge_score["diagnostics"][0]["message"] == str(inference_error)


def _write_rollout(sessions: Path, name: str, session: dict, *events: dict) -> None:
    sessions.mkdir(parents=True, exist_ok=True)
    entries = [_event("session_meta", **session), *events]
    (sessions / f"rollout-{name}.jsonl").write_text("\n".join(map(json.dumps, entries)) + "\n")


def test_root_exec_rollout_ignores_child_usage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    evidence = _load_example(monkeypatch, "codex_evidence")
    agent = tmp_path / "agent"
    sessions = agent / "sessions/2026/11/19"
    meta = {"source": "exec", "cli_version": "0.162.1", "session_id": "root", "id": "root"}
    _write_rollout(sessions, "root", meta, _catalog(None))
    _write_rollout(sessions, "child", {**meta, "source": {"subagent": "review"}, "id": "child"}, _catalog(None))
    records = list(evidence.rollout_events(agent))
    assert records == [_event("session_meta", **meta), _catalog(None)]
    _write_rollout(sessions, "another-root", {**meta, "id": "other", "session_id": "other"}, _catalog(None))
    with pytest.raises(ValueError, match="one Codex root"):
        list(evidence.rollout_events(agent))


def test_treatment_evidence_checks_files_egress_and_probe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scoring = _load_example(monkeypatch, "score")
    bundle = tmp_path / "bundle"
    trial_dir = bundle / "evidence/harbor/job/attempt"
    agent = trial_dir / "agent"
    artifacts = trial_dir / "artifacts"
    artifacts.mkdir(parents=True)
    _write_rollout(
        agent / "sessions/2026/11/19",
        "root",
        {"source": "exec", "cli_version": "0.162.1", "session_id": "root", "id": "root"},
        _catalog(_list_entry(f"- {SKILL}: (file: {REGISTERED})")),
    )
    snapshot = SkillSnapshot(directory_name=SKILL, digest="a" * 64, document=DOCUMENT, files=["SKILL.md"])
    inventory = {SKILL: {"SKILL.md": "b" * 64}}
    (artifacts / "skill-inventory.json").write_text(json.dumps({"staged": inventory, "registered": {}}))
    (agent / "scope-probe.json").write_text('{"enabled": false}')
    log = trial_dir / "trial.log"
    log.write_text("Codex is already available at the requested version\nClosed agent setup egress\n")
    trial = AgentEvalTrial(
        id="attempt",
        task_id="harbor/demo",
        status="completed",
        output=AgentOutput(),
        evidence=CandidateEvidence(
            descriptors=standard_evidence_descriptors(logs_dir=agent, final_state_dir=artifacts)
        ),
    )
    with pytest.raises(ValueError, match="registered skill inventories"):
        scoring._evidence(bundle, trial, snapshot, inventory, expect_setup_egress=True)

    (artifacts / "skill-inventory.json").write_text(json.dumps({"staged": inventory, "registered": inventory}))
    assert scoring._evidence(bundle, trial, snapshot, inventory, expect_setup_egress=True).skill_listed is True
    (agent / "scope-probe.json").write_text('{"enabled": 0}')
    with pytest.raises(ValueError, match="probe must be disabled"):
        scoring._evidence(bundle, trial, snapshot, inventory, expect_setup_egress=True)
    (agent / "scope-probe.json").write_text('{"enabled": false}')
    log.write_text("Codex is already available at the requested version\n")
    with pytest.raises(ValueError, match="egress window closing"):
        scoring._evidence(bundle, trial, snapshot, inventory, expect_setup_egress=True)
    assert scoring._evidence(bundle, trial, snapshot, inventory, expect_setup_egress=False).skill_listed is True


@pytest.mark.asyncio
async def test_package_metrics_use_original_archive_document_and_files(monkeypatch: pytest.MonkeyPatch) -> None:
    scoring = _load_example(monkeypatch, "score")
    snapshot = SkillSnapshot(
        directory_name=SKILL,
        digest="c" * 64,
        document=DOCUMENT + "[Present](references/guide.md) and [missing](references/missing.md)\n",
        files=["SKILL.md", "references/guide.md"],
    )
    report = await scoring._package_metrics(snapshot)
    assert set(report) == {"skill_structure", "skill_reference_integrity", "skill_context_efficiency"}
    integrity = {output["name"]: output["value"] for output in report["skill_reference_integrity"]["outputs"]}
    assert integrity == {"checked": 2, "broken": 1}
