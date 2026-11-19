# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
from pathlib import Path

import pytest
from nhx_evals_sdk.agent_eval.trials import AgentEvalTrial, AgentOutput
from nhx_evals_sdk.values.skill_evidence import SkillContentBundle, SkillScanReport
from nhx_evals_sdk.values.skills import SkillSnapshot

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/harbor_evals_skill"
SKILL = "supercool-agent-eval"
DOCUMENT = "---\nname: supercool-agent-eval\ndescription: Add the run alias.\n---\n# Supercool\n"


def _call(call_id: str, name: str, **arguments: object) -> dict:
    return {
        "type": "response_item",
        "payload": {"type": "function_call", "call_id": call_id, "name": name, "arguments": json.dumps(arguments)},
    }


def _output(call_id: str, text: str) -> dict:
    return {"type": "response_item", "payload": {"type": "function_call_output", "call_id": call_id, "output": text}}


@pytest.mark.asyncio
async def test_replay_measures_real_skill_read_and_correlated_submission(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(EXAMPLE))
    demo = importlib.import_module("supercool_metrics")
    name = "nemo-evals.agent-eva-fixed"
    job = demo.InnerJob("server-job-id", name, demo.EXPECTED_ALIAS, 1.0)
    receipt = "Process exited with code 0\nOutput:\n" + json.dumps({"id": job.id, "name": job.name})
    events = [
        _call("read", "exec_command", cmd=f"cat ~/.agents/skills/{SKILL}/SKILL.md"),
        _output("read", "Process exited with code 0\nOutput:\n" + DOCUMENT),
        _call("submit", "exec_command", cmd="nemo evals agent-evaluate --spec-file /tmp/spec.json"),
        _output("submit", "Process running with session ID 91\nOutput:\n"),
        _call("other", "exec_command", cmd="echo nemo evals agent-evaluate --spec-file /tmp/spec.json"),
        _output("other", receipt),
        _call("poll", "write_stdin", session_id=91, chars=""),
        _output("poll", receipt),
    ]
    captured = demo.trial_trace(events, "trial", DOCUMENT, activated=True, job=job)
    assert [call.component for call in captured.calls] == [f"Skill:{SKILL}", demo.SUBMISSION, None]
    assert captured.calls[1].arguments == {"run_alias": demo.EXPECTED_ALIAS}
    assert captured.complete is False
    assert "cmd" not in captured.model_dump_json()

    trial = AgentEvalTrial(id="trial", task_id="nemo-evals-plugin-002", status="completed", output=AgentOutput())
    result = await demo.score_trial(trial, captured, discovered=True, job=job, evidence=demo.MetricEvidence())
    values = {
        (score["metric_type"], value["name"]): value["value"]
        for score in result["scores"]
        for value in score["outputs"]
    }
    assert values["workflow_compliance", "satisfied"] == 1
    assert values["tool_argument_compliance", "passed"] == 1
    assert ("workflow_compliance", "score") not in values
    assert ("agent_safety", "clean") not in values
    assert {"skill_overactivation", "expected_script_invocation", "skill_outcome"} <= result["unmeasured"].keys()


def test_correlates_agent_eval_in_compound_shell_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(EXAMPLE))
    demo = importlib.import_module("supercool_metrics")
    job = demo.InnerJob("server-job-id", "nemo-evals.agent-eva-fixed", demo.EXPECTED_ALIAS, 1.0)
    shell = "/app/.venv/bin/nemo evals agent-evaluate --workspace default --spec-file /tmp/spec.json"
    prefix = 'Process exited with code 0\nOutput:\n{"labels": {"run_alias": "agent-smoke"}}\nSubmitting evaluation\n'
    receipt = json.dumps({"id": job.id, "name": job.name}, indent=2)
    events = [
        _call("compound", "exec_command", cmd=f"python -c 'prepare()' && cat /tmp/spec.json && {shell}"),
        _output("compound", prefix + receipt),
        _call("quoted", "exec_command", cmd=f"echo '&&' {shell}"),
        _output("quoted", prefix + receipt),
        _call("different-job", "exec_command", cmd=shell),
        _output("different-job", prefix + json.dumps({"id": "other-id", "name": job.name})),
        _call("nested", "exec_command", cmd=shell),
        _output("nested", prefix + json.dumps({"data": {"id": job.id, "name": job.name}}, indent=2)),
    ]
    trace = demo.trial_trace(events, "trial", DOCUMENT, activated=None, job=job)
    assert [call.component for call in trace.calls] == [demo.SUBMISSION, None, None, None]


@pytest.mark.asyncio
async def test_archive_metrics_keep_missing_scanner_evidence_unmeasured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(EXAMPLE))
    demo = importlib.import_module("supercool_metrics")
    snapshot = SkillSnapshot(directory_name=SKILL, digest="a" * 64, document=DOCUMENT, files=["SKILL.md"])
    contents = SkillContentBundle(artifact_digest=snapshot.digest, files={"SKILL.md": DOCUMENT})
    report = await demo.score_package(snapshot, contents, demo.MetricEvidence())
    assert {key: value["outputs"] for key, value in report.items() if key.startswith("skill_security")} == {
        "skill_security_scan": []
    }
    assert report["skill_bundled_test"]["diagnostics"][0]["details"]["reason"] == "missing_test_report"
    assert {value["name"]: value["value"] for value in report["skill_script_quality"]["outputs"]}[
        "scripts_checked"
    ] == 0

    scan = SkillScanReport(
        subject_id=snapshot.digest,
        scanner="fixture",
        scanner_version="1",
        status="completed",
        files_scanned=1,
        files_expected=1,
    )
    scored = await demo.score_package(snapshot, contents, demo.MetricEvidence(skill_scans={"security": scan}))
    assert {value["name"]: value["value"] for value in scored["skill_security_scan"]["outputs"]}["clean"] == 1.0
    mismatched = scan.model_copy(update={"subject_id": "b" * 64})
    with pytest.raises(ValueError, match="another artifact revision"):
        await demo.score_package(snapshot, contents, demo.MetricEvidence(skill_scans={"security": mismatched}))


def test_verified_job_supports_a_missing_alias(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.syspath_prepend(str(EXAMPLE))
    demo = importlib.import_module("supercool_metrics")
    trial = AgentEvalTrial(
        id="trial",
        task_id="nemo-evals-plugin-002",
        status="completed",
        output=AgentOutput(),
        metadata={"reward_details": {"inner_eval_passed": 1, "suffix_applied": 0}},
    )
    artifact = tmp_path / "job-snapshot.json"
    artifact.write_text(
        json.dumps(
            {
                "outcome": "collected",
                "job": {
                    "id": "job-id",
                    "name": "agent-eva",
                    "run_alias": None,
                    "status": "completed",
                    "summary": {"scores": {"scores": [{"name": "exact-match.exact-match", "mean": 1.0}]}},
                },
            }
        )
    )
    assert demo.read_inner_job(artifact, trial) == demo.InnerJob("job-id", "agent-eva", None, 1.0)
