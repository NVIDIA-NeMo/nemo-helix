# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the Pi demo's inline metrics over a native Harbor trial."""

import hashlib
import importlib
import json
from pathlib import Path

import pytest
from nemo_evals.shared.metric_bundles.bundles import bundle_metric, unbundle_metric
from nemo_evals.shared.metric_bundles.inline import InlineMetricBundlePackager
from nhx_evals_sdk.agent_eval.runtimes.harbor.codex_evidence import CODEX_VERSION
from nhx_evals_sdk.agent_eval.trials import standard_evidence_descriptors
from nhx_evals_sdk.metrics.harbor_skill_metrics import HarborSkillMetrics, HarborSkillPackage
from nhx_evals_sdk.metrics.llm_judge import LLMJudgeMetric
from nhx_evals_sdk.metrics.protocol import CandidateOutput, DatasetRow, MetricInput
from nhx_evals_sdk.structured_output import StructuredOutputMode
from nhx_evals_sdk.values.evidence import CandidateEvidence
from nhx_evals_sdk.values.models import Model, ModelRef
from nhx_evals_sdk.values.protocol import validate_metric_result
from nhx_evals_sdk.values.skill_evidence import SkillContentBundle
from nhx_evals_sdk.values.skills import SkillSnapshot

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "harbor_evals_skill"
NAME = "nemo-evals-plugin"
DOCUMENT = f"---\nname: {NAME}\ndescription: Publish a taskset.\n---\n# Publish\n"
DIGEST = "a" * 64
REGISTERED = f"/home/harbor-agent/.agents/skills/{NAME}/SKILL.md"


@pytest.fixture
def package():
    return HarborSkillPackage(
        snapshot=SkillSnapshot(directory_name=NAME, digest=DIGEST, files=["SKILL.md"], document=DOCUMENT),
        contents=SkillContentBundle(artifact_digest=DIGEST, files={"SKILL.md": DOCUMENT}),
        inventory={"SKILL.md": hashlib.sha256(DOCUMENT.encode()).hexdigest()},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("installed", [True, False])
async def test_inline_metric_round_trip_scores_real_skill_read_and_preserves_unknowns(package, installed, tmp_path):
    metric = HarborSkillMetrics(packages=[package], authoring_directory="/workspace/pi-digit")
    bundle = bundle_metric(metric, InlineMetricBundlePackager())
    hydrated = unbundle_metric(bundle)
    agent, verifier = tmp_path / "agent", tmp_path / "verifier"
    (agent / "sessions").mkdir(parents=True)
    verifier.mkdir()
    expected = {NAME: package.inventory} if installed else {}
    (verifier / "skill-inventory.json").write_text(json.dumps({"staged": expected, "registered": expected}))
    body = f"\n### Available skills\n- {NAME}: Publish (file: {REGISTERED})\n" if installed else None
    capture = [
        ("session_meta", {"source": "exec", "cli_version": CODEX_VERSION, "id": "run", "session_id": "run"}),
        ("world_state", {"full": True, "state": {"skills": {"includeInstructions": True, "body": body}}}),
    ]
    if installed:
        capture.extend(
            [
                (
                    "response_item",
                    {
                        "type": "function_call",
                        "name": "exec_command",
                        "call_id": "read",
                        "arguments": json.dumps({"cmd": f"cat {REGISTERED}"}),
                    },
                ),
                (
                    "response_item",
                    {
                        "type": "function_call_output",
                        "call_id": "read",
                        "output": f"Process exited with code 0\nOutput:\n{DOCUMENT}",
                    },
                ),
            ]
        )
    (agent / "sessions" / "rollout-run.jsonl").write_text(
        "\n".join(json.dumps({"type": kind, "payload": payload}) for kind, payload in capture) + "\n"
    )
    candidate = CandidateOutput(
        evidence=CandidateEvidence(descriptors=standard_evidence_descriptors(logs_dir=agent, final_state_dir=verifier)),
    )
    row = DatasetRow(
        data={
            "trial": {
                "id": "trial-a",
                "status": "completed",
                "error": None,
                "metadata": {"harbor_trial_dir": str(tmp_path)},
            }
        }
    )
    scored = validate_metric_result(
        await hydrated.compute_scores(MetricInput(row=row, candidate=candidate)),
        hydrated.output_spec(),
    )

    outputs = {value.name: value.value for value in scored.outputs}
    assert outputs["agent_phase_success.agent_phase_success"] is True
    activation = f"skill.{NAME}.skill_activation.activated"
    if installed:
        assert outputs[activation] == 1
    else:
        assert activation not in outputs
    coverage = scored.diagnostics[-1].details
    assert coverage
    assert coverage["metrics"][f"package.{NAME}.skill_security_scan"] == "unmeasured"
    assert "skill_overactivation" in coverage["unconfigured"]
    assert coverage["metrics"]["tool_execution_outcome"] == "partial"


@pytest.mark.asyncio
async def test_native_judge_uses_harbor_verdicts_and_preserves_each_package_view(monkeypatch, package):
    monkeypatch.syspath_prepend(str(EXAMPLE))
    publisher = importlib.import_module("pi_metrics")
    metric, views = publisher._judge(ModelRef("default/judge"), [package])
    assert {f"{NAME}.documentation_rubric", "expected_behavior", "task_success", "answer_accuracy"} <= views.keys()
    metric = unbundle_metric(bundle_metric(metric, InlineMetricBundlePackager()))
    assert isinstance(metric, LLMJudgeMetric)
    metric.model = Model(name="fixture", url="http://judge.invalid/v1")

    async def no_structured_output(**kwargs):
        return StructuredOutputMode.UNSUPPORTED

    monkeypatch.setattr("nhx_evals_sdk.metrics.llm_judge.detect_structured_output_mode", no_structured_output)

    async def judge(model, request, max_retries, **options):
        evidence = json.loads(request["messages"][-1]["content"])
        assert evidence["documents"] == {NAME: DOCUMENT}
        assert evidence["verifier_rewards"] == {"script_correct": 0}
        assert evidence["candidate_answer"] == "Saved the taskset."
        verdicts = {
            score.name: "8" if score.name.startswith(NAME.replace("-", "_")) else "met" for score in metric.scores
        }
        return {"choices": [{"message": {"content": json.dumps(verdicts)}}]}

    metric.set_inference_fn(judge)
    result = await metric.compute_scores(
        MetricInput(
            row=DatasetRow(
                data={
                    "inputs": {"instruction": "Create a task."},
                    "trial": {"metadata": {"reward_details": {"script_correct": 0}}},
                }
            ),
            candidate=CandidateOutput(output_text="Saved the taskset."),
        )
    )
    assert all(output.value in ("met", "8", 1, 8) for output in result.outputs)
