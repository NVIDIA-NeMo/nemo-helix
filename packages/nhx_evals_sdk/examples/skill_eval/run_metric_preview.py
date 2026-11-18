# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the skill metric prototype on two synthetic revisions without any services."""

import asyncio
import hashlib
import json
from unittest.mock import AsyncMock, patch

from nhx_evals_sdk.agent_eval.evaluator import AgentEvaluator
from nhx_evals_sdk.agent_eval.tasks import AgentEvalTask
from nhx_evals_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus, AgentOutput, TrialMeasurements
from nhx_evals_sdk.metrics.protocol import CandidateOutput, DatasetRow, MetricInput
from nhx_evals_sdk.metrics.skills import (
    SkillActivationMetric,
    SkillContextEfficiencyMetric,
    SkillReferenceIntegrityMetric,
    SkillStructureMetric,
    TokenEfficiencyMetric,
    expected_behavior_metric,
)
from nhx_evals_sdk.structured_output import StructuredOutputMode
from nhx_evals_sdk.values.models import Model
from nhx_evals_sdk.values.skills import SkillSnapshot

_SKILL = "incident-summary"
_PROMPT = "Summarize this incident for the next shift: uploads timed out until the service restarted."
_ASSERTIONS = ("Reports the upload timeouts.", "Identifies the service restart as the action that restored service.")


def mock_snapshot(files: dict[str, str]) -> SkillSnapshot:
    """Adapt fixture files into the grader input, including a digest of their paths and contents."""
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return SkillSnapshot(directory_name=_SKILL, digest=digest, document=files.get("SKILL.md"), files=sorted(files))


async def _preview_revision(
    version: str,
    snapshot: SkillSnapshot,
    *,
    answer: str,
    action_verdict: str,
    activated: bool | None,
    prompt_tokens: int,
) -> None:
    print(f"\n{version} ({snapshot.digest[:12]})")
    artifact_input = MetricInput(
        row=DatasetRow(data={"reference": {"skill_snapshot": snapshot.model_dump(mode="json")}}),
        candidate=CandidateOutput(),
    )
    for metric in (SkillStructureMetric(), SkillReferenceIntegrityMetric(), SkillContextEfficiencyMetric()):
        measured = await metric.compute_scores(artifact_input)
        print(f"  {metric.type.value}: {', '.join(f'{output.name}={output.value}' for output in measured.outputs)}")

    judge, views = expected_behavior_metric(Model(name="fixture-judge", url="http://judge.invalid/v1"), _ASSERTIONS)
    judge.set_inference_fn(
        AsyncMock(
            return_value={
                "choices": [{"message": {"content": json.dumps({"assertion_1": "met", "assertion_2": action_verdict})}}]
            }
        )
    )
    task = AgentEvalTask(
        id="handoff",
        intent="Summarize a resolved incident for the next shift.",
        inputs={"instruction": _PROMPT},
        metrics=[judge, SkillActivationMetric(skill_name=_SKILL, should_activate=True), TokenEfficiencyMetric()],
        views=views,
    )
    trial = AgentEvalTrial(
        id=version,
        task_id=task.id,
        status=AgentEvalTrialStatus.COMPLETED,
        output=AgentOutput(output_text=answer),
        metadata={"skill_activation": {_SKILL: activated}},
        measurements=TrialMeasurements(prompt_tokens=prompt_tokens, completion_tokens=25, cache_read_tokens=100),
    )
    with patch(
        "nhx_evals_sdk.metrics.llm_judge.detect_structured_output_mode", return_value=StructuredOutputMode.UNSUPPORTED
    ):
        run = await AgentEvaluator().run(tasks=[task], trials=[trial])
    print("  trial scores and coverage:")
    for score in run.summary.scores.scores:
        if ".pass@" not in score.name and score.mean is not None:
            print(f"    {score.name}: {score.mean:.4g}")
    if run.summary.metric_coverage["skill_activation"]["activated"].missing:
        print("    skill_activation: unmeasured")


async def main() -> None:
    frontmatter = "---\nname: incident-summary\ndescription: Summarize incidents for shift handoffs.\n---\n"
    await _preview_revision(
        "v1",
        mock_snapshot({"SKILL.md": frontmatter + "Report symptoms. [Template](references/template.md)\n"}),
        answer="Uploads timed out.",
        action_verdict="unmet",
        activated=None,
        prompt_tokens=800,
    )
    await _preview_revision(
        "v2",
        mock_snapshot(
            {
                "SKILL.md": frontmatter
                + "Report symptoms and the recovery action. [Template](references/template.md)\n",
                "references/template.md": "Symptoms, recovery action.\n",
            }
        ),
        answer="Uploads timed out. Restarting the service restored uploads.",
        action_verdict="met",
        activated=True,
        prompt_tokens=650,
    )


if __name__ == "__main__":
    asyncio.run(main())
