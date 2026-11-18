# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from unittest.mock import AsyncMock, patch

import pytest
from nhx_evals_sdk import (
    SkillActivationMetric,
    SkillContextEfficiencyMetric,
    SkillReferenceIntegrityMetric,
    SkillSnapshot,
    SkillStructureMetric,
    TokenEfficiencyMetric,
    expected_behavior_metric,
)
from nhx_evals_sdk.agent_eval.evaluator import AgentEvaluator
from nhx_evals_sdk.agent_eval.tasks import AgentEvalTask
from nhx_evals_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus, AgentOutput, TrialMeasurements
from nhx_evals_sdk.metrics.protocol import CandidateOutput, DatasetRow, Metric, MetricInput, validate_metric_result
from nhx_evals_sdk.metrics.types import MetricsUnion
from nhx_evals_sdk.structured_output import StructuredOutputMode
from nhx_evals_sdk.values.models import Model
from pydantic import TypeAdapter, ValidationError


def mock_snapshot(files: dict[str, str]) -> SkillSnapshot:
    return SkillSnapshot(
        directory_name="incident-summary",
        digest=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        document=files.get("SKILL.md"),
        files=sorted(files),
    )


async def _score(metric: Metric, row: dict) -> dict:
    result = await metric.compute_scores(MetricInput(row=DatasetRow(data=row), candidate=CandidateOutput()))
    validate_metric_result(result, metric.output_spec())
    return {output.name: output.value for output in result.outputs}


@pytest.mark.asyncio
async def test_skill_artifact_checks_use_the_original_snapshot() -> None:
    doc = (
        "---\nname: wrong-name\ndescription: Summarize incidents.\n---\n"
        "[good][ref], [missing](missing.md), [escape](%2E%2E/secret.md), [external](https://example.org),\n"
        "and [normalized](refs/../ok%20file.md#summary).\n"
        "\n~~~md\n[example](ignore.md)\n~~~\n\n"
        "[ref]: ./ok%20file.md\n"
    )
    snapshot = mock_snapshot({"SKILL.md": doc, "ok file.md": "Summary"})
    row = {"reference": {"skill_snapshot": snapshot.model_dump()}}
    assert await _score(SkillStructureMetric(), row) == {"conforms": 0, "defect_count": 1}
    assert await _score(SkillReferenceIntegrityMetric(), row) == {"checked": 4, "broken": 2}
    size = await _score(SkillContextEfficiencyMetric(token_budget=20), row)
    assert size["estimated_tokens"] > 20
    assert size["within_budget"] == 0
    assert await _score(SkillStructureMetric(), {}) == {}
    missing_document = mock_snapshot({})
    missing_doc_row = {"reference": {"skill_snapshot": missing_document.model_dump()}}
    assert await _score(SkillStructureMetric(), missing_doc_row) == {"conforms": 0, "defect_count": 1}
    assert await _score(SkillReferenceIntegrityMetric(), missing_doc_row) == {}


@pytest.mark.asyncio
async def test_token_efficiency_distinguishes_missing_usage_from_measured_zero() -> None:
    metric = TokenEfficiencyMetric(half_life=200)
    for cached in (None, 201):
        measurement = TrialMeasurements(prompt_tokens=200, completion_tokens=50, cache_read_tokens=cached)
        assert await _score(metric, {"trial": {"measurements": measurement.model_dump()}}) == {}
    for prompt, completion, cached, tokens, score in [(0, 0, 0, 0, 1.0), (200, 50, 50, 200, 0.5)]:
        measurement = TrialMeasurements(prompt_tokens=prompt, completion_tokens=completion, cache_read_tokens=cached)
        row = {"trial": {"measurements": measurement.model_dump()}}
        assert await _score(metric, row) == {"efficiency_tokens": tokens, "score": score}


@pytest.mark.asyncio
async def test_activation_unknown_is_neither_a_miss_nor_a_success() -> None:
    positive = SkillActivationMetric(skill_name="incident-summary", should_activate=True)
    negative = SkillActivationMetric(skill_name="incident-summary", should_activate=False)
    unknown = {"trial": {"metadata": {"skill_activation": {"another-skill": True}}}}
    no_activation = {"trial": {"metadata": {"skill_activation": {"incident-summary": False}}}}
    assert await _score(positive, unknown) == await _score(negative, unknown) == {}
    assert await _score(positive, no_activation) == {"activated": 0, "as_expected": 0}
    assert await _score(negative, no_activation) == {"activated": 0, "as_expected": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reply", "expected_mean", "all_met"),
    [('{"assertion_1":"met","assertion_2":"unmet"}', 0.5, 0.0), ('{"assertion_1":"met"}', None, None)],
)
async def test_expected_behavior_views_require_every_judge_verdict(
    reply: str, expected_mean: float | None, all_met: float | None
) -> None:
    judge, views = expected_behavior_metric(
        Model(name="fixture", url="http://judge.invalid/v1"), ["Reports symptoms.", "Reports mitigation."]
    )
    inference = AsyncMock(return_value={"choices": [{"message": {"content": reply}}]})
    judge.set_inference_fn(inference)
    task = AgentEvalTask(
        id="handoff",
        intent="Summarize the incident",
        inputs={"instruction": "Summarize it"},
        metrics=[judge],
        views=views,
    )
    trial = AgentEvalTrial(
        id="v2",
        task_id=task.id,
        status=AgentEvalTrialStatus.COMPLETED,
        output=AgentOutput(output_text="Uploads timed out"),
    )
    with patch(
        "nhx_evals_sdk.metrics.llm_judge.detect_structured_output_mode", return_value=StructuredOutputMode.UNSUPPORTED
    ):
        result = await AgentEvaluator().run(tasks=[task], trials=[trial])
    aggregates = {score.name: score.mean for score in result.summary.scores.scores}
    assert aggregates["view.expected_behavior"] == expected_mean
    assert aggregates["view.all_assertions_met"] == all_met
    inference.assert_awaited_once()
    call = inference.await_args
    assert call is not None
    request = call.args[1]
    assert "Uploads timed out" in request["messages"][-1]["content"]
    assert "Reports mitigation." in request["messages"][-1]["content"]


@pytest.mark.parametrize(
    "metric",
    [
        SkillStructureMetric(),
        SkillReferenceIntegrityMetric(),
        SkillContextEfficiencyMetric(token_budget=1200),
        SkillActivationMetric(skill_name="incident-summary", should_activate=False),
        TokenEfficiencyMetric(half_life=400),
        expected_behavior_metric(Model(name="judge", url="http://judge.invalid/v1"), ["Reports recovery."])[0],
    ],
)
def test_skill_metrics_round_trip_as_builtin_configurations(metric) -> None:
    adapter = TypeAdapter(MetricsUnion)
    restored = adapter.validate_json(adapter.dump_json(metric))
    assert type(restored) is type(metric)
    assert restored.model_dump() == metric.model_dump()
    assert [output.name for output in restored.output_spec()] == [output.name for output in metric.output_spec()]


@pytest.mark.parametrize(
    ("metric_class", "config"),
    [
        (SkillContextEfficiencyMetric, {"token_budget": 0}),
        (SkillContextEfficiencyMetric, {"token_budget": True}),
        (SkillActivationMetric, {"skill_name": "", "should_activate": True}),
        (SkillActivationMetric, {"skill_name": "incident-summary", "should_activate": "false"}),
        (TokenEfficiencyMetric, {"half_life": -1}),
        (TokenEfficiencyMetric, {"half_life": float("inf")}),
    ],
)
def test_skill_metric_configuration_rejects_invalid_values(metric_class, config) -> None:
    with pytest.raises(ValidationError):
        metric_class(**config)
