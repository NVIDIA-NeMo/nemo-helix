# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the reusable agent-eval metrics and the TrialMeasurements contract."""

from __future__ import annotations

from pathlib import Path

import pytest
from nemo_evaluator.shared.metric_bundles.bundles import bundle_metric, unbundle_metric
from nemo_evaluator.shared.metric_bundles.inline import InlineMetricBundlePackager
from nemo_evaluator_sdk.agent_eval.metrics import (
    AgentPhaseSuccessMetric,
    EvidencePresenceMetric,
    SkillUsedMetric,
    ToolArgumentMatchesInputMetric,
    ToolCallCountMetric,
)
from nemo_evaluator_sdk.agent_eval.trials import standard_evidence_descriptors
from nemo_evaluator_sdk.metrics.protocol import CandidateOutput, DatasetRow, MetricInput
from nemo_evaluator_sdk.metrics.runner_rewards import GymRewardMetric, HarborRewardMetric
from nemo_evaluator_sdk.values.evidence import CandidateEvidence
from pydantic import ValidationError


class _BadFloat(float):
    def __float__(self) -> float:
        raise RuntimeError("must not be called")


@pytest.mark.asyncio
async def test_agent_phase_success_metric_reads_metadata_and_bundles_inline() -> None:
    """``type`` is a fixed discriminator now, which is what makes the metric inline-bundleable.

    It used to be overridable per subclass so callers could namespace it. That is incompatible with
    ``MetricsUnion``'s ``Field(discriminator="type")``, and the override bought less than the
    cloudpickle opt-in it cost: storing this metric on a task previously required shipping custom
    code.
    """
    metric = AgentPhaseSuccessMetric()
    assert metric.type == "agent_phase_success"
    ok = await metric.compute_scores(
        MetricInput(row=DatasetRow(data={}), candidate=CandidateOutput(metadata={"agent_ok": True}))
    )
    assert ok.outputs[0].value is True

    bundle = bundle_metric(metric, InlineMetricBundlePackager())
    assert bundle.payload.kind == "inline", "a built-in metric must not need the cloudpickle opt-in"
    assert type(unbundle_metric(bundle)) is AgentPhaseSuccessMetric


@pytest.mark.asyncio
async def test_evidence_presence_metric_scores_over_evidence(tmp_path: Path) -> None:
    final_state = tmp_path / "workspace"
    final_state.mkdir()
    (final_state / "result.txt").write_text("done", encoding="utf-8")
    evidence = CandidateEvidence(
        descriptors=standard_evidence_descriptors(logs_dir=tmp_path / "agent", final_state_dir=final_state)
    )

    metric = EvidencePresenceMetric()
    present = await metric.compute_scores(
        MetricInput(row=DatasetRow(data={}), candidate=CandidateOutput(evidence=evidence))
    )
    assert present.outputs[0].value is True

    # Empty workspace -> non-empty requirement fails; no evidence -> False.
    (final_state / "result.txt").unlink()
    empty = await metric.compute_scores(
        MetricInput(row=DatasetRow(data={}), candidate=CandidateOutput(evidence=evidence))
    )
    assert empty.outputs[0].value is False
    missing = await metric.compute_scores(MetricInput(row=DatasetRow(data={}), candidate=CandidateOutput()))
    assert missing.outputs[0].value is False


@pytest.mark.parametrize(
    ("factory", "expected_type"),
    [
        (lambda: GymRewardMetric(output_name="score"), "gym_reward"),
        (lambda: HarborRewardMetric(output_name="score"), "harbor_reward"),
        (AgentPhaseSuccessMetric, "agent_phase_success"),
        (lambda: EvidencePresenceMetric(evidence_name="workspace", require_non_empty=False), "evidence_presence"),
        (lambda: SkillUsedMetric(trace_evidence="atif"), "skill_used"),
        (lambda: ToolCallCountMetric(tool_name="search", expected_calls=2), "tool_call_count"),
        (
            lambda: ToolArgumentMatchesInputMetric(tool_name="search", argument="query", normalize="exact"),
            "tool_argument_matches_input",
        ),
    ],
)
def test_runner_and_agent_eval_metrics_are_built_in(factory, expected_type: str) -> None:
    """Each of these can be stored on a task without the cloudpickle opt-in.

    Non-default configuration is used deliberately: a metric that survives bundling only with its
    defaults would still lose the caller's settings on the way to a stored task.
    """
    metric = factory()
    assert metric.type == expected_type

    bundle = bundle_metric(metric, InlineMetricBundlePackager())
    assert bundle.payload.kind == "inline"

    restored = unbundle_metric(bundle)
    assert type(restored) is type(metric)
    assert restored == metric


def test_built_in_metrics_reject_a_caller_supplied_type() -> None:
    """The discriminator is fixed. Callers used to namespace it, which the union cannot express."""
    with pytest.raises(ValidationError):
        GymRewardMetric(type="my_namespaced_reward")  # ty: ignore[invalid-argument-type]


def test_harbor_reward_metric_declares_primary_first_and_sparse_secondaries() -> None:
    metric = HarborRewardMetric(
        output_name="score",
        reward_keys=("z_shape", "score", "format_ok", "format_ok"),
    )

    specs = metric.output_spec()

    assert [spec.name for spec in specs] == ["score", "format_ok", "z_shape"]
    assert [spec.required for spec in specs] == [True, False, False]


@pytest.mark.asyncio
async def test_harbor_reward_metric_uses_primary_fallback_and_omits_unusable_secondaries() -> None:
    metric = HarborRewardMetric(output_name="score", reward_keys=("format_ok", "score", "shape_ok"))

    result = await metric.compute_scores(
        MetricInput(
            row=DatasetRow(data={}),
            candidate=CandidateOutput(
                metadata={
                    "reward": float("nan"),
                    "reward_details": {"format_ok": 1.0, "shape_ok": True},
                    "reward_rejections": {"score": "non_finite", "shape_ok": "boolean"},
                    "reward_entry_rejections": ["invalid_key", "reserved_key"],
                }
            ),
        )
    )

    assert [(output.name, output.value) for output in result.outputs] == [("score", 0.0), ("format_ok", 1.0)]
    assert [diagnostic.details for diagnostic in result.diagnostics] == [
        {"output": "score", "reason": "non_finite"},
        {"output": "shape_ok", "reason": "boolean"},
        {"output": None, "reason": "invalid_key"},
        {"output": None, "reason": "reserved_key"},
    ]


@pytest.mark.asyncio
async def test_harbor_reward_metric_defensively_rejects_manual_bad_metadata() -> None:
    metric = HarborRewardMetric(output_name="score", reward_keys=("format_ok", "score"))

    result = await metric.compute_scores(
        MetricInput(
            row=DatasetRow(data={}),
            candidate=CandidateOutput(metadata={"reward": 10**1000, "reward_details": {"format_ok": float("inf")}}),
        )
    )

    assert [(output.name, output.value) for output in result.outputs] == [("score", 0.0)]
    assert [diagnostic.details for diagnostic in result.diagnostics] == [
        {"output": "score", "reason": "unusable"},
        {"output": "format_ok", "reason": "unusable"},
    ]


@pytest.mark.asyncio
async def test_harbor_reward_metric_rejects_hostile_numeric_subclasses_without_calling_them() -> None:
    metric = HarborRewardMetric(output_name="score", reward_keys=("score", "format_ok"))

    result = await metric.compute_scores(
        MetricInput(
            row=DatasetRow(data={}),
            candidate=CandidateOutput(
                metadata={"reward": _BadFloat(1.0), "reward_details": {"format_ok": _BadFloat(1.0)}}
            ),
        )
    )

    assert [(output.name, output.value) for output in result.outputs] == [("score", 0.0)]
    assert [diagnostic.details for diagnostic in result.diagnostics] == [
        {"output": "score", "reason": "unusable"},
        {"output": "format_ok", "reason": "unusable"},
    ]


@pytest.mark.parametrize("bad_name", ["score.pass@2", ""])
def test_harbor_reward_metric_rejects_unsafe_names_on_both_fields(bad_name: str) -> None:
    # The classification rules are the adapter's to test; this proves the validator is wired to both
    # name-bearing fields and fails at construction.
    with pytest.raises(ValidationError, match="reward_key"):
        HarborRewardMetric(output_name=bad_name)
    with pytest.raises(ValidationError, match="reward_key"):
        HarborRewardMetric(reward_keys=(bad_name,))
