# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from nemo_evaluator.api.schemas import EvaluatorTaskDefinition, TaskInputs
from nemo_evaluator.api.task_definitions.evaluator import ResolvedEvaluatorTaskDefinition
from nemo_evaluator.api.task_definitions.provenance import TaskProvenance
from nemo_evaluator.entities import TaskEntity, TaskRevisionEntity
from nemo_evaluator.jobs.agent_spec import (
    AgentEvalTaskInput,
    AgentTarget,
    FabricRunnerTarget,
    GymRunnerTarget,
    HarborRunnerTarget,
    ModelTarget,
    ResolvedTask,
    Target,
)
from nemo_evaluator.jobs.kinds.evaluator import EvaluatorTaskAdapter
from nemo_evaluator.jobs.kinds.registry import KIND_ADAPTERS, get_adapter
from nemo_evaluator.jobs.kinds.types import LoadedTask, PrepareContext, SubmitContext
from nemo_evaluator.jobs.metric_resolution import to_inline
from nemo_evaluator.shared.metric_bundles.bundles import bundle_metric
from nemo_evaluator.shared.metric_bundles.cloudpickle import CloudpickleMetricBundlePackager
from nemo_evaluator_sdk.agent_eval.tasks import SemanticView
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus
from nemo_evaluator_sdk.metrics.exact_match import ExactMatchMetric
from nemo_evaluator_sdk.metrics.llm_judge import LLMJudgeMetric
from nemo_evaluator_sdk.values import GenericAgent, Model, ModelRef
from nemo_evaluator_sdk.values.scores import JSONScoreParser, RangeScore
from nemo_helix_plugin.sdk import AsyncNeMoHelix
from pydantic import BaseModel, field_serializer


@pytest.mark.parametrize("stored", [False, True])
@pytest.mark.parametrize(
    "target",
    [
        None,
        ModelTarget(model=Model(url="http://model.test", name="test")),
        AgentTarget(agent=GenericAgent(url="http://agent.test", name="test", body={}, response_path="$.answer")),
        FabricRunnerTarget(config={}),
        GymRunnerTarget(agent="simple_agent", agent_config="config.yaml", resources_server="mcqa"),
        HarborRunnerTarget(),
    ],
)
def test_evaluator_target_support_matrix(target: Target | None, stored: bool) -> None:
    task = ResolvedTask(
        id="task",
        spec=ResolvedEvaluatorTaskDefinition(
            kind="evaluator",
            intent="Answer",
            provenance=TaskProvenance(entity_name="default/task", revision_digest="a" * 64) if stored else None,
        ),
    )
    assert EvaluatorTaskAdapter().accepts_target(target, [task]) is not (
        stored and isinstance(target, HarborRunnerTarget)
    )


def test_evaluator_rejects_unrecognized_target() -> None:
    class FutureTarget:
        kind = "future"

    task = ResolvedTask(id="task", spec=ResolvedEvaluatorTaskDefinition(kind="evaluator", intent="Answer"))
    assert not EvaluatorTaskAdapter().accepts_target(
        FutureTarget(),  # ty: ignore[invalid-argument-type] -- future adapter input must fail closed
        [task],
    )


async def test_snapshot_serializes_nested_reference_and_isolates_mutation() -> None:
    """Verify nested reference serialization and independence of the returned snapshot."""

    class Reference(BaseModel):
        answers: list[str]

        @field_serializer("answers")
        def serialize_answers(self, answers: list[str]) -> dict[str, list[str]]:
            """Return answers wrapped in the reference's persisted representation."""
            return {"values": answers}

    reference = Reference(answers=["yes"])
    task = AgentEvalTaskInput(id="task", intent="Answer", reference={"nested": reference})
    async with AsyncNeMoHelix(base_url="http://unused.test") as sdk:
        snapshot = await EvaluatorTaskAdapter().resolve(
            LoadedTask(inline=task), SubmitContext("default", None, sdk, KIND_ADAPTERS)
        )
    reference.answers.append("changed")
    task.reference["extra"] = "changed"
    assert snapshot.reference == {"nested": {"answers": {"values": ["yes"]}}}


async def test_stored_adapter_returns_definition_with_selected_revision_origin():
    task = TaskEntity(
        name="stored",
        workspace="other",
        spec=EvaluatorTaskDefinition(kind="evaluator", intent="Answer", metrics=[metric_bundle()]),
    )
    revision = TaskRevisionEntity(
        name="rev.1",
        revision=1,
        workspace="other",
        spec=task.spec,
        content_hash="a" * 64,
    )
    async with AsyncNeMoHelix(base_url="http://unused.test") as sdk:
        definition = await EvaluatorTaskAdapter().resolve(
            LoadedTask(stored=(task, revision)), SubmitContext("default", None, sdk, KIND_ADAPTERS)
        )
    assert definition.provenance is not None
    assert definition.provenance.entity_name == "other/stored"
    assert definition.provenance.revision_digest == "a" * 64
    assert len(definition.metrics) == 1
    revision.content_hash = "b" * 64
    assert definition.provenance.revision_digest == "a" * 64


def metric_bundle(metric=None):
    return to_inline(
        bundle_metric(metric or ExactMatchMetric(reference="yes", candidate="yes"), CloudpickleMetricBundlePackager())
    )


async def test_inline_snapshot_is_independent_and_evaluator_does_not_read_harbor_rewards(tmp_path):
    """Check snapshot isolation, target compatibility, and runtime conversion without requiring Harbor rewards."""
    adapter = EvaluatorTaskAdapter()
    original = AgentEvalTaskInput(
        id="task", intent="Do it", inputs=TaskInputs(instruction="yes"), metrics=[metric_bundle()]
    )
    item = LoadedTask(inline=original)
    async with AsyncNeMoHelix(base_url="http://unused.test") as sdk:
        definition = await adapter.resolve(item, SubmitContext("default", None, sdk, KIND_ADAPTERS))
    task = ResolvedTask(id=adapter.runtime_id(item), spec=definition)
    original.inputs.instruction = "changed"
    original.metrics.clear()
    assert task.spec.kind == "evaluator"
    assert task.spec.inputs.instruction == "yes"
    assert len(task.spec.metrics) == 1
    for trials in (None, [AgentEvalTrial(id="trial", status=AgentEvalTrialStatus.PARTIAL, task_id="task")]):
        adapter.validate_scoring([task], target=None, trials=trials)
    assert adapter.accepts_target(HarborRunnerTarget(), [task])
    stored_payload = task.model_dump()
    stored_payload["spec"]["provenance"] = {"entity_name": "default/task", "revision_digest": "a" * 64}
    stored = ResolvedTask.model_validate(stored_payload)
    assert not adapter.accepts_target(HarborRunnerTarget(), [stored])
    runtime = adapter.prepare([task], PrepareContext(Path(tmp_path), None, None, None, [], KIND_ADAPTERS))
    assert runtime[0].id == "task"
    assert isinstance(runtime[0].metrics[0], ExactMatchMetric)


@pytest.mark.parametrize(
    ("problem", "message"), [("duplicate", "duplicate"), ("view", "unknown"), ("model", "must be resolved")]
)
async def test_evaluator_scoring_validation(problem, message):
    """Reject duplicate metrics, invalid view signals, and unresolved judge models during scoring validation."""
    metrics = [metric_bundle()]
    views = {}
    if problem == "duplicate":
        metrics *= 2
    elif problem == "view":
        views = {
            "quality": SemanticView.model_validate(
                {"reducer": "mean", "signals": [{"metric": "missing", "output": "unknown"}]}
            )
        }
    else:
        metrics = [
            metric_bundle(
                LLMJudgeMetric(
                    model=ModelRef("default/judge"),
                    scores=[
                        RangeScore(name="quality", minimum=0, maximum=1, parser=JSONScoreParser(json_path="quality"))
                    ],
                )
            )
        ]
    from nemo_evaluator.api.task_definitions.evaluator import ResolvedEvaluatorTaskDefinition

    task = ResolvedTask(
        id="task", spec=ResolvedEvaluatorTaskDefinition(kind="evaluator", intent="Do it", metrics=metrics, views=views)
    )
    with pytest.raises(ValueError, match=message):
        EvaluatorTaskAdapter().validate_scoring([task], target=None, trials=[])


def test_loaded_task_is_exactly_one_origin():
    task = TaskEntity(name="task", workspace="default", spec=EvaluatorTaskDefinition(kind="evaluator", intent="Do it"))
    revision = TaskRevisionEntity(name="rev", revision=1, workspace="default", spec=task.spec, content_hash="a" * 64)
    inline = AgentEvalTaskInput(id="task", intent="Do it")
    with pytest.raises(ValueError, match="exactly one"):
        LoadedTask()
    with pytest.raises(ValueError, match="exactly one"):
        LoadedTask(inline=inline, stored=(task, revision))
    assert LoadedTask(inline=inline).kind == "evaluator"
    assert LoadedTask(stored=(task, revision)).kind == "evaluator"
    assert EvaluatorTaskAdapter().runtime_id(LoadedTask(stored=(task, revision))) == "task"


def test_unknown_kind_reports_available_kinds():
    with pytest.raises(ValueError, match="Unsupported task kind 'unknown'.*evaluator.*harbor"):
        get_adapter("unknown")
