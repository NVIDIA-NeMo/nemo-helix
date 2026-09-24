# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stored Harbor scoring crosses submission, compilation and SDK validation."""

import pytest
from nemo_evaluator.api.schemas import HarborTaskDefinition, MetricRef, TaskRef, TasksetRef
from nemo_evaluator.api.task_definitions.harbor import HarborArchiveSource, HarborTaskHash, ResolvedHarborTaskDefinition
from nemo_evaluator.api.task_definitions.provenance import TaskProvenance
from nemo_evaluator.entities import TaskEntity, TaskRevisionEntity, TasksetEntity, TasksetRevisionEntity
from nemo_evaluator.jobs.agent_compiler import _secret_refs
from nemo_evaluator.jobs.agent_evaluate import AgentEvalJob
from nemo_evaluator.jobs.agent_spec import AgentEvalInputSpec, AgentEvalSpec, HarborRunnerTarget, ResolvedTask
from nemo_evaluator.jobs.harbor_scoring import harbor_scoring_task
from nemo_evaluator.jobs.kinds.registry import KIND_ADAPTERS
from nemo_evaluator.jobs.kinds.types import SubmitContext
from nemo_evaluator.jobs.metric_resolution import to_inline
from nemo_evaluator.revisions import publish_revision
from nemo_evaluator.shared.metric_bundles.bundles import bundle_metric
from nemo_evaluator.shared.metric_bundles.cloudpickle import CloudpickleMetricBundlePackager
from nemo_evaluator.task_refs import snapshot_task, validate_execution_support, validate_scoring
from nemo_evaluator_sdk.agent_eval.tasks import SemanticReducer, SemanticView, ViewSignal
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus
from nemo_evaluator_sdk.metrics.exact_match import ExactMatchMetric
from nemo_evaluator_sdk.metrics.llm_judge import LLMJudgeMetric
from nemo_evaluator_sdk.metrics.runner_rewards import HarborRewardMetric
from nemo_evaluator_sdk.metrics.utils import metric_type_name
from nemo_evaluator_sdk.values import Model, ModelRef, SecretRef
from nemo_evaluator_sdk.values.scores import JSONScoreParser, RangeScore
from nemo_helix_plugin.sdk import AsyncNeMoHelix


def _resolved_task(*, metrics=(), views=None):
    return ResolvedTask(
        id="task",
        spec=ResolvedHarborTaskDefinition(
            provenance=TaskProvenance(entity_name="default/task", revision_digest="a" * 64),
            kind="harbor",
            native_task_id="task",
            source=HarborArchiveSource(fileset_ref="default/files#archive", files_hash="b" * 64),
            harbor_hash=HarborTaskHash(digest="c" * 64, harbor_version="test"),
            metrics=list(metrics),
            views=views or {},
        ),
    )


def _metric(metric=None):
    return to_inline(
        bundle_metric(
            metric if metric is not None else ExactMatchMetric(reference="yes", candidate="yes"),
            CloudpickleMetricBundlePackager(),
        )
    )


def _views(output="reward"):
    return {
        "quality": SemanticView(
            reducer=SemanticReducer.MEAN,
            signals=[
                ViewSignal(metric="harbor_reward", output=output),
                ViewSignal(metric="exact-match", output="exact-match"),
            ],
        )
    }


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("offline", [False, True])
@pytest.mark.parametrize("task_workspace", ["default", "other"])
async def test_submission_freezes_scoring_and_compiler_collects_secrets(
    entity_store, monkeypatch, direct, offline, task_workspace
):
    """Verify submission resolves metrics in the task workspace and snapshots scoring data while retaining secret
    references.
    """
    task = TaskEntity(
        name="custom",
        workspace=task_workspace,
        spec=HarborTaskDefinition(
            kind="harbor",
            native_task_id="task",
            source=HarborArchiveSource(fileset_ref="default/files#task/archive", files_hash="a" * 64),
            harbor_hash=HarborTaskHash(digest="b" * 64, harbor_version="0.20.0"),
            metrics=[MetricRef("judge")],
            views=_views("grade"),
        ),
    )
    await entity_store.create(task)
    revision, _, _ = await publish_revision(entity_store, entity_store, task, TaskRevisionEntity)
    ref = TaskRef(f"{task_workspace}/custom#{revision.content_hash}")
    suite = TasksetEntity(name="suite", workspace="default", tasks=[ref])
    await entity_store.create(suite)
    await publish_revision(entity_store, entity_store, suite, TasksetRevisionEntity)
    metric = _metric()
    metric.secrets = {"JUDGE_KEY": SecretRef("default/judge-key")}
    calls = []

    async def resolve(metrics, **kwargs):
        calls.append((metrics, kwargs["workspace"]))
        return [metric.model_copy(deep=True)]

    monkeypatch.setattr("nemo_evaluator.jobs.kinds.harbor.resolve_metrics_to_inline", resolve)
    spec = await AgentEvalJob.to_spec(
        AgentEvalInputSpec(
            tasks=[ref] if direct else TasksetRef("default/suite"),
            target=None if offline else HarborRunnerTarget(reward_key="grade"),
            trials=[
                AgentEvalTrial(
                    id="trial",
                    task_id="task",
                    status=AgentEvalTrialStatus.PARTIAL,
                    metadata={"harbor_primary_reward_key": "grade"},
                )
            ]
            if offline
            else None,
        ),
        workspace="default",
        entity_client=entity_store,
        async_sdk=None,
        is_local=False,
    )
    assert isinstance(spec, AgentEvalSpec)
    assert calls == [([MetricRef("judge")], task_workspace)]
    assert all(task.spec.kind == "harbor" for task in spec.tasks)
    assert spec.tasks[0].spec.provenance is not None
    assert f"{spec.tasks[0].spec.provenance.entity_name}#{spec.tasks[0].spec.provenance.revision_digest}" == ref.root
    assert ("JUDGE_KEY", "default/judge-key") in list(_secret_refs(spec))
    snapshot = spec.model_dump_json()
    metric.secrets.clear()
    assert isinstance(task.spec, HarborTaskDefinition)
    task.spec.config["changed"] = True
    task.spec.instruction = "Changed after submission"
    assert spec.model_dump_json() == snapshot
    restored = AgentEvalSpec.model_validate_json(snapshot)
    assert all(task.spec.kind == "harbor" for task in restored.tasks)
    assert restored.tasks[0].spec.views == _views("grade")
    assert "harbor_scoring" not in restored.model_dump()


@pytest.mark.parametrize(
    "metrics,views,error",
    [
        ([_metric(HarborRewardMetric())], {}, "duplicate task metric types"),
        ([_metric(), _metric()], {}, "duplicate task metric types"),
        ([_metric()], _views("undeclared-secondary"), "unknown output"),
    ],
)
def test_invalid_scoring_rejected_before_execution(metrics, views, error):
    with pytest.raises(ValueError, match=error):
        spec = AgentEvalSpec(
            tasks=[_resolved_task(metrics=metrics, views=views)],
            target=HarborRunnerTarget(),
        )
        validate_scoring(spec.tasks, target=spec.target, trials=spec.trials, adapters=KIND_ADAPTERS)


def test_scoring_projects_snapshot_and_appends_metrics() -> None:
    """Verify a definition produces the expected runtime scoring fields and defaults."""
    scoring = _resolved_task(metrics=[_metric()], views=_views("grade")).spec
    assert isinstance(scoring, ResolvedHarborTaskDefinition)
    scoring.instruction = "Do it"
    result = harbor_scoring_task(scoring, reward_key="grade")
    assert result.id == scoring.native_task_id
    assert result.intent == scoring.native_task_id
    assert result.inputs == {"instruction": scoring.instruction}
    assert [metric_type_name(metric) for metric in result.metrics] == ["harbor_reward", "exact-match"]
    assert result.metrics[0].output_spec()[0].name == "grade"
    assert result.views == _views("grade")
    assert result.reference == {}
    assert result.metadata == {}


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
async def test_offline_selection_rejects_mixed_kinds_and_duplicate_native_ids(entity_store, direct, mixed):
    """Check direct and taskset selections reject mixed task kinds and native ID collisions before offline scoring."""
    from nemo_evaluator.api.schemas import EvaluatorTaskDefinition
    from nemo_evaluator.task_refs import load_tasks

    refs = []
    for index in range(2):
        definition = HarborTaskDefinition(
            kind="harbor",
            native_task_id="TASK" if index else "task",
            source=HarborArchiveSource(fileset_ref="default/files#archive", files_hash="a" * 64),
            harbor_hash=HarborTaskHash(digest="b" * 64, harbor_version="test"),
        )
        task = TaskEntity(
            name=f"stored-{index}",
            workspace="default",
            spec=EvaluatorTaskDefinition(kind="evaluator", intent="Do it") if mixed and index else definition,
        )
        await entity_store.create(task)
        revision, _, _ = await publish_revision(entity_store, entity_store, task, TaskRevisionEntity)
        refs.append(TaskRef(f"default/{task.name}#{revision.content_hash}"))
    suite = TasksetEntity(name="suite", workspace="default", tasks=refs)
    await entity_store.create(suite)
    await publish_revision(entity_store, entity_store, suite, TasksetRevisionEntity)
    with pytest.raises(ValueError, match="cannot mix" if mixed else "task ids must be unique"):
        loaded = await load_tasks(
            refs if direct else TasksetRef("default/suite"),
            SubmitContext(workspace="default", entity_client=entity_store, async_sdk=None, adapters=KIND_ADAPTERS),
        )
        ctx = SubmitContext(workspace="default", entity_client=entity_store, async_sdk=None, adapters=KIND_ADAPTERS)
        snapshots = [await snapshot_task(item, ctx) for item in loaded]
        validate_execution_support(snapshots, target=None, adapters=KIND_ADAPTERS)


@pytest.mark.parametrize("offline", [False, True])
async def test_harbor_judge_model_is_resolved_before_canonical_job(entity_store, monkeypatch, offline):
    """Require judge references to become concrete models during submission for both live runs and offline scoring."""
    from nemo_evaluator.jobs.metric_resolution import HelixMetricModelResolver, to_runtime_bundle
    from nemo_evaluator.shared.metric_bundles.bundles import unbundle_metric

    judge = _metric(
        LLMJudgeMetric(
            model=ModelRef("default/judge"),
            scores=[RangeScore(name="quality", minimum=0, maximum=1, parser=JSONScoreParser(json_path="quality"))],
        )
    )
    with pytest.raises(ValueError, match="models must be resolved"):
        spec = AgentEvalSpec(
            tasks=[_resolved_task(metrics=[judge])],
            target=HarborRunnerTarget(),
        )
        validate_scoring(spec.tasks, target=spec.target, trials=spec.trials, adapters=KIND_ADAPTERS)
    task = TaskEntity(
        name="task",
        workspace="default",
        spec=HarborTaskDefinition(
            kind="harbor",
            native_task_id="task",
            source=HarborArchiveSource(fileset_ref="default/files#archive", files_hash="a" * 64),
            harbor_hash=HarborTaskHash(digest="b" * 64, harbor_version="0.20.0"),
            metrics=[judge],
        ),
    )
    await entity_store.create(task)
    revision, _, _ = await publish_revision(entity_store, entity_store, task, TaskRevisionEntity)
    calls = []

    async def resolve(self, model_ref):
        calls.append(model_ref.root)
        return Model(name="judge", url="https://example.test/v1/chat/completions")

    monkeypatch.setattr(HelixMetricModelResolver, "resolve_model", resolve)
    async with AsyncNeMoHelix(base_url="http://platform.test", workspace="default") as async_sdk:
        spec = await AgentEvalJob.to_spec(
            AgentEvalInputSpec(
                tasks=[TaskRef(f"default/task#{revision.content_hash}")],
                target=None if offline else HarborRunnerTarget(),
                trials=[
                    AgentEvalTrial(
                        id="trial",
                        task_id="task",
                        status=AgentEvalTrialStatus.PARTIAL,
                        metadata={"harbor_primary_reward_key": "reward"},
                    )
                ]
                if offline
                else None,
            ),
            workspace="default",
            entity_client=entity_store,
            async_sdk=async_sdk,
            is_local=False,
        )
    assert calls == ["default/judge"]
    assert isinstance(spec, AgentEvalSpec) and all(task.spec.kind == "harbor" for task in spec.tasks)
    metric = unbundle_metric(to_runtime_bundle(spec.tasks[0].spec.metrics[0]))
    assert isinstance(metric, LLMJudgeMetric) and isinstance(metric.model, Model)
    assert metric.model.url == "https://example.test/v1/chat/completions"
