# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stored Harbor scoring crosses submission, compilation and SDK validation."""

import pytest
from nemo_evaluator.api.schemas import HarborTaskDefinition, MetricRef, TaskRef, TasksetRef
from nemo_evaluator.api.task_definitions.harbor import HarborArchiveSource, HarborTaskHash
from nemo_evaluator.entities import TaskEntity, TaskRevisionEntity, TasksetEntity, TasksetRevisionEntity
from nemo_evaluator.harbor.tasks import HarborTaskScoring, PinnedHarborTaskList, PinnedHarborTaskset
from nemo_evaluator.jobs.agent_compiler import _secret_refs
from nemo_evaluator.jobs.agent_evaluate import AgentEvalJob
from nemo_evaluator.jobs.agent_spec import AgentEvalInputSpec, AgentEvalSpec, HarborRunnerTarget
from nemo_evaluator.jobs.harbor_scoring import apply_harbor_scoring
from nemo_evaluator.jobs.metric_resolution import to_inline
from nemo_evaluator.revisions import publish_revision
from nemo_evaluator.shared.metric_bundles.bundles import bundle_metric
from nemo_evaluator.shared.metric_bundles.cloudpickle import CloudpickleMetricBundlePackager
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalTask, SemanticView, ViewSignal
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial
from nemo_evaluator_sdk.metrics.exact_match import ExactMatchMetric
from nemo_evaluator_sdk.metrics.llm_judge import LLMJudgeMetric
from nemo_evaluator_sdk.metrics.runner_rewards import HarborRewardMetric
from nemo_evaluator_sdk.metrics.utils import metric_type_name
from nemo_evaluator_sdk.values import Model, ModelRef, SecretRef
from nemo_evaluator_sdk.values.scores import JSONScoreParser, RangeScore
from nemo_platform_plugin.sdk import AsyncNeMoPlatform


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
            reducer="mean",
            signals=[
                ViewSignal(metric="harbor_reward", output=output),
                ViewSignal(metric="exact-match", output="exact-match"),
            ],
        )
    }


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("offline", [False, True])
async def test_submission_freezes_scoring_and_compiler_collects_secrets(entity_store, monkeypatch, direct, offline):
    task = TaskEntity(
        name="custom",
        workspace="default",
        spec=HarborTaskDefinition(
            kind="harbor",
            native_task_id="task",
            source=HarborArchiveSource(fileset_ref="default/files#task/archive", files_hash="a" * 64),
            harbor_hash=HarborTaskHash(digest="b" * 64, harbor_version="0.20.0"),
            metrics=[MetricRef("default/judge")],
            views=_views("grade"),
        ),
    )
    await entity_store.create(task)
    revision, _, _ = await publish_revision(entity_store, entity_store, task, TaskRevisionEntity)
    ref = TaskRef(f"default/custom#{revision.content_hash}")
    suite = TasksetEntity(name="suite", workspace="default", tasks=[ref])
    await entity_store.create(suite)
    await publish_revision(entity_store, entity_store, suite, TasksetRevisionEntity)
    metric = _metric()
    metric.secrets = {"JUDGE_KEY": SecretRef("default/judge-key")}
    calls = []

    async def resolve(metrics, **kwargs):
        calls.append(metrics)
        return [metric.model_copy(deep=True)]

    monkeypatch.setattr("nemo_evaluator.jobs.agent_evaluate.resolve_metrics_to_inline", resolve)
    spec = await AgentEvalJob.to_spec(
        AgentEvalInputSpec(
            tasks=[ref] if direct else TasksetRef("default/suite"),
            target=None if offline else HarborRunnerTarget(reward_key="grade"),
            trials=[
                AgentEvalTrial(
                    id="trial", task_id="task", status="partial", metadata={"harbor_primary_reward_key": "grade"}
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
    assert calls == [[MetricRef("default/judge")]]
    assert isinstance(spec.tasks, (PinnedHarborTaskList, PinnedHarborTaskset))
    assert spec.tasks.scoring[0].task_ref == ref
    assert ("JUDGE_KEY", "default/judge-key") in list(_secret_refs(spec))
    snapshot = spec.model_dump_json()
    metric.secrets.clear()
    assert spec.model_dump_json() == snapshot
    restored = AgentEvalSpec.model_validate_json(snapshot)
    assert isinstance(restored.tasks, (PinnedHarborTaskList, PinnedHarborTaskset))
    assert restored.tasks.scoring[0].views == _views("grade")
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
    ref = TaskRef(f"default/task#{'a' * 64}")
    with pytest.raises(ValueError, match=error):
        AgentEvalSpec(
            tasks=PinnedHarborTaskList(
                task_refs=[ref], scoring=[HarborTaskScoring(task_ref=ref, metrics=metrics, views=views)]
            ),
            target=HarborRunnerTarget(),
        )


def test_scoring_cannot_be_associated_with_another_revision():
    with pytest.raises(ValueError, match="must match"):
        AgentEvalSpec(
            tasks=PinnedHarborTaskList(
                task_refs=[TaskRef(f"default/task#{'a' * 64}")],
                scoring=[HarborTaskScoring(task_ref=TaskRef(f"default/task#{'b' * 64}"))],
            ),
            target=HarborRunnerTarget(),
        )


def test_apply_preserves_loaded_task_and_appends_metrics():
    task = AgentEvalTask(
        id="native-id",
        intent="Solve",
        inputs={"instruction": "Do it"},
        metadata={"harbor_dataset_path": "/dataset"},
        metrics=[HarborRewardMetric()],
    )
    scoring = HarborTaskScoring(
        task_ref=TaskRef(f"default/task#{'a' * 64}"), metrics=[_metric()], views=_views("grade")
    )
    result = apply_harbor_scoring(scoring, task, reward_key="grade")
    assert [metric_type_name(metric) for metric in result.metrics] == ["harbor_reward", "exact-match"]
    assert result.metrics[0].output_spec()[0].name == "grade"
    assert result.views == _views("grade")
    assert result.inputs == task.inputs and result.metadata == task.metadata


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
async def test_offline_selection_rejects_mixed_kinds_and_duplicate_native_ids(entity_store, direct, mixed):
    from nemo_evaluator.api.schemas import EvaluatorTaskDefinition
    from nemo_evaluator.task_refs import canonicalize_agent_eval_tasks

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
    with pytest.raises(ValueError, match="Cannot mix" if mixed else "Duplicate Harbor task IDs"):
        await canonicalize_agent_eval_tasks(
            refs if direct else TasksetRef("default/suite"),
            workspace="default",
            entity_client=entity_store,
        )


@pytest.mark.parametrize("offline", [False, True])
async def test_harbor_judge_model_is_resolved_before_canonical_job(entity_store, monkeypatch, offline):
    from nemo_evaluator.jobs.metric_resolution import PlatformMetricModelResolver, to_runtime_bundle
    from nemo_evaluator.shared.metric_bundles.bundles import unbundle_metric

    judge = _metric(
        LLMJudgeMetric(
            model=ModelRef("default/judge"),
            scores=[RangeScore(name="quality", minimum=0, maximum=1, parser=JSONScoreParser(json_path="quality"))],
        )
    )
    ref = TaskRef(f"default/task#{'a' * 64}")
    with pytest.raises(ValueError, match="models must be resolved"):
        AgentEvalSpec(
            tasks=PinnedHarborTaskList(task_refs=[ref], scoring=[HarborTaskScoring(task_ref=ref, metrics=[judge])]),
            target=HarborRunnerTarget(),
        )
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

    monkeypatch.setattr(PlatformMetricModelResolver, "resolve_model", resolve)
    async with AsyncNeMoPlatform(base_url="http://platform.test", workspace="default") as async_sdk:
        spec = await AgentEvalJob.to_spec(
            AgentEvalInputSpec(
                tasks=[TaskRef(f"default/task#{revision.content_hash}")],
                target=None if offline else HarborRunnerTarget(),
                trials=[
                    AgentEvalTrial(
                        id="trial", task_id="task", status="partial", metadata={"harbor_primary_reward_key": "reward"}
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
    assert isinstance(spec, AgentEvalSpec) and isinstance(spec.tasks, PinnedHarborTaskList)
    metric = unbundle_metric(to_runtime_bundle(spec.tasks.scoring[0].metrics[0]))
    assert isinstance(metric, LLMJudgeMetric) and isinstance(metric.model, Model)
    assert metric.model.url == "https://example.test/v1/chat/completions"
