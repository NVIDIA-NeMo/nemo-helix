# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import pytest
from nemo_evaluator.api.task_definitions.evaluator import ResolvedEvaluatorTaskDefinition
from nemo_evaluator.jobs.agent_evaluate import AgentEvalJob
from nemo_evaluator.jobs.agent_spec import (
    AgentEvalInputSpec,
    AgentEvalSpec,
    AgentEvalTaskInput,
    ResolvedTask,
    ResolvedTaskDefinition,
    Target,
)
from nemo_evaluator.jobs.kinds.types import LoadedTask, PrepareContext, SubmitContext
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalTask
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.job_context import JobContext, StoragePaths
from nemo_helix_plugin.job_results import LocalJobResults
from nemo_helix_plugin.sdk import AsyncNeMoHelix


class Prepared(Exception):
    pass


class RecordingKind:
    def __init__(self, kind):
        self.kind = kind
        self.calls = []
        self.invalid = False

    def runtime_id(self, item: LoadedTask) -> str:
        assert item.inline is not None
        return item.inline.id

    async def resolve(self, item: LoadedTask, ctx: SubmitContext) -> ResolvedTaskDefinition:
        assert item.inline is not None
        self.calls.append(("resolve", item.inline.id))
        return ResolvedEvaluatorTaskDefinition(kind="evaluator", intent=item.inline.intent)

    def accepts_target(self, target: Target | None, tasks: Sequence[ResolvedTask]) -> bool:
        self.calls.append(("accept", [t.id for t in tasks]))
        return True

    def validate_scoring(
        self, tasks: Sequence[ResolvedTask], *, target: Target | None, trials: Sequence[AgentEvalTrial] | None
    ) -> None:
        self.calls.append(("validate", [t.id for t in tasks]))
        if self.invalid:
            raise ValueError("Invalid scoring")

    def prepare(self, tasks: Sequence[ResolvedTask], ctx: PrepareContext) -> list[AgentEvalTask]:
        self.calls.append(("prepare", [t.id for t in tasks]))
        raise Prepared


@pytest.mark.parametrize("invalid", [False, True])
async def test_injected_registry_dispatches_submission_and_validates_before_worker_prepare(tmp_path, invalid):
    """Use recording adapters to verify registry dispatch, task order, and validation before worker preparation."""
    evaluator, harbor = RecordingKind("evaluator"), RecordingKind("harbor")
    adapters = {"evaluator": evaluator, "harbor": harbor}

    class TestJob(AgentEvalJob):
        pass

    TestJob.adapters = adapters
    async with AsyncNeMoHelix(base_url="http://unused.test") as sdk:
        spec = await TestJob.to_spec(
            AgentEvalInputSpec(
                tasks=[AgentEvalTaskInput(id=name, intent=name) for name in ("second", "first")], trials=[]
            ),
            workspace="default",
            entity_client=None,
            async_sdk=sdk,
            is_local=True,
        )
    assert isinstance(spec, AgentEvalSpec)
    assert [task.id for task in spec.tasks] == ["second", "first"]
    assert evaluator.calls[-2:] == [("accept", ["second", "first"]), ("validate", ["second", "first"])]
    evaluator.calls.clear()
    evaluator.invalid = invalid

    ctx = JobContext(
        workspace="default",
        storage=StoragePaths(persistent=tmp_path, ephemeral=tmp_path),
        results=LocalJobResults(root=tmp_path / "results"),
    )
    with NemoClient(base_url="http://unused.test", workspace="default") as client:
        with pytest.raises(ValueError if invalid else Prepared):
            TestJob()._run_with_client(spec.model_dump(mode="json"), ctx=ctx, platform_client=client, async_client=None)
    assert [call[0] for call in evaluator.calls] == (
        ["accept", "validate"] if invalid else ["accept", "validate", "prepare"]
    )
    assert harbor.calls == []


@pytest.mark.parametrize("invalid", ["kind", "duplicate-metric"])
def test_worker_rejects_invalid_canonical_json_before_runtime(tmp_path, invalid):
    """Reject missing task kinds and duplicate metrics before the worker creates runtime files."""
    from nemo_evaluator.jobs.metric_resolution import to_inline
    from nemo_evaluator.shared.metric_bundles.bundles import bundle_metric
    from nemo_evaluator.shared.metric_bundles.cloudpickle import CloudpickleMetricBundlePackager
    from nemo_evaluator_sdk.metrics.exact_match import ExactMatchMetric

    metric = to_inline(
        bundle_metric(ExactMatchMetric(reference="yes", candidate="yes"), CloudpickleMetricBundlePackager())
    )
    config = {
        "tasks": [
            {
                "id": "task",
                "spec": {"kind": "evaluator", "intent": "Do it", "metrics": [metric.model_dump(mode="json")] * 2},
            }
        ],
        "trials": [],
    }
    if invalid == "kind":
        config["tasks"][0]["spec"].pop("kind")
    ctx = JobContext(
        workspace="default",
        storage=StoragePaths(persistent=tmp_path, ephemeral=tmp_path),
        results=LocalJobResults(root=tmp_path / "results"),
    )
    with NemoClient(base_url="http://unused.test", workspace="default") as client:
        with pytest.raises(ValueError, match="union_tag_not_found|duplicate task metric types"):
            AgentEvalJob()._run_with_client(config, ctx=ctx, platform_client=client, async_client=None)
    assert not (tmp_path / "agent-eval").exists()
