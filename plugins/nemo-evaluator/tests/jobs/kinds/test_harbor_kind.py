# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_evaluator.api.schemas import HarborTaskDefinition, TaskRef
from nemo_evaluator.api.task_definitions.harbor import HarborArchiveSource, HarborTaskHash
from nemo_evaluator.entities import TaskEntity, TaskRevisionEntity
from nemo_evaluator.jobs.agent_spec import FabricRunnerTarget, HarborRunnerTarget, ResolvedTask
from nemo_evaluator.jobs.kinds.harbor import HarborTaskAdapter
from nemo_evaluator.jobs.kinds.registry import KIND_ADAPTERS
from nemo_evaluator.jobs.kinds.types import PrepareContext, SubmitContext
from nemo_evaluator.revisions import publish_revision
from nemo_evaluator.task_refs import load_tasks, snapshot_task
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus, AgentOutput
from nemo_helix_plugin.sdk import AsyncNeMoHelix


async def test_harbor_adapter_snapshot_and_offline_preparation(entity_store, tmp_path, monkeypatch):
    """Resolve a stored task, then remove its entities and verify offline preparation uses only the snapshot and
    trials.
    """
    task = TaskEntity(
        name="stored",
        workspace="other",
        spec=HarborTaskDefinition(
            kind="harbor",
            native_task_id="native/task",
            instruction="Do it",
            source=HarborArchiveSource(fileset_ref="other/files#archive", files_hash="b" * 64),
            harbor_hash=HarborTaskHash(digest="c" * 64, harbor_version="test"),
        ),
    )
    await entity_store.create(task)
    revision, _, _ = await publish_revision(entity_store, entity_store, task, TaskRevisionEntity)
    async with AsyncNeMoHelix(base_url="http://unused.test") as sdk:
        ctx = SubmitContext("default", entity_store, sdk, KIND_ADAPTERS)
        loaded = await load_tasks([TaskRef("other/stored")], ctx)
        definition = await HarborTaskAdapter().resolve(loaded[0], ctx)
        assert definition.provenance.entity_name == "other/stored"
        assert definition.provenance.revision_digest == revision.content_hash
        snapshot = await snapshot_task(loaded[0], ctx)
    assert snapshot.spec.provenance is not None
    assert snapshot.spec.provenance.revision_digest == revision.content_hash
    assert ResolvedTask.model_validate_json(snapshot.model_dump_json()) == snapshot
    adapter = HarborTaskAdapter()
    assert adapter.runtime_id(loaded[0]) == "native/task"
    assert adapter.accepts_target(None, [snapshot])
    assert adapter.accepts_target(HarborRunnerTarget(), [snapshot])
    assert not adapter.accepts_target(FabricRunnerTarget(config={}), [snapshot])
    trials = [
        AgentEvalTrial(
            id="trial",
            status=AgentEvalTrialStatus.COMPLETED,
            task_id="native/task",
            output=AgentOutput(output_text="yes"),
            metadata={
                "harbor_primary_reward_key": "grade",
                "reward": 1,
                "reward_details": {"grade": 1, "secondary": 0.5},
            },
        )
    ]
    from nemo_evaluator.jobs.kinds import harbor

    original = harbor.saved_harbor_reward_key
    calls = []

    def tracked(values):
        calls.append(values)
        return original(values)

    monkeypatch.setattr(harbor, "saved_harbor_reward_key", tracked)
    adapter.validate_scoring([snapshot] * 10, target=None, trials=trials)
    assert len(calls) == 1
    entity_store.entities.clear()
    runtime = adapter.prepare([snapshot], PrepareContext(tmp_path, None, None, None, trials, KIND_ADAPTERS))
    assert runtime[0].id == "native/task"
    assert {output.name for metric in runtime[0].metrics for output in metric.output_spec()} >= {"grade", "secondary"}
    assert not (tmp_path / "harbor-inputs").exists()
