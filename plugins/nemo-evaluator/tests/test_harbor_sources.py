# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

import pytest
from nemo_evaluator.jobs.agent_spec import AgentEvalInputSpec, AgentEvalSpec
from nemo_evaluator.jobs.kinds.registry import KIND_ADAPTERS
from nemo_evaluator.jobs.kinds.types import SubmitContext
from nemo_evaluator.task_refs import validate_execution_support
from pydantic import ValidationError

DIGEST = "a" * 64


def _snapshot(**changes) -> dict[str, Any]:
    return {
        "id": "task",
        "spec": {
            "provenance": {
                "entity_name": changes.get("entity_name", "default/task"),
                "revision_digest": changes.get("revision_digest", DIGEST),
            },
            "kind": "harbor",
            "native_task_id": "task",
            "instruction": "Do it",
            "source": {"fileset_ref": "default/files#archive", "files_hash": "b" * 64},
            "harbor_hash": {"digest": "c" * 64, "harbor_version": "test"},
            "config": {"verifier": {"timeout_sec": 60}},
            "metrics": [],
            "views": {},
        },
        **{key: value for key, value in changes.items() if key not in {"entity_name", "revision_digest"}},
    }


def test_resolved_tasks_round_trip_with_definitions():
    spec = AgentEvalSpec.model_validate({"tasks": [_snapshot()], "target": {"kind": "harbor"}})
    restored = AgentEvalSpec.model_validate_json(spec.model_dump_json())
    assert restored == spec
    assert restored.model_dump()["tasks"][0]["spec"]["config"] == {"verifier": {"timeout_sec": 60}}


@pytest.mark.parametrize(
    "changes",
    [
        {"entity_name": "task"},
        {"entity_name": "default/task#latest"},
        {"revision_digest": "latest"},
        {"revision_digest": "abc"},
    ],
)
def test_canonical_task_requires_qualified_identity_and_digest(changes):
    with pytest.raises(ValidationError):
        AgentEvalSpec.model_validate({"tasks": [_snapshot(**changes)], "target": {"kind": "harbor"}})


def test_direct_refs_are_public_inputs():
    from nemo_evaluator.api.schemas import TaskRef

    spec = AgentEvalInputSpec.model_validate({"tasks": ["default/task#blessed"], "target": {"kind": "harbor"}})
    assert isinstance(spec.tasks, list)
    assert isinstance(spec.tasks[0], TaskRef)
    assert spec.tasks[0].root == "default/task#blessed"


@pytest.mark.parametrize(
    "case,error",
    [
        ("identity", "Duplicate task identity"),
        ("native_id", "task ids must be unique"),
        ("target", "harbor tasks cannot run"),
        ("metric_ref", "metrics"),
        ("empty", "at least 1"),
        ("mixed", "cannot mix"),
        ("archive", "fileset_ref"),
    ],
)
def test_canonical_tasks_reject_invalid_snapshots(case, error):
    """Exercise snapshot and target validation against malformed identities, sources, metrics, and task collections."""
    tasks: list[dict[str, Any]] = [_snapshot()]
    target = {"kind": "harbor"}
    if case == "identity":
        tasks.append(_snapshot(id="other", revision_digest="d" * 64))
        tasks[-1]["spec"]["native_task_id"] = "other"
    elif case == "native_id":
        other = _snapshot(entity_name="other/task")
        other["spec"]["native_task_id"] = "TASK"
        other["id"] = "TASK"
        tasks.append(other)
    elif case == "target":
        target = {"kind": "fabric", "config": {}}
    elif case == "metric_ref":
        tasks[0]["spec"]["metrics"] = ["default/metric"]
    elif case == "empty":
        tasks = []
    elif case == "mixed":
        tasks.append({"id": "inline", "spec": {"kind": "evaluator", "intent": "Do it", "metrics": []}})
    elif case == "archive":
        tasks[0]["spec"]["source"]["fileset_ref"] = "files#archive"
    with pytest.raises(ValueError, match=error):
        spec = AgentEvalSpec.model_validate({"tasks": tasks, "target": target})
        validate_execution_support(spec.tasks, target=spec.target, adapters=KIND_ADAPTERS)


@pytest.mark.parametrize(
    "source",
    [
        {"kind": "harbor-taskset", "taskset_ref": f"default/suite#{DIGEST}"},
        {"kind": "harbor-task-list", "task_refs": [f"default/task#{DIGEST}"]},
    ],
)
def test_legacy_pinned_jobs_are_rejected(source):
    source["scoring"] = [{"task_ref": f"default/task#{DIGEST}", "metrics": [], "views": {}}]
    with pytest.raises(ValidationError):
        AgentEvalSpec.model_validate({"tasks": source, "target": {"kind": "harbor"}})


def test_preparation_without_clients_fails_before_allocating_inputs(tmp_path):
    from nemo_evaluator.harbor.preparation import prepare_stored_harbor_tasks
    from nemo_evaluator.jobs.agent_spec import ResolvedTask

    with pytest.raises(ValueError, match="authenticated platform client"):
        prepare_stored_harbor_tasks(
            [ResolvedTask.model_validate(_snapshot())],
            destination_root=tmp_path / "persistent" / "harbor-inputs",
            client=None,
            async_client=None,
            reward_key="reward",
        )
    assert not (tmp_path / "persistent" / "harbor-inputs").exists()


def test_worker_without_clients_fails_before_allocating_inputs(tmp_path):
    """Verify a worker lacking an authenticated client fails before allocating Harbor input storage."""
    from nemo_evaluator.jobs.agent_evaluate import AsyncAgentEvalJob
    from nemo_helix_plugin.client.client import AsyncNemoClient
    from nemo_helix_plugin.job_context import JobContext, StoragePaths
    from nemo_helix_plugin.job_results import LocalJobResults

    ctx = JobContext(
        workspace="default",
        job_id="test",
        storage=StoragePaths(
            ephemeral=tmp_path / "ephemeral",
            persistent=tmp_path / "persistent",
        ),
        results=LocalJobResults(root=tmp_path / "results"),
    )
    with pytest.raises(ValueError, match="authenticated platform client"):
        AsyncAgentEvalJob().run(
            {
                "tasks": [_snapshot()],
                "target": {"kind": "harbor"},
            },
            ctx=ctx,
            async_client=cast(AsyncNemoClient, None),
        )
    assert not (tmp_path / "persistent" / "harbor-inputs").exists()


async def test_bounded_resolution_preserves_order_and_drains_failure():
    """Verify concurrent resolution preserves input order, respects its limit, and leaves no work active after
    failure.
    """
    import asyncio

    from nemo_evaluator.harbor.resolution import RESOLUTION_CONCURRENCY, map_with_limited_concurrency

    active = peak = 0

    async def resolve(value):
        """Yield to competing resolutions while tracking active work and peak concurrency."""
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0)
            return value * 2
        finally:
            active -= 1

    assert await map_with_limited_concurrency(resolve, list(range(2000))) == [i * 2 for i in range(2000)]
    assert peak <= RESOLUTION_CONCURRENCY
    assert active == 0

    async def fail(value):
        if value == 1:
            raise ValueError("missing member")
        return await resolve(value)

    with pytest.raises(ValueError, match="missing member"):
        await map_with_limited_concurrency(fail, list(range(2000)))
    assert active == 0


async def test_bounded_resolution_accepts_custom_concurrency():
    """Verify the caller's concurrency limit is reached without changing result order."""
    import asyncio

    from nemo_evaluator.harbor.resolution import map_with_limited_concurrency

    active = peak = 0

    async def resolve(value):
        """Yield to competing resolutions while tracking active work and peak concurrency."""
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0)
            return value
        finally:
            active -= 1

    assert await map_with_limited_concurrency(resolve, list(range(10)), concurrency=3) == list(range(10))
    assert peak == 3


@pytest.mark.parametrize("target", [None, {"kind": "harbor"}, {"kind": "fabric", "config": {}}])
@pytest.mark.parametrize("branch", ["taskset", "inline", "refs"])
def test_homogeneous_input_branches_round_trip(target, branch):
    """Verify JSON round trips preserve taskset references, inline tasks, and direct task references as distinct
    inputs.
    """
    from nemo_evaluator.api.schemas import TaskRef, TasksetRef
    from nemo_evaluator.jobs.agent_spec import AgentEvalTaskInput

    tasks = {
        "taskset": TasksetRef("default/suite"),
        "inline": [AgentEvalTaskInput(id="task", intent="Do task")],
        "refs": [TaskRef("default/task")],
    }[branch]
    spec = AgentEvalInputSpec(tasks=tasks, target=target, trials=[] if target is None else None)
    restored = AgentEvalInputSpec.model_validate_json(spec.model_dump_json())
    assert restored == spec
    assert type(restored.tasks) is type(tasks)
    if isinstance(tasks, list):
        assert isinstance(restored.tasks, list)
        assert type(restored.tasks[0]) is type(tasks[0])


@pytest.mark.parametrize("target", [None, {"kind": "harbor"}, {"kind": "fabric", "config": {}}])
@pytest.mark.parametrize("shape", ["empty", "inline-first", "ref-first"])
def test_input_rejects_empty_and_mixed_lists(target, shape):
    """Reject empty or mixed task selections consistently for Python inputs and serialized JSON."""
    import json

    from nemo_evaluator.api.schemas import TaskRef
    from nemo_evaluator.jobs.agent_spec import AgentEvalTaskInput

    tasks = [] if shape == "empty" else [AgentEvalTaskInput(id="task", intent="Do task"), TaskRef("default/task")]
    if shape == "ref-first":
        tasks.reverse()
    with pytest.raises(ValidationError):
        AgentEvalInputSpec(tasks=tasks, target=target, trials=[] if target is None else None)  # ty: ignore[invalid-argument-type]
    payload = {
        "tasks": [task.model_dump(mode="json") for task in tasks],
        "target": target,
        "trials": [] if target is None else None,
    }
    with pytest.raises(ValidationError):
        AgentEvalInputSpec.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize("reverse", [False, True])
async def test_resolver_rejects_mixed_lists_before_entity_access(reverse):
    from unittest.mock import Mock

    from nemo_evaluator.api.schemas import TaskRef
    from nemo_evaluator.jobs.agent_spec import AgentEvalTaskInput
    from nemo_evaluator.task_refs import load_tasks

    tasks = [AgentEvalTaskInput(id="task", intent="Do task"), TaskRef("default/task")]
    if reverse:
        tasks.reverse()
    client = Mock()
    with pytest.raises(ValueError, match="Cannot mix inline tasks and stored task references"):
        await load_tasks(
            tasks,  # ty: ignore[invalid-argument-type] -- deliberately mixed public branches
            SubmitContext(workspace="default", entity_client=client, async_sdk=None, adapters=KIND_ADAPTERS),
        )
    assert client.mock_calls == []


@pytest.mark.parametrize("change", ["id", "provenance", "kind"])
def test_harbor_snapshot_invariants_on_json_reload(change):
    import json

    snapshot = _snapshot()
    if change == "id":
        snapshot["id"] = "different"
    elif change == "provenance":
        snapshot["spec"].pop("provenance")
    else:
        snapshot["spec"].pop("kind")
    with pytest.raises(ValidationError):
        AgentEvalSpec.model_validate_json(json.dumps({"tasks": [snapshot], "target": {"kind": "harbor"}}))


@pytest.mark.parametrize(
    "task",
    [
        {"id": "task", "intent": "Do it", "metrics": []},
        {"entity_name": "default/task", "revision_digest": DIGEST, "definition": _snapshot()["spec"]},
    ],
)
def test_previous_canonical_list_shapes_are_rejected(task):
    with pytest.raises(ValidationError):
        AgentEvalSpec.model_validate({"tasks": [task], "trials": []})


@pytest.mark.parametrize("mismatch", [None, "instruction", "identity"])
def test_online_preparation_uses_ordered_verified_members(tmp_path, monkeypatch, mismatch):
    from nemo_evaluator.api.task_definitions.harbor import HarborTaskDefinition
    from nemo_evaluator.harbor.materialization import MaterializedHarborMember, MaterializedHarborTasks
    from nemo_evaluator.harbor.preparation import prepare_stored_harbor_tasks
    from nemo_evaluator.harbor.tasks import StoredHarborTask
    from nemo_evaluator.jobs.agent_spec import ResolvedTask
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_archive import NativeTask
    from nemo_helix_plugin.client.client import NemoClient

    snapshots = []
    members = []
    dataset = tmp_path / "dataset"
    # Deliberately nonalphabetical; second member is a multistep package without root instructions.
    for name, instruction in [("z-last", " Do it "), ("a-first", None)]:
        data = _snapshot(id=name, entity_name=f"default/{name}")
        data["spec"].update(native_task_id=name, instruction=instruction)
        snapshot = ResolvedTask.model_validate(data)
        snapshots.append(snapshot)
        source = StoredHarborTask(
            entity_name=f"default/{name}",
            revision_digest=DIGEST,
            definition=HarborTaskDefinition.model_validate(
                {key: value for key, value in data["spec"].items() if key != "provenance"}
            ),
        )
        native = NativeTask(
            "changed" if mismatch == "identity" else name,
            "Changed instruction" if mismatch == "instruction" else instruction,
            {"steps": [{"name": "first"}]} if instruction is None else {},
        )
        members.append(MaterializedHarborMember(source, name, dataset / name, native))

    def materialize(sources, *, files_client, destination_root):
        assert [source.definition.native_task_id for source in sources] == ["z-last", "a-first"]
        return MaterializedHarborTasks(dataset, tuple(members))

    monkeypatch.setattr("nemo_evaluator.harbor.materialization.materialize_harbor_tasks_sync", materialize)
    with NemoClient(base_url="http://test", workspace="default") as client:
        if mismatch:
            with pytest.raises(ValueError):
                prepare_stored_harbor_tasks(
                    snapshots, destination_root=tmp_path, client=client, async_client=None, reward_key="reward"
                )
            return
        tasks = prepare_stored_harbor_tasks(
            snapshots, destination_root=tmp_path, client=client, async_client=None, reward_key="reward"
        )
    assert [task.id for task in tasks] == ["z-last", "a-first"]
    assert [task.inputs for task in tasks] == [{"instruction": "Do it"}, {"instruction": "a-first"}]
    assert [task.metadata for task in tasks] == [
        {"harbor_task_dir": str(member.task_dir), "harbor_dataset_path": str(dataset)} for member in members
    ]
