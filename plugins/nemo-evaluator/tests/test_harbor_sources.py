# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nemo_evaluator.jobs.agent_spec import AgentEvalInputSpec, AgentEvalSpec
from pydantic import ValidationError

DIGEST = "a" * 64
SCORING = [{"task_ref": f"default/task#{DIGEST}", "metrics": [], "views": {}}]


def test_pinned_sources_round_trip_without_definitions():
    for source in [
        {"kind": "harbor-taskset", "taskset_ref": f"default/suite#{DIGEST}"},
        {"kind": "harbor-task-list", "task_refs": [f"default/task#{DIGEST}"]},
    ]:
        source["scoring"] = SCORING
        spec = AgentEvalSpec.model_validate({"tasks": source, "target": {"kind": "harbor"}})
        assert spec.model_dump(mode="json")["tasks"] == source
        assert AgentEvalSpec.model_validate_json(spec.model_dump_json()) == spec


@pytest.mark.parametrize("ref", ["suite", "suite#latest", f"suite#{DIGEST}", "default/suite#blessed"])
@pytest.mark.parametrize("source_kind", ["harbor-taskset", "harbor-task-list"])
def test_canonical_source_requires_qualified_digest(ref, source_kind):
    source = (
        {"kind": source_kind, "taskset_ref": ref}
        if source_kind == "harbor-taskset"
        else {"kind": source_kind, "task_refs": [ref]}
    )
    source["scoring"] = SCORING
    with pytest.raises(ValidationError):
        AgentEvalSpec.model_validate({"tasks": source, "target": {"kind": "harbor"}})


def test_direct_refs_are_public_inputs():
    spec = AgentEvalInputSpec.model_validate({"tasks": ["default/task#blessed"], "target": {"kind": "harbor"}})
    from nemo_evaluator.api.schemas import TaskRef

    assert isinstance(spec.tasks, list)
    assert isinstance(spec.tasks[0], TaskRef)
    assert spec.tasks[0].root == "default/task#blessed"


def test_canonical_source_rejects_duplicate_identity_and_nonharbor_target():
    for payload in [
        {
            "tasks": {"kind": "harbor-task-list", "task_refs": [f"default/task#{DIGEST}", f"default/task#{'b' * 64}"]},
            "target": {"kind": "harbor"},
        },
        {"tasks": {"kind": "harbor-taskset", "taskset_ref": f"default/suite#{DIGEST}"}, "trials": []},
    ]:
        payload["tasks"]["scoring"] = SCORING
        with pytest.raises(ValidationError):
            AgentEvalSpec.model_validate(payload)


def test_worker_without_clients_fails_before_allocating_inputs(tmp_path):
    from nemo_evaluator.api.schemas import TaskRef, TasksetRef
    from nemo_evaluator.harbor.preparation import prepare_stored_harbor_tasks
    from nemo_evaluator.harbor.tasks import HarborTaskScoring, PinnedHarborTaskset

    with pytest.raises(ValueError, match="authenticated platform client"):
        prepare_stored_harbor_tasks(
            PinnedHarborTaskset(
                taskset_ref=TasksetRef(f"default/suite#{DIGEST}"),
                scoring=[HarborTaskScoring(task_ref=TaskRef(f"default/task#{DIGEST}"))],
            ),
            destination_root=tmp_path / "persistent" / "harbor-inputs",
            client=None,
            async_client=None,
            reward_key="reward",
        )
    assert not (tmp_path / "persistent" / "harbor-inputs").exists()


@pytest.mark.parametrize("kind", ["harbor-task-list", "harbor-taskset"])
@pytest.mark.parametrize("scoring", [None, []])
def test_canonical_source_requires_nonempty_scoring(kind, scoring):
    source = (
        {"kind": kind, "task_refs": [f"default/task#{DIGEST}"]}
        if kind == "harbor-task-list"
        else {"kind": kind, "taskset_ref": f"default/suite#{DIGEST}"}
    )
    if scoring is not None:
        source["scoring"] = scoring
    with pytest.raises(ValidationError, match="scoring"):
        AgentEvalSpec.model_validate({"tasks": source, "target": {"kind": "harbor"}})


@pytest.mark.parametrize("kind", ["harbor-task-list", "harbor-taskset"])
def test_source_rejects_duplicate_scoring_references(kind):
    source = (
        {"kind": kind, "task_refs": [f"default/task#{DIGEST}"]}
        if kind == "harbor-task-list"
        else {"kind": kind, "taskset_ref": f"default/suite#{DIGEST}"}
    )
    source["scoring"] = SCORING * 2
    with pytest.raises(ValidationError, match="Duplicate task identity"):
        AgentEvalSpec.model_validate({"tasks": source, "target": {"kind": "harbor"}})


def test_job_rejects_top_level_scoring():
    with pytest.raises(ValidationError, match="harbor_scoring"):
        AgentEvalSpec.model_validate(
            {
                "tasks": {"kind": "harbor-task-list", "task_refs": [f"default/task#{DIGEST}"], "scoring": SCORING},
                "target": {"kind": "harbor"},
                "harbor_scoring": SCORING,
            }
        )


async def test_bounded_resolution_preserves_order_and_drains_failure():
    import asyncio

    from nemo_evaluator.harbor.resolution import RESOLUTION_CONCURRENCY, bounded_ordered_map

    active = peak = 0

    async def resolve(value):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0)
            return value * 2
        finally:
            active -= 1

    assert await bounded_ordered_map(resolve, list(range(2000))) == [i * 2 for i in range(2000)]
    assert peak <= RESOLUTION_CONCURRENCY
    assert active == 0

    async def fail(value):
        if value == 1:
            raise ValueError("missing member")
        return await resolve(value)

    with pytest.raises(ValueError, match="missing member"):
        await bounded_ordered_map(fail, list(range(2000)))
    assert active == 0


@pytest.mark.parametrize("target", [None, {"kind": "harbor"}, {"kind": "fabric", "config": {}}])
@pytest.mark.parametrize("branch", ["taskset", "inline", "refs"])
def test_homogeneous_input_branches_round_trip(target, branch):
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


@pytest.mark.parametrize("target_kind", ["offline", "harbor", "fabric"])
@pytest.mark.parametrize("reverse", [False, True])
async def test_resolver_rejects_mixed_lists_before_entity_access(target_kind, reverse):
    from unittest.mock import Mock

    from nemo_evaluator.api.schemas import TaskRef
    from nemo_evaluator.jobs.agent_spec import AgentEvalTaskInput, FabricRunnerTarget, HarborRunnerTarget
    from nemo_evaluator.task_refs import canonicalize_agent_eval_tasks

    target = {"offline": None, "harbor": HarborRunnerTarget(), "fabric": FabricRunnerTarget(config={})}[target_kind]
    tasks = [AgentEvalTaskInput(id="task", intent="Do task"), TaskRef("default/task")]
    if reverse:
        tasks.reverse()
    client = Mock()
    with pytest.raises(ValueError, match="Cannot mix inline tasks and stored task references"):
        await canonicalize_agent_eval_tasks(tasks, workspace="default", entity_client=client, target=target)  # ty: ignore[invalid-argument-type]
    assert client.mock_calls == []
