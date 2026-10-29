# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submit checks the deployment can run the request and reads every task's images, then accepts the run.

It starts no task Jobs (the controller does) and no image builds.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from nemo_evals.api.fields import TaskRef
from nemo_evals.api.service.taskset_run_service import MAX_CONCURRENT_IMAGE_PREPARATIONS, TasksetRunResolutionError
from nemo_evals.api.taskset_run_schemas import TasksetRunInput
from nemo_evals.entities import TaskEntity, TaskRevisionEntity, TasksetEntity, TasksetRevisionEntity
from nemo_evals.harbor.remote.preparation import prepare_task_images
from nemo_evals.image_builds.models import BuildImage, ImageBuildRequest, TaskImageBuildEntity, build_entity_name
from nemo_evals.image_builds.planning import TaskArchive
from nemo_evals.image_builds.service import ImageBuildService
from nemo_evals.revisions import get_revision, publish_revision
from nemo_evals.taskset_runs.entities import TasksetRunEntity, TasksetRunPhase

pytestmark = pytest.mark.usefixtures("published")


async def submit(service, request_body):
    return await service.submit(TasksetRunInput.model_validate(request_body), workspace="default", caller="user:alice")


async def runs(entity_store):
    return (await entity_store.list(TasksetRunEntity, workspace="default")).data


async def test_runs_off_in_this_deployment(service, request_body, entity_store):
    service._config.runs.harbor = None
    with pytest.raises(TasksetRunResolutionError, match="aren't configured in this deployment"):
        await submit(service, request_body)
    assert not await runs(entity_store)


async def test_omitted_sandbox_freezes_the_default_profile(service, request_body, run_store):
    assert "sandbox" not in request_body["execution"]
    outcome = await submit(service, request_body)
    plan = await run_store.load_plan(await run_store.find("default", outcome.status.run_id))
    assert plan.sandbox_policy.profile == "harbor-standard"
    assert plan.sandbox_policy.egress_allow == ("igw.svc.cluster.local",)


async def test_unknown_sandbox_profile_lists_the_profiles(service, request_body):
    request_body["execution"]["sandbox"] = {"profile": "strict"}
    with pytest.raises(TasksetRunResolutionError, match="harbor-open, harbor-standard"):
        await submit(service, request_body)


async def test_open_profile_is_frozen_into_the_plan(service, request_body, run_store):
    request_body["execution"]["sandbox"] = {"profile": "harbor-open"}
    outcome = await submit(service, request_body)
    plan = await run_store.load_plan(await run_store.find("default", outcome.status.run_id))
    assert plan.sandbox_policy.egress_allow == ("igw.svc.cluster.local", "inference-api.nvidia.com")


async def test_sandbox_overrides_are_frozen_into_the_plan(service, request_body, run_store):
    request_body["execution"]["sandbox"] = {"cpus": 4, "memory_mb": 8192}
    outcome = await submit(service, request_body)
    plan = await run_store.load_plan(await run_store.find("default", outcome.status.run_id))
    assert plan.sandbox_policy.resource_overrides.model_dump(exclude_none=True) == {"cpus": 4, "memory_mb": 8192}


async def test_sandbox_override_over_the_ceiling_is_rejected(service, request_body, entity_store):
    service._config.runs.harbor.sandbox_profiles["harbor-standard"].max_cpus = 2
    request_body["execution"]["sandbox"] = {"cpus": 4}
    with pytest.raises(TasksetRunResolutionError, match="execution.sandbox.cpus=4 exceeds .*'s max_cpus=2"):
        await submit(service, request_body)
    assert not await runs(entity_store)


async def test_dispatcher_profile_must_be_allowed(service, request_body):
    request_body["execution"]["dispatcher_profile"] = "gpu"
    with pytest.raises(TasksetRunResolutionError, match="isn't allowed; use one of: local-k8s, k8s-cpu"):
        await submit(service, request_body)


async def test_omitted_dispatcher_profile_uses_the_operators_first(
    service, request_body, harbor_config, fake_task_jobs, run_store
):
    request_body["execution"].pop("dispatcher_profile", None)
    service._config.runs.harbor = harbor_config.model_copy(update={"dispatcher_profiles": ("k8s-cpu", "local-k8s")})
    outcome = await submit(service, request_body)
    plan = await run_store.load_plan(await run_store.find("default", outcome.status.run_id))
    assert plan.dispatcher_profile == "k8s-cpu"


async def test_subprocess_dispatcher_profile_is_rejected(service, request_body, fake_task_jobs):
    fake_task_jobs.subprocess_profiles = {"local-k8s"}
    with pytest.raises(TasksetRunResolutionError, match="runs as a host subprocess"):
        await submit(service, request_body)


@pytest.fixture
async def two_task_body(request_body, entity_store, published):
    """``request_body`` over a published taskset ``default/pair`` of task-1 and task-2."""
    task_head = await entity_store.get(TaskEntity, name="task-1", workspace="default")
    second = TaskEntity(
        name="task-2", workspace="default", spec=task_head.spec.model_copy(update={"native_task_id": "task-2"})
    )
    second, _, _ = await publish_revision(
        entity_store, entity_store, await entity_store.create(second), TaskRevisionEntity
    )
    pair = TasksetEntity(
        name="pair",
        workspace="default",
        tasks=[*published[1].tasks, TaskRef(f"default/task-2#{second.content_hash}")],
    )
    await publish_revision(entity_store, entity_store, await entity_store.create(pair), TasksetRevisionEntity)
    request_body["spec"]["tasks"] = "default/pair"
    return request_body


async def test_more_tasks_than_the_concurrency_cap_are_accepted(service, two_task_body, run_store, fake_task_jobs):
    """The cap is how many task Jobs run at once; the controller queues the rest."""
    two_task_body["execution"]["max_concurrent_task_jobs"] = 1
    outcome = await submit(service, two_task_body)
    plan = await run_store.load_plan(await run_store.find("default", outcome.status.run_id))
    assert outcome.created and len(plan.spec.tasks) == 2 and plan.max_concurrent_task_jobs == 1


async def test_task_without_images_is_rejected_before_acceptance(service, request_body, entity_store, run_preparer):
    run_preparer.fail = {"task-1": "image_unavailable"}
    with pytest.raises(TasksetRunResolutionError, match="tasks without images: task-1: image_unavailable"):
        await submit(service, request_body)
    assert not await runs(entity_store)


class _BuildJobs:
    def __init__(self) -> None:
        self.created: list[str] = []

    async def create(self, request) -> None:
        self.created.append(request.name)

    async def get(self, name: str):
        return None


@pytest.fixture
def real_preparer(service, monkeypatch):
    """Submit with the real preparer and image build service; a task archive read returns a Dockerfile task."""

    async def archive(files, source):
        return TaskArchive(root="t", files=frozenset({"t/environment/Dockerfile"}), compose=None)

    monkeypatch.setattr("nemo_evals.harbor.remote.preparation.read_task_archive", archive)
    service._preparer = prepare_task_images
    return service


async def build(entity_store, task: str, status: str, *, rebuild: bool = False) -> None:
    """Start the next build of ``task``'s newest revision, as the CLI does before submit, and record ``status``."""
    head = await entity_store.get(TaskEntity, name=task, workspace="default")
    digest = (await get_revision(entity_store, TaskRevisionEntity, head)).content_hash

    async def allow(workspace: str) -> None:
        return None

    builds = ImageBuildService(entity_store, _BuildJobs(), allow)
    started = await builds.request_build("default", task, digest, ImageBuildRequest(rebuild=rebuild))
    assert started is not None
    record = await entity_store.get(
        TaskImageBuildEntity, name=build_entity_name(digest, int(started.id)), workspace="default", parent=head.id
    )
    record.status = status
    if status == "ready":
        ref = f"registry.invalid/{task}@sha256:{'d' * 64}"
        record.images = [
            BuildImage(
                type="main", source="dockerfile", status="ready", image_ref=ref, builder_image="b", build_job="j"
            )
        ]
    await entity_store.update(record)


def builder_calls(authorizer) -> list:
    return [call for call in authorizer.await_args_list if "/apis/builder/" in str(call)]


async def test_an_unbuilt_task_is_rejected_without_starting_a_build(
    real_preparer, two_task_body, entity_store, authorizer, fake_task_jobs
):
    """The real preparer only reads builds: task-2 has none, so the run is refused and nothing is started."""
    await build(entity_store, "task-1", "ready")
    with pytest.raises(
        TasksetRunResolutionError, match=r"tasks without images: task-2: image_unavailable: .* has no image build"
    ):
        await submit(real_preparer, two_task_body)
    assert not await runs(entity_store)
    assert not builder_calls(authorizer)  # never needs, nor asks for, the caller's Builder permission


async def test_every_task_ready_is_accepted_with_its_images(
    real_preparer, two_task_body, entity_store, run_store, authorizer, fake_task_jobs
):
    await build(entity_store, "task-1", "ready")
    await build(entity_store, "task-2", "ready")
    outcome = await submit(real_preparer, two_task_body)
    plan = await run_store.load_plan(await run_store.find("default", outcome.status.run_id))
    assert outcome.created and [env.agent.image_ref for env in plan.task_environments] == [
        f"registry.invalid/task-{i}@sha256:{'d' * 64}" for i in (1, 2)
    ]
    assert not builder_calls(authorizer)


async def test_a_rebuild_in_flight_is_rejected_although_an_older_build_is_ready(
    real_preparer, request_body, entity_store, fake_task_jobs
):
    """The CLI waits for the rebuild; runs already accepted keep the digests they froze."""
    await build(entity_store, "task-1", "ready")
    await build(entity_store, "task-1", "pending", rebuild=True)
    with pytest.raises(TasksetRunResolutionError, match=r"task-1: image_unavailable: image build 2 .* is pending"):
        await submit(real_preparer, request_body)
    assert not await runs(entity_store)


async def test_a_task_over_the_profiles_ceiling_is_rejected_before_any_build(
    service, request_body, entity_store, harbor_config, monkeypatch
):
    """The real preparer: task-1's task.toml asks for 1024 MB, over the profile's max_memory_mb."""
    profiles = dict(harbor_config.sandbox_profiles)
    profiles["harbor-standard"] = profiles["harbor-standard"].model_copy(update={"max_memory_mb": 512})
    service._config.runs.harbor = harbor_config.model_copy(update={"sandbox_profiles": profiles})
    monkeypatch.setattr(service, "_image_builds", lambda workspace: pytest.fail("no build may start"))
    service._preparer = prepare_task_images
    with pytest.raises(
        TasksetRunResolutionError,
        match=r"task-1: resources_exceed_policy: agent sandbox memory_mb=1024 \(task.toml\) exceeds sandbox profile "
        r"'harbor-standard''s max_memory_mb=512",
    ):
        await submit(service, request_body)
    assert not await runs(entity_store)


async def test_the_default_resources_notice_is_on_the_accept_response(
    service, request_body, fake_task_jobs, harbor_config, monkeypatch
):
    """task-1 sets no resources. A keyed replay returns the stored status only."""
    request = TasksetRunInput.model_validate(request_body)
    pinned, spec = await service._resolve(request, "default")
    spec.tasks[0].spec.config = {"environment": {}}
    monkeypatch.setattr(service, "_resolve", AsyncMock(return_value=(pinned, spec)))
    profiles = dict(harbor_config.sandbox_profiles)
    profiles["harbor-standard"] = profiles["harbor-standard"].model_copy(update={"max_memory_mb": 1024})
    service._config.runs.harbor = harbor_config.model_copy(update={"sandbox_profiles": profiles})
    notice = (
        "1 task relies on default sandbox resources: task-1 agent cpus=1 memory_mb=1024. "
        "Set them in task.toml or pass --cpus/--memory-mb."
    )

    accepted = await submit(service, request_body)
    assert accepted.created is True and accepted.status.warnings == [notice]
    replay = await submit(service, request_body)
    assert replay.created is False and replay.status.warnings == []


@pytest.mark.parametrize("rejected", [0, 1], ids=["first-rejected", "second-rejected"])
async def test_a_rejected_task_stops_every_preparation(service, two_task_body, run_preparer, rejected):
    """Every task is checked before any task's builds are read."""
    tasks = ["task-1", "task-2"]
    run_preparer.reject = {tasks[rejected]: "gpu_unsupported"}
    with pytest.raises(TasksetRunResolutionError, match=f"{tasks[rejected]}: gpu_unsupported"):
        await submit(service, two_task_body)
    assert sorted(run_preparer.checked) == tasks and run_preparer.calls == []


async def test_a_replay_never_reads_images_again(service, request_body, fake_task_jobs, run_preparer, run_store):
    outcome = await submit(service, request_body)
    calls = len(run_preparer.calls)

    # Before the retry the task's images change (a moved tag, a newer build), then can't be prepared at all.
    run_preparer.with_verifier = {"task-1"}
    run_preparer.fail = {"task-1": "image_unavailable"}
    replay = await submit(service, request_body)
    assert replay.created is False and len(run_preparer.calls) == calls
    plan = await run_store.load_plan(await run_store.find("default", outcome.status.run_id))
    (environment,) = plan.task_environments
    assert environment.verifier is None and environment.agent.image_ref.startswith(
        "registry.invalid/fake/task-1-agent@"
    )


@pytest.mark.parametrize("miss_replay", [False, True], ids=["replay", "accept-conflict"])
async def test_an_ended_run_is_returned_as_is(
    service, request_body, fake_task_jobs, run_store, run_preparer, monkeypatch, miss_replay
):
    await submit(service, request_body)
    # The controller expired the run before the caller retried.
    await run_store.set_state(await run_store.find("default", "run-11f77762178b0e05"), phase=TasksetRunPhase.EXPIRED)
    prepared = len(run_preparer.calls)
    if miss_replay:
        # Stands in for a submit that raced the winner past the replay lookup: accept returns the winner's run.
        monkeypatch.setattr(service._store, "replay", AsyncMock(return_value=None))
    outcome = await submit(service, request_body)
    # Only the accept path prepares images again; this proves which path ran.
    assert (len(run_preparer.calls) > prepared) is miss_replay
    assert outcome.created is False and outcome.status.phase == TasksetRunPhase.EXPIRED
    assert outcome.status.warnings == []


class SlowPreparer:
    """Prepares every task after a pause, recording how many preparations ran at once."""

    def __init__(self, *, raise_for: str | None = None) -> None:
        self.raise_for = raise_for
        self.running = self.most_running = 0
        self.cancelled: list[str] = []

    async def check(self, task, **_):
        return None

    async def __call__(self, task, recorded, **_):
        self.running += 1
        self.most_running = max(self.most_running, self.running)
        try:
            await asyncio.sleep(0.01)
            if task.id == self.raise_for:
                raise RuntimeError(f"registry down for {task.id}")
            await asyncio.sleep(0.05)
            return SimpleNamespace(environment=f"images of {task.id}", error=None)  # what submit reads
        except asyncio.CancelledError:
            self.cancelled.append(task.id)
            raise
        finally:
            self.running -= 1


async def prepare(service, preparer, count):
    service._preparer = preparer
    tasks = [SimpleNamespace(id=f"task-{i}") for i in range(count)]
    return await service._prepare_images(tasks, None, caller="user:alice")


async def test_images_are_prepared_ten_at_a_time_in_task_order(service):
    preparer = SlowPreparer()
    results = await prepare(service, preparer, 25)
    assert preparer.most_running == MAX_CONCURRENT_IMAGE_PREPARATIONS == 10
    assert results == [f"images of task-{i}" for i in range(25)]


async def test_a_failed_preparation_cancels_the_others_and_fails_submit(service):
    preparer = SlowPreparer(raise_for="task-3")
    with pytest.raises(RuntimeError, match="registry down for task-3"):
        await prepare(service, preparer, 25)
    assert preparer.cancelled and preparer.running == 0  # the in-flight ones were stopped, none left running
