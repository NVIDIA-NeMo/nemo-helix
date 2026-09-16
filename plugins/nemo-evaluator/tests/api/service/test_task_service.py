# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import Mock

import pytest
from nemo_evaluator.api.schemas import (
    EvaluatorTaskDefinition,
    HarborTaskDefinition,
    MetadataItem,
    MetricInline,
    MetricRef,
    Task,
    TaskInput,
    TaskInputs,
)
from nemo_evaluator.api.service.task_service import MetricRefNotFoundError, TaskService
from nemo_evaluator.api.task_definitions.harbor import HarborArchiveSource, HarborTaskHash
from nemo_evaluator.shared.metric_bundles.bundles import bundle_metric
from nemo_evaluator.shared.metric_bundles.cloudpickle import CloudpickleMetricBundlePackager
from nemo_evaluator_sdk.metrics.exact_match import ExactMatchMetric
from nemo_platform_plugin.entity_client import NemoEntityConflictError, NemoEntityNotFoundError


class _FakeMetricService:
    """Records both metric-service entry points.

    ``stored`` covers inline-metric normalization, so a test can assert a task stores refs rather
    than bundles. ``looked_up`` covers ref validation — recorded separately because "this task never
    touched the metric service" is a claim about *both* calls, and asserting only on ``stored``
    would leave a lookup-only path silently passing.
    """

    def __init__(self, existing: set[tuple[str, str]] | None = None) -> None:
        self.stored: list[MetricInline] = []
        self.looked_up: list[tuple[str, str]] = []
        self.existing = existing if existing is not None else {("default", "stored-metric")}

    async def store_derived_metric(self, metric: MetricInline, *, workspace: str) -> MetricRef:
        self.stored.append(metric)
        return MetricRef(f"{workspace}/derived.{metric.payload.digest}")

    async def get_metric(self, workspace: str, name: str) -> object | None:
        self.looked_up.append((workspace, name))
        return object() if (workspace, name) in self.existing else None


def _inline_metric() -> MetricInline:
    bundle = bundle_metric(
        ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}"),
        CloudpickleMetricBundlePackager(),
    )
    return MetricInline.model_validate(bundle.model_dump(mode="json"))


def _evaluator_spec(task: Task) -> EvaluatorTaskDefinition:
    """Narrow ``Task.spec`` to the evaluator variant before reading a variant-specific field.

    ``spec`` is a discriminated union, so a test that reads ``intent``/``metrics``/``reference`` has
    to say which kind it expects. Asserting it rather than assuming it means a change that routed
    the wrong variant here fails on the kind, not with an ``AttributeError`` mid-assertion.
    """
    assert isinstance(task.spec, EvaluatorTaskDefinition)
    return task.spec


def _harbor_spec(task: Task) -> HarborTaskDefinition:
    """The Harbor half of :func:`_evaluator_spec`."""
    assert isinstance(task.spec, HarborTaskDefinition)
    return task.spec


def _ref(metric: MetricRef | MetricInline) -> MetricRef:
    """Narrow a stored task's metric to a reference — inline bundles are offloaded on create."""
    assert isinstance(metric, MetricRef)
    return metric


def _task_input() -> TaskInput:
    return TaskInput(
        spec=EvaluatorTaskDefinition(
            kind="evaluator",
            intent="Answer the question.",
            inputs=TaskInputs(instruction="What is 2+2?"),
            metrics=[MetricRef("default/stored-metric")],
        ),
        metadata=[MetadataItem(key="suite", value="smoke")],
    )


@pytest.fixture(autouse=True)
def verified_archive_projection(monkeypatch):
    # These tests isolate entity/API semantics; real archive verification has separate tests.
    from nemo_evaluator.harbor.archive import NativeTask

    async def verify(spec, files_client):
        return NativeTask("fixture", spec.instruction, spec.config)

    monkeypatch.setattr("nemo_evaluator.api.service.task_service.verify_definition", verify)


@pytest.fixture
def metric_service() -> _FakeMetricService:
    return _FakeMetricService()


@pytest.fixture
def service(metric_service: _FakeMetricService, entity_store) -> TaskService:
    return TaskService(entity_store, metric_service, files_client=Mock())


async def test_create_then_get(service: TaskService) -> None:
    created, _ = await service.create_task("task-1", _task_input(), workspace="default")

    assert isinstance(created, Task)
    assert created.name == "task-1"
    assert created.id == "task-task-1"
    assert _evaluator_spec(created).intent == "Answer the question."
    assert isinstance(_evaluator_spec(created).metrics[0], MetricRef)
    assert created.created_at is not None

    got = await service.get_task("default", "task-1")
    assert got is not None and got.name == "task-1"


async def test_create_normalizes_inline_metrics_to_refs(
    service: TaskService, metric_service: _FakeMetricService
) -> None:
    inline = _inline_metric()
    task_input = TaskInput(
        spec=EvaluatorTaskDefinition(
            kind="evaluator",
            intent="Answer the question.",
            inputs=TaskInputs(instruction="What is 2+2?"),
            metrics=[MetricRef("default/stored-metric"), inline],
        )
    )

    created, _ = await service.create_task("task-1", task_input, workspace="default")

    # The inline metric was offloaded to the metric service (stored as a derived metric)...
    assert metric_service.stored == [inline]
    # ...and the persisted task holds only refs — the passthrough ref plus the derived one.
    assert all(isinstance(m, MetricRef) for m in _evaluator_spec(created).metrics)
    assert _ref(_evaluator_spec(created).metrics[0]).root == "default/stored-metric"
    assert _ref(_evaluator_spec(created).metrics[1]).root == f"default/derived.{inline.payload.digest}"


async def test_create_preserves_grader_only_reference(service: TaskService) -> None:
    """Normalization narrows ``metrics`` and must leave the rest of the spec alone.

    ``_normalize_spec`` rebuilds the spec with ``model_copy(update=...)``, so a field it does not
    name rides along untouched — this pins that, since silently dropping ground truth would leave
    metrics grading against nothing while the run still reported a score.
    """
    reference = {"expected": "Paris", "held_out_tests": ["test_capital.py"]}
    task_input = TaskInput(
        spec=EvaluatorTaskDefinition(
            kind="evaluator",
            intent="Answer the question.",
            inputs=TaskInputs(instruction="What is the capital of France?"),
            reference=reference,
            metrics=[_inline_metric()],
        )
    )

    created, _ = await service.create_task("task-1", task_input, workspace="default")

    assert _evaluator_spec(created).reference == reference, "normalizing metrics must not disturb the reference"
    got = await service.get_task("default", "task-1")
    assert got is not None and _evaluator_spec(got).reference == reference


async def test_create_rejects_missing_metric_ref(service: TaskService) -> None:
    task_input = TaskInput(
        spec=EvaluatorTaskDefinition(
            kind="evaluator", intent="x", inputs=TaskInputs(instruction="?"), metrics=[MetricRef("default/nope")]
        )
    )
    with pytest.raises(MetricRefNotFoundError, match="not found"):
        await service.create_task("task-1", task_input, workspace="default")


async def test_create_canonicalizes_bare_metric_ref(service: TaskService) -> None:
    # A bare "stored-metric" ref resolves against the task workspace and is persisted as "default/stored-metric".
    task_input = TaskInput(
        spec=EvaluatorTaskDefinition(
            kind="evaluator", intent="x", inputs=TaskInputs(instruction="?"), metrics=[MetricRef("stored-metric")]
        )
    )
    created, _ = await service.create_task("task-1", task_input, workspace="default")
    assert _ref(_evaluator_spec(created).metrics[0]).root == "default/stored-metric"


async def test_create_rejects_duplicate(service: TaskService) -> None:
    await service.create_task("task-1", _task_input(), workspace="default")
    with pytest.raises(ValueError, match="already exists"):
        await service.create_task("task-1", _task_input(), workspace="default")


async def test_get_returns_none_when_missing(service: TaskService) -> None:
    assert await service.get_task("default", "nope") is None


async def test_list_returns_workspace_tasks(service: TaskService) -> None:
    await service.create_task("a", _task_input(), workspace="default")
    await service.create_task("b", _task_input(), workspace="default")

    page = await service.list_tasks(workspace="default")

    assert {t.name for t in page.data} == {"a", "b"}
    assert page.pagination is not None and page.pagination.total_results == 2


async def test_delete(service: TaskService) -> None:
    await service.create_task("task-1", _task_input(), workspace="default")
    assert await service.delete_task("default", "task-1") is True
    assert await service.get_task("default", "task-1") is None


async def test_delete_returns_false_when_missing(service: TaskService) -> None:
    assert await service.delete_task("default", "nope") is False


# --- Failure handling ---------------------------------------------------------


async def test_create_rolls_back_the_head_when_publishing_fails(
    metric_service: _FakeMetricService, entity_store
) -> None:
    """A head with no revision would break the invariant every consumer relies on — `#latest`
    always resolves and `revision` is never 0. There is no cross-entity transaction, so create
    must undo itself rather than leave a half-created task behind."""
    service = TaskService(entity_store, metric_service, files_client=Mock())

    real_create = type(entity_store).create

    async def _boom(entity):
        if entity.__entity_type__ == "task_revision":
            raise RuntimeError("store unavailable")
        return await real_create(entity_store, entity)

    entity_store.create = _boom

    with pytest.raises(RuntimeError):
        await service.create_task("task-1", _task_input(), workspace="default")

    entity_store.create = real_create.__get__(entity_store)
    assert await service.get_task("default", "task-1") is None, "the orphaned head must not survive"


async def test_replace_propagates_a_concurrent_write_conflict(metric_service: _FakeMetricService, entity_store) -> None:
    """Losing the optimistic lock is a client-retryable conflict, not a server fault — the route
    maps this to 409, so the service must let it through rather than swallowing it."""
    service = TaskService(entity_store, metric_service, files_client=Mock())
    await service.create_task("task-1", _task_input(), workspace="default")

    async def _stale(entity, *, original_name=None):
        raise NemoEntityConflictError("modified by another request")

    entity_store.update = _stale

    with pytest.raises(NemoEntityConflictError):
        await service.replace_task("task-1", _task_input(), workspace="default")


async def test_replace_leaves_no_uncovered_head_content_when_publishing_fails(
    metric_service: _FakeMetricService, entity_store
) -> None:
    """A failed publish must not leave the head serving content no revision covers.

    Replace stages content in memory and lets publishing commit it, so a publish that never
    happens leaves nothing behind. Committing the head first would instead make a plain GET —
    which reads the head — return content that `#latest` does not resolve to.
    """
    service = TaskService(entity_store, metric_service, files_client=Mock())
    await service.create_task("task-1", _task_input(), workspace="default")

    real_create = type(entity_store).create

    async def _boom(entity):
        if entity.__entity_type__ == "task_revision":
            raise RuntimeError("store unavailable")
        return await real_create(entity_store, entity)

    entity_store.create = _boom

    changed = TaskInput(
        spec=EvaluatorTaskDefinition(
            kind="evaluator",
            intent="Rewritten.",
            inputs=TaskInputs(instruction="?"),
            metrics=[MetricRef("default/stored-metric")],
        )
    )
    with pytest.raises(RuntimeError):
        await service.replace_task("task-1", changed, workspace="default")

    entity_store.create = real_create.__get__(entity_store)
    head = await service.get_task("default", "task-1")
    assert head is not None
    assert _evaluator_spec(head).intent == "Answer the question.", "the head must still hold the last published content"


async def test_tag_revision_returns_none_for_a_missing_task(service: TaskService) -> None:
    assert await service.tag_revision("default", "nope", "blessed", "latest") is None


async def test_list_revisions_returns_none_for_a_missing_task(service: TaskService) -> None:
    assert await service.list_revisions("default", "nope") is None


async def test_replace_applies_project(service: TaskService) -> None:
    """`project` used to be accepted on replace and silently discarded."""
    await service.create_task("task-1", _task_input(), workspace="default", project="proj-a")
    replaced, _ = await service.replace_task("task-1", _task_input(), workspace="default", project="proj-b")
    assert replaced.project == "proj-b"


async def test_replace_without_project_leaves_it_unchanged(service: TaskService) -> None:
    """An omitted query parameter means "leave it alone", not "clear it"."""
    await service.create_task("task-1", _task_input(), workspace="default", project="proj-a")
    replaced, _ = await service.replace_task("task-1", _task_input(), workspace="default")
    assert replaced.project == "proj-a"


async def test_rollback_failure_does_not_mask_the_original_error(
    metric_service: _FakeMetricService, entity_store
) -> None:
    """If cleanup also fails, the caller must still see *why* the publish failed."""
    service = TaskService(entity_store, metric_service, files_client=Mock())
    real_create = type(entity_store).create

    async def _boom(entity):
        if entity.__entity_type__ == "task_revision":
            raise RuntimeError("the original failure")
        return await real_create(entity_store, entity)

    async def _delete_also_fails(*args, **kwargs):
        raise RuntimeError("rollback failed too")

    entity_store.create = _boom
    entity_store.delete = _delete_also_fails

    with pytest.raises(RuntimeError, match="the original failure"):
        await service.create_task("task-1", _task_input(), workspace="default")


async def test_resolve_revision_defaults_to_the_current_revision(service: TaskService) -> None:
    """No fragment means ``latest`` — the bare-member case taskset publishing hits most often."""
    await service.create_task("task-1", _task_input(), workspace="default")

    digest = await service.resolve_revision("default", "task-1")

    revisions = await service.list_revisions("default", "task-1")
    assert revisions is not None
    assert digest == revisions.data[0].content_hash


async def test_resolve_revision_honours_a_tag_naming_an_older_revision(service: TaskService) -> None:
    """The hook taskset publishing uses to turn a member's tag into an exact digest.

    Needs a task with *more than one* revision and a tag left behind on the older one: against a
    single-revision task every fragment resolves to the same digest, so the test would pass even if
    the fragment were ignored entirely.
    """
    await service.create_task("task-1", _task_input(), workspace="default")
    first = (await service.list_revisions("default", "task-1")).data[0].content_hash
    await service.tag_revision("default", "task-1", "blessed", "latest")

    revised = _task_input()
    revised.spec.intent = "Answer differently."
    await service.replace_task("task-1", revised, workspace="default")

    latest = await service.resolve_revision("default", "task-1")
    blessed = await service.resolve_revision("default", "task-1", "blessed")

    assert latest != first, "the task must actually have moved on, or this proves nothing"
    assert blessed == first, "a tag must resolve to the revision it names, not the current one"


async def test_resolve_revision_round_trips_a_digest_fragment(service: TaskService) -> None:
    """A member submitted already-pinned must resolve to itself rather than to the head."""
    await service.create_task("task-1", _task_input(), workspace="default")
    first = (await service.list_revisions("default", "task-1")).data[0].content_hash

    revised = _task_input()
    revised.spec.intent = "Answer differently."
    await service.replace_task("task-1", revised, workspace="default")

    assert await service.resolve_revision("default", "task-1", first) == first


async def test_resolve_revision_raises_for_a_missing_task(service: TaskService) -> None:
    """Existence surfaces from resolution itself — taskset publishing relies on this to reject a
    member that does not exist, now that the separate existence check is gone."""
    with pytest.raises(NemoEntityNotFoundError):
        await service.resolve_revision("default", "nope")


# --- Harbor-kind tasks --------------------------------------------------------


def _harbor_input(digest: str = "a" * 64) -> TaskInput:
    return TaskInput(
        spec=HarborTaskDefinition(
            kind="harbor",
            harbor_hash=HarborTaskHash(digest="b" * 64, harbor_version="0.20.0"),
            source=HarborArchiveSource(
                fileset_ref="default/harbor-tasks#packages/org-name/abc/files",
                files_hash=digest,
            ),
            instruction="Fix the failing test.",
            config={"verifier": {"type": "pytest"}},
        ),
        metadata=[MetadataItem(key="suite", value="swe")],
    )


async def test_stores_a_harbor_task(service: TaskService) -> None:
    """Both kinds live in one record type, so a user manages every evaluation unit in one place."""
    created, published = await service.create_task("fix-test", _harbor_input(), workspace="default")

    assert published
    assert created.spec.kind == "harbor"
    assert created.spec.source.fileset_ref.endswith("files")
    assert created.spec.config == {"verifier": {"type": "pytest"}}


async def test_harbor_task_publishes_revisions_like_any_other(service: TaskService) -> None:
    await service.create_task("fix-test", _harbor_input(), workspace="default")

    same, published_again = await service.replace_task("fix-test", _harbor_input(), workspace="default")
    assert not published_again, "identical content must not cut a revision"

    changed, published = await service.replace_task("fix-test", _harbor_input(digest="b" * 64), workspace="default")
    assert published and changed.revision == 2


async def test_a_harbor_task_never_reaches_the_metric_service(
    service: TaskService, metric_service: _FakeMetricService
) -> None:
    """Metric normalization is agent-eval-specific: a Harbor task is scored by Harbor's own reward,
    and its spec arrives already in stored form.

    Both entry points, not just the write: a Harbor spec must not be validated against stored
    metrics either, so ``_normalize_spec`` has to short-circuit before ref resolution rather than
    merely find nothing to offload.
    """
    await service.create_task("fix-test", _harbor_input(), workspace="default")
    assert metric_service.stored == []
    assert metric_service.looked_up == []


async def test_kinds_with_matching_metadata_do_not_share_a_digest(service: TaskService) -> None:
    """The revision digest covers the whole spec, so two kinds cannot collide on content."""
    harbor, _ = await service.create_task("a", _harbor_input(), workspace="default")
    agent, _ = await service.create_task("b", _task_input(), workspace="default")

    harbor_revisions = await service.list_revisions("default", "a")
    agent_revisions = await service.list_revisions("default", "b")
    assert harbor_revisions is not None and agent_revisions is not None
    assert harbor_revisions.data[0].content_hash != agent_revisions.data[0].content_hash


async def test_harbor_config_is_stored_but_not_hashed(service: TaskService) -> None:
    """`config` is a projection of task.toml, which lives inside the tree — a real change moves
    `files_hash`. Hashing the projection too would make our history sensitive to Harbor's
    serialization: a release that reordered keys would cut a revision for byte-identical files."""
    await service.create_task("fix-test", _harbor_input(), workspace="default")

    reserialized = TaskInput(
        spec=HarborTaskDefinition(
            kind="harbor",
            harbor_hash=HarborTaskHash(digest="b" * 64, harbor_version="0.20.0"),
            source=HarborArchiveSource(
                fileset_ref="default/harbor-tasks#packages/org-name/abc/files",
                files_hash="a" * 64,
            ),
            instruction="Fix the failing test.",
            config={"verifier": {"type": "pytest"}, "added_by_a_new_harbor_release": True},
        ),
        metadata=[MetadataItem(key="suite", value="swe")],
    )
    same, published = await service.replace_task("fix-test", reserialized, workspace="default")

    assert not published, "a config-only change must not cut a revision"
    assert same.revision == 1

    # ...but the new config is *persisted*, so the queryable projection stays current. Re-read
    # rather than trusting the returned object: `replace_task` builds its result from the head it
    # already mutated in memory, so asserting on `same` would pass even if nothing were written.
    # On this path the write is a lone `entity_client.update` whose comment justifies it by
    # `project` alone — drop it as a redundant round trip and only a re-read notices.
    refetched = await service.get_task("default", "fix-test")
    assert refetched is not None
    assert _harbor_spec(refetched).config["added_by_a_new_harbor_release"] is True


async def test_reference_only_change_publishes_a_revision(service: TaskService) -> None:
    """The mirror of the Harbor ``config`` case, and the reason the two differ.

    ``config`` is excluded because it is a projection of content ``files_hash`` already covers.
    ``reference`` is nothing of the sort: it is the ground truth a metric grades against, so a task
    whose reference changed scores differently and must be a distinct revision. Deduping it onto the
    old digest would let a pinned taskset silently re-grade.
    """

    def _graded(expected: str) -> TaskInput:
        return TaskInput(
            spec=EvaluatorTaskDefinition(
                kind="evaluator",
                intent="Answer the question.",
                inputs=TaskInputs(instruction="What is the capital of France?"),
                reference={"expected": expected},
                metrics=[MetricRef("default/stored-metric")],
            )
        )

    await service.create_task("capital", _graded("Paris"), workspace="default")

    same, published_again = await service.replace_task("capital", _graded("Paris"), workspace="default")
    assert not published_again and same.revision == 1, "identical content must still dedup"

    changed, published = await service.replace_task("capital", _graded("Lyon"), workspace="default")
    assert published, "changing the ground truth must cut a new revision"
    assert changed.revision == 2
    assert _evaluator_spec(changed).reference == {"expected": "Lyon"}


async def test_a_real_tree_change_does_cut_a_revision(service: TaskService) -> None:
    """The flip side: `files_hash` is the authoritative identity, so it must still move."""
    await service.create_task("fix-test", _harbor_input(), workspace="default")
    changed, published = await service.replace_task("fix-test", _harbor_input(digest="b" * 64), workspace="default")
    assert published and changed.revision == 2
