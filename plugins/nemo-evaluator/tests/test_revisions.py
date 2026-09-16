# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Revision publishing and resolution.

Exercised against an in-memory store that reproduces the two entity-store behaviors this logic
leans on: parent-scoped name uniqueness (which is what serializes concurrent ordinal allocation)
and conflict-on-duplicate-create.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import TypeVar

import pytest
from nemo_evaluator.api.schemas import LATEST_TAG, EvaluatorTaskDefinition, MetricRef, TaskInputs, TaskRef
from nemo_evaluator.entities import TaskEntity, TaskRevisionEntity, TasksetEntity
from nemo_evaluator.revisions import (
    RevisionConflictError,
    RevisionContentMismatchError,
    RevisionNotFoundError,
    apply_tag,
    get_revision,
    head_digest,
    list_revisions,
    publish_revision,
    revision_name,
)
from nemo_platform_plugin.entities import EntityBase
from nemo_platform_plugin.entity_client import NemoEntityConflictError, NemoEntityNotFoundError
from nemo_platform_plugin.filter_ops import FilterOperator, LogicalOperation

_E = TypeVar("_E", bound=EntityBase)


class FakeStore:
    """In-memory stand-in keyed the way the entity store keys records."""

    def __init__(self) -> None:
        self.records: dict[tuple[str, str, str, str | None], EntityBase] = {}
        self._next_id = 0
        #: Monotonic tick for creation timestamps. ``list``/``find_by_digest`` order on
        #: ``-created_at``, so leaving it unset would make ordering undefined — and sorting several
        #: ``None`` timestamps raises rather than quietly returning insertion order.
        self._tick = 0
        #: Ordinals to fail the first create for, simulating a lost allocation race.
        self.contend_ordinals: set[int] = set()
        #: Ordinals to lose to a publisher of *identical* content whose pointer write has not landed
        #: yet — the interleaving that makes two concurrent identical requests each look novel.
        self.contend_identically: set[int] = set()
        #: One-shot hook fired just before a *head* update, modelling a competing publisher that
        #: lands in the window between our child create and our pointer write. Lives on the fake
        #: rather than a monkeypatch so the race is expressed the same way ``contend_ordinals`` is.
        self.before_head_update: Callable[[], Awaitable[None]] | None = None

    def _key(self, entity_type: type[EntityBase], name: str, workspace: str, parent: str | None):
        return (entity_type.__entity_type__, workspace, name, parent)

    async def create(self, entity: _E) -> _E:
        key = self._key(type(entity), entity.name, entity.workspace, entity._parent)
        if key in self.records:
            raise NemoEntityConflictError(f"{entity.name} exists")
        ordinal = getattr(entity, "revision", None)
        if ordinal in self.contend_ordinals:
            self.contend_ordinals.discard(ordinal)
            self._win_race(entity, ordinal)
            raise NemoEntityConflictError(f"{entity.name} taken by a concurrent publisher")
        if ordinal in self.contend_identically:
            self.contend_identically.discard(ordinal)
            self._win_race(entity, ordinal, digest=entity.content_hash, advance_head=False)
            raise NemoEntityConflictError(f"{entity.name} taken by a concurrent publisher")
        self._next_id += 1
        self._tick += 1
        entity._id = f"id-{self._next_id}"
        entity._created_at = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=self._tick)
        entity._updated_at = entity._created_at
        self.records[key] = entity.model_copy(deep=True)
        return entity

    def _win_race(self, losing_entity, ordinal: int, *, digest: str | None = None, advance_head: bool = True) -> None:
        """Model a *complete* competing publish, not just a failed create.

        By default the winner's record exists under the contended name and the head has advanced.
        Without both, the loser re-reads an unchanged head and recomputes the same ordinal — which
        no real race would do, and which would make the retry look broken when it isn't. The
        competitor publishes *different* content (its own digest), so the loser genuinely needs a
        new ordinal rather than discovering its content already published.

        ``digest`` and ``advance_head`` express the opposite interleaving: a winner publishing the
        *same* content whose pointer write has not landed yet. The loser then sees a head that still
        names the previous revision, so nothing but the contended child itself reveals that its
        content is already published.
        """
        winner = losing_entity.model_copy(update={"content_hash": digest or f"{ordinal:064x}"})
        winner._parent = losing_entity._parent
        self._next_id += 1
        winner._id = f"id-{self._next_id}"
        self._tick += 1
        winner._created_at = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=self._tick)
        self.records[self._key(type(winner), winner.name, winner.workspace, winner._parent)] = winner
        if not advance_head:
            return
        head = self.records.get(self._key(TaskEntity, "task-1", winner.workspace, None))
        if head is not None:
            head.latest_revision = max(head.latest_revision, ordinal)
            head.tags = {**head.tags, LATEST_TAG: ordinal}

    async def get(self, entity_type: type[_E], name, *, workspace=None, parent=None) -> _E:
        """Hand back a *copy*, as the real client does — it rebuilds entities from an HTTP response
        and cannot share objects with its caller. Returning the stored instance would let code
        under test mutate the store just by touching what it read."""
        key = self._key(entity_type, name, workspace or "default", parent)
        if key not in self.records:
            raise NemoEntityNotFoundError(f"{name} not found")
        stored = self.records[key]
        copy = stored.model_copy(deep=True)
        copy._parent, copy._id, copy._db_version = stored._parent, stored._id, stored._db_version
        return copy

    async def update(self, entity: _E, *, original_name=None) -> _E:
        """Enforce the ``db_version`` optimistic lock the real store enforces.

        A fake that accepts every update makes correct retry logic untestable and incorrect retry
        logic look fine, so this rejects a write whose base version is stale.
        """
        if self.before_head_update is not None and isinstance(entity, (TaskEntity, TasksetEntity)):
            hook, self.before_head_update = self.before_head_update, None
            await hook()
        key = self._key(type(entity), original_name or entity.name, entity.workspace, entity._parent)
        stored = self.records.get(key)
        if stored is not None and entity._db_version != stored._db_version:
            raise NemoEntityConflictError(
                f"stale update for {entity.name}: base version {entity._db_version}, "
                f"stored version {stored._db_version}"
            )
        saved = entity.model_copy(deep=True)
        saved._parent = entity._parent
        saved._id = entity._id
        saved._db_version = (stored._db_version if stored is not None else 0) + 1
        self.records[key] = saved
        return saved

    async def delete(self, entity_type, name, *, workspace, parent=None, expected_db_version=None) -> object:
        """Part of the standard client surface, so the fake carries it even though publishing and
        resolving never delete — a stand-in that omits it would not be substitutable."""
        key = self._key(entity_type, name, workspace, parent)
        if key not in self.records:
            raise NemoEntityNotFoundError(f"{name} not found")
        return self.records.pop(key)

    async def list(self, entity_type, *, workspace, filter_operation=None, sort=None, page=1, page_size=100):
        """Evaluate the filter the way the store does, so the query path is actually exercised.

        Supports exactly what this module emits: an AND of equality comparisons over ``parent`` and
        ``data.<field>``. Anything else raises rather than silently matching everything — a fake
        that ignores filters would make a broken query look correct. ``sort`` and ``page`` are
        honoured for the same reason: accepting them and returning insertion order would make a
        caller that ordered wrongly look right.
        """
        rows = [
            record
            for (entity_type_name, record_workspace, _, _), record in self.records.items()
            if entity_type_name == entity_type.__entity_type__ and record_workspace == workspace
        ]
        for comparison in self._comparisons(filter_operation):
            if comparison.operator is not FilterOperator.EQ:
                raise NotImplementedError(f"fake supports only EQ, got {comparison.operator}")
            if comparison.field == "parent":
                rows = [r for r in rows if r.parent == comparison.value]
            elif comparison.field.startswith("data."):
                attribute = comparison.field.removeprefix("data.")
                rows = [r for r in rows if getattr(r, attribute, None) == comparison.value]
            else:
                raise NotImplementedError(f"fake cannot filter on {comparison.field!r}")
        if sort and rows:
            field = sort.lstrip("-")
            if not hasattr(rows[0], field):
                raise NotImplementedError(f"fake cannot sort on {field!r}")
            rows = sorted(rows, key=lambda record: getattr(record, field), reverse=sort.startswith("-"))
        start = (page - 1) * page_size
        return SimpleNamespace(data=rows[start : start + page_size])

    def _comparisons(self, operation):
        if operation is None:
            return []
        if isinstance(operation, LogicalOperation):
            if operation.operator is not FilterOperator.AND:
                raise NotImplementedError(f"fake supports only AND, got {operation.operator}")
            return [c for child in operation.operations for c in self._comparisons(child)]
        return [operation]

    def concurrent_head_write(self, head: EntityBase, *, tags: dict[str, int]) -> None:
        """Simulate another publisher committing to the head between our read and our write."""
        key = self._key(type(head), head.name, head.workspace, None)
        stored = self.records[key]
        stored.tags = {**stored.tags, **tags}
        stored.latest_revision = max([stored.latest_revision, *tags.values()], default=stored.latest_revision)
        stored._db_version += 1


def _head(store: FakeStore, *, intent: str = "Answer the question.") -> TaskEntity:
    head = TaskEntity(
        spec=EvaluatorTaskDefinition(
            kind="evaluator",
            intent=intent,
            inputs=TaskInputs(instruction="What is 2+2?"),
            metrics=[MetricRef("default/stored-metric")],
        ),
        name="task-1",
        workspace="default",
    )
    head._id = "head-1"
    head._db_version = 0
    # Store a *distinct* copy: the real client returns freshly deserialized objects, so a caller's
    # in-memory head and the stored record are never the same object. Aliasing them would make
    # every optimistic-lock conflict invisible (bumping one bumps the other).
    stored = head.model_copy(deep=True)
    stored._id = head.id
    stored._db_version = 0
    store.records[store._key(TaskEntity, "task-1", "default", None)] = stored
    return head


def _head_named(store: FakeStore, name: str) -> TaskEntity:
    """A second record with content identical to :func:`_head`'s — same digest, different parent."""
    head = TaskEntity(
        spec=EvaluatorTaskDefinition(
            kind="evaluator",
            intent="Answer the question.",
            inputs=TaskInputs(instruction="What is 2+2?"),
            metrics=[MetricRef("default/stored-metric")],
        ),
        name=name,
        workspace="default",
    )
    head._id = f"head-{name}"
    head._db_version = 0
    stored = head.model_copy(deep=True)
    stored._id = head.id
    stored._db_version = 0
    store.records[store._key(TaskEntity, name, "default", None)] = stored
    return head


async def _publish(store: FakeStore, head: TaskEntity, *, tags: set[str] | None = None):
    """Publish and drop the returned head — these tests assert against their own head object."""
    revision, _head, created = await publish_revision(store, store, head, TaskRevisionEntity, tags=tags)
    return revision, created


# --- First publish -----------------------------------------------------------


@pytest.mark.asyncio
async def test_first_publish_creates_revision_one() -> None:
    store = FakeStore()
    head = _head(store)
    revision, created = await _publish(store, head)
    assert created
    assert revision.revision == 1
    assert revision.name == revision_name(1)
    assert revision.content_hash == head_digest(head)


@pytest.mark.asyncio
async def test_first_publish_points_latest_at_revision_one() -> None:
    store = FakeStore()
    head = _head(store)
    revision, _ = await _publish(store, head)
    assert head.tags[LATEST_TAG] == revision.revision
    assert head.latest_revision == 1


@pytest.mark.asyncio
async def test_revision_is_a_child_of_its_head() -> None:
    """Parent scoping is what makes ordinals collide instead of silently duplicating."""
    store = FakeStore()
    head = _head(store)
    revision, _ = await _publish(store, head)
    assert revision.parent == head.id


# --- Idempotency -------------------------------------------------------------


@pytest.mark.asyncio
async def test_republishing_identical_content_creates_nothing() -> None:
    """The property that makes a re-publish cheap: same content, same digest, no new revision."""
    store = FakeStore()
    head = _head(store)
    first, created_first = await _publish(store, head)
    second, created_second = await _publish(store, head)
    assert created_first and not created_second
    assert second.revision == first.revision
    assert head.latest_revision == 1


@pytest.mark.asyncio
async def test_republishing_identical_content_still_applies_new_tags() -> None:
    """A no-op publish is not a no-op tag operation — that's how you tag an existing revision."""
    store = FakeStore()
    head = _head(store)
    revision, _ = await _publish(store, head)
    _, created = await _publish(store, head, tags={"blessed"})
    assert not created
    assert head.tags["blessed"] == revision.revision


@pytest.mark.asyncio
async def test_changed_content_allocates_the_next_ordinal() -> None:
    store = FakeStore()
    head = _head(store)
    first, _ = await _publish(store, head)
    head.spec.intent = "Do something else."
    second, created = await _publish(store, head)
    assert created
    assert second.revision == 2
    assert second.content_hash != first.content_hash
    assert head.tags[LATEST_TAG] == second.revision
    assert head.latest_revision == 2


# --- Concurrency -------------------------------------------------------------


@pytest.mark.asyncio
async def test_contended_ordinal_is_retried() -> None:
    """A publisher that loses the race for ordinal N re-reads and takes N+1 rather than failing."""
    store = FakeStore()
    head = _head(store)
    await _publish(store, head)
    head.spec.intent = "Changed."
    store.contend_ordinals = {2}
    revision, created = await _publish(store, head)
    assert created
    assert revision.revision == 3


@pytest.mark.asyncio
async def test_identical_contended_publish_adopts_the_winners_revision() -> None:
    """Two identical publishes that overlap must still cut exactly one revision.

    The dedup check reads the revision ``latest`` names, so it misses a winner whose pointer write
    has not landed: the loser sees a head still naming ``N-1``, computes the same digest, and would
    step past the contended ordinal onto a byte-identical ``rev.N+1``. Both callers would then
    report a new revision for the same content, which is exactly what publishing is supposed to be
    idempotent against.
    """
    store = FakeStore()
    head = _head(store)
    await _publish(store, head)

    head.spec.intent = "Changed."
    store.contend_identically = {2}
    revision, created = await _publish(store, head)

    assert not created, "the loser must report the winner's revision, not a publish of its own"
    assert revision.revision == 2, "adopt the contended ordinal rather than allocating past it"

    page = await list_revisions(store, TaskRevisionEntity, head)
    assert [entry.revision for entry in page.data] == [2, 1], "no duplicate-content revision was cut"

    stored = store.records[store._key(TaskEntity, head.name, head.workspace, None)]
    assert isinstance(stored, TaskEntity)
    assert stored.tags[LATEST_TAG] == 2, "adopting must still point latest at the revision it adopted"


@pytest.mark.asyncio
async def test_contended_publish_of_different_content_still_allocates_a_new_ordinal() -> None:
    """Adoption is keyed on the digest, not on losing the race.

    The guard added for identical publishes must not swallow a genuine one: a loser whose content
    differs from the winner's still needs its own ordinal.
    """
    store = FakeStore()
    head = _head(store)
    await _publish(store, head)

    head.spec.intent = "Changed."
    store.contend_ordinals = {2}  # winner publishes *different* content
    revision, created = await _publish(store, head)

    assert created
    assert revision.revision == 3


@pytest.mark.asyncio
async def test_publishing_recovers_when_a_revision_exists_but_the_head_never_advanced() -> None:
    """The one case where losing a create does *not* mean someone else advanced the head.

    If a publisher creates ``rev.N`` and then exhausts its pointer-write retries, the child exists
    while the head still names ``N-1``. A later publish computes N, loses the create, re-reads the
    head — which still says ``N-1`` — and computes N again, every attempt until it gives up. Every
    publish after that does the same, so the record would be permanently unpublishable rather than
    merely wasteful. The failed create is proof enough that N is taken.
    """
    store = FakeStore()
    head = _head(store)
    await _publish(store, head)

    # Simulate the pointer write never landing: rev.1 exists, the head still names revision 0.
    head.latest_revision, head.tags = 0, {}
    stored = store.records[store._key(TaskEntity, head.name, head.workspace, None)]
    assert isinstance(stored, TaskEntity)
    stored.latest_revision, stored.tags = 0, {}

    head.spec.intent = "Changed."
    revision, created = await _publish(store, head)

    assert created
    assert revision.revision == 2, "must step past the orphaned rev.1 instead of retrying it"


@pytest.mark.asyncio
async def test_losing_the_head_race_does_not_leave_the_head_on_an_older_revision() -> None:
    """A publisher that loses the *pointer* race must not drag the head's content backwards.

    A creates rev.2 and stalls; B publishes rev.3 and commits first; A's pointer write then loses
    the lock and retries. ``latest`` correctly stays at 3 — but if A's staged content still landed
    on the head, a plain read would return rev.2's content while ``#latest`` returned rev.3's, and
    the record would report itself as revision 3 while serving something else.
    """
    store = FakeStore()
    head = _head(store)
    await publish_revision(store, store, head, TaskRevisionEntity)  # rev.1

    a = await store.get(TaskEntity, name="task-1", workspace="default")
    b = await store.get(TaskEntity, name="task-1", workspace="default")
    a.spec.intent, b.spec.intent = "A's content.", "B's content."

    async def b_publishes() -> None:
        await publish_revision(store, store, b, TaskRevisionEntity)

    store.before_head_update = b_publishes
    a_revision, _, _ = await publish_revision(store, store, a, TaskRevisionEntity)

    stored = store.records[store._key(TaskEntity, "task-1", "default", None)]
    assert isinstance(stored, TaskEntity)
    latest = await get_revision(store, TaskRevisionEntity, stored, LATEST_TAG)

    assert stored.tags[LATEST_TAG] == 3
    assert stored.spec.intent == latest.spec.intent == "B's content."
    assert stored.latest_revision == latest.revision, "the reported revision must describe the content served"

    # A's publish is not lost — it is a real revision, still resolvable by digest.
    assert a_revision.revision == 2
    pinned = await get_revision(store, TaskRevisionEntity, stored, a_revision.content_hash)
    assert pinned.spec.intent == "A's content."


@pytest.mark.asyncio
async def test_persistent_contention_raises_rather_than_looping() -> None:
    store = FakeStore()
    head = _head(store)
    store.contend_ordinals = set(range(1, 50))
    with pytest.raises(RevisionConflictError):
        await _publish(store, head)


@pytest.mark.asyncio
async def test_latest_revision_never_rewinds() -> None:
    """Re-tagging an older revision must not hand the next publish an ordinal already in use.

    ``latest_revision`` is the allocation watermark, not a pointer: pointing a user tag backwards
    is legitimate, but if that dragged the watermark back with it, the next publish would compute
    an ordinal that already exists and lose its create.
    """
    store = FakeStore()
    head = _head(store)
    first, _ = await _publish(store, head)
    head.spec.intent = "Changed."
    await _publish(store, head)

    head = await store.get(TaskEntity, "task-1", workspace="default")
    await apply_tag(store, store, TaskRevisionEntity, head, "rollback", first.content_hash)

    stored = store.records[store._key(TaskEntity, "task-1", "default", None)]
    assert stored.tags["rollback"] == first.revision
    assert stored.latest_revision == 2, "the watermark must not follow a backwards tag"
    assert stored.tags[LATEST_TAG] == 2


@pytest.mark.asyncio
async def test_head_update_retries_on_optimistic_lock_conflict() -> None:
    """A publisher that loses the head write re-reads and folds its pointers into the winner's
    record, rather than propagating the conflict or overwriting with a stale copy."""
    store = FakeStore()
    head = _head(store)
    store.concurrent_head_write(head, tags={"other": 1})
    revision, created = await _publish(store, head)
    assert created
    stored = store.records[store._key(TaskEntity, "task-1", "default", None)]
    assert stored.tags["other"] == 1, "the winner's tag must survive our retry"
    assert stored.tags[LATEST_TAG] == revision.revision


@pytest.mark.asyncio
async def test_persistent_head_contention_raises() -> None:
    """Bounded retries: a head that keeps moving must fail loudly, not loop forever."""

    store = FakeStore()
    head = _head(store)
    original_update = store.update

    async def always_stale(entity, *, original_name=None):
        store.concurrent_head_write(entity, tags={})
        return await original_update(entity, original_name=original_name)

    store.update = always_stale  # type: ignore[method-assign]
    with pytest.raises(RevisionConflictError):
        await _publish(store, head)


@pytest.mark.asyncio
async def test_latest_does_not_regress_when_writes_interleave() -> None:
    """The interleaving that motivates forward-only ``latest``: A creates rev.1 and B creates
    rev.2, but B commits its head pointers first. A's later write must not drag ``latest`` back
    onto rev.1 while ``latest_revision`` says 2."""
    store = FakeStore()
    head = _head(store)

    # A has staged content and is about to publish as rev.1. B publishes rev.2 and commits its head
    # pointers first, so A's pointer write loses the optimistic lock and retries against B's head.
    store.concurrent_head_write(head, tags={LATEST_TAG: 2})

    revision, created = await _publish(store, head)

    assert created and revision.revision == 1
    stored = store.records[store._key(TaskEntity, "task-1", "default", None)]
    assert stored.tags[LATEST_TAG] == 2, "latest must not regress onto A's older revision"
    assert stored.latest_revision == 2


@pytest.mark.asyncio
async def test_user_tags_may_be_moved_backwards() -> None:
    """Only ``latest`` is forward-only. Retagging onto an older revision is a rollback — explicit
    user intent, not a race — and must be honoured."""
    store = FakeStore()
    head = _head(store)
    older, _ = await _publish(store, head, tags={"blessed"})
    head.spec.intent = "Newer content."
    await _publish(store, head)

    head = await store.get(TaskEntity, "task-1", workspace="default")
    await apply_tag(store, store, TaskRevisionEntity, head, "blessed", older.content_hash)

    stored = store.records[store._key(TaskEntity, "task-1", "default", None)]
    assert stored.tags["blessed"] == older.revision
    assert stored.tags[LATEST_TAG] == 2, "only the user tag moves; latest stays put"


@pytest.mark.asyncio
async def test_reverting_to_earlier_content_publishes_a_new_revision() -> None:
    """Dedup is against the *current* revision, not the whole history.

    Reverting is a real change to what the record is now. If it deduped onto the old revision the
    head would hold that content while ``latest`` — forward-only — kept naming the newer one, so a
    plain read and a ``#latest`` read would disagree about the same record.
    """
    store = FakeStore()
    head = _head(store)
    await _publish(store, head)  # rev.1: "Answer the question."
    head.spec.intent = "Changed."
    await _publish(store, head)  # rev.2

    head.spec.intent = "Answer the question."  # back to rev.1's content
    revision, created = await _publish(store, head)

    assert created, "a revert is a publish, not a no-op"
    assert revision.revision == 3
    assert head.tags[LATEST_TAG] == 3

    latest = await get_revision(store, TaskRevisionEntity, head, LATEST_TAG)
    assert latest.spec.intent == head.spec.intent, "the head and #latest must describe the same content"


@pytest.mark.asyncio
async def test_a_digest_shared_by_two_revisions_resolves_to_the_newer_one() -> None:
    """Reverting makes one record hold two revisions with the same digest, so a digest lookup has
    to pick deterministically. Both carry identical content — all a pin promises — but a pin must
    not resolve to a different ordinal from one call to the next."""
    store = FakeStore()
    head = _head(store)
    first, _ = await _publish(store, head)  # rev.1
    head.spec.intent = "Changed."
    await _publish(store, head)  # rev.2
    head.spec.intent = "Answer the question."
    third, _ = await _publish(store, head)  # rev.3, same digest as rev.1

    assert third.content_hash == first.content_hash

    for _ in range(3):
        resolved = await get_revision(store, TaskRevisionEntity, head, first.content_hash)
        assert resolved.revision == third.revision


@pytest.mark.asyncio
async def test_republishing_the_current_content_is_still_a_no_op() -> None:
    """The idempotency the narrower dedup must not cost: re-PUTting unchanged content."""
    store = FakeStore()
    head = _head(store)
    first, _ = await _publish(store, head)
    revision, created = await _publish(store, head)

    assert not created
    assert revision.revision == first.revision


# --- Resolution --------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolves_latest_by_default() -> None:
    store = FakeStore()
    head = _head(store)
    await _publish(store, head)
    head.spec.intent = "Changed."
    second, _ = await _publish(store, head)
    assert (await get_revision(store, TaskRevisionEntity, head)).content_hash == second.content_hash


@pytest.mark.asyncio
async def test_resolves_a_digest_to_its_revision() -> None:
    store = FakeStore()
    head = _head(store)
    first, _ = await _publish(store, head)
    head.spec.intent = "Changed."
    await _publish(store, head)
    resolved = await get_revision(store, TaskRevisionEntity, head, first.content_hash)
    assert resolved.revision == 1


@pytest.mark.asyncio
async def test_resolves_a_user_tag() -> None:
    store = FakeStore()
    head = _head(store)
    revision, _ = await _publish(store, head, tags={"blessed"})
    assert (await get_revision(store, TaskRevisionEntity, head, "blessed")).revision == revision.revision


@pytest.mark.asyncio
async def test_unknown_fragment_raises() -> None:
    store = FakeStore()
    head = _head(store)
    await _publish(store, head)
    with pytest.raises(RevisionNotFoundError):
        await get_revision(store, TaskRevisionEntity, head, "nonexistent")


@pytest.mark.asyncio
async def test_index_pointing_at_a_missing_record_raises() -> None:
    """A torn write must surface, not resolve to nothing."""
    store = FakeStore()
    head = _head(store)
    revision, _ = await _publish(store, head)
    del store.records[store._key(TaskRevisionEntity, revision.name, "default", head.id)]
    with pytest.raises(RevisionNotFoundError):
        await get_revision(store, TaskRevisionEntity, head)


@pytest.mark.asyncio
async def test_unknown_digest_raises() -> None:
    store = FakeStore()
    head = _head(store)
    await _publish(store, head)
    with pytest.raises(RevisionNotFoundError):
        await get_revision(store, TaskRevisionEntity, head, "c" * 64)


@pytest.mark.asyncio
async def test_digest_resolution_is_scoped_to_the_parent() -> None:
    """Identical content under two records yields one digest, so the digest alone does not identify
    a revision. Without parent scoping this would resolve to the other record's revision."""
    store = FakeStore()
    mine = _head(store)
    theirs = _head_named(store, "task-2")
    published, _, _ = await publish_revision(store, store, theirs, TaskRevisionEntity)

    assert head_digest(mine) == published.content_hash, "same content, same digest"
    with pytest.raises(RevisionNotFoundError):
        await get_revision(store, TaskRevisionEntity, mine, published.content_hash)


@pytest.mark.asyncio
async def test_identical_content_under_two_records_publishes_independently() -> None:
    """The flip side: one record's publish must not make another's look already-published."""
    store = FakeStore()
    mine = _head(store)
    theirs = _head_named(store, "task-2")
    await publish_revision(store, store, theirs, TaskRevisionEntity)
    revision, created = await _publish(store, mine)
    assert created and revision.revision == 1


@pytest.mark.asyncio
async def test_reading_a_revision_whose_content_was_tampered_with_is_refused() -> None:
    """The digest has to be checked on the way out or it is only a label.

    A revision is immutable by convention; the store will still accept a write to one. If reading
    trusted the recorded digest, a pinned ref would keep resolving and quietly serve content nobody
    pinned — the exact failure content-addressing exists to prevent.
    """
    store = FakeStore()
    head = _head(store)
    revision, _ = await _publish(store, head)

    stored = store.records[store._key(TaskRevisionEntity, revision_name(1), "default", head.id)]
    assert isinstance(stored, TaskRevisionEntity)
    stored.spec.intent = "Tampered with after publication."

    with pytest.raises(RevisionContentMismatchError, match="does not match its recorded digest"):
        await get_revision(store, TaskRevisionEntity, head, LATEST_TAG)


@pytest.mark.asyncio
async def test_a_digest_pinned_read_is_verified_too() -> None:
    """Both resolution paths verify — a digest lookup filters on the *recorded* digest, so without
    re-hashing it would happily return a record whose content no longer matches it."""
    store = FakeStore()
    head = _head(store)
    revision, _ = await _publish(store, head)

    stored = store.records[store._key(TaskRevisionEntity, revision_name(1), "default", head.id)]
    assert isinstance(stored, TaskRevisionEntity)
    stored.spec.intent = "Tampered with after publication."

    with pytest.raises(RevisionContentMismatchError):
        await get_revision(store, TaskRevisionEntity, head, revision.content_hash)


# --- Tag validation -----------------------------------------------------------


@pytest.mark.asyncio
async def test_a_digest_shaped_tag_is_rejected() -> None:
    """Such a tag could be stored but never resolved: a digest-shaped reference is looked up as a
    digest, so the tag map would never be consulted. Rejected rather than silently useless."""
    store = FakeStore()
    head = _head(store)
    with pytest.raises(ValueError, match="looks like a content digest"):
        await _publish(store, head, tags={"a" * 64})


@pytest.mark.asyncio
async def test_latest_cannot_be_moved_by_hand() -> None:
    """Refused where it matters: pointing ``latest`` somewhere would break the forward-only rule."""
    store = FakeStore()
    head = _head(store)
    revision, _ = await _publish(store, head)
    with pytest.raises(ValueError, match="managed automatically"):
        await apply_tag(store, store, TaskRevisionEntity, head, LATEST_TAG, revision.content_hash)


@pytest.mark.asyncio
async def test_latest_in_publish_tags_is_tolerated() -> None:
    """Listing ``latest`` at publish time asks for exactly what the server does anyway, so it is a
    no-op rather than an error. Rejecting it would break clients that pass it defensively — which
    is what an earlier version of this validation did."""
    store = FakeStore()
    head = _head(store)
    revision, created = await _publish(store, head, tags={LATEST_TAG, "blessed"})
    assert created
    assert head.tags[LATEST_TAG] == revision.revision
    assert head.tags["blessed"] == revision.revision


@pytest.mark.asyncio
async def test_an_empty_tag_is_rejected() -> None:
    """An absent fragment already means ``latest``, so an empty tag could never be addressed."""
    store = FakeStore()
    head = _head(store)
    with pytest.raises(ValueError, match="must not be empty"):
        await _publish(store, head, tags={"   "})


@pytest.mark.asyncio
async def test_a_nearly_digest_shaped_tag_is_allowed() -> None:
    """The rejection is exact — 64 lowercase hex — so ordinary names stay usable."""
    store = FakeStore()
    head = _head(store)
    _, created = await _publish(store, head, tags={"a" * 63, "deadbeef"})
    assert created
    assert head.tags["deadbeef"] == 1


@pytest.mark.parametrize("tag", ["release/2026", "release candidate"])
@pytest.mark.asyncio
async def test_a_tag_outside_the_fragment_charset_is_rejected(tag: str) -> None:
    """A tag exists to be written after ``#`` in a member reference. ``TaskRef`` admits only
    ``[\\w\\-.]+`` there, so a tag with a slash or a space would apply and list cleanly and then be
    unusable for the one thing it is for — the same silent dead end as a digest-shaped tag."""
    store = FakeStore()
    head = _head(store)
    with pytest.raises(ValueError, match="cannot appear in a reference fragment"):
        await _publish(store, head, tags={tag})


@pytest.mark.asyncio
async def test_an_accepted_tag_is_usable_as_a_ref_fragment() -> None:
    """The other half of the rule: whatever publishing accepts, ``TaskRef`` must also accept. Pins
    the two charsets together so neither can drift into rejecting the other's output."""
    store = FakeStore()
    head = _head(store)
    await _publish(store, head, tags={"blessed", "v1.2.3", "rc_2"})

    for tag in head.tags:
        TaskRef(f"{head.workspace}/{head.name}#{tag}")


@pytest.mark.parametrize("kind", ["task", "taskset"])
async def test_service_response_is_own_revision_during_head_race(kind):
    from unittest.mock import AsyncMock

    from nemo_evaluator.api.schemas import TaskInput, TasksetInput
    from nemo_evaluator.api.service.task_service import TaskService
    from nemo_evaluator.api.service.taskset_service import TasksetService

    store = FakeStore()
    tasks = TaskService(store, AsyncMock())
    if kind == "task":
        create, replace, get = tasks.create_task, tasks.replace_task, tasks.get_task

        def value(label):
            return TaskInput(
                spec=EvaluatorTaskDefinition(kind="evaluator", intent=label, inputs=TaskInputs(), metrics=[])
            )

        def label(result):
            return result.spec.intent
    else:
        service = TasksetService(store, tasks)
        create, replace, get = service.create_taskset, service.replace_taskset, service.get_taskset

        def value(label):
            return TasksetInput(description=label)

        def label(result):
            return result.description

    await create("race", value("initial"), workspace="default")

    async def advance():
        await replace("race", value("B"), workspace="default")

    store.before_head_update = advance
    result, _ = await replace("race", value("A"), workspace="default")
    assert result.revision == 2
    assert label(result) == "A"
    latest = await get("default", "race")
    assert latest.revision == 3
    assert label(latest) == "B"
