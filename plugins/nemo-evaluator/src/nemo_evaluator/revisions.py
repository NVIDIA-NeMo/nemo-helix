# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publishing and resolving revisions of tasks and tasksets.

A *head* record (``TaskEntity`` / ``TasksetEntity``) holds the current content plus revision
bookkeeping. Publishing freezes that content into an immutable *revision* child
(``TaskRevisionEntity`` / ``TasksetRevisionEntity``) addressed by a digest of the content, and
points the requested tags at it. Resolution goes the other way: a ref's ``#fragment`` — a tag or a
digest — becomes a concrete revision.

Two ways to address a revision, two lookup costs. A **tag** is a mutable pointer stored on the head
as ``tag → ordinal``, so resolving one is a direct child lookup with no query — which matters
because ``#latest`` is the common path. A **digest** is content-addressed and resolved by querying
the children on ``(parent, content_hash)``. Nothing is denormalized: the head stores only pointers,
never a copy of what the children already say.

Concurrency. Two publishers racing on the same record both compute ordinal N; the entity store's
parent-scoped uniqueness — unique within ``(workspace, entity_type, parent, name)`` — makes the
second ``rev.N`` create conflict rather than silently duplicate. The loser refreshes its allocation
state and retries against N+1, keeping the content it was asked to publish. The child create, not
the head update, is the serialization point, so a revision is never allocated twice even if the
head update is slow.
"""

from __future__ import annotations

import logging
import re
from typing import Protocol, TypeVar

from nemo_evaluator.api.schemas import LATEST_TAG, REF_FRAGMENT_CHARSET
from nemo_evaluator.content_hash import DIGEST_PATTERN, content_hash
from nemo_evaluator.entities import (
    REVISION_POINTER_EXCLUDE,
    REVISION_POINTER_FIELDS,
    REVISION_SELF_EXCLUDE,
    TaskEntity,
    TaskRevisionEntity,
    TasksetEntity,
    TasksetRevisionEntity,
)
from nemo_helix_plugin.entities import (
    EntityBase,
    EntityClientProtocol,
    EntityGetterProtocol,
    EntityUpdateClientProtocol,
    ListResponse,
)
from nemo_helix_plugin.entity_client import (
    NemoEntityConflictError,
    NemoEntityNotFoundError,
)
from nemo_helix_plugin.filter_ops import ComparisonOperation, FilterOperator, LogicalOperation

logger = logging.getLogger(__name__)

#: Attempts to allocate an ordinal before giving up. Each retry costs one head re-read; contention
#: on a single task is expected to be rare (publishes are human- or pipeline-paced), so a small
#: bound is enough to absorb a race without masking a genuine, persistent conflict.
_MAX_ALLOCATION_ATTEMPTS = 5

#: Prefix for revision entity names. Entity names must start with a lowercase letter and cap at 63
#: chars, so a revision is ``rev.<ordinal>`` — a bare ordinal is not a legal name, and the full
#: 64-char digest is both too long and usually digit-leading.
_REVISION_NAME_PREFIX = "rev."

#: Shape of a content digest in a ref fragment — what distinguishes a digest from a tag name.
_DIGEST_FRAGMENT = re.compile(DIGEST_PATTERN)

#: A tag has to be usable as a ref ``#fragment``, so it is bound by the same charset. Sharing the
#: constant keeps the two from drifting into a state where a tag is mintable but unreferenceable.
_TAG_NAME = re.compile(REF_FRAGMENT_CHARSET)

HeadT = TypeVar("HeadT", TaskEntity, TasksetEntity)
RevisionT = TypeVar("RevisionT", TaskRevisionEntity, TasksetRevisionEntity)
_EntityT = TypeVar("_EntityT", bound=EntityBase)


class HeadStoreProtocol(EntityGetterProtocol[_EntityT], EntityUpdateClientProtocol[_EntityT], Protocol[_EntityT]):
    """The head-record surface: read it, then write its pointers under the optimistic lock.

    Composed from the shared protocols rather than restated — ``EntityUpdateClientProtocol`` is
    deliberately separate from the CRUD one precisely so callers needing ``update`` can add it.
    """


#: The revision-child surface: create a revision, fetch one by ``(name, parent)``, query by digest.
#: The shared CRUD protocol covers all three, parameterised on the revision type rather than erased
#: to "any entity", so a wrong entity type at a call site is a type error rather than a cast.
RevisionStoreProtocol = EntityClientProtocol


class RevisionNotFoundError(LookupError):
    """A ref names a revision — by tag or digest — that the record does not have."""


class RevisionConflictError(RuntimeError):
    """Ordinal allocation lost too many races to concurrent publishers."""


class RevisionContentMismatchError(RuntimeError):
    """A stored revision's content no longer hashes to the digest recorded with it.

    Corruption rather than absence: the revision resolved, but serving it would break the promise
    a pinned reference makes, so it is refused instead.
    """


def revision_name(ordinal: int) -> str:
    """Entity name for a revision ordinal."""
    return f"{_REVISION_NAME_PREFIX}{ordinal}"


def head_digest(head: EntityBase) -> str:
    """Digest of a head record's content, excluding its revision bookkeeping.

    The exclusion is what makes this comparable to a revision's own digest: pointers describe
    *which* content is current, not what the content is.
    """
    return content_hash(head, exclude=REVISION_POINTER_EXCLUDE)


def validate_tag_name(tag: str) -> str:
    """Reject tag names that could be stored but never resolved.

    - **Empty**: an absent fragment already means ``latest``, so an empty tag is unaddressable.
    - **Outside the fragment charset** (a slash, a space): a ref's ``#fragment`` admits only
      ``[\\w\\-.]+``, so such a tag could be applied and listed but never written into a member
      reference — the one thing tags exist for.
    - **Digest-shaped** (64 lowercase hex): resolution checks the digest form *first*, so such a
      tag would be queried as a content hash and the tag map never consulted.

    Each is silently useless rather than obviously wrong, which is why they are refused here
    rather than left to surprise someone later. ``latest`` is *not* rejected: whether it is
    acceptable depends on the operation, so that check lives with the caller.
    """
    if not tag.strip():
        raise ValueError("tag name must not be empty")
    if not _TAG_NAME.fullmatch(tag):
        raise ValueError(
            f"tag {tag!r} contains characters that cannot appear in a reference fragment; such a "
            "tag could be stored but never used to pin a member, because a ref's '#fragment' "
            "admits only letters, digits, underscores, hyphens and dots"
        )
    if is_digest(tag):
        raise ValueError(
            f"tag {tag!r} looks like a content digest; such a tag could be stored but never "
            "resolved, because a digest-shaped reference is looked up as a digest, not as a tag"
        )
    return tag


def validate_movable_tag(tag: str) -> str:
    """Validate a tag a caller is asking to *point somewhere* by hand.

    Adds the ``latest`` restriction on top of :func:`validate_tag_name`: ``latest`` is
    machine-managed and only ever advances on publish, so moving it manually would break the
    forward-only guarantee that keeps concurrent publishes consistent.
    """
    if tag == LATEST_TAG:
        raise ValueError(
            f"{LATEST_TAG!r} is managed automatically and always names the most recently published "
            "revision; it cannot be moved by hand"
        )
    return validate_tag_name(tag)


def is_digest(fragment: str) -> bool:
    """Whether a ref fragment is a content digest rather than a tag.

    Unambiguous by construction: a digest is exactly 64 lowercase hex chars, and tag names are
    bound to the same charset a fragment allows, which cannot produce that shape by accident.

    ``fullmatch`` rather than ``match``: Python's ``$`` also matches before a trailing newline, so
    ``match`` would classify a 64-hex string with a newline glued on as a digest and send it down
    the query path instead of the tag path.
    """
    return bool(_DIGEST_FRAGMENT.fullmatch(fragment))


async def find_by_digest(
    entity_client: RevisionStoreProtocol[RevisionT],
    revision_type: type[RevisionT],
    head: TaskEntity | TasksetEntity,
    digest: str,
) -> RevisionT | None:
    """Find this record's newest revision with a given content digest, or ``None``.

    Scoped by parent as well as digest: identical content published under two different records
    yields the same digest, so the digest alone does not identify a revision.

    One record *can* hold two revisions with the same digest — reverting to earlier content
    republishes it — so this orders newest-first rather than taking whichever row came back. The
    two are interchangeable in content, which is all a digest pin promises, but a pin should not
    resolve to a different ordinal from one call to the next.
    """
    result = await entity_client.list(
        revision_type,
        workspace=head.workspace,
        filter_operation=LogicalOperation(
            operator=FilterOperator.AND,
            operations=[
                ComparisonOperation(field="parent", operator=FilterOperator.EQ, value=head.id),
                ComparisonOperation(field="data.content_hash", operator=FilterOperator.EQ, value=digest),
            ],
        ),
        sort="-created_at",
        page_size=1,
    )
    return result.data[0] if result.data else None


async def _revision_at(
    revision_client: RevisionStoreProtocol[RevisionT],
    revision_type: type[RevisionT],
    head: TaskEntity | TasksetEntity,
    ordinal: int,
) -> RevisionT | None:
    """This record's revision ``ordinal``, or ``None`` if no such child exists.

    A direct child lookup rather than a query: a revision's identity within its parent is its
    ordinal, so the name is already known.
    """
    try:
        return await revision_client.get(
            revision_type, name=revision_name(ordinal), workspace=head.workspace, parent=head.id
        )
    except NemoEntityNotFoundError:
        return None


async def _current_revision(
    revision_client: RevisionStoreProtocol[RevisionT],
    revision_type: type[RevisionT],
    head: TaskEntity | TasksetEntity,
) -> RevisionT | None:
    """The revision ``latest`` names, or ``None`` if this record has none yet.

    ``None`` also covers the record whose ``latest`` names a revision that is missing: publishing
    is then the recovery, not an error to propagate.
    """
    ordinal = head.tags.get(LATEST_TAG)
    if ordinal is None:
        return None
    return await _revision_at(revision_client, revision_type, head, ordinal)


async def get_revision(
    entity_client: RevisionStoreProtocol[RevisionT],
    revision_type: type[RevisionT],
    head: TaskEntity | TasksetEntity,
    fragment: str = LATEST_TAG,
) -> RevisionT:
    """Fetch the revision a ref fragment names, verifying its content on the way out.

    A tag resolves through ``tags`` to an ordinal and then a direct child lookup — no query. A
    digest is queried by ``(parent, content_hash)``. Either way the content that comes back is
    re-hashed and compared against the digest stored beside it, because a pinned ref is a claim
    about what the consumer will get and this is the one place that claim can be checked. Without
    it the digest is only a label: a revision mutated in place would keep resolving and quietly
    serve content nobody pinned.
    """
    if is_digest(fragment):
        revision = await find_by_digest(entity_client, revision_type, head, fragment)
        if revision is None:
            raise RevisionNotFoundError(f"'{head.workspace}/{head.name}' has no revision with digest {fragment!r}")
        _verify_content(head, revision)
        return revision

    ordinal = head.tags.get(fragment)
    if ordinal is None:
        known = ", ".join(sorted(head.tags)) or "none"
        raise RevisionNotFoundError(
            f"'{head.workspace}/{head.name}' has no revision tagged {fragment!r} (known tags: {known})"
        )
    try:
        revision = await entity_client.get(
            revision_type, name=revision_name(ordinal), workspace=head.workspace, parent=head.id
        )
    except NemoEntityNotFoundError as exc:
        raise RevisionNotFoundError(
            f"'{head.workspace}/{head.name}' tags {fragment!r} as revision {ordinal}, but that "
            "revision record is missing"
        ) from exc
    _verify_content(head, revision)
    return revision


def _verify_content(head: TaskEntity | TasksetEntity, revision: TaskRevisionEntity | TasksetRevisionEntity) -> None:
    """Re-hash a revision's content and check it against the digest stored alongside it.

    A revision is immutable by convention, not by enforcement — the store will happily accept a
    write to one. This turns that convention into something detectable rather than something the
    reader has to assume.
    """
    actual = content_hash(revision, exclude=REVISION_SELF_EXCLUDE)
    if actual != revision.content_hash:
        raise RevisionContentMismatchError(
            f"revision {revision.revision} of '{head.workspace}/{head.name}' does not match its "
            f"recorded digest (recorded {revision.content_hash}, actual {actual}); the stored "
            "content has been modified since it was published"
        )


async def list_revisions(
    entity_client: RevisionStoreProtocol[RevisionT],
    revision_type: type[RevisionT],
    head: TaskEntity | TasksetEntity,
    *,
    page: int = 1,
    page_size: int = 100,
) -> ListResponse[RevisionT]:
    """Return a page of a record's published revisions, newest first.

    Scoped by parent, so it returns this record's history rather than every revision of every
    record in the workspace.

    Ordering is server-side on ``-created_at`` rather than sorted client-side by ordinal: sorting
    one page locally would order *within* the page while paging silently returned an arbitrary
    slice. Creation order matches ordinal order because ordinal allocation is serialized by the
    child create, so the two agree.

    Returns the store's ``ListResponse`` — including pagination counts — so a caller can tell a
    complete history from a truncated one. Returning a bare list would make a capped result
    indistinguishable from the whole thing.
    """
    return await entity_client.list(
        revision_type,
        workspace=head.workspace,
        filter_operation=ComparisonOperation(field="parent", operator=FilterOperator.EQ, value=head.id),
        sort="-created_at",
        page=page,
        page_size=page_size,
    )


async def apply_tag(
    head_client: HeadStoreProtocol[HeadT],
    revision_client: RevisionStoreProtocol[RevisionT],
    revision_type: type[RevisionT],
    head: HeadT,
    tag: str,
    fragment: str,
) -> HeadT:
    """Point an existing tag at the revision ``fragment`` names.

    Separate from publishing because a tag is not always applied at publish time — blessing a
    revision after it has been evaluated is the common case, and Harbor exposes the same operation
    (``tag_package_version``) independently of publish.

    ``latest`` is refused: it is machine-managed and moves only forward, on publish. Letting a
    caller point it at an arbitrary revision would make it disagree with ``latest_revision`` and
    break the forward-only guarantee that keeps concurrent publishes consistent.
    """
    validate_movable_tag(tag)
    revision = await get_revision(revision_client, revision_type, head, fragment)
    return await _point_tags(head_client, head, {tag}, revision.revision)


async def publish_revision(
    head_client: HeadStoreProtocol[HeadT],
    revision_client: RevisionStoreProtocol[RevisionT],
    head: HeadT,
    revision_type: type[RevisionT],
    *,
    tags: set[str] | None = None,
) -> tuple[RevisionT, HeadT, bool]:
    """Freeze a head record's current content as a revision and point tags at it.

    Returns ``(revision, head, created)``. The head is returned already carrying the new pointers,
    so callers do not have to re-read it. ``created`` is ``False`` when the content was already
    published — republishing identical content is a no-op that still applies any newly requested
    tags, which is what makes a re-publish cheap and idempotent.

    ``latest`` is always applied, on top of any caller-supplied tags.
    """
    # ``latest`` is applied regardless, so a caller who lists it explicitly is asking for what
    # already happens — tolerated rather than rejected. Only *moving* latest by hand is refused
    # (see :func:`validate_movable_tag`).
    applied = {LATEST_TAG} | {validate_tag_name(tag) for tag in tags or set() if tag != LATEST_TAG}

    for attempt in range(_MAX_ALLOCATION_ATTEMPTS):
        digest = head_digest(head)

        # Dedup against the revision ``latest`` names, *not* against any revision ever published.
        # Matching any would make reverting to older content a no-op: the head would be rewritten
        # with that content while ``latest`` — which only moves forward — kept naming the newer
        # revision, so a plain read and a ``#latest`` read would disagree. A revert is a genuine
        # change to what this record currently is, so it publishes.
        current = await _current_revision(revision_client, revision_type, head)
        if current is not None and current.content_hash == digest:
            if any(head.tags.get(tag) != current.revision for tag in applied):
                head = await _point_tags(head_client, head, applied, current.revision)
            return current, head, False

        ordinal = head.latest_revision + 1
        # Copy *all* content, including fields the digest excludes: a revision stores the full
        # published spec, and only its hash ignores the derived parts.
        content = head.model_dump(exclude=set(REVISION_POINTER_FIELDS) | set(head.__base_fields__), mode="json")
        revision = revision_type(
            name=revision_name(ordinal),
            workspace=head.workspace,
            project=head.project,
            content_hash=digest,
            revision=ordinal,
            **content,
        )
        revision._parent = head.id

        try:
            created_revision = await revision_client.create(revision)
        except NemoEntityConflictError:
            # Another publisher took this ordinal. Refresh only the allocation state — re-reading
            # the whole head would replace the content the caller staged for publishing with
            # whatever is currently stored, and we'd publish the wrong thing.
            logger.info(
                "Revision ordinal contended, retrying",
                extra={"record": f"{head.workspace}/{head.name}", "ordinal": ordinal, "attempt": attempt + 1},
            )
            _refresh_pointers(head, await head_client.get(type(head), name=head.name, workspace=head.workspace))

            # The winner may have published exactly what we are publishing. Two identical requests
            # that overlap both compute this digest; the dedup check above missed it only because
            # the winner's pointer write had not landed when we read ``latest``. Stepping past their
            # ordinal here would cut a byte-identical duplicate and report ``created`` to both
            # callers, so idempotency has to be re-checked against the revision that actually beat
            # us rather than against a head that may still be stale.
            contended = await _revision_at(revision_client, revision_type, head, ordinal)
            if contended is not None and contended.content_hash == digest:
                if any(head.tags.get(tag) != contended.revision for tag in applied):
                    head = await _point_tags(head_client, head, applied, contended.revision)
                return contended, head, False

            # The refresh earlier adopts the *stored* allocation state, which does not necessarily
            # account for the ordinal we just lost: if the winner's own pointer write never landed,
            # the stored head still names N-1 and we would recompute N and lose again, every attempt
            # until the retries run out — and every later publish would do the same, leaving the
            # record permanently unpublishable. The failed create is itself proof that N is taken,
            # so step past it explicitly rather than trusting the head to say so.
            head.latest_revision = max(head.latest_revision, ordinal)
            continue

        # If the pointer write below exhausts its retries, the revision child still exists while
        # `latest_revision` never advanced. That is safe but wasteful: the next publish computes the
        # same ordinal, loses the create, and — via the step-past above — retries onto N+1. It
        # self-heals rather than corrupting anything, so it is left alone deliberately.
        head = await _point_tags(head_client, head, applied, ordinal)
        return created_revision, head, True

    raise RevisionConflictError(
        f"could not allocate a revision ordinal for '{head.workspace}/{head.name}' after "
        f"{_MAX_ALLOCATION_ATTEMPTS} attempts"
    )


def _refresh_pointers(head: HeadT, fresh: HeadT) -> HeadT:
    """Adopt another writer's revision bookkeeping without touching our content.

    A publish carries the caller's staged content; only the allocation state (which ordinals exist
    and the version we're writing against) can go stale underneath it. So a retry refreshes exactly
    that and leaves the content alone.
    """
    head.latest_revision = fresh.latest_revision
    head.tags = fresh.tags
    head._db_version = fresh._db_version
    return head


def _apply_pointers(head: HeadT, tags: set[str], ordinal: int) -> HeadT:
    """Fold one publish's pointers into a head record (pure; no I/O).

    ``latest`` moves **forward only**. Two publishers can create their revisions in one order and
    reach the head update in the other; without this guard the second writer's ``latest`` wins even
    though it names the older revision, leaving ``latest`` and ``latest_revision`` disagreeing.
    ``latest`` is machine-managed, so forward-only is the right rule. User tags are explicit intent
    and may legitimately be moved backwards — retagging ``blessed`` onto an older revision is a
    rollback, not a race.
    """
    moves_forward = ordinal >= head.tags.get(LATEST_TAG, 0)
    applied = {tag: ordinal for tag in tags if tag != LATEST_TAG or moves_forward}
    head.tags = {**head.tags, **applied}
    head.latest_revision = max(head.latest_revision, ordinal)
    return head


async def _point_tags(
    head_client: HeadStoreProtocol[HeadT],
    head: HeadT,
    tags: set[str],
    ordinal: int,
) -> HeadT:
    """Point tags at a revision ordinal, under the head's optimistic lock.

    The read-modify-write is retried on conflict rather than propagated: losing this race means
    another publisher updated the head between our read and our write, and their pointers must be
    preserved — so we re-read *their* head and fold our pointers into it, instead of overwriting
    with a copy that predates them.

    Which content survives depends on who is newer. If the revision we are pointing at is the
    newest, the head takes our content. If the winner published a *later* revision, the head has to
    keep theirs: writing ours would leave the head serving one revision's content while ``latest``
    named another, which is the same divergence a plain read and a ``#latest`` read must never show.
    Our non-``latest`` tags still apply either way — pointing ``blessed`` at an older revision is
    legitimate; dragging the head's content back with it is not.
    """
    for attempt in range(_MAX_ALLOCATION_ATTEMPTS):
        try:
            return await head_client.update(_apply_pointers(head, tags, ordinal))
        except NemoEntityConflictError:
            logger.info(
                "Head pointer update contended, re-reading",
                extra={"record": f"{head.workspace}/{head.name}", "attempt": attempt + 1},
            )
            fresh = await head_client.get(type(head), name=head.name, workspace=head.workspace)
            if ordinal < fresh.tags.get(LATEST_TAG, 0):
                head = fresh
            else:
                _refresh_pointers(head, fresh)
    raise RevisionConflictError(
        f"could not update revision pointers for '{head.workspace}/{head.name}' after "
        f"{_MAX_ALLOCATION_ATTEMPTS} attempts"
    )
