# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference syntax for revisioned entities: ``workspace/name#<tag-or-digest>``.

Shared by every plugin that publishes immutable revisions (evals tasks and tasksets, agents
skills), so a ref means the same thing everywhere. What counts as a ``workspace/name`` is the
platform's (:data:`~nemo_helix_plugin.refs.ENTITY_REF_PATTERN`); this module adds only the
``#fragment`` that selects a revision — a tag (``#latest``, ``#candidate``) or a full 64-char
content digest — and the rules for minting tags that can appear in one.
"""

from __future__ import annotations

import re

from nemo_helix_plugin.content_hash import DIGEST_PATTERN
from nemo_helix_plugin.refs import ENTITY_REF_PATTERN, parse_entity_ref

#: The charset a ``#fragment`` may use. Exported because anything that *mints* a fragment — notably
#: revision tag names — has to be constrained by it: a value outside this set can be stored happily
#: and then never appear in a reference, which is a silent dead end rather than an error.
REF_FRAGMENT_CHARSET = r"[\w\-.]+"

#: A *sub-entity* reference adds an optional ``#fragment`` to the platform's ``ENTITY_REF_PATTERN``,
#: the standard way of addressing something contained within an entity (filesets address a contained
#: file the same way: ``workspace/fileset#path``). For a revisioned entity the fragment selects a
#: revision.
#:
#: Deliberately a sibling of ``ENTITY_REF_PATTERN`` rather than a widening of it: that constant is
#: shared by refs to entities without revisions, and admitting a fragment there would accept input
#: nothing is built to resolve. The base alternation is spliced in from the shared constant (minus
#: its anchors) so the two shapes cannot drift.
SUBENTITY_REF_PATTERN = rf"^{ENTITY_REF_PATTERN.removeprefix('^').removesuffix('$')}(#{REF_FRAGMENT_CHARSET})?$"

#: A sub-entity ref whose fragment is a full content digest — the only form a published snapshot may
#: store, because tags move.
PINNED_SUBENTITY_REF_PATTERN = rf"^{ENTITY_REF_PATTERN.removeprefix('^').removesuffix('$')}#[0-9a-f]{{64}}$"

#: The fragment separator for sub-entity references. Matches the fileset/job ref convention.
REF_FRAGMENT_SEPARATOR = "#"

#: The tag applied to every publish and used when a ref carries no fragment.
LATEST_TAG = "latest"

#: Shape of a content digest in a ref fragment — what distinguishes a digest from a tag name.
_DIGEST_FRAGMENT = re.compile(DIGEST_PATTERN)

#: A tag has to be usable as a ref ``#fragment``, so it is bound by the same charset.
_TAG_NAME = re.compile(REF_FRAGMENT_CHARSET)


def parse_subentity_ref(root: str, default_workspace: str) -> tuple[str, str, str]:
    """Split a reference into ``(workspace, name, fragment)``.

    The ``workspace/name`` split is delegated to :func:`~nemo_helix_plugin.refs.parse_entity_ref`;
    this only adds the revision fragment on top. Callers that don't care about revisions discard the
    third element — that, rather than a second parser, is how a pinned ref is read unpinned.

    An absent fragment resolves to :data:`LATEST_TAG` — a bare ``workspace/name`` means "the current
    revision", never "unpinned". The fragment is returned verbatim: it may be a tag or a content
    digest, and telling them apart is resolution's job, not parsing's.
    """
    base, separator, fragment = root.partition(REF_FRAGMENT_SEPARATOR)
    parsed = parse_entity_ref(base, default_workspace)
    return parsed.workspace, parsed.name, fragment if separator and fragment else LATEST_TAG


def is_digest(fragment: str) -> bool:
    """Whether a ref fragment is a content digest rather than a tag.

    Unambiguous by construction: a digest is exactly 64 lowercase hex chars, and tag names are
    bound to the same charset a fragment allows, which cannot produce that shape by accident.

    ``fullmatch`` rather than ``match``: Python's ``$`` also matches before a trailing newline, so
    ``match`` would classify a 64-hex string with a newline glued on as a digest and send it down
    the query path instead of the tag path.
    """
    return bool(_DIGEST_FRAGMENT.fullmatch(fragment))


def validate_tag_name(tag: str) -> str:
    """Reject tag names that could be stored but never resolved.

    - **Empty**: an absent fragment already means ``latest``, so an empty tag is unaddressable.
    - **Outside the fragment charset** (a slash, a space): a ref's ``#fragment`` admits only
      ``[\\w\\-.]+``, so such a tag could be applied and listed but never written into a reference.
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
