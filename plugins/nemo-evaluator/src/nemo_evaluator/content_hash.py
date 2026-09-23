# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical content hashing for revisioned entities (tasks and tasksets).

A revision is addressed by a digest of its *content* — the same projection the entity store
persists (``model_dump(exclude=__base_fields__)``), serialized canonically and hashed with SHA-256.
Two independent computations of the same content must agree, because a consumer reading a pinned
ref (``workspace/task-name#<digest>``) recomputes the digest and compares.

Three properties this deliberately has:

**Content only, never identity.** The name, workspace, and revision index are *not* inputs. Salting
the digest with them would break the two things it exists for: republishing identical content would
produce a new digest (defeating publish-time dedup), and verify-on-read would pass whenever
identity matched, regardless of whether the content beneath it had changed. Identity is carried by
the ref, which already names the entity; a cross-entity digest collision can't cause a
misresolution because lookups are scoped by name before the digest is consulted.

**Full digest, never truncated.** SHA-256's collision resistance is bounded at 2**128 by the
birthday paradox, which is ample. A truncated prefix is not: 12 hex chars is 48 bits, with a
birthday bound near 2**24 — reachable by accident at scale. Short forms belong in display and
prefix *matching* against stored full digests (git's model), never in the stored value.

**Bare hex, no algorithm prefix.** Matches the existing derived-metric digest idiom in
``metric_service`` and keeps the ``#`` ref fragment free of ``:``, which the entity ref charset
does not admit. See ``docs`` in the backend design for the algorithm-agility tradeoff.

The weak link is not SHA-256 — it is *canonicalization*. If two semantically different revisions
serialize identically they collide with probability 1 and no hash strength helps. Hence the
explicit rules below, and the near-miss tests that pin them.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Set
from typing import Any

from nemo_helix_plugin.entities import EntityBase

#: Length of a SHA-256 digest rendered as lowercase hex.
DIGEST_LENGTH = 64

#: Charset/shape of a digest as it appears in a ``#`` ref fragment.
DIGEST_PATTERN = r"^[0-9a-f]{64}$"


def _as_exclude_map(exclude: Set[str] | Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize either accepted ``exclude`` form to pydantic's dict form."""
    if exclude is None:
        return {}
    if isinstance(exclude, Mapping):
        return {str(name): nested for name, nested in exclude.items()}
    return {str(name): True for name in exclude}


def canonical_payload(entity: EntityBase, *, exclude: Set[str] | Mapping[str, Any] | None = None) -> str:
    """Return the canonical JSON serialization that :func:`content_hash` digests.

    Exposed separately because it is the actual compatibility contract: if this string changes
    shape for unchanged content, every stored digest is invalidated. Tests assert on it directly.

    Canonicalization rules:

    - Server-owned fields are dropped via ``EntityBase.__base_fields__`` — the same exclusion the
      entity store itself uses when persisting custom fields, so the hash input tracks what is
      actually stored rather than a parallel hand-maintained list.
    - ``mode="json"`` normalizes rich types (datetimes, enums, sub-models) to JSON primitives.
    - ``sort_keys=True`` makes mapping order irrelevant.
    - ``separators=(",", ":")`` removes insignificant whitespace.

    Rules this does NOT impose, deliberately:

    - **Sequence order is significant.** A list that is semantically a set must be normalized by
      its own model (as ``TaskRefList`` does) before hashing; this function will not reorder it,
      because it cannot tell a set from an ordered sequence.
    - **``1`` and ``1.0`` hash differently**, per JSON. That is desired: an int and a float are
      distinguishable values, not the same value formatted twice.
    - **Absent and default-valued fields collapse**, because ``model_dump`` materializes defaults.
      A model that needs to distinguish "unset" from "set to the default" must model that
      explicitly (e.g. with an optional field defaulting to ``None``).

    Args:
        entity: The entity whose content to serialize.
        exclude: Extra field names to drop on top of ``__base_fields__``. Revisioned entities pass
            their own revision/tag bookkeeping here — a revision's digest must not cover the
            revision index that was assigned *because of* that digest.
    """
    # Pydantic's dict form lets a caller exclude a *nested* field (``{"spec": {"config"}}``), which
    # a flat set cannot express. Both forms are accepted so simple cases stay simple.
    excluded: dict[str, Any] = {str(name): True for name in entity.__base_fields__}
    excluded.update(_as_exclude_map(exclude))
    payload = entity.model_dump(exclude=excluded, exclude_computed_fields=True, mode="json")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def content_hash(entity: EntityBase, *, exclude: Set[str] | Mapping[str, Any] | None = None) -> str:
    """Return the full 64-char lowercase hex SHA-256 digest of an entity's content.

    See :func:`canonical_payload` for what is and is not included.
    """
    return hashlib.sha256(canonical_payload(entity, exclude=exclude).encode("utf-8")).hexdigest()
