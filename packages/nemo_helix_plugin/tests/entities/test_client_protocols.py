# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The entity-client protocols must describe the client they stand in for.

A protocol that has drifted from its implementation is worse than no protocol: a service typed
against it either fails type-checking on correct code, or type-checks against a method the real
client does not have. These tests pin that relationship so the two cannot silently diverge.
"""

from __future__ import annotations

import inspect
from typing import Protocol, TypeVar

from nemo_helix_plugin.entities import (
    EntityBase,
    EntityClient,
    EntityClientProtocol,
    EntityGetterProtocol,
    EntityUpdateClientProtocol,
)

EntityT = TypeVar("EntityT", bound=EntityBase)


class _Entity(EntityBase):
    __entity_type__ = "protocol_conformance_probe"


class _ReadWriteStore(
    EntityClientProtocol[EntityT],
    EntityUpdateClientProtocol[EntityT],
    Protocol[EntityT],
):
    """The shape a service needing the wider surface composes for itself.

    Exists here to prove the pieces *compose*: ``update`` is a separate protocol precisely so a
    service can opt into it alongside the CRUD one, rather than declaring a private protocol that
    restates the whole surface.
    """


def _static_conformance(client: EntityClient) -> _ReadWriteStore[_Entity]:
    """Static assertion, checked by ``ty`` rather than at runtime.

    If ``EntityClient`` ever stops satisfying the composed protocols — a renamed method, a changed
    signature — this return fails type-checking. The runtime tests below document *which* parts
    matter and why; this is what actually catches drift, because structural conformance is a
    type-level property no ``hasattr`` check can verify.
    """
    return client


def _signature(owner: object, method: str) -> inspect.Signature:
    return inspect.signature(getattr(owner, method))


#: Parameters the client has that a protocol deliberately does not expose, with the reason. Anything
#: not listed here is treated as accidental under-specification by the signature check below.
_INTENTIONAL_OMISSIONS = {
    # ``filter_operation`` is the sanctioned structured form; these two are a JSON-string variant
    # and an exact-match shorthand kept for older callers.
    ("list", "filter_str"),
    ("list", "filter_obj"),
}


def test_protocols_expose_every_client_parameter() -> None:
    """Catch *under*-specification, which conformance alone cannot.

    Structural conformance is one-directional: a class satisfies a protocol by providing at least
    what it declares, so extra parameters on the client pass silently. That is how ``parent`` went
    missing from ``delete`` while ``_static_conformance`` reported success — a service typed against
    the protocol could not delete a child entity even though its client could.
    """
    gaps: dict[str, set[str]] = {}
    for method in ("get", "create", "update", "delete", "list"):
        protocol = EntityUpdateClientProtocol if method == "update" else EntityClientProtocol
        declared = set(_signature(protocol, method).parameters)
        available = set(_signature(EntityClient, method).parameters)
        missing = {p for p in available - declared if (method, p) not in _INTENTIONAL_OMISSIONS}
        if missing:
            gaps[method] = missing
    assert not gaps, f"protocol is missing client parameters: {gaps}"


def test_update_is_its_own_protocol() -> None:
    """``update`` is opt-in. Most services never modify an entity in place, and folding ``update``
    into the CRUD protocol would force each of them — and every one of their test doubles — to
    satisfy a method they do not use."""
    assert hasattr(EntityUpdateClientProtocol, "update")
    assert not hasattr(EntityClientProtocol, "update")


def test_update_protocol_matches_the_client() -> None:
    protocol = _signature(EntityUpdateClientProtocol, "update").parameters
    client = _signature(EntityClient, "update").parameters
    assert set(protocol) == set(client)
    assert protocol["original_name"].kind is inspect.Parameter.KEYWORD_ONLY


def test_getter_accepts_parent_for_child_entities() -> None:
    """Child records are unique within ``(workspace, entity_type, parent, name)``, so the parent is
    part of their address. It is optional, so fetching a root entity is unchanged."""
    getter = _signature(EntityGetterProtocol, "get").parameters
    assert "parent" in getter
    assert getter["parent"].default is None
    assert "parent" in _signature(EntityClient, "get").parameters
