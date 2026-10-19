# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example entity definitions — stored in the NeMo Helix entity store.

This module contains only entity classes (subclasses of
:class:`~nemo_helix_plugin.entity.NemoEntity`).  API request/response schemas live
in :mod:`nemo_example_plugin.schema` — keep the two concerns separate.
"""

from __future__ import annotations

from nemo_helix_plugin.entity import NemoEntity


class ExampleItem(NemoEntity, entity_type="example_item"):
    """A simple example entity that demonstrates the :class:`~nemo_helix_plugin.entity.NemoEntity` pattern.

    Stored in the entity store under entity type ``"example_item"``.
    The ``name`` and ``workspace`` fields are inherited from ``EntityBase``
    and are managed by the entity store.  Only domain-specific fields are
    declared here.
    """

    title: str
    body: str = ""
    tags: list[str] = []
