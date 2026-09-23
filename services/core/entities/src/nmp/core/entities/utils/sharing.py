# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry of entity types shared out of the global workspace.

The ``default`` workspace doubles as the installation-wide GLOBAL workspace: entities
of a shared type that live there are readable from every workspace, so teams do not
duplicate expensive resources (deployed models, LoRA adapters, datasets) per workspace.

Sharing is opt-in per entity type. Read access widens only for the types listed here;
everything else keeps strict per-workspace isolation. Writes are never widened — a
mutation in ``default`` still requires access to ``default``.

Phase 0: hardcoded registry. Phase 1: dynamic registration via entity type schemas.
"""

from nmp.common.entities.global_workspace import GLOBAL_WORKSPACE as GLOBAL_WORKSPACE

_GLOBALLY_SHAREABLE: frozenset[str] = frozenset(
    {
        "model",
        "adapter",
        "model_provider",
        "virtual_model",
    }
)


def is_globally_shareable(entity_type: str | None) -> bool:
    """True when *entity_type* is readable from any workspace once it lives in ``default``."""
    return entity_type is not None and entity_type in _GLOBALLY_SHAREABLE
