# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entity types readable from every workspace when they live in the global (``default``) workspace."""

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
