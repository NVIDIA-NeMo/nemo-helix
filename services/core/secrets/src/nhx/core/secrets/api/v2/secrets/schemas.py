# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Secrets API schemas.

The request/response contract lives in ``nemo_helix_plugin.secrets.types`` —
the shared single source of truth imported by both the server (here) and the
typed ``SecretsClient``. This module re-exports those models and adds the
server-only ``from_entity`` conversion helpers.
"""

from __future__ import annotations

from nemo_helix_plugin.secrets.types import (
    PlatformSecretAccessResponse as PlatformSecretAccessResponse,
)
from nemo_helix_plugin.secrets.types import (
    PlatformSecretCreateRequest as PlatformSecretCreateRequest,
)
from nemo_helix_plugin.secrets.types import (
    PlatformSecretResponse as _PlatformSecretResponse,
)
from nemo_helix_plugin.secrets.types import (
    PlatformSecretUpdateRequest as PlatformSecretUpdateRequest,
)
from nhx.core.secrets.entities import PlatformSecret


class PlatformSecretResponse(_PlatformSecretResponse):
    """Response model for a platform secret."""

    # NB: the docstring mirrors the base model's on purpose — FastAPI uses the
    # response_model's docstring as the OpenAPI schema description, so a
    # server-only note here would leak into the public API spec. This subclass
    # exists solely to add ``from_entity``.

    @classmethod
    def from_entity(cls, secret: PlatformSecret) -> PlatformSecretResponse:
        """Create a PlatformSecretResponse from a PlatformSecret entity."""
        return cls(
            name=secret.name,
            workspace=secret.workspace,
            description=secret.description,
            created_at=secret.created_at,
            updated_at=secret.updated_at,
        )
