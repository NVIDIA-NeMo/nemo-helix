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
    HelixSecretAccessResponse as HelixSecretAccessResponse,
)
from nemo_helix_plugin.secrets.types import (
    HelixSecretCreateRequest as HelixSecretCreateRequest,
)
from nemo_helix_plugin.secrets.types import (
    HelixSecretResponse as _HelixSecretResponse,
)
from nemo_helix_plugin.secrets.types import (
    HelixSecretUpdateRequest as HelixSecretUpdateRequest,
)
from nhx.core.secrets.entities import HelixSecret


class HelixSecretResponse(_HelixSecretResponse):
    """Response model for a platform secret."""

    # NB: the docstring mirrors the base model's on purpose — FastAPI uses the
    # response_model's docstring as the OpenAPI schema description, so a
    # server-only note here would leak into the public API spec. This subclass
    # exists solely to add ``from_entity``.

    @classmethod
    def from_entity(cls, secret: HelixSecret) -> HelixSecretResponse:
        """Create a HelixSecretResponse from a HelixSecret entity."""
        return cls(
            name=secret.name,
            workspace=secret.workspace,
            description=secret.description,
            created_at=secret.created_at,
            updated_at=secret.updated_at,
        )
