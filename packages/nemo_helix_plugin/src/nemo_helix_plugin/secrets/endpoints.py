# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed endpoint definitions for the Secrets service.

These are the single source of truth for the HTTP contract. Paths include the
``/apis/secrets`` gateway prefix.

The service exposes two routers:
- the ``secrets`` CRUD + access router (workspace-scoped), and
- the privileged ``admin`` router (``rotate-encryption-keys``), which the server
  gates on the ``secrets.rotate`` permission in the ``system`` workspace.
"""

from __future__ import annotations

from abc import abstractmethod

from nemo_helix_plugin.client.endpoint import delete, get, patch, post
from nemo_helix_plugin.client.types import Paginated, PreparedRequest
from nemo_helix_plugin.secrets.types import (
    HelixSecretAccessResponse,
    HelixSecretAdminRotationResponse,
    HelixSecretCreateRequest,
    HelixSecretResponse,
    HelixSecretUpdateRequest,
    ListSecretsQueryParams,
)

# ---------------------------------------------------------------------------
# Secret CRUD
# ---------------------------------------------------------------------------


@get("/apis/secrets/v2/workspaces/{workspace}/secrets/{name}")
@abstractmethod
def get_secret(*, workspace: str | None = None, name: str) -> HelixSecretResponse: ...


@get("/apis/secrets/v2/workspaces/{workspace}/secrets")
@abstractmethod
def list_secrets(
    *, workspace: str | None = None, query_params: ListSecretsQueryParams | None = None
) -> Paginated[HelixSecretResponse]: ...


def _get_secret_on_conflict(
    body: HelixSecretCreateRequest, workspace: str | None
) -> PreparedRequest[HelixSecretResponse]:
    """Build the retrieve request replayed when ``create_secret(exist_ok=True)`` 409s."""
    return get_secret(name=body.name, workspace=workspace)


@post("/apis/secrets/v2/workspaces/{workspace}/secrets", get_on_conflict=_get_secret_on_conflict)
@abstractmethod
def create_secret(
    *, workspace: str | None = None, body: HelixSecretCreateRequest, exist_ok: bool = False
) -> HelixSecretResponse: ...


@patch("/apis/secrets/v2/workspaces/{workspace}/secrets/{name}")
@abstractmethod
def update_secret(
    *, workspace: str | None = None, name: str, body: HelixSecretUpdateRequest
) -> HelixSecretResponse: ...


@delete("/apis/secrets/v2/workspaces/{workspace}/secrets/{name}")
@abstractmethod
def delete_secret(*, workspace: str | None = None, name: str) -> None: ...


# ---------------------------------------------------------------------------
# Secret value access
# ---------------------------------------------------------------------------


@get("/apis/secrets/v2/workspaces/{workspace}/secrets/{name}/access")
@abstractmethod
def access_secret(*, workspace: str | None = None, name: str) -> HelixSecretAccessResponse: ...


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------


@post("/apis/secrets/v2/rotate-encryption-keys")
@abstractmethod
def rotate_encryption_keys() -> HelixSecretAdminRotationResponse: ...
