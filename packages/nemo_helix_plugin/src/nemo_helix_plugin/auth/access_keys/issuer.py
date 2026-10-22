# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Protocol

from nemo_helix_plugin.auth.access_keys.types import (
    AccessKeyCreateRequest,
    AccessKeyCreateResponse,
    AccessKeyListResponse,
    AccessKeyRevokeResponse,
    AccessKeyRotateResponse,
    AccessKeyStatusChangeResponse,
)
from nemo_helix_plugin.client.errors import ConflictError, PermissionDeniedError


class AccessKeyOperationNotImplementedError(RuntimeError):
    """Raised when a Scoped Access Key lifecycle operation is not implemented by the selected issuer."""


class AccessKeyFeatureDisabledError(RuntimeError):
    """Raised when Scoped Access Keys are disabled in platform config."""


# Ends the detail of a 409 workspace_bound_keys_exist response, for API clients; the CLI asks instead.
WORKSPACE_BOUND_KEYS_HINT = "Set acknowledge_workspace_bound_keys to confirm."


class AccessKeyWorkspaceBoundKeysError(ConflictError):
    """Raised when creating a service-bound key needs acknowledge_workspace_bound_keys.

    The service account already has live keys bound to workspaces, and the new key would grant the
    account a role it doesn't hold yet, which those keys would get too; ``detail`` names them. Resend the request with acknowledge_workspace_bound_keys=True to
    proceed. Still a ConflictError, so code that handles the API's 409s (or any NemoHTTPError)
    keeps handling it.
    """


class AccessKeyBoundWorkspacesNotAdministeredError(PermissionDeniedError):
    """Raised when managing a service-bound key bound to workspaces the caller isn't Admin of.

    The caller is Admin of some of the key's bound workspaces, but not all of them, so the key was
    left unchanged; ``detail`` names the others. Still a PermissionDeniedError, so code that handles
    the API's 403s (or any NemoHTTPError) keeps handling it.
    """


class AccessKeyIssuer(Protocol):
    """Scoped Access Key implementation interface shared by service and client implementations."""

    def create(self, request: AccessKeyCreateRequest) -> AccessKeyCreateResponse: ...

    def list(self, *, page: int = 1, page_size: int = 100) -> AccessKeyListResponse: ...

    def revoke(self, jti: str) -> AccessKeyRevokeResponse: ...

    def suspend(self, jti: str) -> AccessKeyStatusChangeResponse: ...

    def unsuspend(self, jti: str) -> AccessKeyStatusChangeResponse: ...

    def rotate(self, jti: str, *, grace_period_seconds: int | None = None) -> AccessKeyRotateResponse: ...
