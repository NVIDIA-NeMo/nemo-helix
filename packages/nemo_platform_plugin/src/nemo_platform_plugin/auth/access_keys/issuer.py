# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Protocol

from nemo_platform_plugin.auth.access_keys.types import (
    AccessKeyCreateRequest,
    AccessKeyCreateResponse,
    AccessKeyListResponse,
    AccessKeyRevokeResponse,
    AccessKeyRotateResponse,
    AccessKeyStatusChangeResponse,
)


class AccessKeyOperationNotImplementedError(RuntimeError):
    """Raised when a Scoped Access Key lifecycle operation is not implemented by the selected issuer."""


class AccessKeyFeatureDisabledError(RuntimeError):
    """Raised when Scoped Access Keys are disabled in platform config."""


class AccessKeyIssuer(Protocol):
    """Scoped Access Key implementation interface shared by service and client implementations."""

    def create(self, request: AccessKeyCreateRequest) -> AccessKeyCreateResponse: ...

    def list(self, *, page: int = 1, page_size: int = 100) -> AccessKeyListResponse: ...

    def revoke(self, jti: str) -> AccessKeyRevokeResponse: ...

    def suspend(self, jti: str) -> AccessKeyStatusChangeResponse: ...

    def unsuspend(self, jti: str) -> AccessKeyStatusChangeResponse: ...

    def rotate(self, jti: str, *, grace_period_seconds: int | None = None) -> AccessKeyRotateResponse: ...
