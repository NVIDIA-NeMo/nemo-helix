# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_platform_plugin.auth.access_keys import endpoints
from nemo_platform_plugin.auth.access_keys.issuer import (
    AccessKeyFeatureDisabledError,
    AccessKeyIssuer,
    AccessKeyOperationNotImplementedError,
)
from nemo_platform_plugin.auth.access_keys.types import (
    AccessKeyCreateRequest,
    AccessKeyCreateResponse,
    AccessKeyListResponse,
    AccessKeyRevokeResponse,
    AccessKeyRotateRequest,
    AccessKeyRotateResponse,
    AccessKeyStatusChangeResponse,
)
from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_platform_plugin.client.errors import NemoHTTPError
from nemo_platform_plugin.client.method import method


class _AccessKeyMethods:
    create_access_key = method(endpoints.create_access_key)
    list_access_keys = method(endpoints.list_access_keys)
    revoke_access_key = method(endpoints.revoke_access_key)
    suspend_access_key = method(endpoints.suspend_access_key)
    unsuspend_access_key = method(endpoints.unsuspend_access_key)
    rotate_access_key = method(endpoints.rotate_access_key)


class AccessKeysClient(_AccessKeyMethods, NemoClient):
    """Sync client for the Scoped Access Key API."""


class AsyncAccessKeysClient(_AccessKeyMethods, AsyncNemoClient):
    """Async client for the Scoped Access Key API."""


class AccessKeyIssuerClient(AccessKeyIssuer):
    """AccessKeyIssuer implementation that calls the auth service over HTTP."""

    def __init__(self, client: AccessKeysClient) -> None:
        self._client = client

    def create(self, request: AccessKeyCreateRequest) -> AccessKeyCreateResponse:
        try:
            return self._client.create_access_key(body=request).data()
        except NemoHTTPError as exc:
            _raise_known_domain_error_from_http(exc)
            raise

    def list(self, *, page: int = 1, page_size: int = 100) -> AccessKeyListResponse:
        try:
            return self._client.list_access_keys(query_params={"page": page, "page_size": page_size}).data()
        except NemoHTTPError as exc:
            _raise_known_domain_error_from_http(exc)
            raise

    def revoke(self, jti: str) -> AccessKeyRevokeResponse:
        try:
            return self._client.revoke_access_key(jti=jti).data()
        except NemoHTTPError as exc:
            _raise_known_domain_error_from_http(exc)
            raise

    def suspend(self, jti: str) -> AccessKeyStatusChangeResponse:
        try:
            return self._client.suspend_access_key(jti=jti).data()
        except NemoHTTPError as exc:
            _raise_known_domain_error_from_http(exc)
            raise

    def unsuspend(self, jti: str) -> AccessKeyStatusChangeResponse:
        try:
            return self._client.unsuspend_access_key(jti=jti).data()
        except NemoHTTPError as exc:
            _raise_known_domain_error_from_http(exc)
            raise

    def rotate(self, jti: str, *, grace_period_seconds: int | None = None) -> AccessKeyRotateResponse:
        try:
            body = AccessKeyRotateRequest(grace_period_seconds=grace_period_seconds)
            return self._client.rotate_access_key(jti=jti, body=body).data()
        except NemoHTTPError as exc:
            _raise_known_domain_error_from_http(exc)
            raise


def _raise_known_domain_error_from_http(exc: NemoHTTPError) -> None:
    if exc.status_code == 501:
        raise AccessKeyOperationNotImplementedError(exc.detail) from exc
    if exc.status_code == 404:
        disabled_code = isinstance(exc.body, dict) and exc.body.get("code") == "access_keys_disabled"
        legacy_disabled_detail = exc.detail == "Scoped Access Keys are not enabled"
        if disabled_code or legacy_disabled_detail:
            raise AccessKeyFeatureDisabledError(exc.detail) from exc
