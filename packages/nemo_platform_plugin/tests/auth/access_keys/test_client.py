# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime
from typing import cast
from unittest.mock import MagicMock

import httpx
import pytest
from nemo_platform_plugin.auth.access_keys.client import AccessKeyIssuerClient, AccessKeysClient
from nemo_platform_plugin.auth.access_keys.issuer import (
    AccessKeyFeatureDisabledError,
    AccessKeyOperationNotImplementedError,
)
from nemo_platform_plugin.auth.access_keys.types import (
    AccessKeyCreateRequest,
    AccessKeyCreateResponse,
    AccessKeyRevokeResponse,
    AccessKeyRotateRequest,
    AccessKeyRotateResponse,
    AccessKeyStatusChangeResponse,
)
from nemo_platform_plugin.client.errors import NemoHTTPError


class _AccessKeysClientStub:
    def __init__(self) -> None:
        self.create_access_key = MagicMock()
        self.list_access_keys = MagicMock()
        self.revoke_access_key = MagicMock()
        self.suspend_access_key = MagicMock()
        self.unsuspend_access_key = MagicMock()
        self.rotate_access_key = MagicMock()

    def as_client(self) -> AccessKeysClient:
        return cast(AccessKeysClient, self)


def test_access_key_issuer_client_delegates_create_to_client() -> None:
    created = AccessKeyCreateResponse(
        jti="ak_example",
        name=None,
        token="signed.jwt.token",
        token_type="Bearer",
        principal="alice@example.com",
        created_at=datetime(2026, 7, 28, 12, 0, tzinfo=UTC),
        expires_at=None,
        description=None,
        status="ACTIVE",
        issuer="https://platform.example.com/apis/auth",
        audiences=["nemo-platform-access-key"],
    )
    client = _AccessKeysClientStub()
    client.create_access_key.return_value.data.return_value = created

    issuer = AccessKeyIssuerClient(client.as_client())
    result = issuer.create(AccessKeyCreateRequest())

    assert result == created
    client.create_access_key.assert_called_once_with(body=AccessKeyCreateRequest())


def test_access_key_issuer_client_revokes_by_jti() -> None:
    client = _AccessKeysClientStub()
    revoked = AccessKeyRevokeResponse(jti="ak_example", revoked=True)
    client.revoke_access_key.return_value.data.return_value = revoked

    issuer = AccessKeyIssuerClient(client.as_client())
    result = issuer.revoke("ak_example")

    assert result == revoked
    client.revoke_access_key.assert_called_once_with(jti="ak_example")


def test_access_key_issuer_client_suspends_and_unsuspends_by_jti() -> None:
    client = _AccessKeysClientStub()
    client.suspend_access_key.return_value.data.return_value = AccessKeyStatusChangeResponse(
        jti="ak_example", status="SUSPENDED", changed=True
    )
    client.unsuspend_access_key.return_value.data.return_value = AccessKeyStatusChangeResponse(
        jti="ak_example", status="ACTIVE", changed=True
    )
    issuer = AccessKeyIssuerClient(client.as_client())

    assert issuer.suspend("ak_example").changed
    assert issuer.unsuspend("ak_example").changed
    client.suspend_access_key.assert_called_once_with(jti="ak_example")
    client.unsuspend_access_key.assert_called_once_with(jti="ak_example")


def test_access_key_issuer_client_rotates_by_jti() -> None:
    client = _AccessKeysClientStub()
    rotated = AccessKeyRotateResponse(
        new_key=AccessKeyCreateResponse(
            jti="ak_successor",
            name=None,
            token="signed.jwt.token",
            token_type="Bearer",
            principal="alice@example.com",
            created_at=datetime(2026, 7, 28, 12, 0, tzinfo=UTC),
            expires_at=None,
            description=None,
            status="ACTIVE",
            issuer="https://platform.example.com/apis/auth",
            audiences=["nemo-platform-access-key"],
        ),
        previous_jti="ak_example",
        previous_status="ROTATING",
        grace_period_seconds=3600,
    )
    client.rotate_access_key.return_value.data.return_value = rotated

    issuer = AccessKeyIssuerClient(client.as_client())
    result = issuer.rotate("ak_example")

    assert result == rotated
    client.rotate_access_key.assert_called_once_with(jti="ak_example", body=AccessKeyRotateRequest())


def test_access_key_issuer_client_rotates_with_explicit_grace_period() -> None:
    client = _AccessKeysClientStub()
    client.rotate_access_key.return_value.data.return_value = None

    issuer = AccessKeyIssuerClient(client.as_client())
    issuer.rotate("ak_example", grace_period_seconds=3600)

    client.rotate_access_key.assert_called_once_with(
        jti="ak_example", body=AccessKeyRotateRequest(grace_period_seconds=3600)
    )


def test_access_key_issuer_client_lists_requested_page() -> None:
    client = _AccessKeysClientStub()
    issuer = AccessKeyIssuerClient(client.as_client())

    issuer.list(page=3, page_size=25)

    client.list_access_keys.assert_called_once_with(query_params={"page": 3, "page_size": 25})


def test_access_key_issuer_client_translates_http_501_to_domain_error() -> None:
    response = httpx.Response(
        501,
        json={"detail": "Scoped Access Key listing is not implemented."},
        request=httpx.Request("GET", "https://cluster.example.com/apis/auth/v2/access-keys"),
    )
    client = _AccessKeysClientStub()
    client.list_access_keys.side_effect = NemoHTTPError(response)

    issuer = AccessKeyIssuerClient(client.as_client())

    with pytest.raises(AccessKeyOperationNotImplementedError, match="not implemented"):
        issuer.list()


@pytest.mark.parametrize(
    "body",
    [
        {"detail": "Scoped Access Keys are not enabled", "code": "access_keys_disabled"},
        {"detail": "Scoped Access Keys are not enabled"},
    ],
    ids=["structured-code", "legacy-detail"],
)
def test_access_key_issuer_client_translates_disabled_feature_to_domain_error(body: dict[str, str]) -> None:
    response = httpx.Response(
        404,
        json=body,
        request=httpx.Request("POST", "https://cluster.example.com/apis/auth/v2/access-keys"),
    )
    client = _AccessKeysClientStub()
    client.create_access_key.side_effect = NemoHTTPError(response)

    issuer = AccessKeyIssuerClient(client.as_client())

    with pytest.raises(AccessKeyFeatureDisabledError, match="not enabled"):
        issuer.create(AccessKeyCreateRequest())


def test_access_key_issuer_client_propagates_not_found_as_http_error() -> None:
    """Plain 404 (key not found) propagates as NemoHTTPError so @handle_errors renders it correctly."""
    response = httpx.Response(
        404,
        json={"detail": "Scoped Access Key ak_" + "0" * 32 + " was not found"},
        request=httpx.Request("DELETE", "https://cluster.example.com/apis/auth/v2/access-keys/ak_" + "0" * 32),
    )
    client = _AccessKeysClientStub()
    client.revoke_access_key.side_effect = NemoHTTPError(response)

    issuer = AccessKeyIssuerClient(client.as_client())

    with pytest.raises(NemoHTTPError) as exc_info:
        issuer.revoke("ak_" + "0" * 32)
    assert exc_info.value.status_code == 404
    assert "was not found" in exc_info.value.detail
