# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A provider that calls models back as its caller gets the caller's credential, and only a narrow one."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from nemo_helix_plugin.models.types import ModelProvider
from nhx.common.auth.dependencies import auth_client_context
from nhx.core.inference_gateway.api.provider_request import upstream_credential

TOKEN = "eyJhbGciOiJub25lIn0.eyJzdWIiOiJ0cmlhbCJ9."


def _provider(*, forward: bool) -> ModelProvider:
    now = datetime.now(UTC)
    return ModelProvider(
        name="switchyard",
        workspace="default",
        host_url="http://switchyard:4002",
        forward_caller_credential=forward,
        created_at=now,
        updated_at=now,
    )


def _auth(*, enabled: bool = True, scopes: list[str] | None = None) -> Any:
    resolved = SimpleNamespace(scopes=scopes) if scopes is not None else None
    return SimpleNamespace(auth_enabled=enabled, resolved_bearer_token=resolved)


@pytest.fixture(autouse=True)
def _clear_auth_context() -> Iterator[None]:
    token = auth_client_context.set(None)
    try:
        yield
    finally:
        auth_client_context.reset(token)


def test_an_ordinary_provider_gets_its_stored_key() -> None:
    assert upstream_credential(_provider(forward=False), "stored-key", {"authorization": f"Bearer {TOKEN}"}) == (
        "stored-key"
    )


@pytest.mark.parametrize(
    "headers",
    [{"Authorization": f"Bearer {TOKEN}"}, {"x-api-key": TOKEN}],
    ids=["bearer", "x-api-key"],
)
def test_with_auth_on_an_inference_scoped_credential_is_forwarded(headers: dict[str, str]) -> None:
    auth_client_context.set(_auth(scopes=["inference:read", "inference:write"]))

    assert upstream_credential(_provider(forward=True), None, headers) == TOKEN


def test_behind_envoy_the_trusted_scopes_header_decides() -> None:
    auth_client_context.set(_auth())

    assert (
        upstream_credential(
            _provider(forward=True),
            None,
            {"authorization": f"Bearer {TOKEN}", "x-nhx-scopes": "inference:read inference:write"},
        )
        == TOKEN
    )


@pytest.mark.parametrize(
    "scopes",
    [[], ["inference:read", "files:read"], ["platform:read", "platform:write"]],
    ids=["unscoped session", "broader key", "other scope"],
)
def test_with_auth_on_a_broader_credential_is_refused_without_echoing_it(scopes: list[str]) -> None:
    """A user's full-power token must never reach a router, which could then act as that user anywhere."""
    auth_client_context.set(_auth(scopes=scopes))

    with pytest.raises(HTTPException) as exc_info:
        upstream_credential(_provider(forward=True), None, {"authorization": f"Bearer {TOKEN}"})

    assert exc_info.value.status_code == 403
    assert TOKEN not in str(exc_info.value.detail)


@pytest.mark.parametrize("auth", [None, _auth(enabled=False)], ids=["no auth context", "auth off"])
def test_with_auth_off_the_callers_credential_is_forwarded_as_is(auth: Any) -> None:
    auth_client_context.set(auth)

    assert upstream_credential(_provider(forward=True), None, {"authorization": f"Bearer {TOKEN}"}) == TOKEN


@pytest.mark.parametrize("headers", [{}, {"authorization": "Bearer a b"}], ids=["no credential", "malformed bearer"])
def test_without_a_usable_credential_nothing_is_forwarded(headers: dict[str, str]) -> None:
    auth_client_context.set(_auth(scopes=["inference:read", "inference:write"]))

    assert upstream_credential(_provider(forward=True), "stored-key", headers) is None
