# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""With platform auth on, a router's call back is authenticated and authorized as the caller it carries."""

from collections.abc import Generator, Iterator
from unittest.mock import patch

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest
from nhx.common.auth.access_keys import public_jwk_from_private_key_pem, validate_access_key_token
from nhx.common.auth.token_claims import TokenClaims
from nhx.common.config import AuthConfig, HelixConfig
from nhx.common.config.base import AccessKeyConfig, TokenSigningConfig
from nhx.core.auth.config import AuthServiceConfig
from nhx.core.inference_gateway.api.caller_context import CALLER_CONTEXT_HEADER, gateway_seal
from nhx.core.inference_gateway.api.model_cache import ModelCache
from nhx.core.inference_gateway.service import InferenceGatewayService
from nhx.core.models.service import ModelsService
from nhx.testing import (
    TEST_ADMIN_EMAIL,
    ClientContext,
    as_user,
    create_test_client,
    grant_workspace_role,
    short_unique_name,
    unique_email,
)

pytestmark = [pytest.mark.integration, pytest.mark.xdist_group("igw_caller_context_with_auth")]

ROUTER = "routers/router"
ROUTER_PRINCIPAL = "router@example.com"


@pytest.fixture(scope="module")
def ctx(tmp_path_factory: pytest.TempPathFactory) -> Generator[ClientContext, None, None]:
    """A platform with auth on that issues access keys."""
    signing_key = tmp_path_factory.mktemp("signing") / "private.pem"
    signing_key.write_bytes(
        rsa.generate_private_key(public_exponent=65537, key_size=2048).private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    auth = AuthConfig(
        enabled=True,
        policy_decision_point_provider="embedded",
        policy_decision_point_base_url="http://testserver",
        propagation_poll_interval_seconds=0.05,
        token_signing=TokenSigningConfig(
            issuer="http://testserver/apis/auth", key_id="test-access-key", private_key_file=str(signing_key)
        ),
        access_keys=AccessKeyConfig(enabled=True),
    )
    with create_test_client(
        InferenceGatewayService,
        ModelsService,
        service_configs={
            HelixConfig: HelixConfig(base_url="http://127.0.0.1"),
            AuthConfig: auth,
            AuthServiceConfig: AuthServiceConfig(
                **auth.model_dump(),
                policy_data_refresh_interval=0.2,
                bundle_cache_seconds=0.1,
                admin_email=TEST_ADMIN_EMAIL,
                allowed_service_principals=["integration-test"],
            ),
        },
        auth_enabled=True,
        igw_mock_provider_mode=True,
        client_type=ClientContext,
    ) as ctx:
        jwks = {"keys": [public_jwk_from_private_key_pem(auth)]}

        async def validate_with_local_jwks(config: AuthConfig, token: str) -> TokenClaims | None:
            return await validate_access_key_token(config, token, jwks_override=jwks)

        with patch("nhx.common.auth.access_keys.validate_access_key_token", validate_with_local_jwks):
            yield ctx


@pytest.fixture(scope="module")
def router_key(ctx: ClientContext) -> Iterator[str]:
    """A router holding a live platform key of its own, with access to every workspace these tests make."""
    key, _ = _access_key(ctx, ROUTER_PRINCIPAL)
    with patch.object(ModelCache, "router_keys", return_value={ROUTER: key}):
        yield key


def _as(email: str) -> dict[str, str]:
    return {"X-NHX-Principal-Id": email, "X-NHX-Principal-Email": email}


def _workspace(ctx: ClientContext, member: str | None = None) -> str:
    workspace = short_unique_name("igw-cc")
    WorkspacesClient(base_url="http://testserver", http_client=ctx.test_client).with_headers(
        _as(TEST_ADMIN_EMAIL)
    ).create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
    for principal in (ROUTER_PRINCIPAL, *([member] if member is not None else [])):
        grant_workspace_role(
            as_user(ctx.client, TEST_ADMIN_EMAIL), workspace=workspace, principal=principal, roles=["Editor"]
        )
    return workspace


def _access_key(ctx: ClientContext, email: str) -> tuple[str, str]:
    response = ctx.test_client.post(
        "/apis/auth/v2/access-keys",
        json={"name": short_unique_name("trial"), "expires_in_seconds": 600},
        headers=_as(email),
    )
    assert response.status_code == 200, response.text
    return response.json()["token"], response.json()["jti"]


def _call_back(ctx: ClientContext, router_key: str, workspace: str, caller: str) -> int:
    return ctx.test_client.get(
        f"/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/v1/models",
        headers={"Authorization": f"Bearer {router_key}", CALLER_CONTEXT_HEADER: gateway_seal().seal(ROUTER, caller)},
    ).status_code


def test_a_call_back_is_allowed_what_its_caller_is_allowed(ctx: ClientContext, router_key: str) -> None:
    caller = unique_email("caller")
    workspace = _workspace(ctx, member=caller)
    token, _ = _access_key(ctx, caller)

    assert _call_back(ctx, router_key, workspace, token) == 200


def test_a_call_back_is_refused_where_only_its_router_has_access(ctx: ClientContext, router_key: str) -> None:
    caller = unique_email("caller")
    _workspace(ctx, member=caller)
    elsewhere = _workspace(ctx)
    token, _ = _access_key(ctx, caller)

    assert _call_back(ctx, router_key, elsewhere, token) == 403


def test_a_call_back_is_refused_once_its_callers_key_is_revoked(ctx: ClientContext, router_key: str) -> None:
    """The caller context outlives the caller's key by up to its lifetime, so the key itself must be checked."""
    caller = unique_email("caller")
    workspace = _workspace(ctx, member=caller)
    token, jti = _access_key(ctx, caller)
    assert ctx.test_client.delete(f"/apis/auth/v2/access-keys/{jti}", headers=_as(caller)).status_code == 200

    assert _call_back(ctx, router_key, workspace, token) == 401


def test_a_routers_key_alone_is_refused_before_platform_auth(ctx: ClientContext, router_key: str) -> None:
    workspace = _workspace(ctx)

    response = ctx.test_client.get(
        f"/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/v1/models",
        headers={"Authorization": f"Bearer {router_key}"},
    )

    assert response.status_code == 403
    assert "caller context" in response.json()["detail"]
