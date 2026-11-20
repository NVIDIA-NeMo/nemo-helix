# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from nhx.common.auth.client import AuthClient
from nhx.common.auth.models import Principal
from nhx.common.auth.token_resolver import ResolvedBearerToken
from nhx.common.auth.workload_tokens import (
    issue_service_workload_access_token_async,
    resolve_workload_access_token,
    resolve_workload_subject_token,
    workload_signing_key_async,
)
from nhx.common.config import AuthConfig, Configuration
from nhx.common.config.base import OIDCConfig, OIDCWorkloadConfig, TokenSigningConfig

_DELEGATION_CLAIM = {"nhx_service_attributes": {"auth": {"delegation_id": "workload_delegation-abc123"}}}


def _write_private_key(path: Path) -> None:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )


def _new_private_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _token_exchange_config(tmp_path: Path) -> AuthConfig:
    key_path = tmp_path / "workload-private.pem"
    _write_private_key(key_path)
    return AuthConfig(
        enabled=True,
        token_signing=TokenSigningConfig(
            issuer="https://nemo.example.test/apis/auth",
            key_id="test-workload",
            private_key_file=str(key_path),
        ),
        oidc=OIDCConfig(
            enabled=True,
            audience="test-audience",
            workload=OIDCWorkloadConfig(
                client_id="nemo-workload",
                audience="nemo-helix",
                scope="openid email groups",
            ),
        ),
    )


async def _sign_with_workload_key(config: AuthConfig, claims: dict[str, object]) -> str:
    """A token with ``claims`` signed by the platform's workload key, as any process holding that key can make."""
    signing_key = await workload_signing_key_async(config)
    now = int(time.time())
    return jwt.encode(
        {
            "iss": config.token_signing.issuer,
            "aud": "nemo-helix",
            "iat": now,
            "nbf": now,
            "exp": now + 300,
            **claims,
        },
        signing_key.private_key,
        algorithm="RS256",
        headers={"kid": signing_key.kid},
    )


def _auth_client_for(config: AuthConfig, resolved: ResolvedBearerToken | None) -> AuthClient:
    """The auth client the middleware builds for a request authenticated with ``resolved``."""
    assert resolved is not None
    return AuthClient(principal=resolved.principal, config=config, resolved_bearer_token=resolved)


@pytest.fixture(autouse=True)
def _clear_configuration_overrides():
    Configuration.clear_override(AuthConfig)
    yield
    Configuration.clear_override(AuthConfig)


@pytest.mark.asyncio
async def test_service_workload_access_token_resolves_to_service_principal(tmp_path: Path) -> None:
    config = _token_exchange_config(tmp_path)

    token = await issue_service_workload_access_token_async(config, service_name="models")
    resolved = await resolve_workload_access_token(config, token)

    assert resolved is not None
    assert resolved.token_kind == "workload_access_token"
    assert resolved.principal.id == "service:models"
    assert resolved.principal.authz_aliases == ["service:models"]
    assert resolved.scopes == ["openid", "email", "groups"]


@pytest.mark.asyncio
async def test_delegated_service_workload_access_token_preserves_obo_principal(tmp_path: Path) -> None:
    config = _token_exchange_config(tmp_path)
    creator = Principal(
        id="user:alice",
        email="alice@example.test",
        groups=["workspace-editors"],
    )

    token = await issue_service_workload_access_token_async(config, service_name="jobs", on_behalf_of=creator)
    resolved = await resolve_workload_access_token(config, token)

    assert resolved is not None
    assert resolved.principal.id == "service:jobs"
    assert resolved.principal.on_behalf_of == "user:alice"
    assert resolved.principal.on_behalf_of_email == "alice@example.test"
    assert resolved.principal.on_behalf_of_groups == ["workspace-editors"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "on_behalf_of",
    [None, Principal(id="user:alice", email="alice@example.test")],
    ids=["service token", "service token on behalf of a user"],
)
async def test_service_tokens_name_no_workload_delegation(tmp_path: Path, on_behalf_of: Principal | None) -> None:
    config = _token_exchange_config(tmp_path)

    token = await issue_service_workload_access_token_async(config, service_name="jobs", on_behalf_of=on_behalf_of)
    auth_client = _auth_client_for(config, await resolve_workload_access_token(config, token))

    assert auth_client.workload_delegation_verifiable is True
    assert auth_client.workload_delegation_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra_claims",
    [
        {},
        # The workload token validator does not look at the token type today, so a token typed as an access key
        # is believed too. This case flips to "does not resolve" once the validator refuses that type.
        {"nhx_token_type": "access_key", "jti": "ak_" + "a" * 32},
    ],
    ids=["delegated token", "delegated token typed as an access key"],
)
async def test_platform_signed_token_with_actor_and_claim_names_its_workload_delegation(
    tmp_path: Path, extra_claims: dict[str, object]
) -> None:
    """The trust base: whatever the workload key signed with an actor and the delegation claim is believed."""
    config = _token_exchange_config(tmp_path)
    token = await _sign_with_workload_key(
        config,
        {
            "sub": "creator@example.test",
            "act": {"sub": "system:serviceaccount:nemo-runs:job-runner"},
            **_DELEGATION_CLAIM,
            **extra_claims,
        },
    )

    auth_client = _auth_client_for(config, await resolve_workload_access_token(config, token))

    assert auth_client.workload_delegation_id == "workload_delegation-abc123"


@pytest.mark.asyncio
async def test_access_key_shaped_token_names_no_workload_delegation(tmp_path: Path) -> None:
    """An access key reaches the workload resolver when access keys are off at the verifier and audiences overlap.

    It is signed with the same key by default, so it resolves as a workload access token. It acts for nobody, so
    the delegation claim it carries is not believed.
    """
    config = _token_exchange_config(tmp_path)
    token = await _sign_with_workload_key(
        config,
        {
            "sub": "alice@example.test",
            "nhx_token_type": "access_key",
            "jti": "ak_" + "a" * 32,
            **_DELEGATION_CLAIM,
        },
    )

    resolved = await resolve_workload_access_token(config, token)

    assert resolved is not None
    assert resolved.token_kind == "workload_access_token"
    assert _auth_client_for(config, resolved).workload_delegation_id is None


@pytest.mark.asyncio
async def test_workload_subject_token_resolves_with_configured_subject_issuer(tmp_path: Path) -> None:
    private_key = _new_private_key()
    now = datetime.now(tz=UTC)
    token = jwt.encode(
        {
            "iss": "https://sso.example.test/application/o/nemo-workload/",
            "sub": "svc-nemo",
            "email": "svc-nemo@example.test",
            "groups": "nemo-workloads",
            "scope": "openid email groups",
            "aud": "nemo-helix-workload",
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(minutes=5)).timestamp()),
        },
        private_key,
        algorithm="RS256",
        headers={"kid": "subject-key"},
    )
    workload_private_key_file = tmp_path / "workload-private.pem"
    _write_private_key(workload_private_key_file)
    config = AuthConfig(
        enabled=True,
        token_signing=TokenSigningConfig(
            key_id="test-workload",
            private_key_file=str(workload_private_key_file),
        ),
        oidc=OIDCConfig(
            enabled=True,
            audience="test-audience",
            issuer="https://sso.example.test/application/o/nemo/",
            workload=OIDCWorkloadConfig(
                client_id="nemo-helix-workload",
                subject_jwks_uri="https://sso.example.test/application/o/nemo-workload/jwks/",
                subject_issuers=["https://sso.example.test/application/o/nemo-workload/"],
            ),
        ),
    )

    with patch("nhx.common.auth.workload_tokens.AsyncJWKSClient") as client_cls:
        client_cls.return_value.get_signing_key_from_jwt = AsyncMock(
            return_value=SimpleNamespace(key=private_key.public_key())
        )

        resolved = await resolve_workload_subject_token(config, token)

    assert resolved is not None
    assert resolved.token_kind == "workload_subject_token"
    assert resolved.principal.id == "svc-nemo"
    assert resolved.principal.email == "svc-nemo@example.test"
    assert resolved.principal.groups == ["nemo-workloads"]
    assert resolved.scopes == ["openid", "email", "groups"]
