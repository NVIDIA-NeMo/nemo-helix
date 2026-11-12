# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from nemo_helix_plugin.auth.workload_delegations import JOB_WORKLOAD_CLAIM, JobWorkloadBinding, current_workload_job
from nhx.common.auth.client import AuthClient
from nhx.common.auth.dependencies import auth_client_context
from nhx.common.auth.models import Principal
from nhx.common.auth.token_resolver import ResolvedTokenKind
from nhx.common.auth.workload_tokens import (
    issue_service_workload_access_token,
    issue_service_workload_access_token_async,
    resolve_workload_access_token,
    resolve_workload_subject_token,
)
from nhx.common.config import AuthConfig, Configuration
from nhx.common.config.base import OIDCConfig, OIDCWorkloadConfig, TokenSigningConfig


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("token_kind", "claim", "named"),
    [
        ("workload_access_token", None, True),
        ("oidc_access_token", None, False),
        ("access_key", None, False),
        ("workload_access_token", {"workspace": "team"}, False),
    ],
    ids=["workload-token", "oidc-token", "access-key", "malformed-claim"],
)
async def test_only_a_workload_access_token_names_its_job(
    tmp_path: Path, token_kind: ResolvedTokenKind, claim: dict | None, named: bool
) -> None:
    config = _token_exchange_config(tmp_path)
    job = JobWorkloadBinding(workspace="team", job="job-one")
    token = await issue_service_workload_access_token_async(
        config, service_name="jobs", on_behalf_of=Principal(id="user:alice"), job=job
    )
    resolved = await resolve_workload_access_token(config, token)
    assert resolved is not None
    raw_claims = resolved.claims.raw_claims | ({JOB_WORKLOAD_CLAIM: claim} if claim is not None else {})
    resolved = replace(resolved, token_kind=token_kind, claims=replace(resolved.claims, raw_claims=raw_claims))
    marker = auth_client_context.set(
        AuthClient(principal=resolved.principal, config=config, resolved_bearer_token=resolved)
    )
    try:
        assert current_workload_job() == (job if named else None)
    finally:
        auth_client_context.reset(marker)


def test_a_job_bound_service_token_must_act_for_the_jobs_creator(tmp_path: Path) -> None:
    config = _token_exchange_config(tmp_path)

    with pytest.raises(ValueError, match="creator"):
        issue_service_workload_access_token(
            config, service_name="jobs", job=JobWorkloadBinding(workspace="team", job="job-one")
        )
