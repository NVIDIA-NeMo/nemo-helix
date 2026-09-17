# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import uuid
from collections.abc import Callable
from pathlib import Path
from time import monotonic, sleep
from unittest.mock import patch

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from httpx import Response
from nmp.common.auth.access_keys import public_jwk_from_private_key_pem, validate_access_key_token
from nmp.common.auth.token_claims import TokenClaims
from nmp.common.config import AuthConfig
from nmp.common.config.base import AccessKeyConfig, TokenSigningConfig
from nmp.core.auth.config import AuthServiceConfig
from nmp.testing.client import create_test_client

# Keep this test in its own xdist group while embedded PDP uses process-wide
# wasmtime state. This limits cross-test scheduling noise under --dist loadgroup.
pytestmark = pytest.mark.xdist_group("auth_scoped_access_keys")

ACCESS_KEYS_PATH = "/apis/auth/v2/access-keys"
IAM_ROLE_BINDINGS_PATH = "/apis/auth/v2/iam/role-bindings"
WORKSPACES_PATH = "/apis/entities/v2/workspaces"
SERVICE_HEADERS = {"X-NMP-Principal-Id": "service:integration-test"}
AUTHZ_PROPAGATION_TIMEOUT_SECONDS = 5.0
AUTHZ_PROPAGATION_POLL_INTERVAL_SECONDS = 0.05


def _wait_for_authorization_response(
    request: Callable[[], Response],
    *,
    expected_status_code: int,
) -> Response:
    return _wait_for_condition(request, lambda response: response.status_code == expected_status_code)


def _wait_for_condition(
    request: Callable[[], Response],
    condition: Callable[[Response], bool],
) -> Response:
    deadline = monotonic() + AUTHZ_PROPAGATION_TIMEOUT_SECONDS
    response = request()
    while not condition(response) and monotonic() < deadline:
        sleep(AUTHZ_PROPAGATION_POLL_INTERVAL_SECONDS)
        response = request()

    if not condition(response):
        raise AssertionError(f"Timed out waiting for expected authorization response: {response.text}")
    return response


def _tamper_jwt(token: str) -> str:
    """Return a JWT with its signature bytes changed so validation must fail."""
    parts = token.split(".")
    assert len(parts) == 3
    signature = parts[2]
    replacement = "A" if signature[0] != "A" else "B"
    parts[2] = f"{replacement}{signature[1:]}"
    return ".".join(parts)


def _write_private_key(path: Path) -> None:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )


def _auth_configs(private_key_file: str) -> tuple[AuthConfig, AuthServiceConfig]:
    shared_config = AuthConfig(
        enabled=True,
        policy_decision_point_provider="embedded",
        policy_decision_point_base_url="http://testserver",
        propagation_poll_interval_seconds=0.05,
        token_signing=TokenSigningConfig(
            issuer="http://testserver/apis/auth",
            key_id="integration-access-key",
            private_key_file=private_key_file,
        ),
        access_keys=AccessKeyConfig(enabled=True),
    )
    service_config = AuthServiceConfig(
        **shared_config.model_dump(),
        policy_data_refresh_interval=0.2,
        bundle_cache_seconds=0.1,
        admin_email="admin@example.com",
    )
    return shared_config, service_config


def test_legacy_access_key_without_identity_fields_is_listed_as_user(tmp_path: Path) -> None:
    private_key_file = tmp_path / "legacy-access-key-private.pem"
    _write_private_key(private_key_file)
    shared_config, service_config = _auth_configs(str(private_key_file))
    principal = f"legacy-access-key-{uuid.uuid4().hex[:8]}@example.com"
    jti = f"ak_legacy_{uuid.uuid4().hex}"
    user_headers = {
        "X-NMP-Principal-Id": principal,
        "X-NMP-Principal-Email": principal,
    }

    with create_test_client(
        client_type=TestClient,
        auth_enabled=True,
        service_configs={AuthConfig: shared_config, AuthServiceConfig: service_config},
    ) as client:
        legacy_data = {
            "key_name": "legacy-key",
            "description": "Pre-migration access key",
            "principal": principal,
            "issuer": "http://testserver/apis/auth",
            "audiences": ["nemo-platform-access-key"],
            "issued_at": "2026-08-25T00:00:00Z",
            "expires_at": "2030-01-01T00:00:00Z",
            "status": "ACTIVE",
        }
        assert "entity_type" not in legacy_data
        assert "subject_principal" not in legacy_data
        created = client.post(
            "/apis/entities/v2/workspaces/system/entities/access_key",
            json={"name": jti, "data": legacy_data},
            headers=SERVICE_HEADERS,
        )
        assert created.status_code == 201, created.text

        listed = client.get(ACCESS_KEYS_PATH, headers=user_headers)

        assert listed.status_code == 200, listed.text
        legacy_key = next(key for key in listed.json()["data"] if key["jti"] == jti)
        assert legacy_key["entity_type"] == "USER"


def test_scoped_access_key_created_by_auth_service_authenticates_platform_requests(tmp_path: Path) -> None:
    private_key_file = tmp_path / "access-key-private.pem"
    _write_private_key(private_key_file)
    shared_config, service_config = _auth_configs(str(private_key_file))
    assert shared_config.oidc.workload_token_exchange_enabled is False

    with create_test_client(
        client_type=TestClient,
        auth_enabled=True,
        service_configs={
            AuthConfig: shared_config,
            AuthServiceConfig: service_config,
        },
    ) as client:
        workspace = f"access-key-smoke-{uuid.uuid4().hex[:8]}"
        group = f"access-key-group-{uuid.uuid4().hex[:8]}"
        user = f"access-key-user-{uuid.uuid4().hex[:8]}@example.com"
        user_headers = {
            "X-NMP-Principal-Id": user,
            "X-NMP-Principal-Email": user,
            "X-NMP-Principal-Groups": group,
        }

        create_workspace = client.post(
            WORKSPACES_PATH,
            json={"name": workspace, "description": "Scoped Access Key integration smoke"},
            headers=SERVICE_HEADERS,
        )
        assert create_workspace.status_code in {200, 201}, create_workspace.text

        try:
            create_key = client.post(
                ACCESS_KEYS_PATH,
                json={"name": "integration-smoke"},
                headers=user_headers,
            )
            assert create_key.status_code == 200, create_key.text
            access_key_body = create_key.json()
            access_key = access_key_body["token"]
            access_key_jti = access_key_body["jti"]

            role_binding = client.post(
                IAM_ROLE_BINDINGS_PATH,
                json={"principal": group, "role": "Viewer", "workspace": workspace},
                headers=SERVICE_HEADERS,
            )
            assert role_binding.status_code in {200, 201}, role_binding.text

            jwks = {"keys": [public_jwk_from_private_key_pem(shared_config)]}

            async def validate_with_local_jwks(config: AuthConfig, token: str) -> TokenClaims | None:
                return await validate_access_key_token(config, token, jwks_override=jwks)

            def get_workspace_with_access_key(token: str) -> Response:
                with patch("nmp.common.auth.access_keys.validate_access_key_token", validate_with_local_jwks):
                    return client.get(
                        f"{WORKSPACES_PATH}/{workspace}",
                        headers={"Authorization": f"Bearer {token}"},
                    )

            def authenticate_with_access_key(token: str) -> Response:
                with patch("nmp.common.auth.access_keys.validate_access_key_token", validate_with_local_jwks):
                    return client.get(
                        "/apis/auth/authenticate",
                        headers={"Authorization": f"Bearer {token}"},
                    )

            response = _wait_for_authorization_response(
                lambda: get_workspace_with_access_key(access_key),
                expected_status_code=200,
            )

            assert response.status_code == 200, response.text
            assert response.json()["name"] == workspace

            suspend_key = client.post(f"{ACCESS_KEYS_PATH}/{access_key_jti}/suspend", headers=user_headers)
            assert suspend_key.status_code == 200, suspend_key.text
            assert suspend_key.json() == {
                "jti": access_key_jti,
                "status": "SUSPENDED",
                "changed": True,
            }
            _wait_for_authorization_response(
                lambda: authenticate_with_access_key(access_key),
                expected_status_code=401,
            )

            unsuspend_key = client.post(f"{ACCESS_KEYS_PATH}/{access_key_jti}/unsuspend", headers=user_headers)
            assert unsuspend_key.status_code == 200, unsuspend_key.text
            assert unsuspend_key.json() == {
                "jti": access_key_jti,
                "status": "ACTIVE",
                "changed": True,
            }
            _wait_for_authorization_response(
                lambda: authenticate_with_access_key(access_key),
                expected_status_code=200,
            )

            authenticate_response = authenticate_with_access_key(access_key)
            invalid_authenticate_response = authenticate_with_access_key(_tamper_jwt(access_key))

            assert authenticate_response.status_code == 200, authenticate_response.text
            assert authenticate_response.json()["principal"] == user
            assert authenticate_response.json()["token_kind"] == "access_key"
            assert invalid_authenticate_response.status_code == 401, invalid_authenticate_response.text

            invalid_workspace_response = get_workspace_with_access_key(_tamper_jwt(access_key))

            assert invalid_workspace_response.status_code == 401, invalid_workspace_response.text

            revoke_key = client.delete(f"{ACCESS_KEYS_PATH}/{access_key_jti}", headers=user_headers)
            assert revoke_key.status_code == 200, revoke_key.text
            revoked_keys = client.get(ACCESS_KEYS_PATH, headers=user_headers).json()["data"]
            assert len(revoked_keys) == 1
            assert revoked_keys[0]["status"] == "REVOKED"

            revoked_authenticate_response = _wait_for_authorization_response(
                lambda: authenticate_with_access_key(access_key),
                expected_status_code=401,
            )
            revoked_workspace_response = _wait_for_authorization_response(
                lambda: get_workspace_with_access_key(access_key),
                expected_status_code=401,
            )

            assert revoked_authenticate_response.status_code == 401, revoked_authenticate_response.text
            assert revoked_workspace_response.status_code == 401, revoked_workspace_response.text
        finally:
            client.delete(f"{WORKSPACES_PATH}/{workspace}", headers=SERVICE_HEADERS)


def test_rotated_access_key_keeps_working_during_grace_period_and_new_key_authenticates(tmp_path: Path) -> None:
    private_key_file = tmp_path / "rotate-access-key-private.pem"
    _write_private_key(private_key_file)
    shared_config, service_config = _auth_configs(str(private_key_file))
    user = f"rotate-user-{uuid.uuid4().hex[:8]}@example.com"
    user_headers = {
        "X-NMP-Principal-Id": user,
        "X-NMP-Principal-Email": user,
    }

    with create_test_client(
        client_type=TestClient,
        auth_enabled=True,
        service_configs={AuthConfig: shared_config, AuthServiceConfig: service_config},
    ) as client:
        created = client.post(
            ACCESS_KEYS_PATH,
            json={"name": "rotate-smoke"},
            headers=user_headers,
        )
        assert created.status_code == 200, created.text
        old_key = created.json()["token"]
        old_jti = created.json()["jti"]

        rotated = client.post(f"{ACCESS_KEYS_PATH}/{old_jti}/rotate", headers=user_headers)
        assert rotated.status_code == 200, rotated.text
        rotated_body = rotated.json()
        assert rotated_body["previous_jti"] == old_jti
        assert rotated_body["previous_status"] == "ROTATING"
        new_key = rotated_body["new_key"]["token"]
        new_jti = rotated_body["new_key"]["jti"]
        assert new_jti != old_jti

        jwks = {"keys": [public_jwk_from_private_key_pem(shared_config)]}

        async def validate_with_local_jwks(config: AuthConfig, token: str) -> TokenClaims | None:
            return await validate_access_key_token(config, token, jwks_override=jwks)

        def authenticate_with_access_key(token: str) -> Response:
            with patch("nmp.common.auth.access_keys.validate_access_key_token", validate_with_local_jwks):
                return client.get(
                    "/apis/auth/authenticate",
                    headers={"Authorization": f"Bearer {token}"},
                )

        # Dual-active grace period: both the rotated-out key and its successor
        # authenticate until the old key is finalized (revoked or grace expires).
        old_key_response = authenticate_with_access_key(old_key)
        new_key_response = authenticate_with_access_key(new_key)
        assert old_key_response.status_code == 200, old_key_response.text
        assert new_key_response.status_code == 200, new_key_response.text

        listing = {key["jti"]: key for key in client.get(ACCESS_KEYS_PATH, headers=user_headers).json()["data"]}
        assert listing[old_jti]["status"] == "ROTATING"
        assert listing[new_jti]["status"] == "ACTIVE"

        # Cannot rotate an already-rotating key; must finalize (revoke) first.
        second_rotate = client.post(f"{ACCESS_KEYS_PATH}/{old_jti}/rotate", headers=user_headers)
        assert second_rotate.status_code == 409, second_rotate.text

        finalize = client.delete(f"{ACCESS_KEYS_PATH}/{old_jti}", headers=user_headers)
        assert finalize.status_code == 200, finalize.text

        finalized_old_key_response = _wait_for_authorization_response(
            lambda: authenticate_with_access_key(old_key),
            expected_status_code=401,
        )
        assert finalized_old_key_response.status_code == 401, finalized_old_key_response.text
        assert authenticate_with_access_key(new_key).status_code == 200


def test_platform_admin_creates_service_bound_key_with_independent_identity(tmp_path: Path) -> None:
    private_key_file = tmp_path / "service-access-key-private.pem"
    _write_private_key(private_key_file)
    shared_config, service_config = _auth_configs(str(private_key_file))
    admin_headers = {
        "X-NMP-Principal-Id": "admin@example.com",
        "X-NMP-Principal-Email": "admin@example.com",
    }
    user_headers = {
        "X-NMP-Principal-Id": "user@example.com",
        "X-NMP-Principal-Email": "user@example.com",
    }

    with create_test_client(
        client_type=TestClient,
        auth_enabled=True,
        service_configs={AuthConfig: shared_config, AuthServiceConfig: service_config},
    ) as client:
        denied = client.post(
            ACCESS_KEYS_PATH,
            json={"service_account_id": "otel-collector"},
            headers=user_headers,
        )
        assert denied.status_code == 403, denied.text

        created = client.post(
            ACCESS_KEYS_PATH,
            json={"name": "otel", "service_account_id": "otel-collector"},
            headers=admin_headers,
        )
        assert created.status_code == 200, created.text
        body = created.json()
        assert body["principal"] == "service-account:otel-collector"
        assert body["entity_type"] == "SERVICE_ACCOUNT"

        jwks = {"keys": [public_jwk_from_private_key_pem(shared_config)]}

        async def validate_with_local_jwks(config: AuthConfig, token: str) -> TokenClaims | None:
            return await validate_access_key_token(config, token, jwks_override=jwks)

        with patch("nmp.common.auth.access_keys.validate_access_key_token", validate_with_local_jwks):
            authenticated = client.get(
                "/apis/auth/authenticate",
                headers={"Authorization": f"Bearer {body['token']}"},
            )
        assert authenticated.status_code == 200, authenticated.text
        assert authenticated.json()["principal"] == "service-account:otel-collector"

        workspace = f"service-key-scope-{uuid.uuid4().hex[:8]}"
        create_workspace = client.post(
            WORKSPACES_PATH,
            json={"name": workspace, "description": "Service-bound key scope check"},
            headers=SERVICE_HEADERS,
        )
        assert create_workspace.status_code in {200, 201}, create_workspace.text

        def get_workspace_with_service_key() -> Response:
            with patch("nmp.common.auth.access_keys.validate_access_key_token", validate_with_local_jwks):
                return client.get(
                    f"{WORKSPACES_PATH}/{workspace}",
                    headers={"Authorization": f"Bearer {body['token']}"},
                )

        assert get_workspace_with_service_key().status_code == 403
        role_binding = client.post(
            IAM_ROLE_BINDINGS_PATH,
            json={"principal": body["principal"], "role": "Viewer", "workspace": workspace},
            headers=SERVICE_HEADERS,
        )
        assert role_binding.status_code in {200, 201}, role_binding.text
        _wait_for_authorization_response(get_workspace_with_service_key, expected_status_code=200)

        listed = client.get(ACCESS_KEYS_PATH, headers=admin_headers)
        assert listed.status_code == 200, listed.text
        assert listed.json()["data"][0]["principal"] == "service-account:otel-collector"

        # Service-bound keys are platform-owned, not creator-owned (AIRCORE-986): a second
        # PlatformAdmin who did not create this key must still see it in their own listing,
        # not just be able to look it up by jti.
        other_admin_headers = {
            "X-NMP-Principal-Id": "admin2@example.com",
            "X-NMP-Principal-Email": "admin2@example.com",
        }
        other_admin_role_binding = client.post(
            IAM_ROLE_BINDINGS_PATH,
            json={"principal": "admin2@example.com", "role": "PlatformAdmin", "workspace": "system"},
            headers=SERVICE_HEADERS,
        )
        assert other_admin_role_binding.status_code in {200, 201}, other_admin_role_binding.text
        other_admin_listed = _wait_for_condition(
            lambda: client.get(ACCESS_KEYS_PATH, headers=other_admin_headers),
            lambda response: (
                response.status_code == 200
                and any(key["principal"] == "service-account:otel-collector" for key in response.json()["data"])
            ),
        )
        assert other_admin_listed.status_code == 200, other_admin_listed.text

        revoked = client.delete(f"{ACCESS_KEYS_PATH}/{body['jti']}", headers=admin_headers)
        assert revoked.status_code == 200, revoked.text
        client.delete(f"{WORKSPACES_PATH}/{workspace}", headers=SERVICE_HEADERS)
