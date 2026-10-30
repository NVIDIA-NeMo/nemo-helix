# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Driver bearer authentication and creator permission checks through real Auth/Secrets."""

# ruff: noqa: E402 -- Check the optional SDK before imports.

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest

pytest.importorskip("openshell")

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from nemo_deployments_plugin.backends.openshell.credential_driver import wire
from nemo_deployments_plugin.backends.openshell.credential_driver.registry import BindingRegistry, ProviderOwner
from nemo_deployments_plugin.backends.openshell.credential_driver.service import CredentialDriver
from nemo_helix_plugin.client.auth import StaticToken
from nemo_helix_plugin.openshell.registration import BindingScope, RegistrationDenied, RegistrationTickets
from nemo_helix_plugin.secrets.client import AsyncSecretsClient, SecretsClient
from nemo_helix_plugin.secrets.types import HelixSecretCreateRequest
from nhx.common.auth import Principal
from nhx.common.auth.workload_tokens import issue_service_workload_access_token_async
from nhx.common.config import get_auth_config
from nhx.common.config.base import OIDCConfig, OIDCWorkloadConfig, TokenSigningConfig
from nhx.common.secrets.encryption import get_base64_encoded_random_bytes
from nhx.core.secrets.config import SecretsServiceConfig
from nhx.core.secrets.service import SecretsService
from nhx.testing import TEST_ADMIN_EMAIL, ClientContext, create_test_client
from pydantic import SecretStr


@pytest.fixture
def authorized_platform() -> Iterator[ClientContext]:
    encryption = SecretsServiceConfig(
        encryption={
            "current_provider": "current",
            "providers": {"secret_key": {"current": {"value": get_base64_encoded_random_bytes(32)}}},
        }
    )
    with create_test_client(
        SecretsService,
        client_type=ClientContext,
        auth_enabled=True,
        workspaces=["default", "driver-team"],
        service_configs={SecretsService: encryption},
        # Seeded grants stay fixed while this test enables bearer authentication.
        auth_bundle_cache_seconds=3600,
        auth_policy_data_refresh_interval=3600,
    ) as context:
        yield context


@pytest.mark.asyncio
async def test_service_bearer_requires_signed_creator_permission(
    tmp_path: Path, authorized_platform: ClientContext
) -> None:
    context = authorized_platform
    admin = context.client.with_headers(Principal(id=TEST_ADMIN_EMAIL, email=TEST_ADMIN_EMAIL).get_headers())
    SecretsClient.from_client(admin).create_secret(
        workspace="driver-team", body=HelixSecretCreateRequest(name="bound-token", value=SecretStr("synthetic-secret"))
    )
    tickets = RegistrationTickets(b"synthetic-registration-key-32-bytes", "gateway")
    scope = BindingScope(
        gateway_id="gateway",
        gateway_workspace="gateway-team",
        provider="provider",
        credential_key="API_TOKEN",
        secret_workspace="driver-team",
        secret_name="bound-token",
        job_id="sandbox",
        principal_id=TEST_ADMIN_EMAIL,
    )
    ticket = tickets.sign(scope)
    registry = BindingRegistry(tmp_path / "bindings.db", tickets)
    owner = ProviderOwner(workspace="gateway-team", provider="provider", provider_id="uuid", credential_key="API_TOKEN")
    handle = registry.register(owner, ticket)
    # Issue a second signed binding while permission is available; the changed
    # creator has no current read grant when the driver resolves it.
    denied_scope = scope.model_copy(update={"principal_id": "user-without-permission"})
    denied_handle = registry.register(owner, tickets.sign(denied_scope))
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    key_file = tmp_path / "token-key.pem"
    key_file.write_bytes(
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )
    auth = get_auth_config()
    with (
        patch.object(auth, "token_signing", TokenSigningConfig(key_id="driver-test", private_key_file=str(key_file))),
        patch.object(
            auth,
            "oidc",
            OIDCConfig(
                enabled=True, issuer="https://idp.example.test", workload=OIDCWorkloadConfig(client_id="driver-test")
            ),
        ),
    ):
        token = await issue_service_workload_access_token_async(auth, service_name="deployments")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=context.test_client.app)) as http:
            async with AsyncSecretsClient(
                base_url=str(context.test_client.base_url), http_client=http, auth=StaticToken(token)
            ) as client:
                driver = CredentialDriver(registry, client)

                def request(identifier: str) -> dict[str, Any]:
                    return {
                        "credentials": [
                            {
                                **owner.model_dump(),
                                "request_id": "read",
                                "handle": {"driver": "nemo", "handle": identifier},
                            }
                        ]
                    }

                response = await driver._dispatch("ResolveCredentials", request(handle))
                assert wire.data(response)["credentials"][0]["value"] == "synthetic-secret"
                with patch("nhx.core.secrets.api.v2.secrets.endpoints.envelope_decrypt") as decrypt:
                    with pytest.raises(RegistrationDenied, match="creator"):
                        await driver._dispatch("ResolveCredentials", request(denied_handle))
                    decrypt.assert_not_called()
