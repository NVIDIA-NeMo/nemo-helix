# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Binding preflight against real in-process Secrets/Auth, without a new route."""

from unittest.mock import patch

import pytest
from nemo_deployments_plugin.backends.openshell import authorization as deployment_authorization
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.openshell.registration import BindingScope, RegistrationTickets
from nemo_helix_plugin.secrets.client import AsyncSecretsClient, SecretsClient
from nemo_helix_plugin.secrets.types import HelixSecretCreateRequest
from nhx.common.auth import Principal
from nhx.common.secrets.encryption import get_base64_encoded_random_bytes
from nhx.core.jobs.controllers.backends.openshell import authorization as job_authorization
from nhx.core.secrets.config import SecretsServiceConfig
from nhx.core.secrets.service import SecretsService
from nhx.testing import TEST_ADMIN_EMAIL, ClientContext, create_test_client
from pydantic import SecretStr


@pytest.fixture(params=[True, False])
def platform(request):
    auth_enabled = request.param
    config = SecretsServiceConfig(
        encryption={
            "current_provider": "current",
            "providers": {"secret_key": {"current": {"value": get_base64_encoded_random_bytes(32)}}},
        }
    )
    with create_test_client(
        SecretsService,
        client_type=ClientContext,
        auth_enabled=auth_enabled,
        service_configs={SecretsService: config},
        workspaces=["private-bindings"],
        # The test changes secret metadata, not the seeded permission grants.
        auth_bundle_cache_seconds=3600,
        auth_policy_data_refresh_interval=3600,
    ) as context:
        yield context, auth_enabled


@pytest.mark.parametrize(
    "owner,authorization", [("jobs", job_authorization), ("deployments", deployment_authorization)]
)
async def test_ticket_preflight_uses_existing_permission_and_metadata_services(platform, owner, authorization):
    context, auth_enabled = platform
    admin = context.client.with_headers(
        Principal(id=TEST_ADMIN_EMAIL, email=TEST_ADMIN_EMAIL, authz_aliases=[TEST_ADMIN_EMAIL]).get_headers()
    )
    secrets_client = SecretsClient.from_client(admin)
    secrets_client.create_secret(
        workspace="private-bindings",
        body=HelixSecretCreateRequest(name="bound-token", value=SecretStr("synthetic-value")),
    )
    requester = AsyncSecretsClient.from_client(
        context.async_client.with_headers(
            Principal(id=f"service:{owner}", on_behalf_of=TEST_ADMIN_EMAIL if auth_enabled else None).get_headers()
        )
    )
    tickets = RegistrationTickets(b"synthetic-signing-key-32-bytes!!!!", "gateway")
    scope = BindingScope(
        gateway_id="gateway",
        gateway_workspace="gateway-team",
        provider="provider",
        credential_key="API_TOKEN",
        secret_workspace="private-bindings",
        secret_name="bound-token",
        job_id="sandbox",
        principal_id=TEST_ADMIN_EMAIL if auth_enabled else None,
    )
    with patch("nhx.core.secrets.api.v2.secrets.endpoints.envelope_decrypt") as decrypt:
        await authorization.authorize_secret_binding(
            workspace=scope.secret_workspace,
            name=scope.secret_name,
            principal_id=scope.principal_id,
            secrets_client=requester,
        )
        assert tickets.verify(tickets.sign(scope)).secret_name == "bound-token"
        if auth_enabled:
            # The HTTP client still has service privilege; only the signed
            # creator changes. Metadata access alone would allow this.
            denied = scope.model_copy(update={"principal_id": "user:without-secret-permission"})
            with pytest.raises(PermissionError):
                await authorization.authorize_secret_binding(
                    workspace=denied.secret_workspace,
                    name=denied.secret_name,
                    principal_id=denied.principal_id,
                    secrets_client=requester,
                )
        secrets_client.delete_secret(workspace="private-bindings", name="bound-token")
        with pytest.raises(NotFoundError):
            await authorization.authorize_secret_binding(
                workspace=scope.secret_workspace,
                name=scope.secret_name,
                principal_id=scope.principal_id,
                secrets_client=requester,
            )
        decrypt.assert_not_called()
