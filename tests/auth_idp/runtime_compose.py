# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import replace
from typing import Callable

import httpx
from nemo_platform import NeMoPlatform
from nemo_platform_ext.client.tls import HttpxTLSConfig, httpx_tls_config_from_env

from tests.auth_idp.common import jwt_claims
from tests.auth_idp.device_flow import authenticate_authentik_device_flow
from tests.auth_idp.runtime_contract import AuthIdpCase, DeploymentWorkloadRuntimeConfig, JsonObject, TokenSet

AUTHENTIK_COMPOSE_WORKLOAD_IDENTITY_PASSWORD = "svc-nemo-token-secret-e2e"
AUTHENTIK_DEFAULT_PASSWORDS_BY_ENVVAR = {
    "AUTHENTIK_WORKLOAD_IDENTITY_PASSWORD": AUTHENTIK_COMPOSE_WORKLOAD_IDENTITY_PASSWORD,
}
TOKEN_EXCHANGE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:token-exchange"
JWT_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:jwt"
ACCESS_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"
TOKEN_EXCHANGE_TIMEOUT_SECONDS = 30.0


def _grant_with_password(grant: dict[str, str]) -> dict[str, str]:
    resolved = dict(grant)
    password_env_var = resolved.pop("password_env_var", None)
    if password_env_var and "password" not in resolved:
        password = os.environ.get(password_env_var) or AUTHENTIK_DEFAULT_PASSWORDS_BY_ENVVAR.get(password_env_var)
        if password is None:
            raise AssertionError(f"{password_env_var} must be set for token acquisition")
        resolved["password"] = password
    return resolved


class ComposeAuthIdpRuntime:
    def __init__(self, case: AuthIdpCase, gateway_base_url: str, cleanup: Callable[[], None] | None = None):
        self.case = case
        self._cleanup = cleanup
        self._cleaned_up = False
        self.provider = replace(
            case.provider,
            gateway_base_url=gateway_base_url,
            discovery_url=f"{gateway_base_url}/application/o/nemo/.well-known/openid-configuration",
            token_endpoint=f"{gateway_base_url}/application/o/token/",
        )
        self.gateway_base_url = self.provider.gateway_base_url
        self.discovery_url = self.provider.discovery_url
        self.token_endpoint = self.provider.token_endpoint
        self.workload_token_endpoint = f"{gateway_base_url}/apis/auth/token"

    def e2e_setup_token(self) -> TokenSet:
        assert self.provider.e2e_setup_password_grant is not None
        assert self.token_endpoint is not None
        token = self._exchange_token(self.token_endpoint, _grant_with_password(self.provider.e2e_setup_password_grant))
        return TokenSet(access_token=token, claims=jwt_claims(token))

    def interactive_user_token(self) -> TokenSet:
        assert self.provider.interactive_user_password_grant is not None
        assert self.token_endpoint is not None
        token = self._exchange_token(self.token_endpoint, self.provider.interactive_user_password_grant)
        return TokenSet(access_token=token, claims=jwt_claims(token))

    def workload_provider_token(self) -> TokenSet:
        assert self.provider.workload_provider_password_grant is not None
        assert self.token_endpoint is not None
        token = self._exchange_token(
            self.token_endpoint,
            _grant_with_password(self.provider.workload_provider_password_grant),
        )
        return TokenSet(access_token=token, claims=jwt_claims(token))

    def workload_subject_token(self) -> str:
        return self.workload_provider_token().access_token

    def exchange_workload_token(self, subject_token: str) -> TokenSet:
        assert self.workload_token_endpoint is not None
        assert self.provider.workload_provider_password_grant is not None
        workload_grant = self.provider.workload_provider_password_grant
        response = httpx.post(
            self.workload_token_endpoint,
            data={
                "grant_type": TOKEN_EXCHANGE_GRANT_TYPE,
                "client_id": workload_grant["client_id"],
                "subject_token": subject_token,
                "subject_token_type": JWT_TOKEN_TYPE,
                "requested_token_type": ACCESS_TOKEN_TYPE,
                "audience": self.provider.workload_audience,
                "scope": workload_grant.get("scope", "openid email groups"),
            },
            timeout=TOKEN_EXCHANGE_TIMEOUT_SECONDS,
            **httpx_tls_config_from_env(),
        )
        response.raise_for_status()
        token_response = response.json()
        access_token = token_response["access_token"]
        assert token_response.get("token_type", "").lower() == "bearer"
        return TokenSet(access_token=access_token, claims=jwt_claims(access_token))

    def workload_platform_token(self) -> TokenSet:
        return self.workload_provider_token()

    def deployment_workload_runtime_config(self) -> DeploymentWorkloadRuntimeConfig:
        return DeploymentWorkloadRuntimeConfig(
            env=(
                {"name": "NMP_BASE_URL", "value": "https://nemo-gateway:8080"},
                {"name": "NMP_CLIENT_SSL_CERT_FILE", "value": "/etc/nmp/gateway-tls/tls.crt"},
                {"name": "SSL_CERT_FILE", "value": "/etc/nmp/gateway-tls/tls.crt"},
                {"name": "REQUESTS_CA_BUNDLE", "value": "/etc/nmp/gateway-tls/tls.crt"},
            ),
        )

    def e2e_setup_sdk(self) -> NeMoPlatform:
        token = self.e2e_setup_token().access_token
        return NeMoPlatform(
            base_url=self.gateway_base_url,
            default_headers={"Authorization": f"Bearer {token}"},
            max_retries=0,
        )

    def interactive_user_sdk(self) -> NeMoPlatform:
        token = self.interactive_user_token().access_token
        return NeMoPlatform(
            base_url=self.gateway_base_url,
            default_headers={"Authorization": f"Bearer {token}"},
            max_retries=0,
        )

    def workload_provider_sdk(self) -> NeMoPlatform:
        token = self.workload_platform_token().access_token
        return NeMoPlatform(
            base_url=self.gateway_base_url,
            default_headers={"Authorization": f"Bearer {token}"},
            max_retries=0,
        )

    def workload_role_principals(self) -> list[str]:
        return list(self.provider.workload_expected_groups)

    def authenticate_device_flow(
        self,
        *,
        device_authorization_endpoint: str,
        token_endpoint: str,
        client_id: str,
        scope: str,
        username: str,
        password: str,
        tls_config: HttpxTLSConfig,
    ) -> JsonObject:
        return authenticate_authentik_device_flow(
            gateway_base_url=self.gateway_base_url,
            device_authorization_endpoint=device_authorization_endpoint,
            token_endpoint=token_endpoint,
            client_id=client_id,
            scope=scope,
            username=username,
            password=password,
            tls_config=tls_config,
        )

    def cleanup(self) -> None:
        if self._cleaned_up:
            return
        self._cleaned_up = True
        if self._cleanup is not None:
            self._cleanup()

    def _exchange_token(self, token_endpoint: str, grant: dict[str, str]) -> str:
        from tests.auth_idp.conftest import _exchange_token_with_retries

        return _exchange_token_with_retries(token_endpoint, grant)
