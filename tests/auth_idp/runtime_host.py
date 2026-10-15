# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from dataclasses import replace
from pathlib import Path

import httpx
from nemo_helix_ext.client.tls import NHX_CLIENT_SSL_CERT_FILE_ENVVAR, HttpxTLSConfig
from nemo_helix_plugin.client.client import NemoClient
from nhx.common.auth.token_claims import groups_from_claim

from tests.auth_idp.common import jwt_claims
from tests.auth_idp.deployment_context import AuthIdpDeploymentContext, load_credential_environment
from tests.auth_idp.oidc_test_driver import create_oidc_test_driver
from tests.auth_idp.providers import ProviderConfig, load_provider_config
from tests.auth_idp.runtime_contract import AuthIdpCase, DeploymentWorkloadRuntimeConfig, JsonObject, TokenSet
from tests.auth_idp.token_acquisition import exchange_token_with_retries

TOKEN_EXCHANGE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:token-exchange"
JWT_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:jwt"
ACCESS_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"
ZITADEL_PROJECT_ROLES_CLAIM = "urn:zitadel:iam:org:project:roles"
TOKEN_EXCHANGE_TIMEOUT_SECONDS = 30.0
PLATFORM_API_REQUEST_TIMEOUT_SECONDS = 60.0


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"deployment credentials do not define {name}")
    return value


def _resolved_zitadel_grant(
    grant: dict[str, str] | None,
    *,
    client_id_env: str,
    client_secret_env: str,
    project_id: str,
) -> dict[str, str] | None:
    if grant is None:
        return None
    resolved = dict(grant)
    resolved["client_id"] = _required_env(client_id_env)
    resolved["client_secret"] = _required_env(client_secret_env)
    resolved.pop("client_secret_env_var", None)
    if "scope" in resolved:
        resolved["scope"] = resolved["scope"].replace("__ZITADEL_PROJECT_ID__", project_id)
    resolved["expected_audience"] = project_id
    return resolved


class HostAuthIdpRuntime:
    """HTTP-only contract runtime for a deployment prepared by a provider script."""

    def __init__(self, context: AuthIdpDeploymentContext):
        load_credential_environment(context.credential_env_files)
        self.context = context
        self.gateway_base_url = context.gateway_url
        self.namespace = context.namespace or ""
        self.verify = str(context.ca_bundle)
        os.environ[NHX_CLIENT_SSL_CERT_FILE_ENVVAR] = self.verify
        self.discovery_url = context.provider_discovery_url
        self.token_endpoint = context.provider_token_endpoint
        self.workload_token_endpoint = f"{context.gateway_url}/apis/auth/token"
        self.provider = self._resolved_provider(load_provider_config(context.provider_manifest))
        self.case = AuthIdpCase(
            id=context.runtime_id,
            provider=self.provider,
            backend=context.backend,
            capabilities=frozenset(context.capabilities),
        )
        self._zitadel_login_client_pat = os.environ.get("ZITADEL_LOGIN_CLIENT_PAT")
        self._introspection_client_id = os.environ.get("ZITADEL_INTROSPECTION_CLIENT_ID")
        self._introspection_client_secret = os.environ.get("ZITADEL_INTROSPECTION_CLIENT_SECRET")
        self._oidc_test_driver = create_oidc_test_driver(
            gateway_base_url=self.gateway_base_url,
            provider_name=self.provider.name,
            zitadel_login_client_pat=self._zitadel_login_client_pat,
        )

    def _resolved_provider(self, provider: ProviderConfig) -> ProviderConfig:
        resolved = replace(
            provider,
            gateway_base_url=self.gateway_base_url,
            discovery_url=self.discovery_url,
            token_endpoint=self.token_endpoint,
        )
        if resolved.name != "zitadel":
            return resolved
        project_id = _required_env("ZITADEL_PROJECT_ID")
        return replace(
            resolved,
            e2e_setup_password_grant=_resolved_zitadel_grant(
                resolved.e2e_setup_password_grant,
                client_id_env="ZITADEL_E2E_SETUP_CLIENT_ID",
                client_secret_env="ZITADEL_E2E_SETUP_CLIENT_SECRET",
                project_id=project_id,
            ),
            workload_provider_password_grant=_resolved_zitadel_grant(
                resolved.workload_provider_password_grant,
                client_id_env="ZITADEL_WORKLOAD_CLIENT_ID",
                client_secret_env="ZITADEL_WORKLOAD_CLIENT_SECRET",
                project_id=project_id,
            ),
        )

    def assert_available(self) -> None:
        forward = self.context.port_forward
        if forward is None:
            return
        pid_text = forward.pid_file.read_text(encoding="utf-8").strip()
        if not pid_text.isdigit():
            raise AssertionError(f"invalid Kubernetes port-forward PID file: {forward.pid_file}")
        pid = int(pid_text)
        try:
            os.kill(pid, 0)
        except ProcessLookupError as exc:
            log_tail = _log_tail(forward.log_file)
            raise AssertionError(f"Kubernetes gateway port-forward {pid} exited\n{log_tail}") from exc
        command = subprocess.run(
            ["ps", "-p", str(pid), "-o", "args="],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        required_fragments = (
            "kubectl",
            f"--context {forward.kubernetes_context}",
            f"-n {forward.namespace}",
            "port-forward",
            f"svc/{forward.service}",
            f"{forward.local_port}:{forward.remote_port}",
        )
        missing = [fragment for fragment in required_fragments if fragment not in command]
        if missing:
            raise AssertionError(
                f"Kubernetes gateway port-forward {pid} does not match deployment context; "
                f"missing {missing!r} from {command!r}"
            )

    def e2e_setup_token(self) -> TokenSet:
        return self._token_for_grant(self.provider.e2e_setup_password_grant, "e2e setup")

    def interactive_user_token(self) -> TokenSet:
        return self._token_for_grant(self.provider.interactive_user_password_grant, "interactive user")

    def workload_provider_token(self) -> TokenSet:
        grant = self.provider.workload_provider_password_grant
        if grant is None:
            raise AssertionError("deployment does not configure a workload-provider grant")
        resolved = dict(grant)
        password_env_var = resolved.pop("password_env_var", None)
        if password_env_var and "password" not in resolved:
            resolved["password"] = _required_env(password_env_var)
        return self._token_for_grant(resolved, "workload provider")

    def workload_subject_token(self) -> str:
        if self.context.workload_token_file is not None:
            return self.context.workload_token_file.read_text(encoding="utf-8").strip()
        return self.workload_provider_token().access_token

    def exchange_workload_token(self, subject_token: str) -> TokenSet:
        response = httpx.post(
            self.workload_token_endpoint,
            data={
                "grant_type": TOKEN_EXCHANGE_GRANT_TYPE,
                "client_id": "nemo-helix-workload",
                "subject_token": subject_token,
                "subject_token_type": JWT_TOKEN_TYPE,
                "requested_token_type": ACCESS_TOKEN_TYPE,
                "audience": self.provider.workload_audience,
                "scope": "openid email groups",
            },
            timeout=TOKEN_EXCHANGE_TIMEOUT_SECONDS,
            verify=self.verify,
        )
        response.raise_for_status()
        payload = response.json()
        access_token = payload["access_token"]
        if payload.get("token_type", "").lower() != "bearer":
            raise AssertionError("workload token response did not return a bearer token")
        return TokenSet(access_token=access_token, claims=jwt_claims(access_token))

    def workload_platform_token(self) -> TokenSet:
        if self.case.backend == "compose":
            return self.workload_provider_token()
        return self.exchange_workload_token(self.workload_subject_token())

    def deployment_workload_runtime_config(self) -> DeploymentWorkloadRuntimeConfig:
        env = (
            {"name": "NHX_BASE_URL", "value": self.context.workload_gateway_url},
            {"name": "NHX_CLIENT_SSL_CERT_FILE", "value": self.context.deployment_ca_file},
            {"name": "SSL_CERT_FILE", "value": self.context.deployment_ca_file},
            {"name": "REQUESTS_CA_BUNDLE", "value": self.context.deployment_ca_file},
        )
        if self.case.backend == "compose":
            return DeploymentWorkloadRuntimeConfig(env=env)
        return DeploymentWorkloadRuntimeConfig(
            env=env,
            config_files=(
                {
                    "path": self.context.deployment_ca_file,
                    "content": self.context.ca_bundle.read_text(encoding="utf-8"),
                    "mode": 0o644,
                },
            ),
        )

    def e2e_setup_client(self) -> NemoClient:
        return self._client_for_token(self.e2e_setup_token().access_token)

    def interactive_user_client(self) -> NemoClient:
        return self._client_for_token(self.interactive_user_token().access_token)

    def workload_provider_client(self) -> NemoClient:
        return self._client_for_token(self.workload_platform_token().access_token)

    def workload_role_principals(self) -> list[str]:
        if self.case.backend == "kubernetes":
            return [f"system:serviceaccounts:{self.namespace}"]
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
        with httpx.Client(follow_redirects=False, **tls_config) as client:
            return self._oidc_test_driver.authenticate_device_flow(
                client,
                device_authorization_endpoint=device_authorization_endpoint,
                token_endpoint=token_endpoint,
                client_id=client_id,
                scope=scope,
                username=username,
                password=password,
            )

    def complete_confidential_authorization(
        self,
        *,
        authorization_url: str,
        username: str,
        password: str,
        tls_config: HttpxTLSConfig,
    ) -> str:
        with httpx.Client(follow_redirects=False, **tls_config) as client:
            return self._oidc_test_driver.complete_authorization(
                client,
                authorization_url=authorization_url,
                username=username,
                password=password,
            )

    def approve_device_authorization(
        self,
        *,
        verification_uri_complete: str,
        user_code: str,
        username: str,
        password: str,
        tls_config: HttpxTLSConfig,
    ) -> None:
        with httpx.Client(follow_redirects=False, **tls_config) as client:
            self._oidc_test_driver.user_automation.approve_device_authorization(
                client,
                gateway_base_url=self.gateway_base_url,
                verification_uri_complete=verification_uri_complete,
                user_code=user_code,
                username=username,
                password=password,
            )

    def cleanup(self) -> None:
        return

    def _token_for_grant(self, grant: dict[str, str] | None, description: str) -> TokenSet:
        if grant is None:
            raise AssertionError(f"deployment does not configure an {description} grant")
        token = exchange_token_with_retries(self.token_endpoint, grant, tls_config={"verify": self.verify})
        return TokenSet(access_token=token, claims=self._claims_for_token(token))

    def _claims_for_token(self, token: str) -> JsonObject:
        claims = jwt_claims(token)
        if claims:
            return claims
        if self.provider.name == "zitadel":
            return self._introspect_token(token)
        return {}

    def _introspect_token(self, token: str) -> JsonObject:
        if not self._introspection_client_id or not self._introspection_client_secret:
            raise AssertionError("ZITADEL opaque token introspection credentials are required")
        response = httpx.post(
            f"{self.gateway_base_url}/oauth/v2/introspect",
            data={"token": token},
            auth=(self._introspection_client_id, self._introspection_client_secret),
            timeout=TOKEN_EXCHANGE_TIMEOUT_SECONDS,
            verify=self.verify,
        )
        response.raise_for_status()
        claims = response.json()
        if claims.get("active") is not True:
            raise AssertionError("opaque token introspection returned inactive claims")
        if "groups" not in claims and ZITADEL_PROJECT_ROLES_CLAIM in claims:
            claims["groups"] = groups_from_claim(claims[ZITADEL_PROJECT_ROLES_CLAIM])
        return claims

    def _client_for_token(self, token: str) -> NemoClient:
        return NemoClient(
            base_url=self.gateway_base_url,
            auth=token,
            timeout=PLATFORM_API_REQUEST_TIMEOUT_SECONDS,
            http_client=httpx.Client(verify=self.verify),
            owns_http_client=True,
        )


def _log_tail(path: Path, lines: int = 40) -> str:
    if not path.is_file():
        return f"port-forward log is missing: {path}"
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
