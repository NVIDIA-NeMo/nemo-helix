# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, Protocol

from _pytest.mark.structures import Mark, MarkDecorator
from nemo_helix_ext.client.tls import HttpxTLSConfig
from nemo_helix_plugin.client.client import NemoClient

from tests.auth_idp.providers import ProviderConfig

AuthIdpBackend = Literal["compose", "kubernetes", "external"]
type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


@dataclass(frozen=True)
class AuthIdpCase:
    id: str
    provider: ProviderConfig
    backend: AuthIdpBackend
    capabilities: frozenset[str]
    marks: tuple[Mark | MarkDecorator, ...] = ()


@dataclass(frozen=True)
class TokenSet:
    access_token: str
    claims: JsonObject


@dataclass(frozen=True)
class DeploymentWorkloadRuntimeConfig:
    env: tuple[dict[str, str], ...] = ()
    config_files: tuple[JsonObject, ...] = ()


class AuthIdpRuntime(Protocol):
    @property
    def case(self) -> AuthIdpCase:
        raise NotImplementedError

    @property
    def gateway_base_url(self) -> str:
        raise NotImplementedError

    @property
    def discovery_url(self) -> str:
        raise NotImplementedError

    @property
    def token_endpoint(self) -> str | None:
        raise NotImplementedError

    @property
    def workload_token_endpoint(self) -> str | None:
        raise NotImplementedError

    def e2e_setup_token(self) -> TokenSet:
        raise NotImplementedError

    def interactive_user_token(self) -> TokenSet:
        raise NotImplementedError

    def workload_provider_token(self) -> TokenSet:
        raise NotImplementedError

    def workload_subject_token(self) -> str:
        raise NotImplementedError

    def exchange_workload_token(self, subject_token: str) -> TokenSet:
        raise NotImplementedError

    def workload_platform_token(self) -> TokenSet:
        raise NotImplementedError

    def deployment_workload_runtime_config(self) -> DeploymentWorkloadRuntimeConfig:
        raise NotImplementedError

    def e2e_setup_client(self) -> NemoClient:
        raise NotImplementedError

    def interactive_user_client(self) -> NemoClient:
        raise NotImplementedError

    def workload_provider_client(self) -> NemoClient:
        raise NotImplementedError

    def workload_role_principals(self) -> list[str]:
        raise NotImplementedError

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
        raise NotImplementedError

    def approve_device_authorization(
        self,
        *,
        verification_uri_complete: str,
        user_code: str,
        username: str,
        password: str,
        tls_config: HttpxTLSConfig,
    ) -> None:
        raise NotImplementedError

    def complete_confidential_authorization(
        self,
        *,
        authorization_url: str,
        username: str,
        password: str,
        tls_config: HttpxTLSConfig,
    ) -> str:
        """Complete provider login and return its authorization callback URL."""
        raise NotImplementedError

    def cleanup(self) -> None:
        raise NotImplementedError


class RuntimeManager(Protocol):
    def start(self, case: AuthIdpCase) -> Iterator[AuthIdpRuntime]:
        raise NotImplementedError
