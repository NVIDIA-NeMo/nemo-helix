# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ProviderRuntimeConfig:
    id: str
    backend: str
    capabilities: frozenset[str]
    command: str | None = None


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    mode: str
    compose_file: Path | None
    gateway_base_url: str
    issuer_url: str
    discovery_url: str
    nemo_config: Path
    interactive_user_username: str
    interactive_user_password: str
    interactive_user_expected_email: str
    workload_principal_id: str
    workload_expected_groups: list[str]
    workload_audience: str
    workload_principal_claim: str
    workload_groups_claim: str
    workload_groups_format: str
    workload_token_env_vars: list[str]
    workload_forwarded_headers: dict[str, str]
    token_endpoint: str | None
    e2e_setup_password_grant: dict[str, str] | None
    interactive_user_password_grant: dict[str, str] | None
    workload_provider_password_grant: dict[str, str] | None
    healthchecks: list[dict[str, str]]
    startup_timeouts: dict[str, int]
    test_runtimes: tuple[ProviderRuntimeConfig, ...] = ()
    compose_project_name: str | None = None


@dataclass(frozen=True)
class ProviderManifestSummary:
    name: str
    mode: str


def load_provider_config(manifest_path: Path) -> ProviderConfig:
    data = yaml.safe_load(manifest_path.read_text())
    token_acquisition = data.get("token_acquisition", {})
    return ProviderConfig(
        name=data["provider"],
        mode=data["mode"],
        compose_file=(None if not data.get("compose_file") else manifest_path.parent / data["compose_file"]),
        gateway_base_url=data["gateway_base_url"],
        issuer_url=data["issuer_url"],
        discovery_url=data["discovery_url"],
        nemo_config=manifest_path.parent / data["nemo_config"],
        interactive_user_username=data["interactive_user_identity"]["username"],
        interactive_user_password=_resolve_interactive_user_password(data["interactive_user_identity"]),
        interactive_user_expected_email=data["interactive_user_identity"]["expected_email"],
        workload_principal_id=data["workload_identity"]["principal_id"],
        workload_expected_groups=list(data["workload_identity"]["expected_groups"]),
        workload_audience=data["workload_contract"]["audience"],
        workload_principal_claim=data["workload_contract"]["principal_claim"],
        workload_groups_claim=data["workload_contract"]["groups_claim"],
        workload_groups_format=data["workload_contract"]["groups_format"],
        workload_token_env_vars=list(data["workload_contract"]["token_env_vars"]),
        workload_forwarded_headers=dict(data["workload_contract"]["forwarded_headers"]),
        token_endpoint=token_acquisition.get("token_endpoint"),
        e2e_setup_password_grant=_resolve_grant(
            token_acquisition.get("e2e_setup_grant") or token_acquisition.get("e2e_setup_password_grant")
        ),
        interactive_user_password_grant=_resolve_grant(
            token_acquisition.get("interactive_user_grant") or token_acquisition.get("interactive_user_password_grant")
        ),
        workload_provider_password_grant=_resolve_grant(
            token_acquisition.get("workload_provider_grant")
            or token_acquisition.get("workload_provider_password_grant")
        ),
        healthchecks=list(data.get("healthchecks", [])),
        startup_timeouts=dict(data.get("startup_timeouts", {})),
        test_runtimes=_load_provider_runtimes(data),
    )


def _load_provider_runtimes(data: dict) -> tuple[ProviderRuntimeConfig, ...]:
    return tuple(
        ProviderRuntimeConfig(
            id=runtime["id"],
            backend=runtime["backend"],
            command=runtime.get("command"),
            capabilities=frozenset(runtime["capabilities"]),
        )
        for runtime in data.get("test_runtimes", [])
    )


def _resolve_interactive_user_password(identity: dict[str, str]) -> str:
    password_env_var = identity.get("password_env_var")
    if password_env_var:
        return os.environ.get(password_env_var, "")
    return identity["password"]


def _resolve_grant(grant: dict[str, str] | None) -> dict[str, str] | None:
    if grant is None:
        return None
    resolved = dict(grant)
    password_env_var = resolved.pop("password_env_var", None)
    if password_env_var and "password" not in resolved:
        password = os.environ.get(password_env_var)
        if not password:
            resolved["password_env_var"] = password_env_var
            return resolved
        resolved["password"] = password
    client_secret_env_var = resolved.pop("client_secret_env_var", None)
    if client_secret_env_var and "client_secret" not in resolved:
        client_secret = os.environ.get(client_secret_env_var)
        if not client_secret:
            resolved["client_secret_env_var"] = client_secret_env_var
            return resolved
        resolved["client_secret"] = client_secret
    return resolved


def load_provider_configs() -> list[ProviderConfig]:
    configs: list[ProviderConfig] = []
    for manifest_path in sorted(Path("contrib/auth").glob("*/manifest.yaml")):
        configs.append(load_provider_config(manifest_path))
    return configs


def _load_provider_manifest_summaries() -> list[ProviderManifestSummary]:
    summaries: list[ProviderManifestSummary] = []
    for manifest_path in sorted(Path("contrib/auth").glob("*/manifest.yaml")):
        data = yaml.safe_load(manifest_path.read_text())
        summaries.append(ProviderManifestSummary(name=data["provider"], mode=data["mode"]))
    return summaries


def load_provider_configs_by_mode(mode: str) -> list[ProviderConfig]:
    return [provider for provider in load_provider_configs() if provider.mode == mode]


def load_provider_names() -> list[str]:
    return [provider.name for provider in _load_provider_manifest_summaries()]


def load_provider_names_by_mode(mode: str) -> list[str]:
    return [provider.name for provider in _load_provider_manifest_summaries() if provider.mode == mode]
