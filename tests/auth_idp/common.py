# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import os
import shlex
from typing import Protocol, runtime_checkable

import pytest
from nemo_helix_ext.client.tls import HttpxTLSConfig, httpx_tls_config_from_env

from tests.auth_idp.runtime_contract import AuthIdpCase, AuthIdpRuntime, JsonObject


@runtime_checkable
class RuntimeTLSConfigProvider(Protocol):
    @property
    def verify(self) -> str:
        raise NotImplementedError


def jwt_claims(token: str) -> JsonObject:
    parts = token.split(".")
    if len(parts) != 3:
        return {}
    payload = parts[1] + "=" * (-len(parts[1]) % 4)
    decoded = json.loads(base64.urlsafe_b64decode(payload))
    if not isinstance(decoded, dict):
        return {}
    return decoded


def require_capability(case: AuthIdpCase, capability: str) -> None:
    if capability not in case.capabilities:
        pytest.skip(f"{case.id} does not declare auth-idp capability: {capability}")


def nhx_api_image() -> str:
    registry = os.environ.get("IMAGE_REGISTRY", "my-registry")
    tag = os.environ.get("BAKE_TAG", "local")
    return f"{registry}/nhx-api:{tag}"


def managed_workload_workspace_get_command(*, task_config: JsonObject | None = None) -> str:
    command = (
        'if [ -n "${NHX_PRINCIPAL:-}" ]; then '
        "echo 'Unexpected NHX_PRINCIPAL in managed workload'; exit 42; "
        "fi; "
        'if [ -z "${NHX_WORKLOAD_IDENTITY_TOKEN_FILE:-}" ]; then '
        "echo 'Missing NHX_WORKLOAD_IDENTITY_TOKEN_FILE in managed workload'; exit 43; "
        "fi; "
        'if [ ! -f "${NHX_WORKLOAD_IDENTITY_TOKEN_FILE}" ]; then '
        "echo 'Workload identity token file is missing'; exit 44; "
        "fi; "
        "echo 'Workload auth env: NHX_PRINCIPAL=absent NHX_WORKLOAD_IDENTITY_TOKEN_FILE=present'; "
        "exec nemo-helix run task --task nhx.hello_world.tasks.workload_workspace_get"
    )
    if task_config is None:
        return command
    config_json = json.dumps(task_config, separators=(",", ":"))
    return f"{command} --config {shlex.quote(config_json)}"


def runtime_tls_config(auth_idp_runtime: AuthIdpRuntime) -> HttpxTLSConfig:
    if isinstance(auth_idp_runtime, RuntimeTLSConfigProvider):
        ca_bundle = auth_idp_runtime.verify.strip()
        if not ca_bundle:
            raise AssertionError("Auth IDP runtime verify must be a non-empty CA bundle path")
        return {"verify": ca_bundle}
    return httpx_tls_config_from_env()
