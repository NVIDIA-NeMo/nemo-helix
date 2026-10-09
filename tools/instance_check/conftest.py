# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session fixtures for a manual acceptance run against one platform context."""

from __future__ import annotations

import hashlib
import os
import secrets
import uuid
from collections.abc import Iterator

import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.oidc import NHXOIDCConfig, discover_nhx_config
from nemo_helix_plugin.models.client import ModelsClient
from nemo_helix_plugin.secrets.client import SecretsClient
from nemo_helix_plugin.secrets.types import HelixSecretCreateRequest
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest
from pydantic import SecretStr
from support import (
    ensure_fileset,
    load_settings,
    resolve_check_image,
    retry_until_workspace_granted,
)


@pytest.fixture(scope="session")
def platform_client() -> NemoClient:
    """Client for the context selected by ``tools/instance_check/check.py``."""
    return NemoClient.from_config(context=os.environ.get("NHX_CURRENT_CONTEXT") or None)


@pytest.fixture(scope="session")
def auth_discovery(platform_client: NemoClient) -> NHXOIDCConfig:
    """OIDC settings from the platform. Auth flags drive the workload assertions."""
    return discover_nhx_config(platform_client.base_url)


@pytest.fixture(scope="session")
def workspace(platform_client: NemoClient) -> Iterator[str]:
    """Workspace under test. A user-supplied name is left in place."""
    settings = load_settings()
    name = settings.workspace or f"check-{uuid.uuid4().hex[:8]}"
    created = settings.workspace is None
    workspaces = WorkspacesClient.from_client(platform_client)
    if created:
        workspaces.create_workspace(body=CreateWorkspaceRequest(name=name))
    else:
        workspaces.get_workspace(name=name)
    try:
        ensure_fileset(platform_client, name)
        yield name
    finally:
        if created and not settings.keep:
            try:
                workspaces.delete_workspace(name=name)
            except Exception as exc:
                print(f"cleanup workspace {name} failed: {type(exc).__name__}: {exc}", flush=True)


@pytest.fixture(scope="session")
def check_secret(platform_client: NemoClient, workspace: str) -> Iterator[dict[str, str]]:
    """Secret injected into jobs and deployments. The value is not printed."""
    value = secrets.token_urlsafe(24)
    name = f"check-sec-{uuid.uuid4().hex[:8]}"
    secrets_client = SecretsClient.from_client(platform_client)
    retry_until_workspace_granted(
        lambda: secrets_client.create_secret(
            workspace=workspace,
            body=HelixSecretCreateRequest(name=name, value=SecretStr(value)),
        )
    )
    yield {
        "name": name,
        "value": value,
        "sha256": hashlib.sha256(value.encode()).hexdigest(),
    }
    try:
        secrets_client.delete_secret(workspace=workspace, name=name)
    except Exception as exc:
        print(f"cleanup secret {name} failed: {type(exc).__name__}: {exc}", flush=True)


@pytest.fixture(scope="session")
def selected_model(platform_client: NemoClient, workspace: str) -> str | None:
    """Model used by the optional chat and workload probes."""
    settings = load_settings()
    if not settings.models:
        return None
    if settings.model:
        return settings.model
    listed = ModelsClient.from_client(platform_client).list_models(workspace=workspace)
    for item in listed.items():
        if item.name:
            return item.name if "/" in item.name else f"{workspace}/{item.name}"
    return None


@pytest.fixture(scope="session")
def task_image(platform_client: NemoClient) -> str | None:
    """Image for deployment probes. Jobs can omit it and use the profile default."""
    settings = load_settings()
    image = resolve_check_image(platform_client, settings.image, settings.job_profile or "default")
    if image:
        print(f"deployment task image: {image}", flush=True)
    return image
