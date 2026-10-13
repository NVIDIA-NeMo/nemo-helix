# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E auth tests for the files service.

Non-auth file tests (upload, download, delete, etc.) have been migrated
to nemo-helix (e2e/files/test_files.py). Only auth-enabled tests remain here.
"""

import uuid
from collections.abc import Iterator

import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.storage_config import HuggingfaceStorageConfig
from nemo_helix_plugin.files.types import CreateFilesetRequest, FilesetOutput
from nemo_helix_plugin.schema import SecretRef
from nemo_helix_plugin.secrets.client import SecretsClient
from nemo_helix_plugin.secrets.types import HelixSecretCreateRequest, HelixSecretResponse
from pydantic import SecretStr

pytestmark = [pytest.mark.timeout(1800)]


@pytest.fixture
def fileset(client: NemoClient, workspace: str) -> Iterator[FilesetOutput]:
    """Create a unique fileset for each test with automatic cleanup.

    The fileset is automatically deleted after the test completes.
    """
    files = FilesClient.from_client(client)
    fileset_name = f"e2e-fileset-{uuid.uuid4().hex[:8]}"
    fileset = files.create_fileset(workspace=workspace, body=CreateFilesetRequest(name=fileset_name)).data()
    yield fileset
    try:
        files.delete_fileset(name=fileset_name, workspace=workspace)
    except Exception:
        pass  # Ignore cleanup errors


@pytest.fixture
def secret(client: NemoClient, workspace: str) -> Iterator[HelixSecretResponse]:
    """Create a unique secret for each test with automatic cleanup.

    The secret is automatically deleted after the test completes.
    """
    secrets = SecretsClient.from_client(client)
    secret_name = f"e2e-secret-{uuid.uuid4().hex[:8]}"
    secret = secrets.create_secret(
        workspace=workspace,
        body=HelixSecretCreateRequest(name=secret_name, value=SecretStr(f"e2e-test-value-{uuid.uuid4().hex[:8]}")),
    ).data()
    yield secret
    try:
        secrets.delete_secret(name=secret_name, workspace=workspace)
    except Exception:
        pass  # Ignore cleanup errors


@pytest.mark.feature("auth")
@pytest.mark.external_network
def test_fileset_create_hf_with_token_secret(
    client: NemoClient,
    workspace: str,
    secret: HelixSecretResponse,
):
    """Test auth flow for creating HuggingFace fileset with token_secret."""
    files = FilesClient.from_client(client)
    fileset_name = f"e2e-hf-token-{uuid.uuid4().hex[:8]}"
    try:
        fileset = files.create_fileset(
            workspace=workspace,
            body=CreateFilesetRequest(
                name=fileset_name,
                description="Auth regression coverage for HF token secret create flow",
                storage=HuggingfaceStorageConfig(
                    repo_id="Qwen/Qwen3-0.6B",
                    repo_type="model",
                    token_secret=SecretRef(secret.name),
                ),
            ),
        ).data()
        assert fileset.name == fileset_name
        assert fileset.workspace == workspace
        assert fileset.storage.type == "huggingface"
    finally:
        try:
            files.delete_fileset(name=fileset_name, workspace=workspace)
        except Exception:
            pass
