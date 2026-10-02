# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for external storage backends (NGC, Hugging Face).

These tests verify that the files service can create filesets backed by
external storage providers and read files from them via the SDK.

NGC tests require ``NGC_API_KEY`` in the environment and are skipped
otherwise.  Hugging Face tests use a small public repo; when ``HF_TOKEN``
is set the request is authenticated (avoids rate-limits in CI).
"""

import os
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import BadRequestError
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.storage_config import HuggingfaceStorageConfig, NGCStorageConfig
from nemo_helix_plugin.files.types import CreateFilesetRequest, ListFilesQueryParams
from nemo_helix_plugin.secrets.client import SecretsClient
from nemo_helix_plugin.secrets.types import HelixSecretCreateRequest

# ---------------------------------------------------------------------------
# NGC configuration
# ---------------------------------------------------------------------------
NGC_ORG = "nvidia"
NGC_TEAM = "nemo-microservices"
NGC_TARGET = "nemo-microservices-quickstart"
NGC_TARGET_TYPE = "resource"

# ---------------------------------------------------------------------------
# Hugging Face configuration — small public model
# ---------------------------------------------------------------------------
HF_TOKEN_ENV = "HF_TOKEN"

HF_REPO_ID = "hf-internal-testing/tiny-random-bert"
HF_REPO_TYPE = "model"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ngc_fileset(files_client: FilesClient, workspace: str, ngc_secret: str) -> Iterator[str]:
    """Create an NGC-backed fileset, cleaned up after test."""
    fileset_name = f"e2e-ngc-fs-{uuid.uuid4().hex[:8]}"
    files_client.create_fileset(
        workspace=workspace,
        body=CreateFilesetRequest(
            name=fileset_name,
            description="E2E test NGC-backed fileset",
            storage=NGCStorageConfig(
                api_key_secret=ngc_secret,
                org=NGC_ORG,
                team=NGC_TEAM,
                target=NGC_TARGET,
                target_type=NGC_TARGET_TYPE,
            ),
        ),
    )
    yield fileset_name
    try:
        files_client.delete_fileset(name=fileset_name, workspace=workspace)
    except Exception:
        pass


@pytest.fixture
def hf_token() -> str:
    """Return the HF token from the environment."""
    token = os.environ.get(HF_TOKEN_ENV)
    assert token, f"{HF_TOKEN_ENV} must be set"
    return token


@pytest.fixture
def hf_secret(client: NemoClient, workspace: str, hf_token: str) -> Iterator[str]:
    """Create a secret containing the HF token, cleaned up after test."""
    secret_name = f"e2e-hf-tok-{uuid.uuid4().hex[:8]}"
    secrets = SecretsClient.from_client(client)
    secrets.create_secret(workspace=workspace, body=HelixSecretCreateRequest(name=secret_name, value=hf_token))
    yield secret_name
    try:
        secrets.delete_secret(workspace=workspace, name=secret_name)
    except Exception:
        pass  # Best-effort cleanup; the workspace is deleted anyway


@pytest.fixture
def hf_fileset(files_client: FilesClient, workspace: str, hf_secret: str) -> Iterator[str]:
    """Create a Hugging Face-backed fileset, cleaned up after test."""
    fileset_name = f"e2e-hf-fs-{uuid.uuid4().hex[:8]}"

    storage = HuggingfaceStorageConfig(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        token_secret=hf_secret,
    )

    files_client.create_fileset(
        workspace=workspace,
        body=CreateFilesetRequest(
            name=fileset_name,
            description="E2E test HF-backed fileset",
            storage=storage,
        ),
    )
    yield fileset_name
    try:
        files_client.delete_fileset(name=fileset_name, workspace=workspace)
    except Exception:
        pass


# ===================================================================
# NGC tests
# ===================================================================


class TestNGCFileset:
    """Tests for NGC-backed filesets."""

    def test_list_files(self, client: NemoClient, workspace: str, ngc_fileset: str):
        """Listing an NGC-backed fileset returns files with paths and sizes."""
        files = FilesClient.from_client(client).list_files(name=ngc_fileset, workspace=workspace).data().data
        assert len(files) > 0, "NGC fileset should contain at least one file"

        for f in files:
            assert f.path, "Each file should have a path"
            assert f.size > 0, "Each file should have a non-zero size"

    def test_download_file(self, client: NemoClient, workspace: str, ngc_fileset: str):
        """Downloading the smallest file from an NGC fileset succeeds and size matches."""
        files = FilesClient.from_client(client).list_files(name=ngc_fileset, workspace=workspace).data().data
        assert len(files) > 0

        target = min(files, key=lambda f: f.size)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / target.path.replace("/", "_")
            content = (
                FilesClient.from_client(client)
                .download_file(
                    name=ngc_fileset,
                    workspace=workspace,
                    path=target.path,
                )
                .read()
            )
            local_path.write_bytes(content)
            assert local_path.exists()
            assert local_path.stat().st_size == target.size

    def test_cache_status(self, client: NemoClient, workspace: str, ngc_fileset: str):
        """NGC-backed files report a cacheable status."""
        files = (
            FilesClient.from_client(client)
            .list_files(
                name=ngc_fileset,
                workspace=workspace,
                query_params=ListFilesQueryParams(include_cache_status=True),
            )
            .data()
            .data
        )
        assert len(files) > 0

        for f in files:
            assert f.cache_status is not None
            assert f.cache_status != "not_cacheable"

    # -- error cases --

    @pytest.mark.parametrize(
        ("secret_value", "storage_overrides", "match"),
        [
            pytest.param(
                "not-a-real-key",
                {},
                "Invalid API key. Legacy NGC keys are not supported.",
                id="invalid-key-prefix",
            ),
            pytest.param(
                None,
                {"org": "nvidian", "team": "nemo-llm", "target": "nemo-helix-quickstart"},
                "Error creating NGC storage backend:",
                id="wrong-org",
            ),
            pytest.param(
                None,
                {"target": "this-resource-does-not-exist-12345"},
                "Failed to access NGC resource this-resource-does-not-exist-12345",
                id="nonexistent-resource",
            ),
        ],
    )
    def test_create_error(
        self,
        client: NemoClient,
        files_client: FilesClient,
        workspace: str,
        ngc_api_key: str,
        secret_value: str | None,
        storage_overrides: dict,
        match: str,
    ):
        """Bad NGC configurations are rejected with 400."""
        value = secret_value if secret_value is not None else ngc_api_key
        secret_name = f"e2e-ngc-err-{uuid.uuid4().hex[:8]}"
        secrets = SecretsClient.from_client(client)
        secrets.create_secret(workspace=workspace, body=HelixSecretCreateRequest(name=secret_name, value=value))
        try:
            storage = NGCStorageConfig(
                api_key_secret=secret_name,
                org=storage_overrides.get("org", NGC_ORG),
                team=storage_overrides.get("team", NGC_TEAM),
                target=storage_overrides.get("target", NGC_TARGET),
                target_type=NGC_TARGET_TYPE,
            )
            with pytest.raises(BadRequestError, match=match):
                files_client.create_fileset(
                    workspace=workspace,
                    body=CreateFilesetRequest(
                        name=f"e2e-ngc-err-{uuid.uuid4().hex[:8]}",
                        storage=storage,
                    ),
                )
        finally:
            secrets.delete_secret(workspace=workspace, name=secret_name)

    def test_create_error_nonexistent_secret(self, files_client: FilesClient, workspace: str):
        """Referencing a secret that doesn't exist is rejected with 400."""
        with pytest.raises(BadRequestError, match="Secret not found:"):
            files_client.create_fileset(
                workspace=workspace,
                body=CreateFilesetRequest(
                    name=f"e2e-ngc-err-{uuid.uuid4().hex[:8]}",
                    storage=NGCStorageConfig(
                        api_key_secret="no-such-secret-99999",
                        org=NGC_ORG,
                        team=NGC_TEAM,
                        target=NGC_TARGET,
                        target_type=NGC_TARGET_TYPE,
                    ),
                ),
            )


# ===================================================================
# Hugging Face tests
# ===================================================================


@pytest.mark.skipif(not os.environ.get(HF_TOKEN_ENV), reason=f"{HF_TOKEN_ENV} not set")
class TestHuggingFaceFileset:
    """Tests for Hugging Face-backed filesets."""

    def test_list_files(self, client: NemoClient, workspace: str, hf_fileset: str):
        """Listing an HF-backed fileset returns files with paths and sizes."""
        files = FilesClient.from_client(client).list_files(name=hf_fileset, workspace=workspace).data().data
        assert len(files) > 0, "HF fileset should contain at least one file"

        for f in files:
            assert f.path, "Each file should have a path"
            assert f.size > 0, "Each file should have a non-zero size"

    def test_download_file(self, client: NemoClient, workspace: str, hf_fileset: str):
        """Downloading the smallest file from an HF fileset succeeds and size matches."""
        files = FilesClient.from_client(client).list_files(name=hf_fileset, workspace=workspace).data().data
        assert len(files) > 0

        target = min(files, key=lambda f: f.size)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / target.path.replace("/", "_")
            content = (
                FilesClient.from_client(client)
                .download_file(
                    name=hf_fileset,
                    workspace=workspace,
                    path=target.path,
                )
                .read()
            )
            local_path.write_bytes(content)
            assert local_path.exists()
            assert local_path.stat().st_size == target.size

    def test_cache_status(self, client: NemoClient, workspace: str, hf_fileset: str):
        """HF-backed files report a cacheable status."""
        files = (
            FilesClient.from_client(client)
            .list_files(
                name=hf_fileset,
                workspace=workspace,
                query_params=ListFilesQueryParams(include_cache_status=True),
            )
            .data()
            .data
        )
        assert len(files) > 0

        for f in files:
            assert f.cache_status is not None
            assert f.cache_status != "not_cacheable"

    # -- error cases --

    def test_error_nonexistent_repo(self, files_client: FilesClient, workspace: str):
        """Pointing at a repo that doesn't exist is rejected with 400."""
        with pytest.raises(BadRequestError):
            files_client.create_fileset(
                workspace=workspace,
                body=CreateFilesetRequest(
                    name=f"e2e-hf-err-{uuid.uuid4().hex[:8]}",
                    storage=HuggingfaceStorageConfig(
                        repo_id="this-org-does-not-exist/this-repo-does-not-exist-12345",
                        repo_type="model",
                    ),
                ),
            )
