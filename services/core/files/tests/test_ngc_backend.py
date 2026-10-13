# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NGC storage backend."""

from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest
from nhx.common.api.common import SecretRef
from nhx.core.files.app.backends.base import ByteRange
from nhx.core.files.app.backends.factory import storage_impl_factory
from nhx.core.files.app.backends.ngc import (
    NGCBackendError,
    NGCStorageConfig,
    NGCStorageImpl,
)

_FULL_API_KEY = "nvapi-test-api-key"


@pytest.fixture
def ngc_config() -> NGCStorageConfig:
    """Create a test NGC storage config (private org, resource type)."""
    return NGCStorageConfig(
        org="test-org",
        team="test-team",
        target="test-resource",
        api_key_secret=SecretRef(root="test-api-secret"),
    )


@pytest.fixture
def mock_ngc_client():
    """Mock NGC SDK client and registry API classes (ResourceAPI, ModelAPI, GuestModelAPI).
    Resources always use ResourceAPI; models use GuestModelAPI or ModelAPI."""
    with (
        patch("nhx.core.files.app.backends.ngc.Client") as mock_client_cls,
        patch("nhx.core.files.app.backends.ngc.ResourceAPI") as mock_resource_api_cls,
        patch("nhx.core.files.app.backends.ngc.ModelAPI") as mock_model_api_cls,
        patch("nhx.core.files.app.backends.ngc.GuestModelAPI") as mock_guest_model_api_cls,
    ):
        mock_client = Mock()

        # Create mock API instances with list_versions returning a version
        mock_version = Mock()
        mock_version.versionId = "1.0"

        apis = {}
        for name, cls in [
            ("resource_api", mock_resource_api_cls),
            ("model_api", mock_model_api_cls),
            ("guest_model_api", mock_guest_model_api_cls),
        ]:
            mock_api = Mock()
            mock_api.list_versions.return_value = iter([mock_version])
            cls.return_value = mock_api
            apis[name] = mock_api

        mock_client_cls.return_value = mock_client
        # Client has a config object used for org/team (e.g. set after configure for public org).
        mock_client.config = Mock()
        mock_client.config.org_name = None
        mock_client.config.team_name = None
        apis["client"] = mock_client

        yield apis


@pytest.fixture
def ngc_secrets() -> dict[str, str]:
    """Mock resolved secrets. Key must start with SCOPED_KEY_PREFIX (legacy keys rejected)."""
    return {"api_key": _FULL_API_KEY}


# ---- API class selection tests (2x2 matrix) ----


@pytest.mark.parametrize(
    "org,team,target,target_type,expected_api_key,expected_configure_kwargs",
    [
        # Private org + resource -> ResourceAPI
        (
            "test-org",
            "test-team",
            "test-resource",
            "resource",
            "resource_api",
            {
                "api_key": _FULL_API_KEY,
                "org_name": "test-org",
                "team_name": "test-team",
            },
        ),
        # Private org + model -> ModelAPI
        (
            "test-org",
            "test-team",
            "test-model",
            "model",
            "model_api",
            {
                "api_key": _FULL_API_KEY,
                "org_name": "test-org",
                "team_name": "test-team",
            },
        ),
        # Public org + resource -> ResourceAPI (GuestResourceAPI not used)
        (
            "nvidia",
            "nemo-helix",
            "nemo-quickstart",
            "resource",
            "resource_api",
            {"api_key": _FULL_API_KEY, "team_name": "no-team"},
        ),
        # Public org + model -> GuestModelAPI
        (
            "nvidia",
            "nemo",
            "llama-3_2-1b-instruct",
            "model",
            "guest_model_api",
            {"api_key": _FULL_API_KEY, "team_name": "no-team"},
        ),
    ],
    ids=["private-resource", "private-model", "public-resource", "public-model"],
)
async def test_api_class_selection(
    mock_ngc_client,
    ngc_secrets,
    org,
    team,
    target,
    target_type,
    expected_api_key,
    expected_configure_kwargs,
):
    """Test that the correct NGC API class is selected based on org visibility and target type."""
    config = NGCStorageConfig(
        org=org,
        team=team,
        target=target,
        target_type=target_type,
        api_key_secret=SecretRef(root="test-api-secret"),
    )
    impl = NGCStorageImpl(config, ngc_secrets)

    # Trigger lazy initialization
    await impl._get_version()

    mock_ngc_client["client"].configure.assert_called_once_with(**expected_configure_kwargs)
    assert impl._registry_api is mock_ngc_client[expected_api_key]
    assert impl._version == "1.0"


async def test_target_type_defaults_to_resource(mock_ngc_client, ngc_secrets):
    """target_type should default to 'resource' for backward compatibility."""
    config = NGCStorageConfig(
        org="test-org",
        team="test-team",
        target="test-resource",
        api_key_secret=SecretRef(root="test-api-secret"),
    )
    assert config.target_type == "resource"
    impl = NGCStorageImpl(config, ngc_secrets)

    # Trigger lazy initialization
    await impl._get_version()

    assert impl._registry_api is mock_ngc_client["resource_api"]


def test_legacy_key_rejected(ngc_config, mock_ngc_client):
    """Legacy API keys (without scoped prefix) are rejected with NGCBackendError."""
    from nhx.core.files.app.backends.ngc import NGCBackendError

    legacy_secrets = {"api_key": "ngc-legacy-key"}
    with pytest.raises(NGCBackendError, match="Legacy NGC keys are not supported"):
        NGCStorageImpl(ngc_config, legacy_secrets)


# ---- Initialization behavior tests ----


async def test_initialization_with_latest_version(ngc_config, mock_ngc_client, ngc_secrets):
    """Test that NGC storage impl fetches latest version when not specified."""
    mock_version = Mock()
    mock_version.versionId = "2.5"
    mock_ngc_client["resource_api"].list_versions.return_value = iter([mock_version])

    impl = NGCStorageImpl(ngc_config, ngc_secrets)

    # Trigger lazy initialization
    await impl._get_version()

    # Verify list_versions was called with positional args
    mock_ngc_client["resource_api"].list_versions.assert_called_once_with(
        "test-org",
        "test-team",
        "test-resource",
    )
    assert impl._version == "2.5"


async def test_initialization_no_versions_available(ngc_config, mock_ngc_client, ngc_secrets):
    """Test that initialization fails gracefully when no versions are available."""
    mock_ngc_client["resource_api"].list_versions.return_value = iter([])

    impl = NGCStorageImpl(ngc_config, ngc_secrets)
    with pytest.raises(NGCBackendError, match="No versions found"):
        await impl._get_version()


async def test_initialization_invalid_credentials(ngc_config, mock_ngc_client, ngc_secrets):
    """Test that initialization fails gracefully with invalid NGC credentials."""
    mock_ngc_client["client"].configure.side_effect = ValueError(
        "Invalid apikey for NGC service location [https://api.ngc.nvidia.com]."
    )

    impl = NGCStorageImpl(ngc_config, ngc_secrets)
    with pytest.raises(NGCBackendError, match="Error creating NGC storage backend"):
        await impl._get_version()


# ---- Signed URL tests ----


async def test_get_signed_url_success(ngc_config, mock_ngc_client, ngc_secrets):
    """Test successful signed URL retrieval."""
    mock_ngc_client[
        "resource_api"
    ].get_direct_download_URL.return_value = (
        "v2/org/test-org/team/test-team/resources/test-resource/versions/1.0/files/test.txt"
    )

    with patch(
        "nhx.core.files.app.backends.ngc.download_url",
        new_callable=AsyncMock,
    ) as mock_download:
        mock_download.return_value = {"urls": ["https://signed-url.com/test.txt"]}

        impl = NGCStorageImpl(ngc_config, ngc_secrets)
        signed_url = await impl._get_signed_url("test.txt")

        assert signed_url == "https://signed-url.com/test.txt"
        mock_download.assert_called_once()


async def test_get_signed_url_not_found(ngc_config, mock_ngc_client, ngc_secrets):
    """Test signed URL retrieval with 404 error."""
    mock_ngc_client[
        "resource_api"
    ].get_direct_download_URL.return_value = (
        "v2/org/test-org/team/test-team/resources/test-resource/versions/1.0/files/missing.txt"
    )

    with patch(
        "nhx.core.files.app.backends.ngc.download_url",
        new_callable=AsyncMock,
    ) as mock_download:
        mock_download.side_effect = aiohttp.ClientResponseError(
            request_info=Mock(),
            history=(),
            status=404,
            message="Not Found",
        )

        impl = NGCStorageImpl(ngc_config, ngc_secrets)

        with pytest.raises(NGCBackendError) as exc_info:
            await impl._get_signed_url("missing.txt")

        assert "404" in str(exc_info.value)


async def test_get_signed_url_network_error(ngc_config, mock_ngc_client, ngc_secrets):
    """Test signed URL retrieval with network error."""
    mock_ngc_client[
        "resource_api"
    ].get_direct_download_URL.return_value = (
        "v2/org/test-org/team/test-team/resources/test-resource/versions/1.0/files/test.txt"
    )

    with patch(
        "nhx.core.files.app.backends.ngc.download_url",
        new_callable=AsyncMock,
    ) as mock_download:
        mock_download.side_effect = aiohttp.ClientError("Connection failed")

        impl = NGCStorageImpl(ngc_config, ngc_secrets)

        with pytest.raises(NGCBackendError) as exc_info:
            await impl._get_signed_url("test.txt")

        assert "Network error" in str(exc_info.value)


# ---- File listing tests ----


async def test_list_files_no_filter(ngc_config, mock_ngc_client, ngc_secrets):
    """Test listing all files without path filter."""
    mock_file1 = Mock()
    mock_file1.path = "file1.txt"
    mock_file1.sizeInBytes = 100

    mock_file2 = Mock()
    mock_file2.path = "dir/file2.txt"
    mock_file2.sizeInBytes = 200

    mock_ngc_client["resource_api"].list_files.return_value = [mock_file1, mock_file2]

    impl = NGCStorageImpl(ngc_config, ngc_secrets)
    files = await impl.list_files()

    assert len(files) == 2
    assert files[0].path == "file1.txt"
    assert files[0].size == 100
    assert files[1].path == "dir/file2.txt"
    assert files[1].size == 200


async def test_list_files_with_prefix_filter(ngc_config, mock_ngc_client, ngc_secrets):
    """Test listing files with path prefix filter."""
    mock_file1 = Mock()
    mock_file1.path = "dir/file1.txt"
    mock_file1.sizeInBytes = 100

    mock_file2 = Mock()
    mock_file2.path = "dir/file2.txt"
    mock_file2.sizeInBytes = 200

    mock_file3 = Mock()
    mock_file3.path = "other/file3.txt"
    mock_file3.sizeInBytes = 300

    mock_ngc_client["resource_api"].list_files.return_value = [
        mock_file1,
        mock_file2,
        mock_file3,
    ]

    impl = NGCStorageImpl(ngc_config, ngc_secrets)
    files = await impl.list_files("dir/")

    assert len(files) == 2
    assert all(f.path.startswith("dir/") for f in files)


# ---- Download tests ----


async def test_download_success(ngc_config, mock_ngc_client, ngc_secrets):
    """Test successful file download."""
    mock_ngc_client[
        "resource_api"
    ].get_direct_download_URL.return_value = (
        "v2/org/test-org/team/test-team/resources/test-resource/versions/1.0/files/test.txt"
    )

    with (
        patch(
            "nhx.core.files.app.backends.ngc.download_url",
            new_callable=AsyncMock,
        ) as mock_get_url,
        patch("nhx.core.files.app.backends.ngc.download_url_streaming") as mock_stream,
    ):
        mock_get_url.return_value = {"urls": ["https://signed-url.com/test.txt"]}

        async def mock_chunks():
            yield b"chunk1"
            yield b"chunk2"

        mock_stream.return_value = mock_chunks()

        impl = NGCStorageImpl(ngc_config, ngc_secrets)
        download_iter = await impl.download("test.txt", None)

        chunks = []
        async for chunk in download_iter:
            chunks.append(chunk)

        assert chunks == [b"chunk1", b"chunk2"]


async def test_download_with_byte_range(ngc_config, mock_ngc_client, ngc_secrets):
    """Test file download with byte range."""
    mock_ngc_client[
        "resource_api"
    ].get_direct_download_URL.return_value = (
        "v2/org/test-org/team/test-team/resources/test-resource/versions/1.0/files/test.txt"
    )

    byte_range = ByteRange(start=0, end=100)

    with (
        patch(
            "nhx.core.files.app.backends.ngc.download_url",
            new_callable=AsyncMock,
        ) as mock_get_url,
        patch("nhx.core.files.app.backends.ngc.download_url_streaming") as mock_stream,
    ):
        mock_get_url.return_value = {"urls": ["https://signed-url.com/test.txt"]}

        async def mock_chunks():
            yield b"partial"

        mock_stream.return_value = mock_chunks()

        impl = NGCStorageImpl(ngc_config, ngc_secrets)
        download_iter = await impl.download("test.txt", byte_range)

        chunks = []
        async for chunk in download_iter:
            chunks.append(chunk)

        mock_stream.assert_called_once()
        call_args = mock_stream.call_args
        assert call_args.kwargs["byte_range"] == byte_range
        assert chunks == [b"partial"]


async def test_download_file_not_found(ngc_config, mock_ngc_client, ngc_secrets):
    """Test download when file is not found."""
    mock_ngc_client[
        "resource_api"
    ].get_direct_download_URL.return_value = (
        "v2/org/test-org/team/test-team/resources/test-resource/versions/1.0/files/missing.txt"
    )

    with (
        patch(
            "nhx.core.files.app.backends.ngc.download_url",
            new_callable=AsyncMock,
        ) as mock_get_url,
        patch("nhx.core.files.app.backends.ngc.download_url_streaming") as mock_stream,
    ):
        mock_get_url.return_value = {"urls": ["https://signed-url.com/missing.txt"]}

        async def mock_chunks():
            raise aiohttp.ClientResponseError(
                request_info=Mock(),
                history=(),
                status=404,
                message="Not Found",
            )
            yield  # pragma: no cover

        mock_stream.return_value = mock_chunks()

        impl = NGCStorageImpl(ngc_config, ngc_secrets)
        download_iter = await impl.download("missing.txt", None)

        with pytest.raises(NGCBackendError):
            async for _ in download_iter:
                pass


# ---- Validation tests ----


async def test_validate_storage_success(ngc_config, mock_ngc_client, ngc_secrets):
    """Test successful storage validation."""
    mock_ngc_client["resource_api"].info.return_value = {"name": "test-resource"}

    impl = NGCStorageImpl(ngc_config, ngc_secrets)
    await impl.validate_storage()

    mock_ngc_client["resource_api"].info.assert_called_once()


async def test_validate_storage_not_found(ngc_config, mock_ngc_client, ngc_secrets):
    """Test storage validation when resource not found."""
    from ngcbase.errors import ResourceNotFoundException

    mock_ngc_client["resource_api"].info.side_effect = ResourceNotFoundException("Resource not found")

    impl = NGCStorageImpl(ngc_config, ngc_secrets)

    with pytest.raises(NGCBackendError) as exc_info:
        await impl.validate_storage()

    assert "not found" in str(exc_info.value)


async def test_validate_storage_failure(ngc_config, mock_ngc_client, ngc_secrets):
    """Test storage validation failure."""
    mock_ngc_client["resource_api"].info.side_effect = Exception("Access denied")

    impl = NGCStorageImpl(ngc_config, ngc_secrets)

    with pytest.raises(NGCBackendError) as exc_info:
        await impl.validate_storage()

    assert "Failed to access" in str(exc_info.value)


# ---- Not implemented tests ----


async def test_upload_not_implemented(ngc_config, mock_ngc_client, ngc_secrets):
    """Test that upload raises NotImplementedError."""
    impl = NGCStorageImpl(ngc_config, ngc_secrets)

    async def dummy_stream():
        yield b"data"

    with pytest.raises(NotImplementedError):
        await impl.upload("test.txt", dummy_stream())


async def test_delete_not_implemented(ngc_config, mock_ngc_client, ngc_secrets):
    """Test that delete raises NotImplementedError."""
    impl = NGCStorageImpl(ngc_config, ngc_secrets)

    with pytest.raises(NotImplementedError):
        await impl.delete("test.txt")


def test_ngc_config_does_not_own_storage_data(ngc_config):
    """NGC is read-only external storage; the platform does not own the source data."""
    assert ngc_config.owns_storage_data is False


# ---- Factory tests ----


def test_factory_creates_ngc_impl(ngc_secrets):
    """Test that factory correctly creates NGC storage impl."""
    config = NGCStorageConfig(
        org="test-org",
        team="test-team",
        target="test-resource",
        version="1.0",
        api_key_secret=SecretRef(root="test-api-secret"),
    )

    with (
        patch("nhx.core.files.app.backends.ngc.Client"),
        patch("nhx.core.files.app.backends.ngc.ResourceAPI"),
    ):
        impl = storage_impl_factory(config, ngc_secrets)

        assert isinstance(impl, NGCStorageImpl)
        assert impl.config == config


# ---- Cache path tests ----


@pytest.mark.parametrize(
    "path,expected_suffix",
    [
        ("models/model.bin", "/models/model.bin"),
        (None, ""),
    ],
    ids=["with-path", "without-path"],
)
async def test_get_cache_path_key(ngc_config, mock_ngc_client, ngc_secrets, path, expected_suffix):
    """Test get_cache_path_key returns correct cache path with and without file path."""
    impl = NGCStorageImpl(ngc_config, ngc_secrets)

    cache_key = await impl.get_cache_path_key(path)

    expected = f"cache/ngc/test-org/test-team/test-resource/1.0{expected_suffix}"
    assert cache_key == expected


async def test_get_cache_path_key_with_explicit_version(ngc_secrets):
    """Test get_cache_path_key uses explicit version from config."""
    config = NGCStorageConfig(
        org="nvidia",
        team="nemo",
        target="llama-model",
        target_type="model",
        version="2.5",
        api_key_secret=SecretRef(root="test-secret"),
    )

    with (
        patch("nhx.core.files.app.backends.ngc.Client"),
        patch("nhx.core.files.app.backends.ngc.GuestModelAPI"),
    ):
        impl = NGCStorageImpl(config, ngc_secrets)

        cache_key = await impl.get_cache_path_key("weights.pt")
        assert cache_key == "cache/ngc/nvidia/nemo/llama-model/2.5/weights.pt"

        cache_prefix = await impl.get_cache_path_key()
        assert cache_prefix == "cache/ngc/nvidia/nemo/llama-model/2.5"


async def test_resolve_config_with_latest_version(ngc_config, mock_ngc_client, ngc_secrets):
    """Test resolve_config stores original_version=None when version was not specified.

    When user doesn't specify a version, NGC resolves to the latest available version.
    resolve_config() should record this for auditing.
    """
    # ngc_config has version=None, mock_ngc_client resolves to "1.0"
    impl = NGCStorageImpl(ngc_config, ngc_secrets)
    resolved_config = await impl.resolve_config()

    # Original version was None (user requested "latest")
    assert resolved_config.original_version is None
    # Resolved version is the actual version ID
    assert resolved_config.version == "1.0"
    # Other fields unchanged
    assert resolved_config.org == ngc_config.org
    assert resolved_config.team == ngc_config.team
    assert resolved_config.target == ngc_config.target


async def test_resolve_config_with_explicit_version(ngc_secrets):
    """Test resolve_config stores original_version when version was explicitly provided."""
    config = NGCStorageConfig(
        org="nvidia",
        team="nemo",
        target="llama-model",
        version="2.5",  # User explicitly specified version
        api_key_secret=SecretRef(root="test-secret"),
    )

    with (
        patch("nhx.core.files.app.backends.ngc.Client"),
        patch("nhx.core.files.app.backends.ngc.ResourceAPI"),
    ):
        impl = NGCStorageImpl(config, ngc_secrets)
        resolved_config = await impl.resolve_config()

        # Original version matches what user provided
        assert resolved_config.original_version == "2.5"
        # Resolved version is the same (explicit version is already resolved)
        assert resolved_config.version == "2.5"


async def test_resolve_config_preserves_immutable_version(mock_ngc_client, ngc_secrets):
    """Test that after resolve_config, the version is immutable.

    This ensures cache paths won't change even if "latest" moves to a new version.
    """
    # First request with version=None gets "1.0"
    config1 = NGCStorageConfig(
        org="test-org",
        team="test-team",
        target="test-resource",
        api_key_secret=SecretRef(root="test-secret"),
    )

    impl1 = NGCStorageImpl(config1, ngc_secrets)
    resolved1 = await impl1.resolve_config()

    # Simulate "latest" moving to a new version
    mock_version_new = Mock()
    mock_version_new.versionId = "2.0"
    mock_ngc_client["resource_api"].list_versions.return_value = iter([mock_version_new])

    # New request would get "2.0"
    config2 = NGCStorageConfig(
        org="test-org",
        team="test-team",
        target="test-resource",
        api_key_secret=SecretRef(root="test-secret"),
    )
    impl2 = NGCStorageImpl(config2, ngc_secrets)
    resolved2 = await impl2.resolve_config()

    # First fileset is pinned to 1.0
    assert resolved1.version == "1.0"
    # Second fileset gets 2.0
    assert resolved2.version == "2.0"
    # Both stored original_version=None (requested latest)
    assert resolved1.original_version is None
    assert resolved2.original_version is None
