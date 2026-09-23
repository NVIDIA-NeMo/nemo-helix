# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for config store population - migrated to EntityClient."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nhx.guardrails.app.utils.config_store import populate_config_store
from nhx.guardrails.entities import GuardrailConfig
from nhx.guardrails.entities.values._private import Model, RailsConfig

# Path used when config_store_path is required (populate_config_store no longer uses settings).
MOCK_CONFIG_STORE_PATH = Path("/mock/config/store/path")


@pytest.fixture
def mock_entities_client():
    return AsyncMock()


@pytest.fixture
def mock_fs():
    """Fixture to create a mock fsspec filesystem."""
    fs = MagicMock()
    fs.exists = MagicMock(return_value=True)
    fs.ls = MagicMock(
        return_value=[
            "/path/to/config_store/config1",
            "/path/to/config_store/config2",
            "/path/to/config_store/file.txt",  # not a dir
        ]
    )
    fs.isdir = MagicMock(side_effect=lambda path: not str(path).endswith(".txt"))
    return fs


@pytest.fixture
def mock_fsspec(mock_fs):
    """Fixture to mock fsspec.filesystem."""
    with patch("nhx.guardrails.app.utils.config_store.fsspec.filesystem", return_value=mock_fs):
        yield mock_fs


@pytest.fixture
def mock_rails_config():
    """Fixture to mock RailsConfig.from_path."""
    config = RailsConfig()
    with patch("nhx.guardrails.app.utils.config_store.RailsConfig.from_path", return_value=config):
        yield config


@pytest.fixture
def mock_entity_not_found():
    """Fixture to simulate EntityNotFoundError."""
    from nhx.common.entities.client import EntityNotFoundError

    return EntityNotFoundError("Not found")


@pytest.mark.asyncio
async def test_populate_config_store_normal_operation(
    mock_entities_client, mock_fsspec, mock_entity_not_found, mock_rails_config
):
    """Test normal operation where guardrail configs do not exist."""
    # EntityClient.get raises EntityNotFoundError when config doesn't exist
    mock_entities_client.get = AsyncMock(side_effect=mock_entity_not_found)
    mock_entities_client.create = AsyncMock()

    await populate_config_store(mock_entities_client, MOCK_CONFIG_STORE_PATH)

    # Should have tried to create 2 configs (config1 and config2, not file.txt)
    assert mock_entities_client.create.call_count == 2


@pytest.mark.asyncio
async def test_populate_config_store_configs_already_exist(mock_entities_client, mock_fsspec):
    """Test that existing GuardrailConfigs are not added again."""
    # Return existing config for all get_by_name calls
    mock_config = MagicMock()
    mock_entities_client.get = AsyncMock(return_value=mock_config)
    mock_entities_client.create = AsyncMock()

    await populate_config_store(mock_entities_client, MOCK_CONFIG_STORE_PATH)

    # Should not have created any configs since they all exist
    assert mock_entities_client.create.call_count == 0


@pytest.mark.asyncio
async def test_populate_config_store_with_custom_path(
    mock_entities_client, mock_fsspec, mock_entity_not_found, mock_rails_config
):
    """Test the function with a custom config_store_path."""
    custom_path = "/custom/path/to/config_store"

    mock_entities_client.get = AsyncMock(side_effect=mock_entity_not_found)
    mock_entities_client.create = AsyncMock()

    await populate_config_store(mock_entities_client, config_store_path=Path(custom_path))

    # fsspec.ls is called with the resolved path as str (implementation uses str for fsspec)
    mock_fsspec.ls.assert_called_once_with(str(Path(custom_path).resolve()))


@pytest.mark.asyncio
async def test_populate_config_store_with_no_config_directories(mock_entities_client, mock_fsspec):
    """Test behavior when no directories are found in config_store_path."""
    mock_fsspec.ls.return_value = [
        "/path/to/config_store/file1.txt",
        "/path/to/config_store/file2.log",
    ]
    mock_fsspec.isdir.side_effect = lambda path: False  # noqa

    mock_entities_client.create = AsyncMock()

    await populate_config_store(mock_entities_client, MOCK_CONFIG_STORE_PATH)

    # Should not have created any configs
    assert mock_entities_client.create.call_count == 0


@pytest.mark.asyncio
async def test_populate_config_store_handles_fsspec_exception(mock_entities_client, mock_fsspec):
    """Test that the function handles exceptions from fsspec."""
    mock_fsspec.ls.side_effect = Exception("fsspec error")

    with pytest.raises(Exception) as exc_info:
        await populate_config_store(mock_entities_client, MOCK_CONFIG_STORE_PATH)

    assert "fsspec error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_populate_config_store_deduplicates_models_from_symlinked_directory(
    mock_entities_client, mock_fsspec, mock_entity_not_found
):
    """Regression test: configs loaded from a Kubernetes ConfigMap mount directory can have
    duplicated model entries due to os.walk(followlinks=True) traversing symlinks.
    populate_config_store must deduplicate before persisting."""
    triplicated_config = RailsConfig(
        models=[
            Model(type="main", model="meta/llama-3.1-8b-instruct", engine="nim").model_dump(),
            Model(type="content_safety", model="nvidia/nemoguard", engine="nim").model_dump(),
            Model(type="main", model="meta/llama-3.1-8b-instruct", engine="nim").model_dump(),
            Model(type="content_safety", model="nvidia/nemoguard", engine="nim").model_dump(),
            Model(type="main", model="meta/llama-3.1-8b-instruct", engine="nim").model_dump(),
            Model(type="content_safety", model="nvidia/nemoguard", engine="nim").model_dump(),
        ]
    )

    mock_entities_client.get = AsyncMock(side_effect=mock_entity_not_found)
    mock_entities_client.create = AsyncMock()

    with patch("nhx.guardrails.app.utils.config_store.RailsConfig.from_path", return_value=triplicated_config):
        await populate_config_store(mock_entities_client, MOCK_CONFIG_STORE_PATH)

    assert mock_entities_client.create.call_count == 2
    for create_call in mock_entities_client.create.call_args_list:
        stored: GuardrailConfig = create_call.args[0]
        assert stored.data is not None
        stored_types = [m.type for m in stored.data.models]
        assert len(stored_types) == len(set(stored_types)), (
            f"Persisted config still has duplicate model types: {stored_types}"
        )


@pytest.mark.asyncio
async def test_populate_config_store_assumes_workspace_exists(
    mock_entities_client, mock_fsspec, mock_entity_not_found, mock_rails_config
):
    """Test that populate_config_store assumes workspace already exists.

    Workspace creation is handled by the entities service during startup.
    The guardrails service waits for entities to be ready before calling
    populate_config_store. See architecture/docs/service-startup.md.
    """
    mock_entities_client.get = AsyncMock(side_effect=mock_entity_not_found)
    mock_entities_client.create = AsyncMock()

    await populate_config_store(mock_entities_client, MOCK_CONFIG_STORE_PATH)

    # Should have created the 2 guardrail configs (workspace is assumed to exist)
    assert mock_entities_client.create.call_count == 2
