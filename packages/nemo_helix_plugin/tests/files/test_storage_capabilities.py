# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Storage mutation capability is explicit and independent of ownership."""

import pytest
from nemo_helix_plugin.files.storage_config import (
    BaseStorageConfig,
    GithubStorageConfig,
    GitStorageConfig,
    HuggingfaceStorageConfig,
    LocalStorageConfig,
    NGCStorageConfig,
    S3StorageConfig,
)


@pytest.mark.parametrize(
    "config",
    [
        GithubStorageConfig(owner="org", repo="repo"),
        GitStorageConfig(url="git@example.com:org/repo.git", ssh_key_secret="git-key", known_hosts="example.com key"),
        HuggingfaceStorageConfig(repo_id="org/repo"),
        NGCStorageConfig(org="org", team="team", target="model", api_key_secret="ngc-key"),
    ],
)
def test_external_storage_is_read_only(config):
    """Existing external registries and repositories inherit the safe default."""
    assert config.read_only is True


@pytest.mark.parametrize(
    "config",
    [LocalStorageConfig(path="/tmp/files"), S3StorageConfig(bucket="test", use_sdk_auth=True)],
)
def test_writable_storage_opts_in(config):
    """Writable backends explicitly advertise mutation support without wire changes."""
    assert config.read_only is False
    assert "read_only" not in config.model_dump()
    assert "read_only" not in config.model_json_schema()["properties"]


def test_new_backends_default_to_read_only():
    """Data ownership alone must not grant a new backend permission to mutate."""

    class NewStorageConfig(BaseStorageConfig):
        @property
        def owns_storage_data(self) -> bool:
            return True

    assert NewStorageConfig().read_only is True
