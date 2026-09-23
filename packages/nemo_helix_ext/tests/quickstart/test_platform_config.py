# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for quickstart platform configuration."""

from pathlib import Path

import yaml
from nemo_helix_ext.quickstart.platform_config import HelixConfig


def test_default_platform_config_persists_dev_secrets_key_creation(tmp_path: Path) -> None:
    path = tmp_path / "platform-config.yaml"

    HelixConfig.get_default().save(path)

    config = yaml.safe_load(path.read_text())
    assert config["secrets"] == {
        "allow_key_creation": True,
        "local_key_creation_path": "/data/nhx-encryption-key.txt",
    }


def test_default_platform_config_exports_dev_secrets_env_vars() -> None:
    env = HelixConfig.get_default().to_env_vars()

    assert env["NHX_SECRETS_ALLOW_KEY_CREATION"] == "1"
    assert env["NHX_SECRETS_LOCAL_KEY_CREATION_PATH"] == "/data/nhx-encryption-key.txt"


def test_platform_config_can_disable_quickstart_key_creation(tmp_path: Path) -> None:
    path = tmp_path / "platform-config.yaml"
    path.write_text("secrets:\n  allow_key_creation: false\n")

    config = HelixConfig.load(path)

    assert config.secrets.allow_key_creation is False
    assert config.to_env_vars()["NHX_SECRETS_ALLOW_KEY_CREATION"] == "0"


def test_platform_config_preserves_custom_secrets_encryption_config(tmp_path: Path) -> None:
    path = tmp_path / "platform-config.yaml"
    path.write_text(
        """
secrets:
  encryption:
    current_provider: local_v1
    providers:
      secret_key:
        local_v1:
          from_env: NHX_SECRETS_DEFAULT_ENCRYPTION_KEY
""".lstrip()
    )

    config = HelixConfig.load(path)
    output_path = tmp_path / "saved-platform-config.yaml"
    config.save(output_path)

    saved = yaml.safe_load(output_path.read_text())
    assert saved["secrets"]["encryption"] == {
        "current_provider": "local_v1",
        "providers": {
            "secret_key": {
                "local_v1": {
                    "from_env": "NHX_SECRETS_DEFAULT_ENCRYPTION_KEY",
                },
            },
        },
    }
