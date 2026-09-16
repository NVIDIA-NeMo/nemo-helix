# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from nhx.common.config import PlatformConfig
from nhx.core.secrets.entities import PlatformSecret


def is_default_ngc_api_key(config: PlatformConfig, workspace: str, name: str) -> bool:
    """Check if the secret is the default NGC API key secret."""
    ngc_api_key_secret = config.ngc_api_key_secret.split("/")
    if len(ngc_api_key_secret) != 2:
        raise ValueError("Invalid NGC API key secret configuration")
    return workspace == ngc_api_key_secret[0] and name == ngc_api_key_secret[1]


def get_default_ngc_api_key(config: PlatformConfig) -> PlatformSecret:
    """Create the default NGC API key secret."""
    ngc_api_key = os.environ.get(config.ngc_api_key_env_var)
    if not ngc_api_key:
        raise ValueError(f"NGC API key not found in environment variable {config.ngc_api_key_env_var}")
    ngc_api_key_secret = config.ngc_api_key_secret.split("/")
    if len(ngc_api_key_secret) != 2:
        raise ValueError("Invalid NGC API key secret configuration")
    secret = PlatformSecret(
        workspace=ngc_api_key_secret[0],
        name=ngc_api_key_secret[1],
        description="Default NGC API key secret for the platform",
    )
    secret._data = ngc_api_key
    secret._encrypted_dek = ""
    secret._secret_provider = "env"
    return secret
