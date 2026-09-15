# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Storage backend factory.

NOTE: Backend implementations are intentionally imported inside the match
branches for startup performance. Each backend pulls in heavy dependencies
(boto3, huggingface_hub, etc.) that should only be loaded when the
corresponding storage type is actually used. Do not hoist them to module level.
"""

from __future__ import annotations

import logging

from nmp.common.files.storage_config import (
    GithubStorageConfig,
    HuggingfaceStorageConfig,
    LocalStorageConfig,
    NGCStorageConfig,
    S3StorageConfig,
)
from nmp.common.files.storage_config import StorageConfig as StorageConfig
from nmp.common.files.storage_config import StorageConfigField as StorageConfigField
from nmp.core.files.app.backends.base import StorageImpl

logger = logging.getLogger(__name__)


def storage_impl_factory(
    config: StorageConfig,
    secrets: dict[str, str] | None = None,
) -> StorageImpl:
    if secrets is None:
        secrets = {}
    match config:
        case LocalStorageConfig():
            from nmp.core.files.app.backends.local import LocalStorageImpl

            return LocalStorageImpl(config)
        case NGCStorageConfig():
            from nmp.core.files.app.backends.ngc import NGCStorageImpl

            return NGCStorageImpl(config, secrets)
        case HuggingfaceStorageConfig():
            from nmp.core.files.app.backends.huggingface import HuggingfaceStorageImpl

            return HuggingfaceStorageImpl(config, secrets)
        case S3StorageConfig():
            from nmp.core.files.app.backends.s3 import S3StorageImpl

            return S3StorageImpl(config, secrets)
        case GithubStorageConfig():
            from nmp.core.files.app.backends.github import GithubStorageImpl

            return GithubStorageImpl(config, secrets)
        case _:
            raise TypeError(f"Unsupported storage config type: {type(config).__name__}")
