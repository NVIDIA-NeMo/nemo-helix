# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Storage configuration classes for various backends.

Re-exported from ``nemo_platform_plugin.files.storage_config`` — the canonical
source of truth.  This shim keeps existing ``from nmp.common.files.storage_config
import …`` statements working without changes.
"""

from nemo_platform_plugin.files.storage_config import DEFAULT_READ_CHUNK_SIZE as DEFAULT_READ_CHUNK_SIZE
from nemo_platform_plugin.files.storage_config import BaseStorageConfig as BaseStorageConfig
from nemo_platform_plugin.files.storage_config import GithubStorageConfig as GithubStorageConfig
from nemo_platform_plugin.files.storage_config import HuggingfaceStorageConfig as HuggingfaceStorageConfig
from nemo_platform_plugin.files.storage_config import LocalStorageConfig as LocalStorageConfig
from nemo_platform_plugin.files.storage_config import NGCStorageConfig as NGCStorageConfig
from nemo_platform_plugin.files.storage_config import S3StorageConfig as S3StorageConfig
from nemo_platform_plugin.files.storage_config import StorageConfig as StorageConfig
from nemo_platform_plugin.files.storage_config import StorageConfigField as StorageConfigField
from nemo_platform_plugin.files.storage_config import StorageConfigType as StorageConfigType
