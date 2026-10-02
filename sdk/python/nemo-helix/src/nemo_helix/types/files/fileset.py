# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Dict, Union
from typing_extensions import TypeAlias

from ..._models import BaseModel
from .fileset_purpose import FilesetPurpose
from .s3_storage_config import S3StorageConfig
from .ngc_storage_config import NGCStorageConfig
from .local_storage_config import LocalStorageConfig
from .github_storage_config import GitHubStorageConfig
from ..shared.fileset_metadata import FilesetMetadata
from .huggingface_storage_config import HuggingfaceStorageConfig

__all__ = ["Fileset", "Storage"]

Storage: TypeAlias = Union[
    LocalStorageConfig, NGCStorageConfig, HuggingfaceStorageConfig, S3StorageConfig, GitHubStorageConfig
]


class Fileset(BaseModel):
    """Response DTO for fileset operations."""

    id: str

    created_at: str

    custom_fields: Dict[str, object]

    description: str

    metadata: FilesetMetadata
    """Tagged metadata container - the key indicates the type.

    Example: metadata = FilesetMetadata( dataset=DatasetMetadataContent(
    schema={"columns": ["id", "name"]}, ) )
    """

    name: str

    project: str

    purpose: FilesetPurpose

    storage: Storage

    updated_at: str

    workspace: str
