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

from __future__ import annotations

from typing import Dict, Union
from typing_extensions import Required, TypeAlias, TypedDict

from .fileset_purpose import FilesetPurpose
from .s3_storage_config_param import S3StorageConfigParam
from .ngc_storage_config_param import NGCStorageConfigParam
from .local_storage_config_param import LocalStorageConfigParam
from .github_storage_config_param import GitHubStorageConfigParam
from ..shared_params.fileset_metadata import FilesetMetadata
from .huggingface_storage_config_param import HuggingfaceStorageConfigParam

__all__ = ["FilesetCreateParams", "Storage"]


class FilesetCreateParams(TypedDict, total=False):
    workspace: str

    name: Required[str]
    """The name of the fileset.

    Name must start with a lowercase letter, be 2-63 characters, and use lowercase
    letters, digits, hyphens, and dots (no consecutive hyphens, cannot end with a
    hyphen).
    """

    cache: bool
    """Cache all files after creation. Only applies to external storage."""

    custom_fields: Dict[str, object]
    """Custom fields for the fileset."""

    description: str
    """The description of the fileset."""

    metadata: FilesetMetadata
    """Tagged metadata container - the key indicates the type.

    Example: metadata = FilesetMetadata( dataset=DatasetMetadataContent(
    schema={"columns": ["id", "name"]}, ) )
    """

    project: str
    """The name of the project associated with this fileset."""

    purpose: FilesetPurpose
    """The purpose of the fileset."""

    storage: Storage
    """The storage configuration for the fileset.

    If not provided, uses default storage.
    """


Storage: TypeAlias = Union[
    LocalStorageConfigParam,
    NGCStorageConfigParam,
    HuggingfaceStorageConfigParam,
    S3StorageConfigParam,
    GitHubStorageConfigParam,
]
