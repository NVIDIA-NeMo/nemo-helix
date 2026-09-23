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

from typing import Dict
from typing_extensions import TypedDict

from .fileset_purpose import FilesetPurpose
from ..shared_params.fileset_metadata import FilesetMetadata

__all__ = ["FilesetUpdateParams"]


class FilesetUpdateParams(TypedDict, total=False):
    workspace: str

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
