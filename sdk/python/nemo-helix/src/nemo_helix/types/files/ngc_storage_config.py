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

from typing import Optional
from typing_extensions import Literal

from ..._models import BaseModel
from .secret_ref import SecretRef

__all__ = ["NGCStorageConfig"]


class NGCStorageConfig(BaseModel):
    api_key_secret: SecretRef
    """Reference to a secret.

    Format: 'secret_name' (uses request workspace) or 'workspace/secret_name'
    (explicit workspace).
    """

    org: str
    """NGC organization name"""

    target: str
    """NGC asset name (model or resource)"""

    team: str
    """NGC team name"""

    host: Optional[str] = None
    """NGC API host URL"""

    original_version: Optional[str] = None
    """
    The original version requested by the user before resolution (e.g., 'latest' or
    None). The 'version' field contains the resolved version ID.
    """

    read_chunk_size: Optional[int] = None
    """Chunk size in bytes for reading/streaming files.

    Larger chunks reduce async overhead but increase memory per concurrent download.
    Default: 1MB.
    """

    target_type: Optional[Literal["resource", "model"]] = None
    """Type of NGC asset: 'resource' or 'model'"""

    type: Optional[Literal["ngc"]] = None

    version: Optional[str] = None
    """NGC asset version. If not provided, defaults to latest version"""
