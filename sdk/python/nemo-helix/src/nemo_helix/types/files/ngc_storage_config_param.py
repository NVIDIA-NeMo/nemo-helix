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

from typing_extensions import Literal, Required, TypedDict

from .secret_ref import SecretRef

__all__ = ["NGCStorageConfigParam"]


class NGCStorageConfigParam(TypedDict, total=False):
    api_key_secret: Required[SecretRef]
    """Reference to a secret.

    Format: 'secret_name' (uses request workspace) or 'workspace/secret_name'
    (explicit workspace).
    """

    org: Required[str]
    """NGC organization name"""

    target: Required[str]
    """NGC asset name (model or resource)"""

    team: Required[str]
    """NGC team name"""

    host: str
    """NGC API host URL"""

    original_version: str
    """
    The original version requested by the user before resolution (e.g., 'latest' or
    None). The 'version' field contains the resolved version ID.
    """

    read_chunk_size: int
    """Chunk size in bytes for reading/streaming files.

    Larger chunks reduce async overhead but increase memory per concurrent download.
    Default: 1MB.
    """

    target_type: Literal["resource", "model"]
    """Type of NGC asset: 'resource' or 'model'"""

    type: Literal["ngc"]

    version: str
    """NGC asset version. If not provided, defaults to latest version"""
