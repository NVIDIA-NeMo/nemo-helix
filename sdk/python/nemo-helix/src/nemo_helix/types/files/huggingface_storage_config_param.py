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

__all__ = ["HuggingfaceStorageConfigParam"]


class HuggingfaceStorageConfigParam(TypedDict, total=False):
    repo_id: Required[str]
    """Huggingface repository ID (e.g., 'meta-llama/Llama-2-7b')"""

    endpoint: str
    """Huggingface Hub endpoint URL. Use for self-hosted instances."""

    original_revision: str
    """The original revision requested by the user before resolution (e.g., 'main').

    The 'revision' field contains the resolved commit SHA.
    """

    read_chunk_size: int
    """Chunk size in bytes for reading/streaming files.

    Larger chunks reduce async overhead but increase memory per concurrent download.
    Default: 1MB.
    """

    repo_type: Literal["model", "dataset", "space"]
    """Type of Huggingface repository: 'model', 'dataset', or 'space'"""

    revision: str
    """Branch, tag, or commit SHA. Defaults to 'main'"""

    token_secret: SecretRef
    """Reference to a secret.

    Format: 'secret_name' (uses request workspace) or 'workspace/secret_name'
    (explicit workspace).
    """

    type: Literal["huggingface"]
