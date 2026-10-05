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

__all__ = ["S3StorageConfigParam"]


class S3StorageConfigParam(TypedDict, total=False):
    bucket: Required[str]
    """S3 bucket name"""

    access_key_id_secret: SecretRef
    """Reference to a secret.

    Format: 'secret_name' (uses request workspace) or 'workspace/secret_name'
    (explicit workspace).
    """

    endpoint_url: str
    """Custom endpoint URL for S3-compatible storage (e.g., MinIO, Garage, RustFS).

    If not specified, uses AWS S3.
    """

    prefix: str
    """Optional prefix (folder path) within the bucket.

    All operations will be relative to this prefix.
    """

    read_chunk_size: int
    """Chunk size in bytes for reading/streaming files.

    Larger chunks reduce async overhead but increase memory per concurrent download.
    Default: 1MB.
    """

    region: str
    """AWS region.

    If not specified, uses SDK default (env vars, instance metadata, etc.)
    """

    secret_access_key_secret: SecretRef
    """Reference to a secret.

    Format: 'secret_name' (uses request workspace) or 'workspace/secret_name'
    (explicit workspace).
    """

    signature_version: Literal["s3v4", "s3"]
    """AWS signature version for request signing.

    Use 's3' for legacy systems that only support signature v2.
    """

    type: Literal["s3"]

    use_sdk_auth: bool
    """
    Use AWS SDK credential chain for authentication (env vars like
    AWS_ACCESS_KEY_ID, IAM roles, instance profiles, etc.). This option is only
    available for the platform's default storage backend. User-provided S3 storage
    must use explicit credentials via access_key_id_secret and
    secret_access_key_secret.
    """
