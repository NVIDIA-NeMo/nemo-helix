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

from typing_extensions import TypedDict

from .g_li_ner_detection_options_param import GLiNERDetectionOptionsParam

__all__ = ["GLiNERDetectionParam"]


class GLiNERDetectionParam(TypedDict, total=False):
    """Configuration for GLiNER PII detection."""

    api_key_env_var: str
    """
    Name of the environment variable containing the API key for authenticated
    endpoints (e.g., NVIDIA_API_KEY).
    """

    chunk_length: int
    """Length of text chunks for processing."""

    flat_ner: bool
    """Whether to use flat NER mode. Setting to False allows for nested entities."""

    input: GLiNERDetectionOptionsParam
    """Configuration options for GLiNER."""

    model: str
    """
    Model identifier sent in NIM API requests (only used when server_endpoint ends
    with /v1/chat/completions).
    """

    output: GLiNERDetectionOptionsParam
    """Configuration options for GLiNER."""

    overlap: int
    """Overlap between chunks."""

    retrieval: GLiNERDetectionOptionsParam
    """Configuration options for GLiNER."""

    server_endpoint: str
    """The endpoint for the GLiNER detection server.

    By default, this is for a locally hosted NIM instance running the GLiNER model.
    Changed from http://localhost:1235/v1/extract (custom server) to
    http://localhost:8000/v1/chat/completions (NIM) in this release. If you use the
    custom gliner_server, set this explicitly to http://localhost:1235/v1/extract.
    """

    threshold: float
    """Confidence threshold for entity detection (0.0 to 1.0)."""
