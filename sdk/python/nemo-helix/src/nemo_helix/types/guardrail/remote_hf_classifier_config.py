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

from typing import Dict, List, Optional
from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["RemoteHfClassifierConfig"]


class RemoteHfClassifierConfig(BaseModel):
    """Configuration for a remote HuggingFace classifier (vLLM, KServe, FMS)."""

    base_url: str
    """Base URL for the inference server (e.g. 'http://host:8000')."""

    engine: Literal["vllm", "kserve", "fms"]

    model: str
    """HF model ID, local path, or server-side model identifier."""

    api_key_env_var: Optional[str] = None
    """Environment variable name holding the API key.

    Resolved at runtime to an Authorization: Bearer header.
    """

    blocked_labels: Optional[List[str]] = None
    """Labels that should trigger blocking when detected above threshold."""

    parameters: Optional[Dict[str, object]] = None
    """
    Remote backend parameters: 'timeout' (float, seconds), 'verify_ssl' (bool),
    'ca_cert'/'client_cert'/'client_key' (str, paths). Note: 'ca_cert' replaces (not
    extends) system CAs; use a concatenated bundle to include both custom and system
    CAs.
    """

    threshold: Optional[float] = None
    """Minimum score for a detection to trigger blocking."""
