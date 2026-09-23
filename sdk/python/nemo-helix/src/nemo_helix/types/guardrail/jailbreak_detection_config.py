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

from ..._models import BaseModel

__all__ = ["JailbreakDetectionConfig"]


class JailbreakDetectionConfig(BaseModel):
    """Configuration data for jailbreak detection."""

    api_key_env_var: Optional[str] = None
    """Environment variable containing API key for jailbreak detection model"""

    embedding: Optional[str] = None

    length_per_perplexity_threshold: Optional[float] = None
    """The length/perplexity threshold."""

    nim_base_url: Optional[str] = None
    """Base URL for jailbreak detection model. Example: http://localhost:8000/v1"""

    nim_port: Optional[int] = None
    """DEPRECATED: Include port in nim_base_url instead"""

    nim_server_endpoint: Optional[str] = None
    """Classification path uri. Defaults to 'classify' for NemoGuard JailbreakDetect."""

    nim_url: Optional[str] = None
    """DEPRECATED: Use nim_base_url instead"""

    prefix_suffix_perplexity_threshold: Optional[float] = None
    """The prefix/suffix perplexity threshold."""

    server_endpoint: Optional[str] = None
    """The endpoint for the jailbreak detection heuristics/model container."""
