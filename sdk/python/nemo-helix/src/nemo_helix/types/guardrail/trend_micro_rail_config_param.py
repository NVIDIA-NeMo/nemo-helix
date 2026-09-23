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

__all__ = ["TrendMicroRailConfigParam"]


class TrendMicroRailConfigParam(TypedDict, total=False):
    """Configuration data for the Trend Micro AI Guard API"""

    api_key_env_var: str
    """Environment variable containing API key for Trend Micro AI Guard"""

    application_name: str
    """Application name for TMV1-Application-Name header (REQUIRED).

    Must contain only letters, numbers, hyphens, and underscores, with a maximum
    length of 64 characters.
    """

    detailed_response: bool
    """
    If True, returns detailed AI Guard results with confidence scores (Prefer:
    return=representation). If False, returns minimal response with only action and
    reasons (Prefer: return=minimal).
    """

    v1_url: str
    """The endpoint for the Trend Micro AI Guard API.

    For other regions, use:
    https://api.{region}.xdr.trendmicro.com/v3.0/aiSecurity/applyGuardrails where
    region is eu, jp, au, in, sg, or mea.
    """
