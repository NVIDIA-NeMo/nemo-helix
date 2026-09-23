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

from ..._compat import PYDANTIC_V1, ConfigDict
from ..._models import BaseModel

__all__ = ["APIEndpointData"]


class APIEndpointData(BaseModel):
    """Data about an inference endpoint."""

    api_key: Optional[str] = None
    """API key for authentication"""

    format: Optional[str] = None
    """API format (e.g., openai, nvidia)"""

    model_id: Optional[str] = None
    """Model identifier at the endpoint"""

    url: Optional[str] = None
    """Endpoint URL"""

    if not PYDANTIC_V1:
        # allow fields with a `model_` prefix
        model_config = ConfigDict(protected_namespaces=tuple())
