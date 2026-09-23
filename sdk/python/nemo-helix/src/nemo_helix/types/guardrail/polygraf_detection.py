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
from .polygraf_detection_options import PolygrafDetectionOptions

__all__ = ["PolygrafDetection"]


class PolygrafDetection(BaseModel):
    """Configuration for Polygraf PII detection."""

    input: Optional[PolygrafDetectionOptions] = None
    """Configuration options for Polygraf."""

    output: Optional[PolygrafDetectionOptions] = None
    """Configuration options for Polygraf."""

    retrieval: Optional[PolygrafDetectionOptions] = None
    """Configuration options for Polygraf."""

    server_endpoint: Optional[str] = None
    """The endpoint for the Polygraf detection server."""
