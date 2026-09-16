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

from typing import List, Optional

from ..._models import BaseModel

__all__ = ["SensitiveDataDetectionOptions"]


class SensitiveDataDetectionOptions(BaseModel):
    entities: Optional[List[str]] = None
    """The list of entities that should be detected.

    Check out https://microsoft.github.io/presidio/supported_entities/ forthe list
    of supported entities.
    """

    mask_token: Optional[str] = None
    """The token that should be used to mask the sensitive data."""

    score_threshold: Optional[float] = None
    """The score threshold that should be used to detect the sensitive data."""
