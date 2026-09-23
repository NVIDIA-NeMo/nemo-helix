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

__all__ = ["ClavataRailOptions"]


class ClavataRailOptions(BaseModel):
    """Configuration data for the Clavata API"""

    policy: str
    """The policy alias to use when evaluating inputs or outputs."""

    labels: Optional[List[str]] = None
    """
    A list of labels to match against the policy. If no labels are provided, the
    overall policy result will be returned. If labels are provided, only hits on the
    provided labels will be considered a hit.
    """
