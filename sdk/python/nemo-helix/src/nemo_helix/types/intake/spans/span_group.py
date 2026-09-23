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

from typing import Dict
from datetime import datetime

from ...._models import BaseModel

__all__ = ["SpanGroup"]


class SpanGroup(BaseModel):
    group: Dict[str, str]
    """Group key values, keyed by the requested group-by fields."""

    span_count: int
    """Number of matching spans in this group."""

    started_at: datetime
    """Start time of the earliest matching span in this group.

    Only matching spans count, so a filter narrows this to when the filtered work
    began rather than when the whole trace or session began.
    """
