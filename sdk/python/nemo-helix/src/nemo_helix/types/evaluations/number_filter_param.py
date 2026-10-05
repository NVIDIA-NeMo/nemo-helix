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

from typing_extensions import Annotated, TypedDict

from ..._utils import PropertyInfo

__all__ = ["NumberFilterParam"]


class NumberFilterParam(TypedDict, total=False):
    eq: Annotated[float, PropertyInfo(alias="$eq")]
    """Filter for results equal to this value."""

    gt: Annotated[float, PropertyInfo(alias="$gt")]
    """Filter for results greater than this value."""

    gte: Annotated[float, PropertyInfo(alias="$gte")]
    """Filter for results greater than or equal to this value."""

    lt: Annotated[float, PropertyInfo(alias="$lt")]
    """Filter for results less than this value."""

    lte: Annotated[float, PropertyInfo(alias="$lte")]
    """Filter for results less than or equal to this value."""
