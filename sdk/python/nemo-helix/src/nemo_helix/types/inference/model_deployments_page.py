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

from ..._models import BaseModel
from .model_deployment import ModelDeployment
from ..shared.pagination_data import PaginationData

__all__ = ["ModelDeploymentsPage"]


class ModelDeploymentsPage(BaseModel):
    data: List[ModelDeployment]

    filter: Optional[Dict[str, object]] = None
    """Filtering information."""

    pagination: Optional[PaginationData] = None
    """Pagination information."""

    sort: Optional[str] = None
    """The field on which the results are sorted."""
