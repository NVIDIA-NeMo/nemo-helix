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
from datetime import datetime

from ..._models import BaseModel
from .evaluator_result_data_type import EvaluatorResultDataType

__all__ = ["EvaluatorResult"]


class EvaluatorResult(BaseModel):
    """Response model for evaluator_results read endpoints."""

    created_at: datetime

    data_type: EvaluatorResultDataType

    evaluator_result_id: str

    ingested_at: datetime

    name: str

    session_id: str

    span_id: str

    workspace: str

    comment: Optional[str] = None

    created_by: Optional[str] = None

    string_value: Optional[str] = None

    value: Optional[float] = None
