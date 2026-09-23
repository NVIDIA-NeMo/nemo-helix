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

from .float_filter_param import FloatFilterParam
from .evaluator_result_data_type import EvaluatorResultDataType
from ..shared_params.datetime_filter import DatetimeFilter

__all__ = ["EvaluatorResultFilterParam"]


class EvaluatorResultFilterParam(TypedDict, total=False):
    created_at: DatetimeFilter
    """Filter by row creation time (range supported)."""

    created_by: str
    """Filter by principal/system that wrote the row."""

    data_type: EvaluatorResultDataType
    """Filter by data_type."""

    name: str
    """Filter by evaluator/metric name."""

    session_id: str
    """Filter by target session id."""

    span_id: str
    """Filter by target span id."""

    value: FloatFilterParam
    """Filter by numeric value (range supported)."""
