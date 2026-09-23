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

from typing_extensions import Required, TypedDict

from .evaluator_result_data_type import EvaluatorResultDataType

__all__ = ["EvaluatorResultCreateParams"]


class EvaluatorResultCreateParams(TypedDict, total=False):
    workspace: str

    data_type: Required[EvaluatorResultDataType]
    """Discriminator for which of value / string_value carries the payload."""

    name: Required[str]
    """Evaluator / metric identity (e.g. 'faithfulness/v1')."""

    session_id: Required[str]
    """Session id the target span belongs to.

    Denormalized so session-scoped reads stay fast.
    """

    span_id: Required[str]
    """Target span id. Not validated against existing spans (loose target policy)."""

    comment: str
    """Free-text rationale or explanation."""

    string_value: str
    """String value. Required when data_type is CATEGORICAL or TEXT."""

    value: float
    """Numeric value. Required when data_type is NUMERIC or BOOLEAN (0|1)."""
