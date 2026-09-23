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

from .annotation_kind import AnnotationKind
from .numeric_filter_param import NumericFilterParam
from ..shared_params.datetime_filter import DatetimeFilter

__all__ = ["AnnotationFilterParam"]


class AnnotationFilterParam(TypedDict, total=False):
    created_at: DatetimeFilter
    """Return only annotations created within the given time range."""

    created_by: str
    """Return only annotations created by this user."""

    kind: AnnotationKind
    """
    Return only annotations of this kind (`feedback`, `note`, `label`, or
    `metadata`).
    """

    name: str
    """
    Return only `label` annotations with this `name` (e.g., `severity`,
    `helpfulness`).
    """

    session_id: str
    """Return only annotations attached to this session."""

    span_id: str
    """Return only annotations attached to this span."""

    value_numeric: NumericFilterParam
    """Range filter for numeric annotation values.

    At least one of `$gte` or `$lte` must be supplied — an empty `{}` is not a
    meaningful filter and is rejected.
    """

    value_text: str
    """Return only annotations with this text value.

    For `feedback` annotations this is `positive` or `negative`; for `label`
    annotations with `value_type=text` this is the label's value.
    """
