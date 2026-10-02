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

from ..._types import SequenceNotStr

__all__ = ["ColumnLayoutParam"]


class ColumnLayoutParam(TypedDict, total=False):
    """
    A saved table layout for a group's evaluations list: column order and which columns are hidden.

    Column ids are Studio's and cannot be enumerated here — the table builds a column per evaluator
    and metadata key found in the rows — so ids are stored and echoed back unvalidated.

    Visibility is stored as the *hidden* ids rather than a map over every column, so a column that
    appears later (a new evaluator, a new metadata key) shows up by default.
    """

    hidden: SequenceNotStr[str]
    """Column ids hidden from the table. Any column not listed is shown."""

    order: SequenceNotStr[str]
    """Column ids in display order.

    Empty means no saved order: the table falls back to its natural column order.
    """
