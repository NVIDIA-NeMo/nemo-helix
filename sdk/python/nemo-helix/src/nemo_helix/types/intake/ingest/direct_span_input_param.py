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

from typing import Dict, Union
from datetime import datetime
from typing_extensions import Required, Annotated, TypedDict

from ...._utils import PropertyInfo
from ..span_kind import SpanKind
from ..span_status import SpanStatus
from .json_value_param import JsonValueParam

__all__ = ["DirectSpanInputParam"]


class DirectSpanInputParam(TypedDict, total=False):
    """One provider-neutral span supplied by a historical trace importer."""

    span_id: Required[str]

    started_at: Required[Annotated[Union[str, datetime], PropertyInfo(format="iso8601")]]

    trace_id: Required[str]

    attributes: Dict[str, JsonValueParam]

    ended_at: Annotated[Union[str, datetime], PropertyInfo(format="iso8601")]

    input: JsonValueParam

    kind: SpanKind

    name: str

    output: JsonValueParam

    parent_span_id: str

    session_id: str

    status: SpanStatus
