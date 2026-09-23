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

from typing import Union
from typing_extensions import Literal, Required, TypedDict

__all__ = ["LabelAnnotationParam"]


class LabelAnnotationParam(TypedDict, total=False):
    """Categorical or numeric label attached to a span or session.

    Use `value_type=text` for tag-style labels (e.g., `regression`, `needs-review`) and
    `value_type=numeric` for scored labels (e.g., a 1-5 helpfulness rating). Numeric labels
    must include a `name` to identify what the score measures.
    """

    kind: Required[Literal["label"]]
    """Discriminator. Always `label` for this variant."""

    session_id: Required[str]
    """Id of the session this annotation belongs to. Always required."""

    value: Required[Union[str, float]]
    """The label's value.

    Must be a string when `value_type=text` and a number when `value_type=numeric`.
    """

    value_type: Required[Literal["text", "numeric"]]
    """Whether `value` should be interpreted as text (`text`) or a number (`numeric`)."""

    name: str
    """Name identifying what the label measures (e.g., `severity`, `helpfulness`).

    Optional for text labels; required for numeric labels.
    """

    span_id: str
    """Id of the span this annotation applies to.

    Omit to annotate the whole session instead of a specific span.
    """
