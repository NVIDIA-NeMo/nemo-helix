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

from typing import Dict
from typing_extensions import Literal, Required, TypedDict

__all__ = ["MetadataAnnotationParam"]


class MetadataAnnotationParam(TypedDict, total=False):
    """Structured key/value metadata attached to a span or session."""

    kind: Required[Literal["metadata"]]
    """Discriminator. Always `metadata` for this variant."""

    metadata: Required[Dict[str, object]]
    """Arbitrary key/value pairs. Must contain at least one entry."""

    session_id: Required[str]
    """Id of the session this annotation belongs to. Always required."""

    span_id: str
    """Id of the span this annotation applies to.

    Omit to annotate the whole session instead of a specific span.
    """
