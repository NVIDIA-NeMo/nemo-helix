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

from typing_extensions import Literal, Required, TypedDict

__all__ = ["NoteAnnotationParam"]


class NoteAnnotationParam(TypedDict, total=False):
    """Free-text note attached to a span or session."""

    kind: Required[Literal["note"]]
    """Discriminator. Always `note` for this variant."""

    session_id: Required[str]
    """Id of the session this annotation belongs to. Always required."""

    text: Required[str]
    """The note content. 1 to 10,000 characters."""

    span_id: str
    """Id of the span this annotation applies to.

    Omit to annotate the whole session instead of a specific span.
    """
