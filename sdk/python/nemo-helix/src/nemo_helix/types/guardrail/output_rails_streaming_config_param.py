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

__all__ = ["OutputRailsStreamingConfigParam"]


class OutputRailsStreamingConfigParam(TypedDict, total=False):
    """Configuration for managing streaming output of LLM tokens."""

    chunk_size: int
    """The number of tokens in each processing chunk.

    This is the size of the token block on which output rails are applied.
    """

    context_size: int
    """
    The number of tokens carried over from the previous chunk to provide context for
    continuity in processing.
    """

    enabled: bool
    """Enables streaming mode when True."""

    stream_first: bool
    """If True, token chunks are streamed immediately before output rails are applied."""
