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

__all__ = ["UserMessagesConfigParam"]


class UserMessagesConfigParam(TypedDict, total=False):
    """Configuration for how the user messages are interpreted."""

    embeddings_only: bool
    """Whether to use only embeddings for computing the user canonical form messages."""

    embeddings_only_fallback_intent: str
    """Defines the fallback intent when the similarity is below the threshold.

    If set to None, the user intent is computed normally using the LLM. If set to a
    string value, that string is used as the intent.
    """

    embeddings_only_similarity_threshold: float
    """
    The similarity threshold to use when using only embeddings for computing the
    user canonical form messages.
    """
