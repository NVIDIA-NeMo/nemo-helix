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

from typing_extensions import Literal, TypedDict

__all__ = ["ContextBloatDetectionConfigParam"]


class ContextBloatDetectionConfigParam(TypedDict, total=False):
    """Configuration for context bloat / context manipulation detection."""

    action: Literal["reject", "truncate", "warn"]
    """Action on detection: 'reject', 'truncate', or 'warn'."""

    max_chars: int
    """Size cap in characters. Inputs exceeding this are flagged."""

    max_repetition_ratio: float
    """Max fraction of repeated n-grams (0.0-1.0)."""

    max_run_ratio: float
    """Max fraction of text that is the longest single-char run."""

    min_chars: int
    """Minimum characters before entropy/run/repetition checks apply.

    Shorter texts are only checked against size cap.
    """

    min_entropy: float
    """Shannon entropy floor (bits/char). English prose is ~4.0-4.5."""

    ngram_size: int
    """Size of n-grams used for repetition detection."""
