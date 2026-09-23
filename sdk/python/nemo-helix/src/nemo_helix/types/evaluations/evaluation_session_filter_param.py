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

__all__ = ["EvaluationSessionFilterParam"]


class EvaluationSessionFilterParam(TypedDict, total=False):
    """Filter for listing EvaluationSessions."""

    status: str
    """Filter by root-span status (success, error, cancelled, unknown)."""

    test_case_id: str
    """Deprecated alias for test_case_name. Use test_case_name instead."""

    test_case_name: str
    """Filter by test case name."""
