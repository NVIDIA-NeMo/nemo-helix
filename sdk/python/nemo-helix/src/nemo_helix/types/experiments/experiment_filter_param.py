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
from typing_extensions import TypedDict

__all__ = ["ExperimentFilterParam"]


class ExperimentFilterParam(TypedDict, total=False):
    """Filter for listing Experiments."""

    baseline_evaluation_name: str
    """Filter experiments by the name of their selected baseline Evaluation."""

    insight_id: str
    """Filter experiments by the id of the insight that seeded them."""

    is_deleted: bool
    """When true, returns only soft-deleted experiments.

    Omit (or false) to see only live experiments.
    """

    is_favorite: bool
    """Filter experiments by favorite status."""

    metadata: Dict[str, str]
    """Filter by a metadata key/value pair, e.g.

    filter[metadata.model]=claude-opus-4-8.
    """

    name: str
    """Filter experiments by name."""

    show_evaluations_over_time: bool
    """
    Filter experiments by whether Studio should display Evaluation results over
    time.
    """
