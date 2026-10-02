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
from typing_extensions import Required, Annotated, TypedDict

from ..._types import SequenceNotStr
from ..._utils import PropertyInfo

__all__ = ["EvaluationUpdateParams"]


class EvaluationUpdateParams(TypedDict, total=False):
    workspace: str

    dataset_name: Required[str]
    """Producer-supplied dataset name."""

    body_name: Required[Annotated[str, PropertyInfo(alias="name")]]
    """Producer-supplied, workspace-unique evaluation id."""

    dataset_version: str
    """Producer-supplied dataset version."""

    description: str
    """Human-readable description."""

    experiment_group_id: str
    """Deprecated single-experiment field; provide experiment_ids instead.

    Coalesced into experiment_ids when experiment_ids is omitted.
    """

    experiment_ids: SequenceNotStr[str]
    """Entity ids of the Experiments this Evaluation belongs to (>=1).

    Preferred; each experiment must already exist. When omitted, the deprecated
    experiment_group_id is used instead.
    """

    metadata: Dict[str, str]
    """Free-form producer metadata."""

    parent_evaluation_id: str
    """Entity id of the evaluation this one was derived from (e.g.

    a variant of a baseline), if any.
    """

    root_cause: str
    """Human- or agent-authored explanation of the evaluation's outcome (e.g.

    why it was killed).
    """

    source_link: str
    """Optional URL for the source evaluation."""

    status: str
    """Producer-defined lifecycle status of the evaluation."""
