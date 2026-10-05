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

from typing_extensions import Required, TypedDict

from ..models.lora_param import LoraParam
from ..shared.finetuning_type import FinetuningType

__all__ = ["AdapterCreateParams"]


class AdapterCreateParams(TypedDict, total=False):
    workspace: str

    fileset: Required[str]
    """
    Location where adapter files are stored - expected format
    {workspace}/{fileset_name}
    """

    finetuning_type: Required[FinetuningType]
    """Finetuning types."""

    model: Required[str]
    """
    Base model entity. Use `{workspace}/{model_name}` to reference a model in any
    workspace, or a single `{model_name}` resolved in the path workspace. A single
    name (2-63 characters) or 'workspace/model*name' where each segment is a valid
    name (lowercase, digits, hyphens, and temporarily @ . + *; no leading/trailing
    or consecutive hyphens). If one slash, both sides must be non-empty.
    """

    name: Required[str]
    """Name of the adapter.

    Name must be unique in the workspace. Name must start with a lowercase letter,
    be 2-63 characters, and use lowercase letters, digits, hyphens, and dots (no
    consecutive hyphens, cannot end with a hyphen).
    """

    description: str
    """Optional description of the adapter"""

    enabled: bool
    """Whether to make this adapter available for inference post training"""

    lora_config: LoraParam
    """Lora configuration specifics"""
