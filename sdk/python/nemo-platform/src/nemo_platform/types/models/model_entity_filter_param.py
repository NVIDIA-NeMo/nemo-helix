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
from typing_extensions import TypeAlias, TypedDict

from ..shared.finetuning_type import FinetuningType
from .base_model_filter_param import BaseModelFilterParam
from ..shared_params.string_filter import StringFilter
from .finetuning_type_filter_param import FinetuningTypeFilterParam
from ..shared_params.datetime_filter import DatetimeFilter

__all__ = ["ModelEntityFilterParam", "Adapters", "BaseModel", "Description", "Name"]

Adapters: TypeAlias = Union[FinetuningTypeFilterParam, bool]

BaseModel: TypeAlias = Union[BaseModelFilterParam, bool, str]

Description: TypeAlias = Union[StringFilter, str]

Name: TypeAlias = Union[StringFilter, str]


class ModelEntityFilterParam(TypedDict, total=False):
    """Filter for Model Entity queries."""

    adapters: Adapters
    """Filter models with Parameter Efficient Fine-tuning Adapters."""

    base_model: BaseModel
    """
    Filter by base model: true = has a base model, false = no base model, { name:
    string } or string = match base model name.
    """

    created_at: DatetimeFilter
    """Filter entities based on creation date."""

    description: Description
    """Filter by description."""

    fileset: Union[bool, str]
    """
    Filter by fileset: true = has a fileset, false = no fileset, string = match
    fileset reference in the form {workspace}/{fileset_name}.
    """

    finetuning_type: Union[FinetuningType, bool]
    """Filter models that have been perviously finetuned."""

    lora_enabled: bool
    """Filter models by whether their deployment config has LoRA enabled."""

    name: Name
    """Filter by name."""

    project: str
    """Filter by project name."""

    prompt: bool
    """Filter models with prompt engineering data."""

    updated_at: DatetimeFilter
    """Filter entities based on update date."""

    workspace: str
    """Filter by workspace id."""
