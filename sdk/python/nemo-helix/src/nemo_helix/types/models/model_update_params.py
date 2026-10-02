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

from typing import Dict, Optional
from typing_extensions import TypedDict

from ..._types import SequenceNotStr
from ..shared.backend_format import BackendFormat
from ..shared.finetuning_type import FinetuningType
from ..shared_params.model_spec import ModelSpec
from ..shared_params.prompt_data import PromptData
from ..shared_params.api_endpoint_data import APIEndpointData

__all__ = ["ModelUpdateParams"]


class ModelUpdateParams(TypedDict, total=False):
    workspace: str

    verbose: bool
    """Whether to include full spec details"""

    api_endpoint: APIEndpointData
    """Data about an inference endpoint."""

    backend_format: Optional[BackendFormat]
    """Inference backend API wire formats understood by IGW and middleware plugins."""

    base_model: str
    """Link to another model which is used as a base for the current model"""

    custom_fields: Dict[str, object]
    """Custom fields for additional metadata"""

    description: str
    """Optional description of the model"""

    fileset: str
    """
    A set of checkpoint files, configs, and other auxiliary info associated with
    this model - expected format {workspace}/{fileset_name}
    """

    finetuning_type: FinetuningType
    """Finetuning types."""

    model_providers: SequenceNotStr[str]
    """
    List of ModelProvider workspace/name resource names that provide inference for
    this Model Entity
    """

    ownership: Dict[str, object]
    """Ownership information for the model"""

    prompt: PromptData
    """Configuration for prompt engineering."""

    spec: ModelSpec
    """Detailed specification for a model."""

    trust_remote_code: bool
    """
    Whether to trust remote code for the checkpoint. Some models without support in
    certain libraries such as Transformers require additional custom Python code to
    execute. Due to security ramifications of running arbitrary code, this can only
    be set to true on one of the following conditions: (1) the model's fileset's
    source is pre-approved in the platform config, or (2) the user creating this
    model is an administrator.
    """
