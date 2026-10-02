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

from .model_type import ModelType
from ..shared_params.tool_call_config import ToolCallConfig

__all__ = ["ModelDeploymentConfigModelSpecParam"]


class ModelDeploymentConfigModelSpecParam(TypedDict, total=False):
    """What model to serve and how -- independent of the executor it runs on.

    Executor-invariant facts about the model. The compiler resolves the weight
    source per engine; serving fields override the model entity spec when set.
    """

    chat_template: str
    """Jinja2 chat template string for the model.

    Overrides the chat_template from ModelEntity.spec if both are set. Used by the
    engine to format chat completions.
    """

    lora_enabled: bool
    """Whether to enable LoRA support"""

    model_name: str
    """Model name - model repository name for model weights."""

    model_namespace: str
    """
    Model repository namespace - organization/user namespace as it exists in
    repo_id.
    """

    model_revision: str
    """Model revision (branch, tag, or commit).

    If not specified, parsed from model_name @revision suffix or defaults to 'main'
    """

    model_type: ModelType
    """Model type enum for NIM deployments."""

    tool_call_config: ToolCallConfig
    """Configuration for tool calling support in NIM deployments."""
