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

from typing_extensions import Literal, Required, TypedDict

from .model_parameters_param import ModelParametersParam
from .model_cache_config_param import ModelCacheConfigParam

__all__ = ["ModelParam"]


class ModelParam(TypedDict, total=False):
    """Configuration of a model used by the rails engine.

    If using Inference Gateway, the `model` field should be a Model Entity reference ('workspace/model_name').
    """

    engine: Required[str]

    type: Required[str]

    cache: ModelCacheConfigParam
    """Configuration for model caching."""

    mode: Literal["chat", "text"]
    """Whether the mode is 'text' completion or 'chat' completion.

    Allowed values are 'chat' or 'text'.
    """

    model: str
    """The model name.

    If using Inference Gateway, this should be the Model Entity reference
    ('workspace/model_name').
    """

    parameters: ModelParametersParam
    """Parameters for configuring how to interact with a model in a guardrails config."""
