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

from typing import Iterable
from typing_extensions import TypedDict

from .model_provider_status import ModelProviderStatus
from .served_model_mapping_param import ServedModelMappingParam

__all__ = ["ProviderUpdateStatusParams"]


class ProviderUpdateStatusParams(TypedDict, total=False):
    workspace: str

    model_deployment_id: str
    """
    Reference to the ModelDeployment ID if this provider is associated with a
    deployment
    """

    served_models: Iterable[ServedModelMappingParam]
    """List of models served by this provider with routing information for IGW"""

    status: ModelProviderStatus
    """Status enum for ModelProvider objects."""

    status_message: str
    """Status message.

    If status is provided without status_message, defaults to empty string.
    """
