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

from typing import Dict, Optional

from ..._models import BaseModel

__all__ = ["FunctionDefinition"]


class FunctionDefinition(BaseModel):
    """An OpenAI-compatible function definition for tool calling.

    Mirrors the ``function`` object the Inference Gateway forwards to
    OpenAI-compatible backends.
    """

    name: str
    """The name of the function to be called."""

    description: Optional[str] = None
    """
    A description of what the function does, used by the model to decide when and
    how to call it.
    """

    parameters: Optional[Dict[str, object]] = None
    """The parameters the function accepts, described as a JSON Schema object."""

    strict: Optional[bool] = None
    """Whether to enforce strict schema adherence when generating the function call."""
