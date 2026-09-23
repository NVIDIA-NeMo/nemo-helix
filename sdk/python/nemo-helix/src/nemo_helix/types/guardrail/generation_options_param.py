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

from typing import Dict, Union
from typing_extensions import TypedDict

from ..._types import SequenceNotStr
from .generation_log_options_param import GenerationLogOptionsParam
from .generation_rails_options_param import GenerationRailsOptionsParam

__all__ = ["GenerationOptionsParam"]


class GenerationOptionsParam(TypedDict, total=False):
    """A set of options that should be applied during a generation.

    The GenerationOptions control various things such as what rails are enabled,
    additional parameters for the main LLM, whether the rails should be enforced or
    ran in parallel, what to be included in the generation log, etc.
    """

    llm_output: bool
    """Whether the response should also include any custom LLM output."""

    llm_params: Dict[str, object]
    """Additional parameters that should be used for the LLM call"""

    log: GenerationLogOptionsParam
    """Options for what should be included in the generation log."""

    output_vars: Union[bool, SequenceNotStr[str]]
    """Whether additional context information should be returned.

    When True is specified, the whole context is returned. Otherwise, a list of key
    names can be specified.
    """

    rails: GenerationRailsOptionsParam
    """Options for what rails should be used during the generation."""
