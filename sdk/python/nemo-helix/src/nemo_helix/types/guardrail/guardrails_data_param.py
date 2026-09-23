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
from typing_extensions import TypeAlias, TypedDict

from ..._types import SequenceNotStr
from .rails_config_param import RailsConfigParam
from .generation_options_param import GenerationOptionsParam

__all__ = ["GuardrailsDataParam", "Config"]

Config: TypeAlias = Union[str, RailsConfigParam]


class GuardrailsDataParam(TypedDict, total=False):
    config: Config
    """The id of the configuration or its dict representation to be used."""

    config_id: str
    """The id of the configuration to be used."""

    config_ids: SequenceNotStr[str]
    """The list of configuration ids to be used.

    If set, the configurations will be combined.
    """

    context: Dict[str, object]
    """Additional context data to be added to the conversation."""

    options: GenerationOptionsParam
    """A set of options that should be applied during a generation.

    The GenerationOptions control various things such as what rails are enabled,
    additional parameters for the main LLM, whether the rails should be enforced or
    ran in parallel, what to be included in the generation log, etc.
    """

    return_choice: bool
    """If set, guardrails data will be included as a JSON in the choices array."""

    state: Dict[str, object]
    """A state object that should be used to continue the interaction."""

    stream: bool
    """If set, partial message deltas will be sent, like in ChatGPT.

    Tokens will be sent as data-only server-sent events as they become available,
    with the stream terminated by a data: [DONE] message.
    """
