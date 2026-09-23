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
from typing_extensions import Literal, TypedDict

from .clavata_rail_options_param import ClavataRailOptionsParam

__all__ = ["ClavataRailConfigParam"]


class ClavataRailConfigParam(TypedDict, total=False):
    """Configuration data for the Clavata API"""

    input: ClavataRailOptionsParam
    """Configuration data for the Clavata API"""

    label_match_logic: Literal["ANY", "ALL"]
    """
    The logic to use when deciding whether the evaluation matched. If ANY, only one
    of the configured labels needs to be found in the input or output. If ALL, all
    configured labels must be found in the input or output.
    """

    output: ClavataRailOptionsParam
    """Configuration data for the Clavata API"""

    policies: Dict[str, str]
    """A dictionary of policy aliases and their corresponding IDs."""

    server_endpoint: str
    """The endpoint for the Clavata API"""
