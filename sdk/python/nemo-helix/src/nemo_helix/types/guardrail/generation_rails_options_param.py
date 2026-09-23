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
from typing_extensions import TypedDict

from ..._types import SequenceNotStr

__all__ = ["GenerationRailsOptionsParam"]


class GenerationRailsOptionsParam(TypedDict, total=False):
    """Options for what rails should be used during the generation."""

    dialog: bool
    """Whether the dialog rails are enabled or not."""

    input: Union[bool, SequenceNotStr[str]]
    """Whether the input rails are enabled or not.

    If a list of names is specified, then only the specified input rails will be
    applied.
    """

    output: Union[bool, SequenceNotStr[str]]
    """Whether the output rails are enabled or not.

    If a list of names is specified, then only the specified output rails will be
    applied.
    """

    retrieval: Union[bool, SequenceNotStr[str]]
    """Whether the retrieval rails are enabled or not.

    If a list of names is specified, then only the specified retrieval rails will be
    applied.
    """
