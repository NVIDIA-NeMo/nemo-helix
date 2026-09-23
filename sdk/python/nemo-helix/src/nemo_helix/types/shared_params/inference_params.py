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

from ..._types import SequenceNotStr

__all__ = ["InferenceParams"]


class InferenceParams(  # type: ignore[call-arg]
    TypedDict,
    total=False,
    extra_items=object,  # pyright: ignore[reportGeneralTypeIssues]
):
    """Parameters for model inference.

    Extra fields can be supplied for additional options applied to the inference request directly. Fields not supported by the model may cause inference errors during evaluation.
    """

    max_completion_tokens: int
    """Max tokens to generate"""

    max_tokens: int
    """Max tokens to generate"""

    model: str
    """Model identifier"""

    stop: SequenceNotStr[str]

    temperature: float
    """Float value between 0 and 1.

    temp of 0 indicates greedy decoding, where the token with highest prob is
    chosen. Temperature can't be set to 0.0 currently
    """

    top_p: float
    """
    Float value between 0 and 1; limits to the top tokens within a certain
    probability. top_p=0 means the model will only consider the single most likely
    token for the next prediction
    """
