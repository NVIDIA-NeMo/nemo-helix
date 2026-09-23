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
from typing_extensions import TypedDict

from .patronus_evaluation_success_strategy import PatronusEvaluationSuccessStrategy

__all__ = ["PatronusEvaluateAPIParamsParam"]


class PatronusEvaluateAPIParamsParam(TypedDict, total=False):
    """Config to parameterize the Patronus Evaluate API call"""

    params: Dict[str, object]
    """Parameters to the Patronus Evaluate API"""

    success_strategy: PatronusEvaluationSuccessStrategy
    """
    Strategy for determining whether a Patronus Evaluation API request should pass,
    especially when multiple evaluators are called in a single request. ALL_PASS
    requires all evaluators to pass for success. ANY_PASS requires only one
    evaluator to pass for success.
    """
