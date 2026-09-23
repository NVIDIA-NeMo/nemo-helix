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

from typing import Optional

from ..._models import BaseModel

__all__ = ["EvaluationContext"]


class EvaluationContext(BaseModel):
    """
    Identifies the Evaluation and optional test case associated with ingested telemetry.
    """

    evaluation_id: Optional[str] = None
    """Deprecated alias for evaluation_name. Use evaluation_name instead."""

    evaluation_name: Optional[str] = None
    """Name of an existing Evaluation."""

    test_case_id: Optional[str] = None
    """Deprecated alias for test_case_name. Use test_case_name instead."""

    test_case_name: Optional[str] = None
    """Optional producer-supplied test case name."""
