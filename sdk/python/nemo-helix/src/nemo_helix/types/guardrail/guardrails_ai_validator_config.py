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

__all__ = ["GuardrailsAIValidatorConfig"]


class GuardrailsAIValidatorConfig(BaseModel):
    """Configuration for a single Guardrails AI validator."""

    name: str
    """
    Unique identifier or import path for the Guardrails AI validator (e.g.,
    'toxic_language', 'pii', 'regex_match', or 'guardrails/competitor_check').
    """

    metadata: Optional[Dict[str, object]] = None
    """
    Metadata to pass to the validator during validation (e.g., valid_topics,
    context).
    """

    parameters: Optional[Dict[str, object]] = None
    """
    Parameters to pass to the validator during initialization (e.g., threshold,
    regex pattern).
    """
