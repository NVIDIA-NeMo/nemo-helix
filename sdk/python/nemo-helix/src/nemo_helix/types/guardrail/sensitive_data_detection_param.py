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

from typing import Dict, Iterable
from typing_extensions import TypedDict

from .sensitive_data_detection_options_param import SensitiveDataDetectionOptionsParam

__all__ = ["SensitiveDataDetectionParam"]


class SensitiveDataDetectionParam(TypedDict, total=False):
    """Configuration of what sensitive data should be detected."""

    input: SensitiveDataDetectionOptionsParam
    """Configuration of the entities to be detected on the user input."""

    output: SensitiveDataDetectionOptionsParam
    """Configuration of the entities to be detected on the bot output."""

    recognizers: Iterable[Dict[str, object]]
    """Additional custom recognizers.

    Check out https://microsoft.github.io/presidio/tutorial/08_no_code/ for more
    details.
    """

    retrieval: SensitiveDataDetectionOptionsParam
    """Configuration of the entities to be detected on retrieved relevant chunks."""
