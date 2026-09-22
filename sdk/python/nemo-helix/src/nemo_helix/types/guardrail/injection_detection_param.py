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

from ..._types import SequenceNotStr

__all__ = ["InjectionDetectionParam"]


class InjectionDetectionParam(TypedDict, total=False):
    action: str
    """Action to take.

    Options are 'reject' to offer a rejection message, 'omit' to mask the offending
    content, and 'sanitize' to pass the content as-is in the safest way. These
    options are listed in descending order of relative safety. 'sanitize' is not
    implemented at this time.
    """

    injections: SequenceNotStr[str]
    """The list of injection types to detect.

    Options are 'sqli', 'template', 'code', 'xss'.Currently, only SQL injection,
    template injection, code injection, and markdown cross-site scripting are
    supported. Custom rules can be added, provided they are in the `yara_path` and
    have a `.yara` file extension.
    """

    yara_rules: Dict[str, str]
    """Dictionary mapping rule names to YARA rule strings.

    If provided, these rules will be used instead of loading rules from yara_path.
    Each rule should be a valid YARA rule string.
    """
