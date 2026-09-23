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

__all__ = ["MultilingualConfigParam"]


class MultilingualConfigParam(TypedDict, total=False):
    """Configuration for multilingual refusal messages."""

    enabled: bool
    """
    If True, detect the language of user input and return refusal messages in the
    same language. Supported languages: en (English), es (Spanish), zh (Chinese), de
    (German), fr (French), hi (Hindi), ja (Japanese), ar (Arabic), th (Thai).
    """

    refusal_messages: Dict[str, str]
    """Custom refusal messages per language code.

    If not specified, built-in defaults are used. Example: {'en': 'Sorry, I cannot
    help.', 'es': 'Lo siento, no puedo ayudar.'}
    """
