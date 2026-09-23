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
from .single_call_config import SingleCallConfig
from .user_messages_config import UserMessagesConfig

__all__ = ["DialogRails"]


class DialogRails(BaseModel):
    """Configuration of topical rails."""

    single_call: Optional[SingleCallConfig] = None
    """Configuration for the single LLM call option for topical rails."""

    user_messages: Optional[UserMessagesConfig] = None
    """Configuration for how the user messages are interpreted."""
