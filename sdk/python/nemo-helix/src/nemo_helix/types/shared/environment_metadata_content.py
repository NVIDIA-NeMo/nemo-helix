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

from typing import List, Optional

from ..._models import BaseModel

__all__ = ["EnvironmentMetadataContent"]


class EnvironmentMetadataContent(BaseModel):
    """Content for environment-type filesets (GRPO Gym packages)."""

    format: str
    """Environment package format (native-v1, wheels-v1, adapter-wheels-v1)."""

    name: str

    adapter_agent: Optional[str] = None

    config_paths: Optional[List[str]] = None

    hub_id: Optional[str] = None

    vf_env_id: Optional[str] = None
