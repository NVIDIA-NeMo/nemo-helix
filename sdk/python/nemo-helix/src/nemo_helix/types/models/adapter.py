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
from datetime import datetime

from .lora import Lora
from ..._models import BaseModel
from ..shared.finetuning_type import FinetuningType

__all__ = ["Adapter"]


class Adapter(BaseModel):
    fileset: str
    """
    Fileset where the adapter files are stored expected format
    {workspace}/{fileset_name}
    """

    finetuning_type: FinetuningType
    """Finetuning types."""

    name: str
    """Name of the adapter.

    Name must be unique in the workspace for all Adapters and match the following
    regex: Allowed characters: letters (a-z, A-Z), digits (0-9), underscores,
    hyphens, and dots.
    """

    workspace: str
    """Workspace of the adapter.

    Allowed characters: letters (a-z, A-Z), digits (0-9), underscores, hyphens, and
    dots.
    """

    created_at: Optional[datetime] = None

    description: Optional[str] = None
    """Optional description of the adapter"""

    enabled: Optional[bool] = None
    """Whether to make this adapter available for inference post training"""

    lora_config: Optional[Lora] = None
    """Lora configuration specifics"""

    model: Optional[str] = None
    """Parent model entity reference.

    A single name (2-63 characters) or 'workspace/model*name' where each segment is
    a valid name (lowercase, digits, hyphens, and temporarily @ . + *; no
    leading/trailing or consecutive hyphens). If one slash, both sides must be
    non-empty.
    """

    updated_at: Optional[datetime] = None
