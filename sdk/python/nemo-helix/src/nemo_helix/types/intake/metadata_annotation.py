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
from datetime import datetime
from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["MetadataAnnotation"]


class MetadataAnnotation(BaseModel):
    """Structured key/value metadata attached to a span or session."""

    annotation_id: str

    created_at: datetime

    ingested_at: datetime

    kind: Literal["metadata"]
    """Discriminator. Always `metadata` for this variant."""

    metadata: Dict[str, object]
    """The metadata key/value pairs."""

    session_id: str

    workspace: str

    created_by: Optional[str] = None

    span_id: Optional[str] = None
    """
    Id of the span this annotation applies to, or omitted for session-level
    annotations.
    """
