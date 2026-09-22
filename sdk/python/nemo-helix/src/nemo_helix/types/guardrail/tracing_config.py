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
from .log_adapter_config import LogAdapterConfig

__all__ = ["TracingConfig"]


class TracingConfig(BaseModel):
    adapters: Optional[List[LogAdapterConfig]] = None
    """The list of tracing adapters to use.

    If not specified, the default adapters are used.
    """

    enable_content_capture: Optional[bool] = None
    """
    Capture prompts and responses (user/assistant/tool message content) in
    tracing/telemetry events. Disabled by default for privacy and alignment with
    OpenTelemetry GenAI semantic conventions. WARNING: Enabling this may include PII
    and sensitive data in your telemetry backend.
    """

    enabled: Optional[bool] = None

    span_format: Optional[str] = None
    """The span format to use.

    Options are 'legacy' (simple metrics) or 'opentelemetry' (OpenTelemetry semantic
    conventions).
    """
