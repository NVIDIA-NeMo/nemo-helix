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

__all__ = ["LogQueryParams"]


class LogQueryParams(TypedDict, total=False):
    workspace: str

    artifact_base_path: str
    """
    Folder inside the fileset the logs were nested under (must match the value used
    on write)
    """

    filters: Dict[str, str]
    """Key-value filters to apply to the query"""

    limit: int
    """Maximum number of results to return"""

    page_cursor: str
    """Cursor for pagination"""

    tail: int
    """Number of newest log lines to return"""
