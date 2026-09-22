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
from typing_extensions import Required, TypedDict

__all__ = ["EntityUpdateEntityByNameParams"]


class EntityUpdateEntityByNameParams(TypedDict, total=False):
    workspace: str

    entity_type: Required[str]

    data: Required[Dict[str, object]]
    """Updated entity-specific data"""

    parent: str
    """Parent entity ID for nested entities"""

    expected_db_version: int
    """Optional database version for optimistic locking.

    Update only succeeds if current version matches.
    """

    new_name: str
    """Updated entity name (optional).

    Name must start with a lowercase letter, be 2-63 characters, and use lowercase
    letters, digits, hyphens, and dots (no consecutive hyphens, cannot end with a
    hyphen).
    """

    project: str
    """The name of the project associated with this entity"""
