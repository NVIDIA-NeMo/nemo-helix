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

__all__ = ["AuthContext"]


class AuthContext(BaseModel):
    """Auth context captured at resource creation for delegated access.

    Stores a snapshot of the creating principal's identity so that controllers
    can later act on their behalf (e.g., accessing secrets).
    """

    principal_id: str
    """The principal's unique identifier"""

    principal_email: Optional[str] = None
    """The principal's email address"""

    principal_groups: Optional[List[str]] = None
    """Groups the principal belongs to"""

    principal_on_behalf_of: Optional[str] = None
    """If acting on behalf of another principal, their principal ID"""

    principal_on_behalf_of_email: Optional[str] = None
    """The on-behalf-of principal's email address"""

    principal_on_behalf_of_groups: Optional[List[str]] = None
    """Groups the on-behalf-of principal belongs to"""
