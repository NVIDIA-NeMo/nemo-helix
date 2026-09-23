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

from ..._compat import PYDANTIC_V1, ConfigDict
from ..._models import BaseModel

__all__ = ["ServedModelMapping"]


class ServedModelMapping(BaseModel):
    """Mapping between a Model Entity and how it's served by this provider."""

    model_entity_id: str
    """Model Entity identifier as workspace/name (e.g., 'my-ws/my-model')"""

    served_model_name: str
    """The actual model name to send to the backend endpoint in the 'model' field"""

    if not PYDANTIC_V1:
        # allow fields with a `model_` prefix
        model_config = ConfigDict(protected_namespaces=tuple())
