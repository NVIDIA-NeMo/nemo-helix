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

from typing import Dict, Union, Optional

from pydantic import Field as FieldInfo

from ..._models import BaseModel

__all__ = ["DatasetMetadataContent"]


class DatasetMetadataContent(BaseModel):
    """Content for dataset-type filesets."""

    schema_: Union[Dict[str, object], str, None] = FieldInfo(alias="schema", default=None)
    """
    Default row schema for files in this fileset, either inline JSON Schema or a
    schema_defs key.
    """

    schema_defs: Optional[Dict[str, Dict[str, object]]] = None
    """
    Reusable JSON Schema definitions keyed by name for deduplicating per-file
    dataset schemas.
    """

    schemas_by_path: Optional[Dict[str, Union[Dict[str, object], str]]] = None
    """Optional per-file row schemas keyed by relative path within the fileset.

    Each value may be inline JSON Schema or a schema_defs key.
    """
