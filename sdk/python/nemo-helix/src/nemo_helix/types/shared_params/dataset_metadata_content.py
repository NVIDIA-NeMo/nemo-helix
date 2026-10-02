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

from typing import Dict, Union
from typing_extensions import TypedDict

__all__ = ["DatasetMetadataContent"]


class DatasetMetadataContent(TypedDict, total=False):
    """Content for dataset-type filesets."""

    schema: Union[Dict[str, object], str]
    """
    Default row schema for files in this fileset, either inline JSON Schema or a
    schema_defs key.
    """

    schema_defs: Dict[str, Dict[str, object]]
    """
    Reusable JSON Schema definitions keyed by name for deduplicating per-file
    dataset schemas.
    """

    schemas_by_path: Dict[str, Union[Dict[str, object], str]]
    """Optional per-file row schemas keyed by relative path within the fileset.

    Each value may be inline JSON Schema or a schema_defs key.
    """
