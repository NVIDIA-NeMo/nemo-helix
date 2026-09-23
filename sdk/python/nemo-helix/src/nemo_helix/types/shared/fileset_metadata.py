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

from ..._models import BaseModel
from .model_metadata_content import ModelMetadataContent
from .dataset_metadata_content import DatasetMetadataContent
from .environment_metadata_content import EnvironmentMetadataContent

__all__ = ["FilesetMetadata"]


class FilesetMetadata(BaseModel):
    """Tagged metadata container - the key indicates the type.

    Example:
        metadata = FilesetMetadata(
            dataset=DatasetMetadataContent(
                schema={"columns": ["id", "name"]},
            )
        )
    """

    dataset: Optional[DatasetMetadataContent] = None
    """Content for dataset-type filesets."""

    environment: Optional[EnvironmentMetadataContent] = None
    """Content for environment-type filesets (GRPO Gym packages)."""

    model: Optional[ModelMetadataContent] = None
    """Content for model-type filesets.

    Contains tool calling configuration that is merged into the ModelSpec during
    checkpoint analysis.
    """
