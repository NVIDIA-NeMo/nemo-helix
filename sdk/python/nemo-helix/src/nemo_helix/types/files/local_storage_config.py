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
from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["LocalStorageConfig"]


class LocalStorageConfig(BaseModel):
    path: str

    read_chunk_size: Optional[int] = None
    """Chunk size in bytes for reading/streaming files.

    Larger chunks reduce async overhead but increase memory per concurrent download.
    Default: 1MB.
    """

    type: Optional[Literal["local"]] = None

    write_buffer_size: Optional[int] = None
    """How many bytes to buffer before flushing to disk"""
