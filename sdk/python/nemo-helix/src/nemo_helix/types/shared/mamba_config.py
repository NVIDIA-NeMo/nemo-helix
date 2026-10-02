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

__all__ = ["MambaConfig"]


class MambaConfig(BaseModel):
    """Mamba/State Space Model configuration."""

    is_hybrid: bool
    """Whether model is Mamba-Transformer hybrid"""

    num_mamba_layers: int
    """Number of Mamba/SSM layers"""

    conv_kernel: Optional[int] = None
    """Convolution kernel size for Mamba (d_conv)"""

    num_attention_layers: Optional[int] = None
    """Number of attention layers (for hybrids)"""

    num_mlp_layers: Optional[int] = None
    """Number of standalone MLP layers (for interleaved architectures)"""

    state_size: Optional[int] = None
    """SSM state expansion factor (d_state)"""
