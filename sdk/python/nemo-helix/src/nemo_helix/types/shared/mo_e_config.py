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

__all__ = ["MoEConfig"]


class MoEConfig(BaseModel):
    """Mixture of Experts configuration."""

    num_expert_layers: int
    """Number of layers with MoE"""

    num_experts: int
    """Total number of routed experts (sharded by EP)"""

    num_experts_per_tok: int
    """Number of experts activated per token (top-k routing)"""

    expert_ffn_size: Optional[int] = None
    """FFN size for experts (if different from main FFN)"""

    num_shared_experts: Optional[int] = None
    """Number of shared experts (replicated, not sharded by EP)"""
