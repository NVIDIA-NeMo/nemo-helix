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

from typing_extensions import Required, TypedDict

__all__ = ["MoEConfig"]


class MoEConfig(TypedDict, total=False):
    """Mixture of Experts configuration."""

    num_expert_layers: Required[int]
    """Number of layers with MoE"""

    num_experts: Required[int]
    """Total number of routed experts (sharded by EP)"""

    num_experts_per_tok: Required[int]
    """Number of experts activated per token (top-k routing)"""

    expert_ffn_size: int
    """FFN size for experts (if different from main FFN)"""

    num_shared_experts: int
    """Number of shared experts (replicated, not sharded by EP)"""
