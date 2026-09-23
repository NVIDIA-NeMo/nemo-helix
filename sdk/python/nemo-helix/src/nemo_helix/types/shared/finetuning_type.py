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

from typing_extensions import Literal, TypeAlias

__all__ = ["FinetuningType"]

FinetuningType: TypeAlias = Literal[
    "lora_merged",
    "all_weights",
    "last_layer",
    "top_layers",
    "gradual_unfreezing",
    "bias_only",
    "attention_only",
    "lora",
    "qlora",
    "adalora",
    "dora",
    "lora_plus",
    "prompt_tuning",
    "prefix_tuning",
    "p_tuning",
    "p_tuning_v2",
    "soft_prompt",
    "ppo",
    "dpo",
    "cdpo",
    "ipo",
    "orpo",
    "kto",
    "rrhf",
    "grpo",
]
