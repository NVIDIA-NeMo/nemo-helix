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

from typing import Iterable
from typing_extensions import Literal, Required, TypedDict

from .mo_e_config import MoEConfig
from .mamba_config import MambaConfig
from .tool_call_config import ToolCallConfig
from .linear_layer_spec import LinearLayerSpec
from .sliding_window_config import SlidingWindowConfig

__all__ = ["ModelSpec"]


class ModelSpec(TypedDict, total=False):
    """Detailed specification for a model."""

    base_num_parameters: Required[int]
    """Total model parameters"""

    checkpoint_model_name: Required[str]
    """Checkpoint Model identifier or model path"""

    family: Required[str]
    """Model architecture family (e.g., 'llama', 'mixtral', 'gpt2')"""

    ffn_hidden_size: Required[int]
    """FFN intermediate size"""

    gated_mlp: Required[bool]
    """Whether MLP uses gated activation"""

    hidden_size: Required[int]
    """Hidden dimension size"""

    num_attention_heads: Required[int]
    """Number of attention heads"""

    num_kv_heads: Required[int]
    """Number of key-value heads (for GQA/MQA)"""

    num_layers: Required[int]
    """Number of transformer layers"""

    precision: Required[str]
    """Model precision (e.g., 'float16', 'bfloat16', 'float32', 'int8', 'int4')"""

    tied_embeddings: Required[bool]
    """Whether embeddings are tied"""

    vocab_size: Required[int]
    """Vocabulary size"""

    chat_template: str
    """Jinja2 chat template string for the model.

    Used by NIM to format chat completions. If not set, the model's built-in
    tokenizer template is used.
    """

    context_size: int
    """Context window size"""

    head_type: Literal["causal_lm", "embedding", "cross_encoder", "unknown"]
    """Task-specific head persisted in the checkpoint."""

    is_chat: bool
    """Whether this is a chat model"""

    is_embedding_model: bool
    """Deprecated compatibility alias for head_type == 'embedding'."""

    linear_layers: Iterable[LinearLayerSpec]
    """List of all linear/Conv1D layers with their dimensions.

    Used for LoRA parameter estimation without requiring model instantiation. Each
    entry contains the module name, in_features, and out_features.
    """

    mamba_config: MambaConfig
    """Mamba/State Space Model configuration."""

    minimum_gpus_all_weights: int
    """Minimum GPUs required for full fine-tuning using default configurations."""

    minimum_gpus_lora: int
    """Minimum GPUs required for LoRA fine-tuning using default configurations."""

    moe_config: MoEConfig
    """Mixture of Experts configuration."""

    num_virtual_tokens: int
    """Number of virtual tokens for prompt tuning"""

    sliding_window_config: SlidingWindowConfig
    """Sliding window attention configuration."""

    tool_call_config: ToolCallConfig
    """Configuration for tool calling support in NIM deployments."""
