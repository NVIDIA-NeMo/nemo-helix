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

from typing import List, Optional
from typing_extensions import Literal

from ..._models import BaseModel
from .mo_e_config import MoEConfig
from .mamba_config import MambaConfig
from .tool_call_config import ToolCallConfig
from .linear_layer_spec import LinearLayerSpec
from .sliding_window_config import SlidingWindowConfig

__all__ = ["ModelSpec"]


class ModelSpec(BaseModel):
    """Detailed specification for a model."""

    base_num_parameters: int
    """Total model parameters"""

    checkpoint_model_name: str
    """Checkpoint Model identifier or model path"""

    family: str
    """Model architecture family (e.g., 'llama', 'mixtral', 'gpt2')"""

    ffn_hidden_size: int
    """FFN intermediate size"""

    gated_mlp: bool
    """Whether MLP uses gated activation"""

    hidden_size: int
    """Hidden dimension size"""

    num_attention_heads: int
    """Number of attention heads"""

    num_kv_heads: int
    """Number of key-value heads (for GQA/MQA)"""

    num_layers: int
    """Number of transformer layers"""

    precision: str
    """Model precision (e.g., 'float16', 'bfloat16', 'float32', 'int8', 'int4')"""

    tied_embeddings: bool
    """Whether embeddings are tied"""

    vocab_size: int
    """Vocabulary size"""

    chat_template: Optional[str] = None
    """Jinja2 chat template string for the model.

    Used by NIM to format chat completions. If not set, the model's built-in
    tokenizer template is used.
    """

    context_size: Optional[int] = None
    """Context window size"""

    head_type: Optional[Literal["causal_lm", "embedding", "cross_encoder", "unknown"]] = None
    """Task-specific head persisted in the checkpoint."""

    is_chat: Optional[bool] = None
    """Whether this is a chat model"""

    is_embedding_model: Optional[bool] = None
    """Deprecated compatibility alias for head_type == 'embedding'."""

    linear_layers: Optional[List[LinearLayerSpec]] = None
    """List of all linear/Conv1D layers with their dimensions.

    Used for LoRA parameter estimation without requiring model instantiation. Each
    entry contains the module name, in_features, and out_features.
    """

    mamba_config: Optional[MambaConfig] = None
    """Mamba/State Space Model configuration."""

    minimum_gpus_all_weights: Optional[int] = None
    """Minimum GPUs required for full fine-tuning using default configurations."""

    minimum_gpus_lora: Optional[int] = None
    """Minimum GPUs required for LoRA fine-tuning using default configurations."""

    moe_config: Optional[MoEConfig] = None
    """Mixture of Experts configuration."""

    num_virtual_tokens: Optional[int] = None
    """Number of virtual tokens for prompt tuning"""

    sliding_window_config: Optional[SlidingWindowConfig] = None
    """Sliding window attention configuration."""

    tool_call_config: Optional[ToolCallConfig] = None
    """Configuration for tool calling support in NIM deployments."""
