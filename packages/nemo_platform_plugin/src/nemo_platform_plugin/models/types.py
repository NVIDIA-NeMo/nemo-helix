# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed request/response models for the Models service client.

These models mirror the HTTP contract for model entities, adapters, model
providers, prompts, model deployments, and model deployment configs. The
Models service remains the authoritative wire-schema owner; this plugin module
keeps client DTOs independent of the Stainless-generated SDK.

The plugin package must stay free of an ``nmp_common`` dependency because that
would create a reverse service dependency. Server-only pieces are handled per
the data-vs-behavior split documented in ``client/MIGRATION.md``:

- **Constants** (name regex, max lengths) that live in
  ``nmp.common.entities.constants`` are inlined here with a comment pointing at
  the origin (matching the ``secrets.types`` boundary).
- **``AuthContext``** is a pure-data mirror of ``nmp.common.auth.AuthContext``
  with the same wire shape and no ``from_principal``/``to_principal`` behavior.
- **``InferenceParams``** is a faithful replica of
  ``nmp.common.inference.InferenceParams`` (pure pydantic, no server imports).
- **``BackendFormat``** is reused from ``nemo_platform_plugin.inference_middleware``.
- The Jinja2 ``auth_header_format`` validator needs ``jinja2`` (not a plugin
  dependency) so it stays server-side; the plugin field is a plain string.
- Entity-store ``Filter`` subclasses are genuinely server-only and are not
  mirrored here; the client passes filters as a ``filter`` query-param string.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum, StrEnum
from typing import Any, Literal, NotRequired, Self, TypedDict

from nemo_platform_plugin.inference_middleware import BackendFormat
from pydantic import AnyUrl, BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Inlined constants
#
# Mirror ``nmp.common.entities.constants`` and ``nmp.core.models.constants``.
# Inlined rather than imported so this package stays free of an ``nmp_common``
# (or models-service) dependency -- see the module docstring.
# ---------------------------------------------------------------------------

_NAME_REGEX = r"^[\w\-.]+$"  # constants.REGEX_WORD_CHARACTER_DOT_DASH
_NAME_SLASH_REGEX = r"^[\w\-./]+$"  # constants.REGEX_WORD_CHARACTER_DOT_DASH_SLASH
_NAME_DESC = "Allowed characters: letters (a-z, A-Z), digits (0-9), underscores, hyphens, and dots."
_MAX_LEN_255 = 255  # constants.MAX_LENGTH_255

# nmp.core.models.constants -- adapter/model reference rules.
_MODEL_REF_NAME_SEGMENT = r"[a-z](?!.*--)[a-z0-9\-@.+_]{1,62}(?<!-)"
MODEL_REF_MAX_LEN = 127  # 63 + '/' + 63
MODEL_REF_PATTERN = rf"^{_MODEL_REF_NAME_SEGMENT}(/{_MODEL_REF_NAME_SEGMENT})?$"
MODEL_REF_PATTERN_DESCRIPTION = (
    "A single name (2-63 characters) or 'workspace/model_name' where each segment is a valid name "
    "(lowercase, digits, hyphens, and temporarily @ . + _; no leading/trailing or consecutive hyphens). "
    "If one slash, both sides must be non-empty."
)
_MODEL_REF_RE = re.compile(MODEL_REF_PATTERN)


def is_valid_model_ref(value: str) -> bool:
    """True if *value* matches :data:`MODEL_REF_PATTERN` (entity NAME rules per segment)."""
    return _MODEL_REF_RE.fullmatch(value) is not None


# ---------------------------------------------------------------------------
# Auth context (data-only mirror of nmp.common.auth.AuthContext)
# ---------------------------------------------------------------------------


class AuthContext(BaseModel):
    """Auth context captured at resource creation for delegated access.

    This is the wire/data shape. The server's ``nmp.common.auth.AuthContext``
    adds ``from_principal`` / ``to_principal`` behavior on top of the same
    fields, and the server response models re-type this field to that class.
    """

    principal_id: str = Field(..., description="The principal's unique identifier")
    principal_email: str | None = Field(default=None, description="The principal's email address")
    principal_groups: list[str] = Field(default_factory=list, description="Groups the principal belongs to")
    principal_on_behalf_of: str | None = Field(
        default=None, description="If acting on behalf of another principal, their principal ID"
    )
    principal_on_behalf_of_groups: list[str] | None = Field(
        default=None, description="Groups the on-behalf-of principal belongs to"
    )
    principal_on_behalf_of_email: str | None = Field(
        default=None, description="The on-behalf-of principal's email address"
    )


# ---------------------------------------------------------------------------
# Inference parameters (replica of nmp.common.inference.InferenceParams)
# ---------------------------------------------------------------------------


class InferenceParams(BaseModel):
    """Parameters for model inference.

    Extra fields can be supplied for additional options applied to the inference
    request directly. Fields not supported by the model may cause inference
    errors during evaluation.
    """

    model_config = ConfigDict(extra="allow")

    model: str | None = Field(default=None, description="Model identifier")
    temperature: float | None = Field(
        default=None,
        ge=0,
        le=2,
        description="Float value between 0 and 1. temp of 0 indicates greedy decoding, "
        "where the token with highest prob is chosen. Temperature can't be set to 0.0 currently",
    )
    max_tokens: int | None = Field(default=None, ge=1, description="Max tokens to generate")
    max_completion_tokens: int | None = Field(default=None, ge=1, description="Max tokens to generate")
    top_p: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Float value between 0 and 1; limits to the top tokens within a certain "
        "probability. top_p=0 means the model will only consider the single most likely "
        "token for the next prediction",
    )
    stop: list[str] | None = Field(default=None)

    @model_validator(mode="after")
    def check_max_tokens(self) -> Self:
        if self.max_tokens and self.max_completion_tokens:
            raise ValueError(
                "max_tokens and max_completion_tokens cannot both be configured. "
                "Choose the appropriate tokens parameter for the model."
            )
        return self


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


class ModelPrecision(str, Enum):
    """Type of model precision."""

    INT8 = "int8"
    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"
    FP8_MIXED = "fp8-mixed"
    BF16_MIXED = "bf16-mixed"


class FinetuningType(str, Enum):
    """Finetuning types."""

    LORA_MERGED = "lora_merged"
    ALL_WEIGHTS = "all_weights"

    LAST_LAYER = "last_layer"
    TOP_LAYERS = "top_layers"
    GRADUAL_UNFREEZING = "gradual_unfreezing"
    BIAS_ONLY = "bias_only"  # BitFit
    ATTENTION_ONLY = "attention_only"

    LORA = "lora"
    QLORA = "qlora"
    ADALORA = "adalora"
    DORA = "dora"
    LORA_PLUS = "lora_plus"

    PROMPT_TUNING = "prompt_tuning"
    PREFIX_TUNING = "prefix_tuning"
    P_TUNING = "p_tuning"
    P_TUNING_V2 = "p_tuning_v2"
    SOFT_PROMPT = "soft_prompt"

    PPO = "ppo"
    DPO = "dpo"
    CDPO = "cdpo"
    IPO = "ipo"
    ORPO = "orpo"
    KTO = "kto"
    RRHF = "rrhf"
    GRPO = "grpo"


class MoEConfig(BaseModel):
    """Mixture of Experts configuration."""

    num_experts: int = Field(description="Total number of routed experts (sharded by EP)")
    num_experts_per_tok: int = Field(description="Number of experts activated per token (top-k routing)")
    num_expert_layers: int = Field(description="Number of layers with MoE")
    expert_ffn_size: int | None = Field(default=None, description="FFN size for experts (if different from main FFN)")
    num_shared_experts: int = Field(default=0, description="Number of shared experts (replicated, not sharded by EP)")


class MambaConfig(BaseModel):
    """Mamba/State Space Model configuration."""

    is_hybrid: bool = Field(description="Whether model is Mamba-Transformer hybrid")
    num_mamba_layers: int = Field(description="Number of Mamba/SSM layers")
    num_attention_layers: int = Field(default=0, description="Number of attention layers (for hybrids)")
    num_mlp_layers: int = Field(
        default=0, description="Number of standalone MLP layers (for interleaved architectures)"
    )
    state_size: int = Field(default=16, description="SSM state expansion factor (d_state)")
    conv_kernel: int = Field(default=4, description="Convolution kernel size for Mamba (d_conv)")


class SlidingWindowConfig(BaseModel):
    """Sliding window attention configuration."""

    window_size: int = Field(description="Sliding window size (attends to last N tokens)")


class ToolCallConfig(BaseModel):
    """Configuration for tool calling support in NIM deployments."""

    tool_call_parser: str | None = Field(
        default=None,
        description="Name of the tool call parser to use (e.g., 'openai', 'hermes', 'pythonic', 'llama3_json', 'mistral').",
        max_length=_MAX_LEN_255,
    )
    tool_call_plugin: str | None = Field(
        default=None,
        description="Reference to a fileset containing the custom tool call plugin Python file. "
        "Expected format: '{workspace}/{fileset_name}'. The fileset is mounted separately from "
        "the model checkpoint at deployment time.",
        max_length=_MAX_LEN_255,
    )
    auto_tool_choice: bool | None = Field(
        default=None,
        description="Whether to enable automatic tool choice. When enabled, the model can decide to call tools "
        "without explicit user instruction.",
    )


class LinearLayerSpec(BaseModel):
    """Specification for a single linear layer in the model."""

    name: str = Field(description="Module name (e.g., 'model.layers.0.self_attn.q_proj')")
    in_features: int = Field(description="Input feature dimension")
    out_features: int = Field(description="Output feature dimension")


class ModelSpec(BaseModel):
    """Detailed specification for a model."""

    context_size: int | None = Field(None, description="Context window size")
    num_virtual_tokens: int | None = Field(None, description="Number of virtual tokens for prompt tuning")
    is_chat: bool | None = Field(None, description="Whether this is a chat model")
    chat_template_controls_reasoning: bool | None = Field(
        None,
        description=(
            "Whether the served chat template gives a caller control over reasoning through chat_template_kwargs, determined by rendering the template with a candidate kwarg set two different ways and comparing the result. Covers enable_thinking, thinking, and a template-interpolated reasoning_effort. A False here does not mean reasoning cannot be disabled by some other means: reasoning_effort sent as a request parameter and Anthropic's thinking config belong to the serving API rather than the checkpoint, and prompt-level switches such as Qwen3's /no_think are trained model behaviour that leaves no trace in the template. None when undetermined (checkpoint not analyzed, or the template could not be rendered), which is not the same as False."
        ),
    )
    head_type: Literal["causal_lm", "embedding", "cross_encoder", "unknown"] = Field(
        default="unknown",
        description="Task-specific head persisted in the checkpoint.",
    )
    is_embedding_model: bool = Field(
        False,
        description="Deprecated compatibility alias for head_type == 'embedding'.",
        json_schema_extra={"deprecated": True},
    )

    @model_validator(mode="after")
    def _sync_head_type_and_embedding_alias(self) -> Self:
        if self.head_type in ("causal_lm", "embedding", "cross_encoder"):
            self.is_embedding_model = self.head_type == "embedding"
            return self
        if self.is_embedding_model:
            self.head_type = "embedding"
        return self

    # Basic model information
    checkpoint_model_name: str = Field(description="Checkpoint Model identifier or model path")
    family: str = Field(description="Model architecture family (e.g., 'llama', 'mixtral', 'gpt2')")

    # Architecture dimensions
    num_layers: int = Field(description="Number of transformer layers")
    hidden_size: int = Field(description="Hidden dimension size")
    num_attention_heads: int = Field(description="Number of attention heads")
    num_kv_heads: int = Field(description="Number of key-value heads (for GQA/MQA)")
    ffn_hidden_size: int = Field(description="FFN intermediate size")
    vocab_size: int = Field(description="Vocabulary size")

    # Model properties
    tied_embeddings: bool = Field(description="Whether embeddings are tied")
    gated_mlp: bool = Field(description="Whether MLP uses gated activation")
    base_num_parameters: int = Field(description="Total model parameters")
    precision: str = Field(description="Model precision (e.g., 'float16', 'bfloat16', 'float32', 'int8', 'int4')")

    # Optional configurations
    moe_config: MoEConfig | None = Field(default=None, description="MoE configuration if applicable")
    mamba_config: MambaConfig | None = Field(default=None, description="Mamba/SSM configuration if applicable")
    sliding_window_config: SlidingWindowConfig | None = Field(
        default=None, description="Sliding window attention config if applicable"
    )

    # LoRA-specific metadata (pre-computed to avoid model instantiation)
    linear_layers: list[LinearLayerSpec] | None = Field(
        default=None,
        description="List of all linear/Conv1D layers with their dimensions. "
        "Used for LoRA parameter estimation without requiring model instantiation. "
        "Each entry contains the module name, in_features, and out_features.",
    )

    # Deployment configuration
    chat_template: str | None = Field(
        default=None,
        description="Jinja2 chat template string for the model. Used by NIM to format chat completions. "
        "If not set, the model's built-in tokenizer template is used.",
    )
    tool_call_config: ToolCallConfig | None = Field(
        default=None,
        description="Tool calling configuration for NIM deployments. Controls how the model handles "
        "function/tool calling in chat completions.",
    )

    # GPU requirements (auto-calculated)
    minimum_gpus_all_weights: int | None = Field(
        default=None,
        description="Minimum GPUs required for full fine-tuning using default configurations.",
    )
    minimum_gpus_lora: int | None = Field(
        default=None,
        description="Minimum GPUs required for LoRA fine-tuning using default configurations.",
    )

    def model_precision(self) -> ModelPrecision:
        """Convert the precision string to a :class:`ModelPrecision` enum."""
        precision_map = {
            "bf16-mixed": ModelPrecision.BF16_MIXED,
            "bf16": ModelPrecision.BF16,
            "bfloat16": ModelPrecision.BF16,
            "float16": ModelPrecision.FP16,
            "float32": ModelPrecision.FP32,
            "fp16": ModelPrecision.FP16,
            "fp32": ModelPrecision.FP32,
            "fp8-mixed": ModelPrecision.FP8_MIXED,
            "int4": ModelPrecision.INT8,  # Map int4 to int8 as int4 is not in the enum
            "int8": ModelPrecision.INT8,
        }
        if self.precision in precision_map:
            return precision_map[self.precision]
        return ModelPrecision.BF16


class Lora(BaseModel):
    alpha: int | None = Field(None, description="Alpha scaling used for this adapter")
    rank: int = Field(..., description="LoRA Rank")


class APIEndpointData(BaseModel):
    """Data about an inference endpoint."""

    url: AnyUrl | None = Field(None, description="Endpoint URL")
    model_id: str | None = Field(None, description="Model identifier at the endpoint")
    api_key: str | None = Field(None, description="API key for authentication")
    format: str | None = Field(None, description="API format (e.g., openai, nvidia)")


class PromptData(BaseModel):
    """Configuration for prompt engineering."""

    system_prompt: str | None = Field(None, description="System prompt template")
    icl_few_shot_examples: str | None = Field(None, description="In-context learning examples")
    inference_params: InferenceParams | None = Field(
        default=None, description="Inference parameters that should be overridden."
    )
    system_prompt_template: str | None = Field(
        default=None,
        title="System Prompt Template",
        description="The template which will be used to compile the final prompt used for prompting the LLM. Currently supports only {{icl_few_shot_examples}}",
    )


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------


class ModelEntityBaseModel(BaseModel):
    """Base model for all Models service domain objects."""

    id: str = Field(..., description="Autogenerated id")
    name: str = Field(
        description=f"Name of the entity. Name/workspace combo must be unique across all entities. {_NAME_DESC}",
        max_length=_MAX_LEN_255,
        pattern=_NAME_REGEX,
        examples=["llama-3.1-8b", "my-custom-model"],
    )
    workspace: str = Field(
        description=f"The workspace of the entity. {_NAME_DESC}",
        max_length=_MAX_LEN_255,
        pattern=_NAME_REGEX,
    )
    project: str | None = Field(
        default=None,
        description="The URN of the project associated with this entity.",
        max_length=_MAX_LEN_255,
        pattern=_NAME_SLASH_REGEX,
    )
    created_at: datetime = Field(..., description="The timestamp of model entity creation")
    updated_at: datetime = Field(..., description="The timestamp of the last model entity update")


# ---------------------------------------------------------------------------
# ModelProvider
# ---------------------------------------------------------------------------

_AUTH_HEADER_FORMAT_DESCRIPTION = (
    "Jinja2 template string controlling how the API key secret is sent to the upstream. "
    "Must contain exactly one variable named `auth_secret`, which is substituted with the "
    "resolved secret value at request time. "
    "Example: `'X-Api-Key: {{ auth_secret }}'`. "
    "If not set, defaults to `'Authorization: Bearer {{ auth_secret }}'`."
)


class ModelProviderStatus(str, Enum):
    """Status enum for ModelProvider objects."""

    UNKNOWN = "UNKNOWN"
    CREATED = "CREATED"
    PENDING = "PENDING"
    READY = "READY"
    ERROR = "ERROR"
    DELETING = "DELETING"
    DELETED = "DELETED"
    LOST = "LOST"


class ServedModelMapping(BaseModel):
    """Mapping between a Model Entity and how it's served by this provider."""

    model_entity_id: str = Field(
        description="Model Entity identifier as workspace/name (e.g., 'my-ws/my-model')",
        max_length=_MAX_LEN_255,
    )
    served_model_name: str = Field(
        description="The actual model name to send to the backend endpoint in the 'model' field",
        max_length=_MAX_LEN_255,
    )


class ModelProvider(ModelEntityBaseModel):
    """A reachable network endpoint that provides inference for one or more Model Entities.

    The unique identifier for a ModelProvider is the combination of workspace/name.
    """

    id: str = Field(default="", description="Unique identifier for the model provider")
    description: str | None = Field(
        default=None,
        description="Optional description of the model provider",
        max_length=1000,
    )
    host_url: str = Field(
        description="The network endpoint URL for the model provider",
        max_length=2048,
    )
    api_key_secret_name: str | None = Field(
        default=None,
        description="Reference to the API key stored in Secrets service",
        max_length=_MAX_LEN_255,
    )
    served_models: list[ServedModelMapping] | None = Field(
        default_factory=list,
        description="List of models served by this provider with routing information for IGW",
    )
    enabled_models: list[str] | None = Field(
        default=None,
        description="Optional list of specific models to enable from this provider. If not set, all discovered models are enabled.",
    )
    status: ModelProviderStatus = Field(
        default=ModelProviderStatus.UNKNOWN,
        description="Current status of the model provider, populated by models service",
    )
    status_message: str = Field(
        default="",
        description="Detailed status message, populated by models service",
        max_length=1000,
    )
    default_extra_body: dict[str, Any] | None = Field(
        default=None,
        description="Default body parameters for inference requests. Can be overridden by user requests.",
    )
    default_extra_headers: dict[str, str] | None = Field(
        default=None,
        description="Default headers for inference requests. Can be overridden by user requests.",
    )
    required_extra_body: dict[str, Any] | None = Field(
        default=None,
        description="Required body parameters for inference requests. Cannot be overridden by user requests.",
    )
    required_extra_headers: dict[str, str] | None = Field(
        default=None,
        description="Required headers for inference requests. Cannot be overridden by user requests.",
    )
    model_deployment_id: str | None = Field(
        default=None,
        description="Optional reference to the ModelDeployment ID if this provider was auto-created for a deployment",
        max_length=_MAX_LEN_255,
    )
    auth_context: AuthContext | None = Field(default=None, description="Auth context captured at provider creation.")
    auth_header_format: str | None = Field(
        default=None,
        description=_AUTH_HEADER_FORMAT_DESCRIPTION,
        max_length=1024,
    )


class ModelProviderSort(StrEnum):
    """Sort fields for ModelProvider queries."""

    NAME_ASC = "name"
    NAME_DESC = "-name"
    CREATED_AT_ASC = "created_at"
    CREATED_AT_DESC = "-created_at"
    UPDATED_AT_ASC = "updated_at"
    UPDATED_AT_DESC = "-updated_at"
    STATUS_ASC = "status"
    STATUS_DESC = "-status"


class CreateModelProviderRequest(BaseModel):
    """Request model for creating a ModelProvider."""

    name: str = Field(
        description=f"Name of the model provider. {_NAME_DESC}",
        max_length=_MAX_LEN_255,
        pattern=_NAME_REGEX,
        examples=["my-nim-provider", "openai-endpoint"],
    )
    project: str | None = Field(
        default=None,
        description="The URN of the project associated with this model provider",
        max_length=_MAX_LEN_255,
        pattern=_NAME_SLASH_REGEX,
    )
    description: str | None = Field(
        default=None,
        description="Optional description of the model provider",
        max_length=1000,
    )
    host_url: str = Field(
        description="The network endpoint URL for the model provider",
        max_length=2048,
    )
    api_key_secret_name: str | None = Field(
        default=None,
        description="Reference to an API key secret stored in the Secrets service. "
        "Create the secret first via secrets API, then pass the secret name here.",
        max_length=_MAX_LEN_255,
    )
    enabled_models: list[str] | None = Field(
        default=None, description="Optional list of specific models to enable from this provider"
    )
    default_extra_body: dict[str, Any] | None = Field(
        default=None,
        description="Default body parameters for inference requests. Can be overridden by user requests.",
    )
    default_extra_headers: dict[str, str] | None = Field(
        default=None,
        description="Default headers for inference requests. Can be overridden by user requests.",
    )
    required_extra_body: dict[str, Any] | None = Field(
        default=None,
        description="Required body parameters for inference requests. Cannot be overridden by user requests.",
    )
    required_extra_headers: dict[str, str] | None = Field(
        default=None,
        description="Required headers for inference requests. Cannot be overridden by user requests.",
    )
    model_deployment_id: str | None = Field(
        default=None,
        description="Optional reference to the ModelDeployment ID if this provider is being auto-created for a deployment",
        max_length=_MAX_LEN_255,
    )
    status: ModelProviderStatus | None = Field(default=None, description="Status of the model provider")
    status_message: str | None = Field(
        default=None,
        description="Status message",
        max_length=1000,
    )
    auth_header_format: str | None = Field(
        default=None,
        description=_AUTH_HEADER_FORMAT_DESCRIPTION,
        max_length=1024,
    )


class UpsertModelProviderRequest(BaseModel):
    """Request model for upserting a ModelProvider (PUT).

    All fields must be provided - partial updates are not supported for security reasons.
    Use PUT /status endpoint to update status-related fields only.
    """

    project: str | None = Field(
        default=None,
        description="The URN of the project associated with this model provider",
        max_length=_MAX_LEN_255,
        pattern=_NAME_SLASH_REGEX,
    )
    description: str | None = Field(
        default=None,
        description="Optional description of the model provider",
        max_length=1000,
    )
    host_url: str = Field(
        description="The network endpoint URL for the model provider",
        max_length=2048,
    )
    api_key_secret_name: str | None = Field(
        default=None,
        description="Reference to an API key secret stored in the Secrets service. "
        "Create the secret first via secrets API, then pass the secret name here.",
        max_length=_MAX_LEN_255,
    )
    enabled_models: list[str] | None = Field(
        default=None, description="Optional list of specific models to enable from this provider"
    )
    default_extra_body: dict[str, Any] | None = Field(
        default=None,
        description="Default body parameters for inference requests. Can be overridden by user requests.",
    )
    default_extra_headers: dict[str, str] | None = Field(
        default=None,
        description="Default headers for inference requests. Can be overridden by user requests.",
    )
    required_extra_body: dict[str, Any] | None = Field(
        default=None,
        description="Required body parameters for inference requests. Cannot be overridden by user requests.",
    )
    required_extra_headers: dict[str, str] | None = Field(
        default=None,
        description="Required headers for inference requests. Cannot be overridden by user requests.",
    )
    model_deployment_id: str | None = Field(
        default=None,
        description="Optional reference to the ModelDeployment ID if this provider is associated with a deployment",
        max_length=_MAX_LEN_255,
    )
    status: ModelProviderStatus | None = Field(default=None, description="Status of the model provider")
    status_message: str | None = Field(
        default=None,
        description="Status message",
        max_length=1000,
    )
    auth_header_format: str | None = Field(
        default=None,
        description=_AUTH_HEADER_FORMAT_DESCRIPTION,
        max_length=1024,
    )


class UpdateModelProviderStatusRequest(BaseModel):
    """Request model for updating ModelProvider status and autodiscovery fields."""

    model_deployment_id: str | None = Field(
        default=None,
        description="Reference to the ModelDeployment ID if this provider is associated with a deployment",
        max_length=_MAX_LEN_255,
    )
    served_models: list[ServedModelMapping] | None = Field(
        default=None, description="List of models served by this provider with routing information for IGW"
    )
    status: ModelProviderStatus | None = Field(default=None, description="Status of the model provider")
    status_message: str | None = Field(
        default=None,
        description="Status message. If status is provided without status_message, defaults to empty string.",
        max_length=1000,
    )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


class PromptMessageRole(StrEnum):
    """Role of a message author in a chat prompt."""

    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"


class PromptMessage(BaseModel):
    """A single templated message in a chat prompt."""

    role: PromptMessageRole = Field(description="The role of the message author.")
    content: str = Field(description="Templated message content. May contain template variables.")


class FunctionDefinition(BaseModel):
    """An OpenAI-compatible function definition for tool calling."""

    name: str = Field(
        description="The name of the function to be called.",
        max_length=_MAX_LEN_255,
    )
    description: str | None = Field(
        default=None,
        description="A description of what the function does, used by the model to decide when and how to call it.",
    )
    parameters: dict[str, Any] | None = Field(
        default=None,
        description="The parameters the function accepts, described as a JSON Schema object.",
    )
    strict: bool | None = Field(
        default=None,
        description="Whether to enforce strict schema adherence when generating the function call.",
    )


class ChatCompletionTool(BaseModel):
    """An OpenAI-compatible tool definition (currently always a function tool)."""

    type: str = Field(description="The type of the tool. Currently only 'function' is supported.")
    function: FunctionDefinition = Field(description="The function definition for this tool.")

    @field_validator("type")
    @classmethod
    def _validate_type(cls, v: str) -> str:
        if v != "function":
            raise ValueError("Only 'function' tools are supported")
        return v


class Prompt(ModelEntityBaseModel):
    """A reusable, stored chat prompt. The unique identifier is workspace/name."""

    id: str = Field(default="", description="Unique identifier for the prompt.")
    description: str | None = Field(
        default=None,
        description="Optional description of the prompt.",
        max_length=1000,
    )
    messages: list[PromptMessage] = Field(
        default_factory=list,
        description="Ordered list of chat messages that make up the prompt.",
    )
    input_variables: list[str] = Field(
        default_factory=list,
        description="Names of the Jinja2 template variables the prompt expects.",
    )
    tools: list[ChatCompletionTool] | None = Field(
        default=None,
        description="Optional OpenAI-compatible tool definitions to send with the prompt.",
    )
    tool_choice: str | dict[str, Any] | None = Field(
        default=None,
        description="Controls which (if any) tool is called: 'none', 'auto', 'required', or a named-tool object.",
    )
    response_format: dict[str, Any] | None = Field(
        default=None,
        description="Optional OpenAI-compatible response_format, e.g. a json_schema structured-output spec.",
    )
    inference_params: InferenceParams | None = Field(
        default=None,
        description="Optional default model and sampling parameters (temperature, top_p, max_tokens, ...).",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Optional free-form tags for organizing prompts.",
    )


class PromptSort(StrEnum):
    """Sort fields for Prompt queries."""

    NAME_ASC = "name"
    NAME_DESC = "-name"
    CREATED_AT_ASC = "created_at"
    CREATED_AT_DESC = "-created_at"
    UPDATED_AT_ASC = "updated_at"
    UPDATED_AT_DESC = "-updated_at"


class CreatePromptRequest(BaseModel):
    """Request model for creating a Prompt."""

    name: str = Field(
        description=f"Name of the prompt. {_NAME_DESC}",
        max_length=_MAX_LEN_255,
        pattern=_NAME_REGEX,
        examples=["support-bot-system", "summarizer"],
    )
    project: str | None = Field(
        default=None,
        description="The URN of the project associated with this prompt.",
        max_length=_MAX_LEN_255,
        pattern=_NAME_SLASH_REGEX,
    )
    description: str | None = Field(default=None, max_length=1000)
    messages: list[PromptMessage] = Field(default_factory=list)
    input_variables: list[str] = Field(default_factory=list)
    tools: list[ChatCompletionTool] | None = Field(default=None)
    tool_choice: str | dict[str, Any] | None = Field(default=None)
    response_format: dict[str, Any] | None = Field(default=None)
    inference_params: InferenceParams | None = Field(default=None)
    tags: list[str] | None = Field(default=None)


class UpdatePromptRequest(BaseModel):
    """Request model for replacing a Prompt's mutable fields (full update).

    The prompt name and workspace come from the URL path and cannot be changed.
    """

    project: str | None = Field(
        default=None,
        description="The URN of the project associated with this prompt.",
        max_length=_MAX_LEN_255,
        pattern=_NAME_SLASH_REGEX,
    )
    description: str | None = Field(default=None, max_length=1000)
    messages: list[PromptMessage] = Field(default_factory=list)
    input_variables: list[str] = Field(default_factory=list)
    tools: list[ChatCompletionTool] | None = Field(default=None)
    tool_choice: str | dict[str, Any] | None = Field(default=None)
    response_format: dict[str, Any] | None = Field(default=None)
    inference_params: InferenceParams | None = Field(default=None)
    tags: list[str] | None = Field(default=None)


# ---------------------------------------------------------------------------
# Model entity + Adapter
# ---------------------------------------------------------------------------


class Adapter(BaseModel):
    name: str = Field(
        ...,
        description=f"Name of the adapter. Name must be unique in the workspace for all Adapters and match the following regex: {_NAME_DESC}",
        max_length=_MAX_LEN_255,
        pattern=_NAME_REGEX,
        examples=["lora-adapter-v1", "my-finetune"],
    )
    workspace: str = Field(
        ...,
        description=f"Workspace of the adapter. {_NAME_DESC}",
        max_length=_MAX_LEN_255,
        pattern=_NAME_REGEX,
    )
    description: str | None = Field(
        default=None,
        description="Optional description of the adapter",
        max_length=1000,
    )
    fileset: str = Field(
        ...,
        description="Fileset where the adapter files are stored expected format {workspace}/{fileset_name}",
    )
    finetuning_type: FinetuningType = Field(..., description="Type of finetuning (LORA, P_TUNING, etc.)")
    enabled: bool = Field(
        default=True,
        description="Whether to make this adapter available for inference post training",
    )
    lora_config: Lora | None = Field(None, description="Lora configuration specifics")
    model: str | None = Field(
        default=None,
        description=f"Parent model entity reference. {MODEL_REF_PATTERN_DESCRIPTION}",
        max_length=MODEL_REF_MAX_LEN,
    )
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str | None) -> str | None:
        if v is not None and not is_valid_model_ref(v):
            raise ValueError(MODEL_REF_PATTERN_DESCRIPTION)
        return v


class ModelEntity(ModelEntityBaseModel):
    """A versioned model registered within the platform."""

    project: str | None = Field(
        default=None,
        description="The URN of the project associated with this model entity.",
        max_length=_MAX_LEN_255,
    )
    description: str | None = Field(
        default=None,
        description="Optional description of the model.",
        max_length=1000,
    )
    spec: ModelSpec | None = Field(default=None, description="Detailed specification for the model")
    finetuning_type: FinetuningType | None = Field(None, description="Set for full weight finetuned models")
    fileset: str | None = Field(
        default=None,
        description="A set of checkpoint files, configs, and other auxiliary info associated with this model - expected format {workspace}/{fileset_name}",
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Whether to trust remote code to load this model checkpoint.",
    )
    base_model: str | None = Field(
        default=None, description="Link to another model which is used as a base for the current model"
    )
    api_endpoint: APIEndpointData | None = Field(
        default=None, description="Data about the inference endpoint for this model"
    )
    backend_format: BackendFormat | None = Field(
        default=None,
        description=(
            "Inference API wire format expected by the backend. If unset, inference routing treats the model as "
            "OPENAI_CHAT."
        ),
        json_schema_extra={"nullable": True},
    )
    adapters: list[Adapter] | None = Field(
        default=None,
        description="Adapters that have been created against this model",
    )
    prompt: PromptData | None = Field(default=None, description="Configuration for prompt engineering")
    custom_fields: dict[str, Any] = Field(default_factory=dict, description="Custom fields for additional metadata")
    ownership: dict[str, Any] | None = Field(default=None, description="Ownership information for the model")
    model_providers: list[str] = Field(
        default_factory=list,
        description="List of ModelProvider workspace/name resource names that provide inference for this Model Entity",
    )


class CreateModelEntityRequest(BaseModel):
    """Request model for creating a Model Entity."""

    name: str = Field(
        description=f"Name of the model entity. {_NAME_DESC}",
        max_length=_MAX_LEN_255,
        pattern=_NAME_REGEX,
        examples=["llama-3.1-8b", "my-custom-model"],
    )
    project: str | None = Field(
        default=None,
        description="The URN of the project associated with this model entity",
        max_length=_MAX_LEN_255,
        pattern=_NAME_SLASH_REGEX,
    )
    description: str | None = Field(
        default=None,
        description="Optional description of the model",
        max_length=1000,
    )
    spec: ModelSpec | None = Field(
        default=None,
        description="Detailed specification for the model - Automatically generated by the platform at creation when fileset provided.",
    )
    finetuning_type: FinetuningType | None = Field(None, description="Set for full weight finetuned models")
    fileset: str | None = Field(
        default=None,
        description="A set of checkpoint files, configs, and other auxiliary info associated with this model - expected format {workspace}/{fileset_name}",
    )
    base_model: str | None = Field(
        default=None, description="Link to another model which is used as a base for the current model"
    )
    api_endpoint: APIEndpointData | None = Field(
        default=None, description="Data about the inference endpoint for this model"
    )
    backend_format: BackendFormat | None = Field(
        default=None,
        description=(
            "Inference API wire format expected by the backend. If unset, inference routing treats the model as "
            "OPENAI_CHAT."
        ),
        json_schema_extra={"nullable": True},
    )
    prompt: PromptData | None = Field(default=None, description="Configuration for prompt engineering")
    custom_fields: dict[str, Any] | None = Field(default=None, description="Custom fields for additional metadata")
    ownership: dict[str, Any] | None = Field(default=None, description="Ownership information for the model")
    model_providers: list[str] | None = Field(
        default_factory=list,
        description="List of ModelProvider workspace/name resource names that provide inference for this Model Entity",
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Whether to trust remote code for the checkpoint.",
    )


class CreateModelAdapterRequest(BaseModel):
    """Request body for nested Adapter creation. The base model comes from the URL path, not the body."""

    name: str = Field(
        ...,
        description=f"Name of the adapter. Name must be unique in the workspace. {_NAME_DESC}",
        max_length=_MAX_LEN_255,
        pattern=_NAME_REGEX,
        examples=["lora-adapter-v1", "my-finetune"],
    )
    description: str | None = Field(
        default=None,
        description="Optional description of the adapter",
        max_length=1000,
    )
    fileset: str = Field(
        ...,
        description="Location where adapter files are stored - expected format {workspace}/{fileset_name}",
    )
    finetuning_type: FinetuningType = Field(..., description="Type of finetuning (LORA, P_TUNING, etc.)")
    enabled: bool = Field(
        default=True,
        description="Whether to make this adapter available for inference post training",
    )
    lora_config: Lora | None = Field(None, description="Lora configuration specifics")


class CreateAdapterRequest(CreateModelAdapterRequest):
    """Request body for Adapter creation."""

    model: str = Field(
        ...,
        max_length=MODEL_REF_MAX_LEN,
        description=(
            f"Base model entity. Use `{{workspace}}/{{model_name}}` to reference a model in any workspace, "
            f"or a single `{{model_name}}` resolved in the path workspace. {MODEL_REF_PATTERN_DESCRIPTION}"
        ),
        examples=["llama-3-8b-instruct", "shared-tenant/base-llm"],
    )

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str | None) -> str | None:
        if v is not None and not is_valid_model_ref(v):
            raise ValueError(MODEL_REF_PATTERN_DESCRIPTION)
        return v


class UpdateModelEntityRequest(BaseModel):
    """Request model for updating Model Entity metadata."""

    description: str | None = Field(
        default=None,
        description="Optional description of the model",
        max_length=1000,
    )
    spec: ModelSpec | None = Field(default=None, description="Detailed specification for the model")
    fileset: str | None = Field(
        default=None,
        description="A set of checkpoint files, configs, and other auxiliary info associated with this model - expected format {workspace}/{fileset_name}",
    )
    finetuning_type: FinetuningType | None = Field(None, description="Set for full weight finetuned models")
    base_model: str | None = Field(
        default=None, description="Link to another model which is used as a base for the current model"
    )
    api_endpoint: APIEndpointData | None = Field(
        default=None, description="Data about the inference endpoint for this model"
    )
    backend_format: BackendFormat | None = Field(
        default=None,
        description=(
            "Inference API wire format expected by the backend. If unset, inference routing treats the model as "
            "OPENAI_CHAT."
        ),
        json_schema_extra={"nullable": True},
    )
    prompt: PromptData | None = Field(default=None, description="Configuration for prompt engineering")
    custom_fields: dict[str, Any] | None = Field(default=None, description="Custom fields for additional metadata")
    ownership: dict[str, Any] | None = Field(default=None, description="Ownership information for the model")
    model_providers: list[str] | None = Field(
        default=None,
        description="List of ModelProvider workspace/name resource names that provide inference for this Model Entity",
    )
    trust_remote_code: bool | None = Field(
        default=None,
        description="Whether to trust remote code for the checkpoint.",
    )


class UpdateAdapterRequest(BaseModel):
    """Request model for updating Adapter Sub Entity metadata."""

    description: str | None = Field(
        default=None,
        description="Optional description of the adapter",
        max_length=1000,
    )
    enabled: bool | None = Field(
        default=None,
        description="Whether to make this adapter available for inference post training",
    )
    fileset: str | None = Field(
        default=None,
        description="Updated fileset for the adapter",
    )


class ModelEntitySortField(StrEnum):
    """Sort fields for Model Entity queries."""

    NAME_ASC = "name"
    NAME_DESC = "-name"
    CREATED_AT_ASC = "created_at"
    CREATED_AT_DESC = "-created_at"
    UPDATED_AT_ASC = "updated_at"
    UPDATED_AT_DESC = "-updated_at"


# ---------------------------------------------------------------------------
# ModelDeploymentConfig + ModelDeployment
# ---------------------------------------------------------------------------


class ModelType(str, Enum):
    """Model type enum for NIM deployments."""

    LLM = "llm"
    EMBED = "embed"
    OTHER = "other"


class K8sNIMOperatorConfig(BaseModel):
    """Kubernetes configuration for NIM deployment via k8s-nim-operator."""

    resources: dict[str, Any] | None = Field(
        default=None,
        description="Kubernetes resource requirements including requests and limits. "
        "Example: {'requests': {'cpu': '2', 'memory': '8Gi'}, 'limits': {'memory': '16Gi'}}",
    )
    tolerations: list[dict[str, Any]] | None = Field(
        default=None,
        description="Kubernetes tolerations for pod scheduling. "
        "Example: [{'key': 'nvidia.com/gpu', 'operator': 'Exists', 'effect': 'NoSchedule'}]",
    )
    node_selector: dict[str, str] | None = Field(
        default=None,
        description="Kubernetes node selector for pod placement. "
        "Example: {'node-type': 'gpu-node', 'zone': 'us-west1-a'}",
    )
    startup_probe_grace_seconds: int | None = Field(
        default=None,
        description="Grace period in seconds for NIM startup. "
        "Determines how long Kubernetes will wait for the NIM to become ready before restarting it. "
        "Example: 600 (10 minutes). "
        "Must be a positive integer.",
        gt=0,
    )


class Engine(str, Enum):
    """Inference engine selecting the compiler path for a deployment."""

    NIM = "nim"
    VLLM = "vllm"
    GENERIC = "generic"


class ModelDeploymentConfigModelSpec(BaseModel):
    """What model to serve and how -- independent of the executor it runs on."""

    model_type: ModelType | None = Field(default=None, description="Type of model being deployed")
    model_namespace: str | None = Field(
        default=None,
        description="Model repository namespace - organization/user namespace as it exists in repo_id.",
        max_length=_MAX_LEN_255,
    )
    model_name: str | None = Field(
        default=None,
        description="Model name - model repository name for model weights.",
        max_length=_MAX_LEN_255,
    )
    model_revision: str | None = Field(
        default=None,
        description="Model revision (branch, tag, or commit). If not specified, parsed from model_name @revision suffix or defaults to 'main'",
        max_length=_MAX_LEN_255,
    )
    chat_template: str | None = Field(
        default=None,
        description="Jinja2 chat template string for the model. Overrides the chat_template from ModelEntity.spec "
        "if both are set. Used by the engine to format chat completions.",
    )
    tool_call_config: ToolCallConfig | None = Field(
        default=None,
        description="Tool calling configuration for the deployment. Overrides tool_call_config from "
        "ModelEntity.spec if both are set. Controls how the model handles function/tool calling.",
    )
    lora_enabled: bool = Field(default=False, description="Whether to enable LoRA support")


class ContainerExecutorConfig(BaseModel):
    """Compute + container settings shared by the docker and k8s executors."""

    gpu: int = Field(description="Number of GPUs required for the deployment. 0 = CPU-only.", ge=0)
    disk_size: str = Field(default="50Gi", description="Disk size for the deployment")
    image_name: str | None = Field(
        default=None,
        description="Container image name. If not specified, defaults to the engine's configured image "
        "(e.g. default_vllm_image / default_nimservice_image). Required for engine='generic'.",
        max_length=_MAX_LEN_255,
    )
    image_tag: str | None = Field(
        default=None,
        description="Container image tag. If not specified, defaults to the engine's configured image tag.",
        max_length=_MAX_LEN_255,
    )
    health_check_path: str | None = Field(
        default=None,
        description="HTTP path used for the container readiness probe. If not specified, defaults to the "
        "engine's standard health endpoint (e.g. '/v1/health/ready' for NIM, '/health' for vLLM). "
        "Set this for engine='generic' containers that expose a non-standard health endpoint.",
        max_length=_MAX_LEN_255,
    )
    run_as_user: int | None = Field(
        default=None,
        ge=0,
        description="Pod securityContext runAsUser (uid) for the serving container (k8s backend only). "
        "If unset, the engine default applies (vLLM pins its image's user; generic uses the image's "
        "own user). Ignored by the docker backend.",
    )
    run_as_group: int | None = Field(
        default=None,
        ge=0,
        description="Pod securityContext runAsGroup (gid) for the serving container (k8s backend only). "
        "If unset, the engine default applies. Ignored by the docker backend.",
    )
    additional_envs: dict[str, str] | None = Field(
        default=None, description="Additional environment variables for the deployment"
    )
    additional_args: list[str] = Field(
        default_factory=list,
        description="Raw container/`serve` args appended verbatim to the container's arg vector.",
    )
    k8s_nim_operator_config: K8sNIMOperatorConfig | None = Field(
        default=None,
        description="Typed Kubernetes configuration for common NIMService Spec fields (NIM engine on k8s). "
        "Applied after defaults but before override_config. Ignored by non-NIM engines.",
    )
    override_config: dict[str, Any] | None = Field(
        default=None,
        description="Raw NIMService spec configuration that takes precedence over generated config (NIM engine "
        "on k8s). Allows advanced configuration options directly. Ignored by non-NIM engines.",
    )


class ModelDeploymentStatus(str, Enum):
    """Status enum for ModelDeployment objects."""

    UNKNOWN = "UNKNOWN"  # Terminal
    CREATED = "CREATED"
    PENDING = "PENDING"
    READY = "READY"
    ERROR = "ERROR"  # Terminal
    DELETING = "DELETING"
    DELETED = "DELETED"  # Terminal
    LOST = "LOST"  # Terminal


class ModelDeploymentStatusHistoryItem(BaseModel):
    """Record of a status change in ModelDeployment history."""

    timestamp: datetime = Field(description="When this status was recorded")
    status: ModelDeploymentStatus = Field(description="The status at this point in time")
    status_message: str = Field(default="", description="Status message", max_length=1000)


class ModelDeploymentConfig(ModelEntityBaseModel):
    """Immutable, automatically-versioned deployment config.

    The unique identifier is the combination of workspace/name/entity_version.
    """

    id: str = Field(default="", description="Unique identifier for the deployment config")
    entity_version: int = Field(description="Version of this deployment config. Automatically managed.")
    description: str | None = Field(
        default=None,
        description="Optional description of the deployment configuration",
        max_length=1000,
    )
    engine: Engine = Field(description="Inference engine selecting the compiler path (nim/vllm/generic)")
    model_spec: ModelDeploymentConfigModelSpec = Field(
        description="What model to serve and how -- independent of the executor it runs on"
    )
    executor_config: ContainerExecutorConfig = Field(
        description="Compute + container settings for the executor the deployment runs on"
    )
    model_entity_id: str | None = Field(
        default=None,
        description="Optional reference to the base model entity ID for this deployment",
        max_length=_MAX_LEN_255,
    )


class ModelDeployment(ModelEntityBaseModel):
    """A deployed instance of a model with a specific configuration.

    The unique identifier is the combination of workspace/name/entity_version.
    """

    id: str = Field(default="", description="Unique identifier for the deployment")
    entity_version: int = Field(description="Version of this deployment. Automatically managed.")
    config: str = Field(
        description="Reference to the ModelDeploymentConfig name",
        max_length=_MAX_LEN_255,
    )
    config_version: int = Field(description="Reference to the specific ModelDeploymentConfig version")
    status: ModelDeploymentStatus = Field(
        default=ModelDeploymentStatus.UNKNOWN,
        description="Current status of the deployment, populated by models controller",
    )
    status_message: str = Field(
        default="",
        description="Detailed status message, populated by models controller",
        max_length=1000,
    )
    status_history: list[ModelDeploymentStatusHistoryItem] = Field(
        default_factory=list,
        description="History of status changes, ordered chronologically (oldest first)",
    )
    model_provider_id: str | None = Field(
        default=None,
        description="Optional reference to the auto-created ModelProvider workspace/name (format: workspace/name)",
        max_length=_MAX_LEN_255,
    )
    auth_context: AuthContext | None = Field(default=None, description="Auth context captured at deployment creation. ")


class CreateModelDeploymentConfigRequest(BaseModel):
    """Request model for creating a ModelDeploymentConfig."""

    name: str = Field(
        description=f"Name of the deployment configuration. {_NAME_DESC}",
        max_length=_MAX_LEN_255,
        pattern=_NAME_REGEX,
        examples=["nim-config-v1", "production-config"],
    )
    project: str | None = Field(
        default=None,
        description="The URN of the project associated with this deployment configuration",
        max_length=_MAX_LEN_255,
        pattern=_NAME_SLASH_REGEX,
    )
    description: str | None = Field(
        default=None,
        description="Optional description of the deployment configuration",
        max_length=1000,
    )
    engine: Engine = Field(description="Inference engine selecting the compiler path (nim/vllm/generic)")
    model_spec: ModelDeploymentConfigModelSpec = Field(
        description="What model to serve and how -- independent of the executor it runs on"
    )
    executor_config: ContainerExecutorConfig = Field(
        description="Compute + container settings for the executor the deployment runs on"
    )
    model_entity_id: str | None = Field(
        default=None,
        description="Optional reference to the base model entity ID for this deployment",
        max_length=_MAX_LEN_255,
    )


class UpdateModelDeploymentConfigRequest(BaseModel):
    """Request model for updating a ModelDeploymentConfig (creates new version)."""

    description: str | None = Field(
        default=None,
        description="Optional description of the deployment configuration",
        max_length=1000,
    )
    engine: Engine = Field(description="Inference engine selecting the compiler path (nim/vllm/generic)")
    model_spec: ModelDeploymentConfigModelSpec = Field(
        description="What model to serve and how -- independent of the executor it runs on"
    )
    executor_config: ContainerExecutorConfig = Field(
        description="Compute + container settings for the executor the deployment runs on"
    )
    model_entity_id: str | None = Field(
        default=None,
        description="Optional reference to the base model entity ID for this deployment",
        max_length=_MAX_LEN_255,
    )


class CreateModelDeploymentRequest(BaseModel):
    """Request model for creating a ModelDeployment."""

    name: str = Field(
        description=f"Name of the deployment. {_NAME_DESC}",
        max_length=_MAX_LEN_255,
        pattern=_NAME_REGEX,
        examples=["llama-deploy-v1", "production-nim"],
    )
    project: str | None = Field(
        default=None,
        description="The URN of the project associated with this deployment",
        max_length=_MAX_LEN_255,
        pattern=_NAME_SLASH_REGEX,
    )
    config: str = Field(
        description="Reference to the ModelDeploymentConfig name",
        max_length=_MAX_LEN_255,
    )
    config_version: int | None = Field(
        default=None,
        description="Reference to a specific ModelDeploymentConfig version. If not specified, uses latest.",
    )


class UpdateModelDeploymentRequest(BaseModel):
    """Request model for updating a ModelDeployment (creates new version)."""

    config: str = Field(
        description="Reference to the ModelDeploymentConfig name",
        max_length=_MAX_LEN_255,
    )
    config_version: int | None = Field(
        default=None,
        description="Reference to a specific ModelDeploymentConfig version. If not specified, uses latest.",
    )


class UpdateModelDeploymentStatusRequest(BaseModel):
    """Request model for updating ModelDeployment status."""

    status: ModelDeploymentStatus = Field(description="New status for the deployment")
    status_message: str = Field(default="", description="Detailed status message", max_length=1000)
    model_provider_id: str | None = Field(
        default=None,
        description="Optional reference to the auto-created ModelProvider workspace/name (format: workspace/name)",
        max_length=_MAX_LEN_255,
    )


# ---------------------------------------------------------------------------
# Query parameter types
# ---------------------------------------------------------------------------


class ListModelsQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]
    sort: NotRequired[str]
    filter: NotRequired[str]
    verbose: NotRequired[bool]


class GetModelQueryParams(TypedDict, total=False):
    verbose: NotRequired[bool]


class ListAdaptersQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]
    sort: NotRequired[str]
    filter: NotRequired[str]


class ListProvidersQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]
    sort: NotRequired[str]
    filter: NotRequired[str]


class ListPromptsQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]
    sort: NotRequired[str]
    filter: NotRequired[str]


class ListDeploymentsQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]
    sort: NotRequired[str]
    all_versions: NotRequired[bool]
    filter: NotRequired[str]


class ListDeploymentConfigsQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]
    sort: NotRequired[str]
    filter: NotRequired[str]


class UpdateDeploymentStatusQueryParams(TypedDict, total=False):
    version: NotRequired[str]
