# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from datetime import datetime
from enum import Enum, StrEnum
from typing import Annotated, Any, Dict, List, Literal, Optional, Self, Union

from jinja2 import Environment
from jinja2 import nodes as jinja_nodes
from nemo_platform_plugin.inference_middleware import BackendFormat
from nmp.common.auth import AuthContext
from nmp.common.entities import Filter, constants
from nmp.common.entities.utils import get_random_id
from nmp.common.entities.values import DatetimeFilter, StringFilter, map_entity_field
from nmp.common.inference import InferenceParams
from nmp.core.models.constants import (
    MODEL_REF_MAX_LEN,
    MODEL_REF_PATTERN_DESCRIPTION,
    is_valid_model_ref,
)
from pydantic import AnyUrl, BaseModel, ConfigDict, Field, field_validator, model_validator

# NAME_PATTERN uses lookaround; rust-regex cannot compile it.
ENTITY_NAME_CONFIG = ConfigDict(regex_engine="python-re")


def get_model_id(prefix: str) -> str:
    """Generate a unique model ID with the given prefix."""
    return get_random_id(prefix).lower()


# ============================================================================
# Value Types
# ============================================================================


class ModelPrecision(str, Enum):
    """Type of model precision.

    ## Values
    - `"int8"` - 8-bit integer precision
    - `"bf16"` - Brain floating point precision
    - `"fp16"` - 16-bit floating point precision
    - `"fp32"` - 32-bit floating point precision
    - `"fp8-mixed"` - Mixed 8-bit floating point precision available on Hopper and later architectures.
    - `"bf16-mixed"` - Mixed Brain floating point precision
    """

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
    expert_ffn_size: Optional[int] = Field(
        default=None, description="FFN size for experts (if different from main FFN)"
    )
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

    tool_call_parser: Optional[str] = Field(
        default=None,
        description="Name of the tool call parser to use (e.g., 'openai', 'hermes', 'pythonic', 'llama3_json', 'mistral').",
        max_length=constants.MAX_LENGTH_255,
    )
    tool_call_plugin: Optional[str] = Field(
        default=None,
        description="Reference to a fileset containing the custom tool call plugin Python file. "
        "Expected format: '{workspace}/{fileset_name}'. The fileset is mounted separately from "
        "the model checkpoint at deployment time.",
        max_length=constants.MAX_LENGTH_255,
    )
    auto_tool_choice: Optional[bool] = Field(
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

    context_size: Optional[int] = Field(None, description="Context window size")
    num_virtual_tokens: Optional[int] = Field(None, description="Number of virtual tokens for prompt tuning")
    is_chat: Optional[bool] = Field(None, description="Whether this is a chat model")
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
    moe_config: Optional[MoEConfig] = Field(default=None, description="MoE configuration if applicable")
    mamba_config: Optional[MambaConfig] = Field(default=None, description="Mamba/SSM configuration if applicable")
    sliding_window_config: Optional[SlidingWindowConfig] = Field(
        default=None, description="Sliding window attention config if applicable"
    )

    # LoRA-specific metadata (pre-computed to avoid model instantiation)
    linear_layers: Optional[list[LinearLayerSpec]] = Field(
        default=None,
        description="List of all linear/Conv1D layers with their dimensions. "
        "Used for LoRA parameter estimation without requiring model instantiation. "
        "Each entry contains the module name, in_features, and out_features.",
    )

    # Deployment configuration
    chat_template: Optional[str] = Field(
        default=None,
        description="Jinja2 chat template string for the model. Used by NIM to format chat completions. "
        "If not set, the model's built-in tokenizer template is used.",
    )
    tool_call_config: Optional[ToolCallConfig] = Field(
        default=None,
        description="Tool calling configuration for NIM deployments. Controls how the model handles "
        "function/tool calling in chat completions.",
    )

    # GPU requirements (auto-calculated)
    minimum_gpus_all_weights: Optional[int] = Field(
        default=None,
        description="Minimum GPUs required for full fine-tuning using default configurations.",
    )
    minimum_gpus_lora: Optional[int] = Field(
        default=None,
        description="Minimum GPUs required for LoRA fine-tuning using default configurations.",
    )

    def model_precision(self) -> ModelPrecision:
        """
        Convert the precision string to ModelPrecision enum.

        Returns:
            ModelPrecision enum value corresponding to the stored precision string.

        Raises:
            ValueError: If the precision string cannot be mapped to a ModelPrecision enum value.
        """
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
    alpha: Optional[int] = Field(None, description="Alpha scaling used for this adapter")
    rank: int = Field(..., description="LoRA Rank")


class APIEndpointData(BaseModel):
    """Data about an inference endpoint."""

    url: Optional[AnyUrl] = Field(None, description="Endpoint URL")
    model_id: Optional[str] = Field(None, description="Model identifier at the endpoint")
    api_key: Optional[str] = Field(None, description="API key for authentication")
    format: Optional[str] = Field(None, description="API format (e.g., openai, nvidia)")


class PromptData(BaseModel):
    """Configuration for prompt engineering."""

    system_prompt: Optional[str] = Field(None, description="System prompt template")
    icl_few_shot_examples: Optional[str] = Field(None, description="In-context learning examples")

    inference_params: Optional[InferenceParams] = Field(
        default=None, description="Inference parameters that should be overridden."
    )

    system_prompt_template: Optional[str] = Field(
        default=None,
        title="System Prompt Template",
        description="The template which will be used to compile the final prompt used for prompting the LLM. Currently supports only {{icl_few_shot_examples}}",
    )


# ============================================================================
# Base Models
# ============================================================================


class ModelEntityBaseModel(BaseModel, ABC):
    """Base model for all Models service domain objects."""

    model_config = ENTITY_NAME_CONFIG

    id: str = Field(..., description="Autogenerated id")
    name: str = Field(
        description=f"Name of the entity. Name/workspace combo must be unique across all entities. {constants.NAME_PATTERN_DESCRIPTION}",
        max_length=constants.NAME_MAX_LENGTH,
        pattern=constants.NAME_PATTERN,
        examples=["llama-3.1-8b", "my-custom-model"],
    )
    workspace: str = Field(
        description=f"The workspace of the entity. {constants.REGEX_WORD_CHARACTER_DOT_DASH_DESCRIPTION}",
        max_length=constants.MAX_LENGTH_255,
        pattern=constants.REGEX_WORD_CHARACTER_DOT_DASH,
    )
    project: Optional[str] = Field(
        default=None,
        description="The URN of the project associated with this entity.",
        max_length=constants.MAX_LENGTH_255,
        pattern=constants.REGEX_WORD_CHARACTER_DOT_DASH_SLASH,
    )
    created_at: datetime = Field(..., description="The timestamp of model entity creation")
    updated_at: datetime = Field(..., description="The timestamp of the last model entity update")


# ============================================================================
# ModelProvider Schemas
# ============================================================================

_AUTH_HEADER_FORMAT_DESCRIPTION = (
    "Jinja2 template string controlling how the API key secret is sent to the upstream. "
    "Must contain exactly one variable named `auth_secret`, which is substituted with the "
    "resolved secret value at request time. "
    "Example: `'X-Api-Key: {{ auth_secret }}'`. "
    "If not set, defaults to `'Authorization: Bearer {{ auth_secret }}'`."
)


def _validate_auth_header_format(v: str | None) -> str | None:
    """Validate that an auth_header_format value is a well-formed Jinja2 template
    containing exactly one ``{{ auth_secret }}`` substitution and that it renders
    to a ``<Header-Name>: <Header-Value>`` shape the IGW proxy can split on
    ``": "``.

    Two checks beyond Jinja2 syntax:

    1. **Exactly one** ``auth_secret`` substitution. A duplicate placeholder
       would leak the resolved secret into the rendered header twice, which is
       never intended and a security concern.
    2. **Renderable to ``<Header-Name>: <Header-Value>``.** A dry-run render
       with a sentinel value must produce a string that ``str.partition(": ")``
       splits into two non-empty parts. Otherwise the runtime proxy silently
       sends an empty header value to the upstream.
    """
    if v is None:
        return v
    # Validates an auth-header template (rendered to HTTP, not HTML).
    # Autoescape would corrupt secrets containing `&`, `<`, `>`, or quotes.
    env = Environment(autoescape=False)  # noqa: S701  # nosec B701
    try:
        ast = env.parse(v)
    except Exception as exc:
        raise ValueError(f"Invalid Jinja2 template: {exc}") from exc
    variable_names = [node.name for node in ast.find_all(jinja_nodes.Name)]
    if set(variable_names) != {"auth_secret"} or len(variable_names) != 1:
        found = ", ".join(sorted(set(variable_names))) if variable_names else "none"
        raise ValueError(
            f"auth_header_format must contain exactly one Jinja2 variable named 'auth_secret' (found: {found})"
        )
    rendered = env.from_string(v).render(auth_secret="__AUTH_SECRET__")
    header_name, sep, header_value = rendered.partition(": ")
    if not sep or not header_name.strip() or not header_value.strip():
        raise ValueError("auth_header_format must render to '<Header-Name>: <Header-Value>'")
    return v


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
        max_length=constants.MAX_LENGTH_255,
    )
    served_model_name: str = Field(
        description="The actual model name to send to the backend endpoint in the 'model' field",
        max_length=constants.MAX_LENGTH_255,
    )


class ModelProvider(ModelEntityBaseModel):
    """
    A ModelProvider defines a reachable network endpoint that provides an inference
    service for one or more Model Entities. Examples of Model Providers include
    OpenAI, NIMs, Bedrock, NVIDIA Build, etc. A ModelProvider may be provisioned
    automatically by Models Controller for ModelDeployments, or it may be provisioned
    manually by an end user for an endpoint that does not have its lifecycle managed
    by models service (like an external provider.)

    The unique identifier for a ModelProvider is the combination of workspace/name.
    """

    id: str = Field(
        default_factory=lambda: get_model_id("modelprovider"), description="Unique identifier for the model provider"
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional description of the model provider",
        max_length=1000,
    )
    host_url: str = Field(
        description="The network endpoint URL for the model provider",
        max_length=2048,
    )
    api_key_secret_name: Optional[str] = Field(
        default=None,
        description="Reference to the API key stored in Secrets service",
        max_length=constants.MAX_LENGTH_255,
    )
    served_models: Optional[List[ServedModelMapping]] = Field(
        default_factory=list,
        description="List of models served by this provider with routing information for IGW",
    )
    enabled_models: Optional[List[str]] = Field(
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
    default_extra_body: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Default body parameters for inference requests. Can be overridden by user requests.",
    )
    default_extra_headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="Default headers for inference requests. Can be overridden by user requests.",
    )
    required_extra_body: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Required body parameters for inference requests. Cannot be overridden by user requests.",
    )
    required_extra_headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="Required headers for inference requests. Cannot be overridden by user requests.",
    )
    model_deployment_id: Optional[str] = Field(
        default=None,
        description="Optional reference to the ModelDeployment ID if this provider was auto-created for a deployment",
        max_length=constants.MAX_LENGTH_255,
    )
    auth_context: Optional[AuthContext] = Field(default=None, description="Auth context captured at provider creation.")
    auth_header_format: Optional[str] = Field(
        default=None,
        description=_AUTH_HEADER_FORMAT_DESCRIPTION,
        max_length=1024,
    )

    @field_validator("auth_header_format")
    @classmethod
    def validate_auth_header_format(cls, v: str | None) -> str | None:
        return _validate_auth_header_format(v)


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

    model_config = ENTITY_NAME_CONFIG

    name: str = Field(
        description=f"Name of the model provider. {constants.NAME_PATTERN_DESCRIPTION}",
        max_length=constants.NAME_MAX_LENGTH,
        pattern=constants.NAME_PATTERN,
        examples=["my-nim-provider", "openai-endpoint"],
    )
    project: Optional[str] = Field(
        default=None,
        description="The URN of the project associated with this model provider",
        max_length=constants.MAX_LENGTH_255,
        pattern=constants.REGEX_WORD_CHARACTER_DOT_DASH_SLASH,
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional description of the model provider",
        max_length=1000,
    )
    host_url: str = Field(
        description="The network endpoint URL for the model provider",
        max_length=2048,
    )
    api_key_secret_name: Optional[str] = Field(
        default=None,
        description="Reference to an API key secret stored in the Secrets service. "
        "Create the secret first via secrets API, then pass the secret name here.",
        max_length=constants.MAX_LENGTH_255,
    )
    enabled_models: Optional[List[str]] = Field(
        default=None, description="Optional list of specific models to enable from this provider"
    )
    default_extra_body: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Default body parameters for inference requests. Can be overridden by user requests.",
    )
    default_extra_headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="Default headers for inference requests. Can be overridden by user requests.",
    )
    required_extra_body: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Required body parameters for inference requests. Cannot be overridden by user requests.",
    )
    required_extra_headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="Required headers for inference requests. Cannot be overridden by user requests.",
    )
    model_deployment_id: Optional[str] = Field(
        default=None,
        description="Optional reference to the ModelDeployment ID if this provider is being auto-created for a deployment",
        max_length=constants.MAX_LENGTH_255,
    )
    status: Optional[ModelProviderStatus] = Field(default=None, description="Status of the model provider")
    status_message: Optional[str] = Field(
        default=None,
        description="Status message",
        max_length=1000,
    )
    auth_header_format: Optional[str] = Field(
        default=None,
        description=_AUTH_HEADER_FORMAT_DESCRIPTION,
        max_length=1024,
    )

    @field_validator("auth_header_format")
    @classmethod
    def validate_auth_header_format(cls, v: str | None) -> str | None:
        return _validate_auth_header_format(v)


class UpsertModelProviderRequest(BaseModel):
    """Request model for upserting a ModelProvider (PUT /apis/models/v2/workspaces/{workspace}/providers/{name}).

    All fields must be provided - partial updates are not supported for security reasons.
    Use PUT /status endpoint to update status-related fields only.
    """

    project: Optional[str] = Field(
        default=None,
        description="The URN of the project associated with this model provider",
        max_length=constants.MAX_LENGTH_255,
        pattern=constants.REGEX_WORD_CHARACTER_DOT_DASH_SLASH,
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional description of the model provider",
        max_length=1000,
    )
    host_url: str = Field(
        description="The network endpoint URL for the model provider",
        max_length=2048,
    )
    api_key_secret_name: Optional[str] = Field(
        default=None,
        description="Reference to an API key secret stored in the Secrets service. "
        "Create the secret first via secrets API, then pass the secret name here.",
        max_length=constants.MAX_LENGTH_255,
    )
    enabled_models: Optional[List[str]] = Field(
        default=None, description="Optional list of specific models to enable from this provider"
    )
    default_extra_body: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Default body parameters for inference requests. Can be overridden by user requests.",
    )
    default_extra_headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="Default headers for inference requests. Can be overridden by user requests.",
    )
    required_extra_body: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Required body parameters for inference requests. Cannot be overridden by user requests.",
    )
    required_extra_headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="Required headers for inference requests. Cannot be overridden by user requests.",
    )
    model_deployment_id: Optional[str] = Field(
        default=None,
        description="Optional reference to the ModelDeployment ID if this provider is associated with a deployment",
        max_length=constants.MAX_LENGTH_255,
    )
    status: Optional[ModelProviderStatus] = Field(default=None, description="Status of the model provider")
    status_message: Optional[str] = Field(
        default=None,
        description="Status message",
        max_length=1000,
    )
    auth_header_format: Optional[str] = Field(
        default=None,
        description=_AUTH_HEADER_FORMAT_DESCRIPTION,
        max_length=1024,
    )

    @field_validator("auth_header_format")
    @classmethod
    def validate_auth_header_format(cls, v: str | None) -> str | None:
        return _validate_auth_header_format(v)


class UpdateModelProviderStatusRequest(BaseModel):
    """Request model for updating ModelProvider status and autodiscovery fields.

    This endpoint supports partial updates for fields managed by Models Controller.
    """

    model_deployment_id: Optional[str] = Field(
        default=None,
        description="Reference to the ModelDeployment ID if this provider is associated with a deployment",
        max_length=constants.MAX_LENGTH_255,
    )
    served_models: Optional[List[ServedModelMapping]] = Field(
        default=None, description="List of models served by this provider with routing information for IGW"
    )
    status: Optional[ModelProviderStatus] = Field(default=None, description="Status of the model provider")
    status_message: Optional[str] = Field(
        default=None,
        description="Status message. If status is provided without status_message, defaults to empty string.",
        max_length=1000,
    )


class GetModelProviderRequest(BaseModel):
    """Request model for getting a ModelProvider."""

    workspace: str = Field(
        description=f"The workspace of the model provider. {constants.REGEX_WORD_CHARACTER_DOT_DASH_DESCRIPTION}",
        max_length=constants.MAX_LENGTH_255,
        pattern=constants.REGEX_WORD_CHARACTER_DOT_DASH,
    )
    name: str = Field(
        description=f"Name of the model provider. {constants.REGEX_WORD_CHARACTER_DOT_DASH_DESCRIPTION}",
        max_length=constants.MAX_LENGTH_255,
        pattern=constants.REGEX_WORD_CHARACTER_DOT_DASH,
    )


class ListModelProvidersRequest(BaseModel):
    """Request model for listing ModelProviders.

    Supports both:
    - GET /apis/models/v2/workspaces/{workspace}/providers (lists by workspace)
    - GET /apis/models/v2/workspaces/{workspace}/providers (lists all when workspace is omitted)
    """

    workspace: Optional[str] = Field(
        default=None,
        description=f"Optional workspace to filter model providers. If not provided, lists all workspaces. {constants.REGEX_WORD_CHARACTER_DOT_DASH_DESCRIPTION}",
        max_length=constants.MAX_LENGTH_255,
        pattern=constants.REGEX_WORD_CHARACTER_DOT_DASH,
    )
    project: Optional[str] = Field(
        default=None,
        description="Optional project URN to filter model providers",
        max_length=constants.MAX_LENGTH_255,
        pattern=constants.REGEX_WORD_CHARACTER_DOT_DASH_SLASH,
    )
    status: Optional[ModelProviderStatus] = Field(
        default=None, description="Optional status to filter model providers (e.g., to list healthy ModelProviders)"
    )
    models: Optional[List[str]] = Field(
        default=None,
        description="Optional list of model names to discover ModelProviders based on the models they're advertising",
    )


class DeleteModelProviderRequest(BaseModel):
    """Request model for deleting a ModelProvider."""

    workspace: str = Field(
        description=f"The workspace of the model provider. {constants.REGEX_WORD_CHARACTER_DOT_DASH_DESCRIPTION}",
        max_length=constants.MAX_LENGTH_255,
        pattern=constants.REGEX_WORD_CHARACTER_DOT_DASH,
    )
    name: str = Field(
        description=f"Name of the model provider. {constants.REGEX_WORD_CHARACTER_DOT_DASH_DESCRIPTION}",
        max_length=constants.MAX_LENGTH_255,
        pattern=constants.REGEX_WORD_CHARACTER_DOT_DASH,
    )


# ============================================================================
# Prompt Schemas
# ============================================================================


class PromptMessageRole(StrEnum):
    """Role of a message author in a chat prompt.

    Follows the OpenAI chat schema the Inference Gateway speaks
    (``/v1/chat/completions``).
    """

    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"


class PromptMessage(BaseModel):
    """A single templated message in a chat prompt.

    ``content`` is a Jinja2 template body that may reference the prompt's
    declared ``input_variables`` (e.g. ``{{ topic }}``).
    """

    role: PromptMessageRole = Field(description="The role of the message author.")
    content: str = Field(description="Templated message content. May contain template variables.")


class FunctionDefinition(BaseModel):
    """An OpenAI-compatible function definition for tool calling.

    Mirrors the ``function`` object the Inference Gateway forwards to
    OpenAI-compatible backends.
    """

    name: str = Field(
        description="The name of the function to be called.",
        max_length=constants.MAX_LENGTH_255,
    )
    description: Optional[str] = Field(
        default=None,
        description="A description of what the function does, used by the model to decide when and how to call it.",
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="The parameters the function accepts, described as a JSON Schema object.",
    )
    strict: Optional[bool] = Field(
        default=None,
        description="Whether to enforce strict schema adherence when generating the function call.",
    )


class ChatCompletionTool(BaseModel):
    """An OpenAI-compatible tool definition (currently always a function tool)."""

    type: Literal["function"] = Field(
        description="The type of the tool. Currently only 'function' is supported.",
    )
    function: FunctionDefinition = Field(description="The function definition for this tool.")


class Prompt(ModelEntityBaseModel):
    """A reusable, stored chat prompt.

    A Prompt captures the messages, declared template variables, optional tool
    definitions, and default inference parameters needed to invoke a model
    through the Inference Gateway. The unique identifier is workspace/name.
    """

    id: str = Field(
        default_factory=lambda: get_model_id("prompt"),
        description="Unique identifier for the prompt.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional description of the prompt.",
        max_length=1000,
    )
    messages: List[PromptMessage] = Field(
        default_factory=list,
        description="Ordered list of chat messages that make up the prompt.",
    )
    input_variables: List[str] = Field(
        default_factory=list,
        description="Names of the Jinja2 template variables the prompt expects.",
    )
    tools: Optional[List[ChatCompletionTool]] = Field(
        default=None,
        description="Optional OpenAI-compatible tool definitions to send with the prompt.",
    )
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(
        default=None,
        description="Controls which (if any) tool is called: 'none', 'auto', 'required', or a named-tool object.",
    )
    response_format: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional OpenAI-compatible response_format, e.g. a json_schema structured-output spec.",
    )
    inference_params: Optional[InferenceParams] = Field(
        default=None,
        description="Optional default model and sampling parameters (temperature, top_p, max_tokens, ...).",
    )
    tags: List[str] = Field(
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

    model_config = ENTITY_NAME_CONFIG

    name: str = Field(
        description=f"Name of the prompt. {constants.NAME_PATTERN_DESCRIPTION}",
        max_length=constants.NAME_MAX_LENGTH,
        pattern=constants.NAME_PATTERN,
        examples=["support-bot-system", "summarizer"],
    )
    project: Optional[str] = Field(
        default=None,
        description="The URN of the project associated with this prompt.",
        max_length=constants.MAX_LENGTH_255,
        pattern=constants.REGEX_WORD_CHARACTER_DOT_DASH_SLASH,
    )
    description: Optional[str] = Field(default=None, max_length=1000)
    messages: List[PromptMessage] = Field(default_factory=list)
    input_variables: List[str] = Field(default_factory=list)
    tools: Optional[List[ChatCompletionTool]] = Field(default=None)
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(default=None)
    response_format: Optional[Dict[str, Any]] = Field(default=None)
    inference_params: Optional[InferenceParams] = Field(default=None)
    tags: Optional[List[str]] = Field(default=None)


class UpdatePromptRequest(BaseModel):
    """Request model for replacing a Prompt's mutable fields (full update).

    The prompt name and workspace come from the URL path and cannot be changed.
    """

    project: Optional[str] = Field(
        default=None,
        description="The URN of the project associated with this prompt.",
        max_length=constants.MAX_LENGTH_255,
        pattern=constants.REGEX_WORD_CHARACTER_DOT_DASH_SLASH,
    )
    description: Optional[str] = Field(default=None, max_length=1000)
    messages: List[PromptMessage] = Field(default_factory=list)
    input_variables: List[str] = Field(default_factory=list)
    tools: Optional[List[ChatCompletionTool]] = Field(default=None)
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(default=None)
    response_format: Optional[Dict[str, Any]] = Field(default=None)
    inference_params: Optional[InferenceParams] = Field(default=None)
    tags: Optional[List[str]] = Field(default=None)


class GetPromptRequest(BaseModel):
    """Request model for getting a Prompt."""

    workspace: str = Field(
        description=f"The workspace of the prompt. {constants.REGEX_WORD_CHARACTER_DOT_DASH_DESCRIPTION}",
        max_length=constants.MAX_LENGTH_255,
        pattern=constants.REGEX_WORD_CHARACTER_DOT_DASH,
    )
    name: str = Field(
        description=f"Name of the prompt. {constants.REGEX_WORD_CHARACTER_DOT_DASH_DESCRIPTION}",
        max_length=constants.MAX_LENGTH_255,
        pattern=constants.REGEX_WORD_CHARACTER_DOT_DASH,
    )


class DeletePromptRequest(BaseModel):
    """Request model for deleting a Prompt."""

    workspace: str = Field(
        description=f"The workspace of the prompt. {constants.REGEX_WORD_CHARACTER_DOT_DASH_DESCRIPTION}",
        max_length=constants.MAX_LENGTH_255,
        pattern=constants.REGEX_WORD_CHARACTER_DOT_DASH,
    )
    name: str = Field(
        description=f"Name of the prompt. {constants.REGEX_WORD_CHARACTER_DOT_DASH_DESCRIPTION}",
        max_length=constants.MAX_LENGTH_255,
        pattern=constants.REGEX_WORD_CHARACTER_DOT_DASH,
    )


# ============================================================================
# Model Entity Schemas
# ============================================================================


class Adapter(BaseModel):
    name: str = Field(
        ...,
        description=f"Name of the adapter. Name must be unique in the workspace for all Adapters and match the following regex: {constants.REGEX_WORD_CHARACTER_DOT_DASH_DESCRIPTION}",
        max_length=constants.MAX_LENGTH_255,
        pattern=constants.REGEX_WORD_CHARACTER_DOT_DASH,
        examples=["lora-adapter-v1", "my-finetune"],
    )

    workspace: str = Field(
        ...,
        description=f"Workspace of the adapter. {constants.REGEX_WORD_CHARACTER_DOT_DASH_DESCRIPTION}",
        max_length=constants.MAX_LENGTH_255,
        pattern=constants.REGEX_WORD_CHARACTER_DOT_DASH,
    )

    description: Optional[str] = Field(
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
    lora_config: Optional[Lora] = Field(None, description="Lora configuration specifics")

    model: Optional[str] = Field(
        default=None,
        description=f"Parent model entity reference. {MODEL_REF_PATTERN_DESCRIPTION}",
        max_length=MODEL_REF_MAX_LEN,
    )

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @field_validator("model")
    def validate_model(cls, v: str | None) -> str | None:
        if v is not None and not is_valid_model_ref(v):
            raise ValueError(MODEL_REF_PATTERN_DESCRIPTION)
        return v


class ModelEntity(ModelEntityBaseModel):
    """
    Model Entity represents a versioned model registered within the platform.
    Uses EntityBase for entity store compatibility.
    """

    project: Optional[str] = Field(
        default=None,
        description="The URN of the project associated with this model entity.",
        max_length=constants.MAX_LENGTH_255,
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional description of the model.",
        max_length=1000,
    )
    spec: Optional[ModelSpec] = Field(default=None, description="Detailed specification for the model")

    finetuning_type: Optional[FinetuningType] = Field(None, description="Set for full weight finetuned models")

    # TODO Replace this with Optional[Union[str, Fileset]] when fileset is accessible outside of the
    # so that a user can inline the fileset definition
    fileset: Optional[str] = Field(
        default=None,
        description="A set of checkpoint files, configs, and other auxiliary info associated with this model - expected format {workspace}/{fileset_name}",
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Whether to trust remote code to load this model checkpoint.",
    )
    base_model: Optional[str] = Field(
        default=None, description="Link to another model which is used as a base for the current model"
    )
    api_endpoint: Optional[APIEndpointData] = Field(
        default=None, description="Data about the inference endpoint for this model"
    )
    backend_format: Optional[BackendFormat] = Field(
        default=None,
        description=(
            "Inference API wire format expected by the backend. If unset, inference routing treats the model as "
            "OPENAI_CHAT."
        ),
        json_schema_extra={"nullable": True},
    )
    adapters: Optional[list[Adapter]] = Field(
        default=None,
        description="Adapters that have been created against this model",
    )
    prompt: Optional[PromptData] = Field(default=None, description="Configuration for prompt engineering")
    custom_fields: Dict[str, Any] = Field(default_factory=dict, description="Custom fields for additional metadata")
    ownership: Optional[Dict[str, Any]] = Field(default=None, description="Ownership information for the model")
    model_providers: List[str] = Field(
        default_factory=list,
        description="List of ModelProvider workspace/name resource names that provide inference for this Model Entity",
    )


class CreateModelEntityRequest(BaseModel):
    """Request model for creating a Model Entity."""

    model_config = ENTITY_NAME_CONFIG

    name: str = Field(
        description=f"Name of the model entity. {constants.NAME_PATTERN_DESCRIPTION}",
        max_length=constants.NAME_MAX_LENGTH,
        pattern=constants.NAME_PATTERN,
        examples=["llama-3.1-8b", "my-custom-model"],
    )
    project: Optional[str] = Field(
        default=None,
        description="The URN of the project associated with this model entity",
        max_length=constants.MAX_LENGTH_255,
        pattern=constants.REGEX_WORD_CHARACTER_DOT_DASH_SLASH,
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional description of the model",
        max_length=1000,
    )
    spec: Optional[ModelSpec] = Field(
        default=None,
        description="Detailed specification for the model - Automatically generated by the platform at creation when fileset provided.",
    )
    finetuning_type: Optional[FinetuningType] = Field(None, description="Set for full weight finetuned models")
    # TODO Replace this with Optional[Union[str, Fileset]] when fileset is accessible outside of the
    # so that a user can inline the fileset definition
    fileset: Optional[str] = Field(
        default=None,
        description="A set of checkpoint files, configs, and other auxiliary info associated with this model - expected format {workspace}/{fileset_name}",
    )

    base_model: Optional[str] = Field(
        default=None, description="Link to another model which is used as a base for the current model"
    )
    api_endpoint: Optional[APIEndpointData] = Field(
        default=None, description="Data about the inference endpoint for this model"
    )
    backend_format: Optional[BackendFormat] = Field(
        default=None,
        description=(
            "Inference API wire format expected by the backend. If unset, inference routing treats the model as "
            "OPENAI_CHAT."
        ),
        json_schema_extra={"nullable": True},
    )
    prompt: Optional[PromptData] = Field(default=None, description="Configuration for prompt engineering")
    custom_fields: Optional[Dict[str, Any]] = Field(default=None, description="Custom fields for additional metadata")
    ownership: Optional[Dict[str, Any]] = Field(default=None, description="Ownership information for the model")
    model_providers: Optional[List[str]] = Field(
        default_factory=list,
        description="List of ModelProvider workspace/name resource names that provide inference for this Model Entity",
    )
    trust_remote_code: bool = Field(
        default=False,
        description="""Whether to trust remote code for the checkpoint.
        Some models without support in certain libraries such as Transformers require additional custom Python code to execute.
        Due to security ramifications of running arbitrary code, this can only be set to true on one of the following conditions:
        (1) the model's fileset's source is pre-approved in the platform config, or
        (2) the user creating this model is an administrator.
        """,
    )


class CreateModelAdapterRequest(BaseModel):
    """Request body for nested Adapter creation. The base model comes from the URL path, not the body."""

    model_config = ENTITY_NAME_CONFIG

    name: str = Field(
        ...,
        description=f"Name of the adapter. Name must be unique in the workspace. {constants.NAME_PATTERN_DESCRIPTION}",
        max_length=constants.NAME_MAX_LENGTH,
        pattern=constants.NAME_PATTERN,
        examples=["lora-adapter-v1", "my-finetune"],
    )

    description: Optional[str] = Field(
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
    lora_config: Optional[Lora] = Field(None, description="Lora configuration specifics")


class CreateAdapterRequest(CreateModelAdapterRequest):
    """Request body for Adapter creation."""

    model: str = Field(
        ...,
        max_length=MODEL_REF_MAX_LEN,
        description=(
            f"""Base model entity.
            Use `{{workspace}}/{{model_name}}` to reference a model in any workspace, or a single `{{model_name}}` resolved in the path workspace. {MODEL_REF_PATTERN_DESCRIPTION}"""
        ),
        examples=["llama-3-8b-instruct", "shared-tenant/base-llm"],
    )

    @field_validator("model")
    def validate_model(cls, v: str | None) -> str | None:
        if v is not None and not is_valid_model_ref(v):
            raise ValueError(MODEL_REF_PATTERN_DESCRIPTION)
        return v


class UpdateModelEntityRequest(BaseModel):
    """Request model for updating Model Entity metadata."""

    description: Optional[str] = Field(
        default=None,
        description="Optional description of the model",
        max_length=1000,
    )
    spec: Optional[ModelSpec] = Field(default=None, description="Detailed specification for the model")
    # TODO Replace this with Optional[Union[str, Fileset]] when fileset is accessible outside of the
    # so that a user can inline the fileset definition
    fileset: Optional[str] = Field(
        default=None,
        description="A set of checkpoint files, configs, and other auxiliary info associated with this model - expected format {workspace}/{fileset_name}",
    )
    finetuning_type: Optional[FinetuningType] = Field(None, description="Set for full weight finetuned models")
    base_model: Optional[str] = Field(
        default=None, description="Link to another model which is used as a base for the current model"
    )
    api_endpoint: Optional[APIEndpointData] = Field(
        default=None, description="Data about the inference endpoint for this model"
    )
    backend_format: Optional[BackendFormat] = Field(
        default=None,
        description=(
            "Inference API wire format expected by the backend. If unset, inference routing treats the model as "
            "OPENAI_CHAT."
        ),
        json_schema_extra={"nullable": True},
    )
    prompt: Optional[PromptData] = Field(default=None, description="Configuration for prompt engineering")
    custom_fields: Optional[Dict[str, Any]] = Field(default=None, description="Custom fields for additional metadata")
    ownership: Optional[Dict[str, Any]] = Field(default=None, description="Ownership information for the model")
    model_providers: Optional[List[str]] = Field(
        default=None,
        description="List of ModelProvider workspace/name resource names that provide inference for this Model Entity",
    )
    trust_remote_code: Optional[bool] = Field(
        default=None,
        description="""Whether to trust remote code for the checkpoint.
        Some models without support in certain libraries such as Transformers require additional custom Python code to execute.
        Due to security ramifications of running arbitrary code, this can only be set to true on one of the following conditions:
        (1) the model's fileset's source is pre-approved in the platform config, or
        (2) the user creating this model is an administrator.
        """,
    )


class UpdateAdapterRequest(BaseModel):
    """Request model for updating Adapter Sub Entity metadata."""

    description: Optional[str] = Field(
        default=None,
        description="Optional description of the adapter",
        max_length=1000,
    )

    enabled: Optional[bool] = Field(
        default=None,
        description="Whether to make this adapter available for inference post training",
    )

    fileset: Optional[str] = Field(
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


class GetModelEntityRequest(BaseModel):
    """Request model for getting a Model Entity."""

    workspace: str = Field(
        description=f"The workspace of the model entity. {constants.REGEX_WORD_CHARACTER_DOT_DASH_DESCRIPTION}",
        max_length=constants.MAX_LENGTH_255,
        pattern=constants.REGEX_WORD_CHARACTER_DOT_DASH,
    )
    name: str = Field(
        description=f"Name of the model entity. {constants.REGEX_WORD_CHARACTER_DOT_DASH_DESCRIPTION}",
        max_length=constants.MAX_LENGTH_255,
        pattern=constants.REGEX_WORD_CHARACTER_DOT_DASH,
    )


# Filter classes for Model Entity queries
class BaseModelFilter(Filter):
    """Filter for base model properties."""

    name: StringFilter | str | None = Field(None, description="Filter by name of the base model.")


class FinetuningTypeFilter(Filter):
    """Filter for PEFT-related properties."""

    finetuning_type: Optional[FinetuningType] = Field(
        None, description="Filter models with adapters with this fine-tuning type."
    )


class ModelEntityFilter(Filter):
    """Filter for Model Entity queries."""

    name: StringFilter | str | None = Field(None, description="Filter by name.")
    project: Optional[str] = Field(None, description="Filter by project name.")
    workspace: Optional[str] = Field(None, description="Filter by workspace id.")
    base_model: Optional[Union[BaseModelFilter, bool, str]] = Field(
        default=None,
        description="Filter by base model: true = has a base model, false = no base model, "
        "{ name: string } or string = match base model name.",
    )
    adapters: Annotated[Optional[Union[FinetuningTypeFilter, bool]], map_entity_field("adapters")] = Field(
        default=None,
        description="Filter models with Parameter Efficient Fine-tuning Adapters.",
    )
    finetuning_type: Optional[Union[FinetuningType, bool]] = Field(
        None, description="Filter models that have been perviously finetuned."
    )
    prompt: Optional[bool] = Field(default=None, description="Filter models with prompt engineering data.")
    lora_enabled: Optional[bool] = Field(
        default=None,
        description="Filter models by whether their deployment config has LoRA enabled.",
    )
    description: StringFilter | str | None = Field(None, description="Filter by description.")
    fileset: Optional[Union[bool, str]] = Field(
        None,
        description="Filter by fileset: true = has a fileset, false = no fileset, "
        "string = match fileset reference in the form {workspace}/{fileset_name}.",
    )
    created_at: Optional[DatetimeFilter] = Field(None, description="Filter entities based on creation date.")
    updated_at: Optional[DatetimeFilter] = Field(None, description="Filter entities based on update date.")


class AdapterEntityFilter(Filter):
    """Filter for Adapter list queries."""

    name: StringFilter | str | None = Field(None, description="Filter by adapter name.")
    model: Annotated[StringFilter | str | None, map_entity_field("data.model")] = Field(
        default=None,
        description="Filter by parent (base) model entity reference in the form {workspace}/{model_name}.",
    )
    description: StringFilter | str | None = Field(None, description="Filter by description.")
    fileset: Optional[str] = Field(
        None,
        description="Filter by fileset reference in the form {workspace}/{fileset_name}.",
    )
    finetuning_type: Optional[FinetuningType] = Field(None, description="Filter by fine-tuning / PEFT type.")
    enabled: Optional[bool] = Field(
        None, description="Filter by whether the adapter is enabled for inference after training."
    )
    created_at: Optional[DatetimeFilter] = Field(None, description="Filter entities based on creation date.")
    updated_at: Optional[DatetimeFilter] = Field(None, description="Filter entities based on update date.")


# ============================================================================
# ModelDeploymentConfig and ModelDeployment Schemas
# ============================================================================


class ModelType(str, Enum):
    """Model type enum for NIM deployments."""

    LLM = "llm"
    EMBED = "embed"
    OTHER = "other"


class K8sNIMOperatorConfig(BaseModel):
    """Kubernetes configuration for NIM deployment via k8s-nim-operator.

    These fields provide typed access to commonly-used NIMService Spec fields
    and are applied before override_config in the compilation precedence.
    """

    resources: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Kubernetes resource requirements including requests and limits. "
        "Example: {'requests': {'cpu': '2', 'memory': '8Gi'}, 'limits': {'memory': '16Gi'}}",
    )
    tolerations: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Kubernetes tolerations for pod scheduling. "
        "Example: [{'key': 'nvidia.com/gpu', 'operator': 'Exists', 'effect': 'NoSchedule'}]",
    )
    node_selector: Optional[Dict[str, str]] = Field(
        default=None,
        description="Kubernetes node selector for pod placement. "
        "Example: {'node-type': 'gpu-node', 'zone': 'us-west1-a'}",
    )
    startup_probe_grace_seconds: Optional[int] = Field(
        default=None,
        description="Grace period in seconds for NIM startup. "
        "Determines how long Kubernetes will wait for the NIM to become ready before restarting it. "
        "Example: 600 (10 minutes). "
        "Must be a positive integer.",
        gt=0,
    )


class Engine(str, Enum):
    """Inference engine selecting the compiler path for a deployment.

    The engine determines what command, image, and env a deployment compiles to.
    The fields a compiler consumes are not engine-specific; engines take the same
    inputs (model_spec + executor_config) and differ in what they do with them.
    """

    NIM = "nim"
    VLLM = "vllm"
    # Plain container: run image + args + env, no inference-engine compiler.
    GENERIC = "generic"


class ModelDeploymentConfigModelSpec(BaseModel):
    """What model to serve and how -- independent of the executor it runs on.

    Executor-invariant facts about the model. The compiler resolves the weight
    source per engine; serving fields override the model entity spec when set.
    """

    model_type: Optional[ModelType] = Field(default=None, description="Type of model being deployed")

    # Model source configuration (the compiler resolves these per engine)
    model_namespace: Optional[str] = Field(
        default=None,
        description="Model repository namespace - organization/user namespace as it exists in repo_id.",
        max_length=constants.MAX_LENGTH_255,
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Model name - model repository name for model weights.",
        max_length=constants.MAX_LENGTH_255,
    )
    model_revision: Optional[str] = Field(
        default=None,
        description="Model revision (branch, tag, or commit). If not specified, parsed from model_name @revision suffix or defaults to 'main'",
        max_length=constants.MAX_LENGTH_255,
    )

    # Serving configuration (overrides model entity spec if set)
    chat_template: Optional[str] = Field(
        default=None,
        description="Jinja2 chat template string for the model. Overrides the chat_template from ModelEntity.spec "
        "if both are set. Used by the engine to format chat completions.",
    )
    tool_call_config: Optional[ToolCallConfig] = Field(
        default=None,
        description="Tool calling configuration for the deployment. Overrides tool_call_config from "
        "ModelEntity.spec if both are set. Controls how the model handles function/tool calling.",
    )

    # LoRA -- drives the adapter sidecar wiring (see LoRA Hot-Reload)
    lora_enabled: bool = Field(default=False, description="Whether to enable LoRA support")


class ContainerExecutorConfig(BaseModel):
    """Compute + container settings shared by the docker and k8s executors.

    Both the docker and k8s executors run containers and share this shape.
    A future non-container executor (e.g. subprocess) would warrant turning
    ``executor_config`` into a discriminated union.
    """

    gpu: int = Field(description="Number of GPUs required for the deployment. 0 = CPU-only.", ge=0)
    disk_size: str = Field(default="50Gi", description="Disk size for the deployment")

    # Image -- None falls back to the engine's configured default
    # (e.g. default_vllm_image / default_nimservice_image). Required for
    # engine="generic" (no platform default exists).
    image_name: Optional[str] = Field(
        default=None,
        description="Container image name. If not specified, defaults to the engine's configured image "
        "(e.g. default_vllm_image / default_nimservice_image). Required for engine='generic'.",
        max_length=constants.MAX_LENGTH_255,
    )
    image_tag: Optional[str] = Field(
        default=None,
        description="Container image tag. If not specified, defaults to the engine's configured image tag.",
        max_length=constants.MAX_LENGTH_255,
    )

    # Readiness probe -- None falls back to the engine's default health path
    # (NIM: /v1/health/ready, vLLM: /health). Required for engine="generic"
    # (no engine default exists for an arbitrary container).
    health_check_path: Optional[str] = Field(
        default=None,
        description="HTTP path used for the container readiness probe. If not specified, defaults to the "
        "engine's standard health endpoint (e.g. '/v1/health/ready' for NIM, '/health' for vLLM). "
        "Set this for engine='generic' containers that expose a non-standard health endpoint.",
        max_length=constants.MAX_LENGTH_255,
    )

    # Pod securityContext user override (k8s backend only). When unset, the engine
    # picks an appropriate default (vLLM pins its image's user; generic runs as the
    # image's own user). Set these to run an arbitrary container as a specific
    # uid/gid -- e.g. a generic image that requires a particular user. Ignored by
    # the docker backend, which does not set a container user.
    run_as_user: Optional[int] = Field(
        default=None,
        ge=0,
        description="Pod securityContext runAsUser (uid) for the serving container (k8s backend only). "
        "If unset, the engine default applies (vLLM pins its image's user; generic uses the image's "
        "own user). Ignored by the docker backend.",
    )
    run_as_group: Optional[int] = Field(
        default=None,
        ge=0,
        description="Pod securityContext runAsGroup (gid) for the serving container (k8s backend only). "
        "If unset, the engine default applies. Ignored by the docker backend.",
    )

    # Escape hatches -- for anything not surfaced as a first-class field above
    additional_envs: Optional[Dict[str, str]] = Field(
        default=None, description="Additional environment variables for the deployment"
    )
    additional_args: list[str] = Field(
        default_factory=list,
        description="Raw container/`serve` args appended verbatim to the container's arg vector.",
    )

    # Kubernetes configuration for k8s-nim-operator (NIM engine on k8s only).
    # Carried here so the k8s NIM operator backend keeps working; ignored by the
    # docker and vLLM paths.
    k8s_nim_operator_config: Optional[K8sNIMOperatorConfig] = Field(
        default=None,
        description="Typed Kubernetes configuration for common NIMService Spec fields (NIM engine on k8s). "
        "Applied after defaults but before override_config. Ignored by non-NIM engines.",
    )

    # Raw per-container override map (NIM engine on k8s only).
    override_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Partial NIMService Spec fragments deep-merged after generated defaults and "
        "k8s_nim_operator_config (NIM engine on k8s only). Supported keys: image, command, args, "
        "resources, env, readinessProbe, livenessProbe, startupProbe, nodeSelector, tolerations, "
        "userID, groupID, labels, initContainers, sidecarContainers. Unsupported keys are rejected "
        "at compile time. Ignored by non-NIM engines and docker runtime. One key is a platform "
        "directive rather than a Spec fragment and is honored on both runtimes: nimLegacy (default "
        "true) emits the legacy NIM_MODEL_NAME / NIM_MODEL_PATH weight env vars; set it false for "
        "images that retired those names and expect NIM_ENGINE_MODEL_NAME / NIM_ENGINE_MODEL_PATH, "
        "since such images fail config validation when a retired name is set.",
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
    """
    ModelDeploymentConfig stores the configuration details for deploying a model.
    These objects are immutable with automatic versioning.

    The unique identifier is the combination of workspace/name/entity_version.
    """

    id: str = Field(
        default_factory=lambda: get_model_id("deploymentconfig"),
        description="Unique identifier for the deployment config",
    )
    entity_version: int = Field(
        description="Version of this deployment config. Automatically managed.",
    )
    description: Optional[str] = Field(
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
    model_entity_id: Optional[str] = Field(
        default=None,
        description="Optional reference to the base model entity ID for this deployment",
        max_length=constants.MAX_LENGTH_255,
    )


class ModelDeployment(ModelEntityBaseModel):
    """
    ModelDeployment represents a deployed instance of a model with a specific configuration.
    These objects are immutable with automatic versioning, except for status updates.

    The unique identifier is the combination of workspace/name/entity_version.
    """

    id: str = Field(
        default_factory=lambda: get_model_id("deployment"), description="Unique identifier for the deployment"
    )
    entity_version: int = Field(
        description="Version of this deployment. Automatically managed.",
    )
    config: str = Field(
        description="Reference to the ModelDeploymentConfig name",
        max_length=constants.MAX_LENGTH_255,
    )
    config_version: int = Field(
        description="Reference to the specific ModelDeploymentConfig version",
    )
    status: ModelDeploymentStatus = Field(
        default=ModelDeploymentStatus.UNKNOWN,
        description="Current status of the deployment, populated by models controller",
    )
    status_message: str = Field(
        default="",
        description="Detailed status message, populated by models controller",
        max_length=1000,
    )
    status_history: List[ModelDeploymentStatusHistoryItem] = Field(
        default_factory=list,
        description="History of status changes, ordered chronologically (oldest first)",
    )
    model_provider_id: Optional[str] = Field(
        default=None,
        description="Optional reference to the auto-created ModelProvider workspace/name (format: workspace/name)",
        max_length=constants.MAX_LENGTH_255,
    )
    auth_context: Optional[AuthContext] = Field(
        default=None, description="Auth context captured at deployment creation. "
    )


# ============================================================================
# Request/Response Models for ModelDeploymentConfig
# ============================================================================


class CreateModelDeploymentConfigRequest(BaseModel):
    """Request model for creating a ModelDeploymentConfig."""

    model_config = ENTITY_NAME_CONFIG

    name: str = Field(
        description=f"Name of the deployment configuration. {constants.NAME_PATTERN_DESCRIPTION}",
        max_length=constants.NAME_MAX_LENGTH,
        pattern=constants.NAME_PATTERN,
        examples=["nim-config-v1", "production-config"],
    )
    project: Optional[str] = Field(
        default=None,
        description="The URN of the project associated with this deployment configuration",
        max_length=constants.MAX_LENGTH_255,
        pattern=constants.REGEX_WORD_CHARACTER_DOT_DASH_SLASH,
    )
    description: Optional[str] = Field(
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
    model_entity_id: Optional[str] = Field(
        default=None,
        description="Optional reference to the base model entity ID for this deployment",
        max_length=constants.MAX_LENGTH_255,
    )


class UpdateModelDeploymentConfigRequest(BaseModel):
    """Request model for updating a ModelDeploymentConfig (creates new version)."""

    description: Optional[str] = Field(
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
    model_entity_id: Optional[str] = Field(
        default=None,
        description="Optional reference to the base model entity ID for this deployment",
        max_length=constants.MAX_LENGTH_255,
    )


class ListModelDeploymentConfigsRequest(BaseModel):
    """Request model for listing ModelDeploymentConfigs."""

    workspace: Optional[str] = Field(
        default=None,
        description=f"Optional workspace to filter deployment configs. {constants.REGEX_WORD_CHARACTER_DOT_DASH_DESCRIPTION}",
        max_length=constants.MAX_LENGTH_255,
        pattern=constants.REGEX_WORD_CHARACTER_DOT_DASH,
    )
    project: Optional[str] = Field(
        default=None,
        description="Optional project URN to filter deployment configs",
        max_length=constants.MAX_LENGTH_255,
        pattern=constants.REGEX_WORD_CHARACTER_DOT_DASH_SLASH,
    )


# ============================================================================
# Request/Response Models for ModelDeployment
# ============================================================================


class CreateModelDeploymentRequest(BaseModel):
    """Request model for creating a ModelDeployment."""

    model_config = ENTITY_NAME_CONFIG

    name: str = Field(
        description=f"Name of the deployment. {constants.NAME_PATTERN_DESCRIPTION}",
        max_length=constants.NAME_MAX_LENGTH,
        pattern=constants.NAME_PATTERN,
        examples=["llama-deploy-v1", "production-nim"],
    )
    project: Optional[str] = Field(
        default=None,
        description="The URN of the project associated with this deployment",
        max_length=constants.MAX_LENGTH_255,
        pattern=constants.REGEX_WORD_CHARACTER_DOT_DASH_SLASH,
    )
    config: str = Field(
        description="Reference to the ModelDeploymentConfig name",
        max_length=constants.MAX_LENGTH_255,
    )
    config_version: Optional[int] = Field(
        default=None,
        description="Reference to a specific ModelDeploymentConfig version. If not specified, uses latest.",
    )


class UpdateModelDeploymentRequest(BaseModel):
    """Request model for updating a ModelDeployment (creates new version)."""

    config: str = Field(
        description="Reference to the ModelDeploymentConfig name",
        max_length=constants.MAX_LENGTH_255,
    )
    config_version: Optional[int] = Field(
        default=None,
        description="Reference to a specific ModelDeploymentConfig version. If not specified, uses latest.",
    )


class UpdateModelDeploymentStatusRequest(BaseModel):
    """Request model for updating ModelDeployment status."""

    status: ModelDeploymentStatus = Field(description="New status for the deployment")
    status_message: str = Field(default="", description="Detailed status message", max_length=1000)
    model_provider_id: Optional[str] = Field(
        default=None,
        description="Optional reference to the auto-created ModelProvider workspace/name (format: workspace/name)",
        max_length=constants.MAX_LENGTH_255,
    )


class ListModelDeploymentsRequest(BaseModel):
    """Request model for listing ModelDeployments."""

    workspace: Optional[str] = Field(
        default=None,
        description=f"Optional workspace to filter deployments. {constants.REGEX_WORD_CHARACTER_DOT_DASH_DESCRIPTION}",
        max_length=constants.MAX_LENGTH_255,
        pattern=constants.REGEX_WORD_CHARACTER_DOT_DASH,
    )
    project: Optional[str] = Field(
        default=None,
        description="Optional project URN to filter deployments",
        max_length=constants.MAX_LENGTH_255,
        pattern=constants.REGEX_WORD_CHARACTER_DOT_DASH_SLASH,
    )
    status: Optional[ModelDeploymentStatus] = Field(default=None, description="Optional status to filter deployments")
    all_versions: bool = Field(
        default=False,
        description="If False (default), return only the latest version of each deployment. "
        "If True, return all versions matching the filters.",
    )


# ============================================================================
# Filter and Search classes for Pagination Support
# ============================================================================


class PromptFilter(Filter):
    """Filter for Prompt queries."""

    workspace: Optional[str] = Field(None, description="Filter by workspace.")
    project: Optional[str] = Field(None, description="Filter by project URN.")
    name: Optional[str] = Field(None, description="Filter by name.")
    description: Optional[str] = Field(None, description="Filter by description.")
    created_at: Optional[DatetimeFilter] = Field(None, description="Filter by creation date.")
    updated_at: Optional[DatetimeFilter] = Field(None, description="Filter by update date.")


class ModelProviderFilter(Filter):
    """Filter for ModelProvider queries."""

    workspace: Optional[str] = Field(None, description="Filter by workspace.")
    project: Optional[str] = Field(None, description="Filter by project URN.")
    status: Optional[ModelProviderStatus] = Field(None, description="Filter by status.")
    model_deployment_id: Optional[str] = Field(None, description="Filter by associated deployment ID.")
    name: StringFilter | str | None = Field(None, description="Filter by name.")
    description: StringFilter | str | None = Field(None, description="Filter by description.")
    host_url: StringFilter | str | None = Field(None, description="Filter by host URL.")
    created_at: Optional[DatetimeFilter] = Field(None, description="Filter by creation date.")
    updated_at: Optional[DatetimeFilter] = Field(None, description="Filter by update date.")


class ModelDeploymentConfigFilter(Filter):
    """Filter for ModelDeploymentConfig queries."""

    workspace: Optional[str] = Field(None, description="Filter by workspace.")
    project: Optional[str] = Field(None, description="Filter by project URN.")
    model_entity_id: Optional[str] = Field(None, description="Filter by associated model entity ID.")
    name: Annotated[StringFilter | str | None, map_entity_field("data.base_name")] = Field(
        None, description="Filter by config name."
    )
    description: StringFilter | str | None = Field(None, description="Filter by description.")
    created_at: Optional[DatetimeFilter] = Field(None, description="Filter by creation date.")
    updated_at: Optional[DatetimeFilter] = Field(None, description="Filter by update date.")


class ModelDeploymentFilter(Filter):
    """Filter for ModelDeployment queries."""

    workspace: Optional[str] = Field(None, description="Filter by workspace.")
    project: Optional[str] = Field(None, description="Filter by project URN.")
    status: Optional[ModelDeploymentStatus] = Field(None, description="Filter by status.")
    config: StringFilter | str | None = Field(None, description="Filter by config name.")
    model_provider_id: Optional[str] = Field(None, description="Filter by model provider ID.")
    name: Annotated[StringFilter | str | None, map_entity_field("data.base_name")] = Field(
        None, description="Filter by deployment name."
    )
    status_message: StringFilter | str | None = Field(None, description="Filter by status message.")
    created_at: Optional[DatetimeFilter] = Field(None, description="Filter by creation date.")
    updated_at: Optional[DatetimeFilter] = Field(None, description="Filter by update date.")
