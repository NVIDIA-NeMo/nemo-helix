# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Domain entities for the Models service using EntityBase."""

from typing import Any, ClassVar, Optional, Self

from nhx.common.auth import AuthContext
from nhx.common.entities import constants
from nhx.common.entities.client import EntityBase
from nhx.common.inference import InferenceParams
from nhx.core.models.constants import (
    MODEL_REF_MAX_LEN,
    MODEL_REF_PATTERN_DESCRIPTION,
    is_valid_model_ref,
)
from nhx.core.models.schemas import (
    APIEndpointData,
    BackendFormat,
    ChatCompletionTool,
    ContainerExecutorConfig,
    Engine,
    FinetuningType,
    Lora,
    ModelDeploymentConfigModelSpec,
    ModelDeploymentStatus,
    ModelProviderStatus,
    ModelSpec,
    PromptData,
    PromptMessage,
    ServedModelMapping,
)
from pydantic import Field, PrivateAttr, computed_field, field_validator, model_validator


class Adapter(EntityBase):
    __entity_type__: ClassVar[str] = "adapter"

    @model_validator(mode="before")
    @classmethod
    def _migrate_auto_deploy(cls, data: Any) -> Any:
        """Translate legacy ``auto_deploy`` field to ``enabled`` for existing DB rows."""
        if isinstance(data, dict) and "auto_deploy" in data and "enabled" not in data:
            data["enabled"] = data.pop("auto_deploy")
        return data

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
    model: Optional[str] = Field(
        None,
        description=f"Parent model entity reference. {MODEL_REF_PATTERN_DESCRIPTION}",
        max_length=MODEL_REF_MAX_LEN,
    )

    @field_validator("model")
    def validate_model(cls, v: str) -> str:
        if v is not None and not is_valid_model_ref(v):
            raise ValueError(f"Invalid model reference: {v}")
        return v


class Model(EntityBase):
    """Model entity representing a versioned model registered within the platform.

    This entity stores model metadata including specifications, artifacts,
    and configuration for inference endpoints.
    """

    __entity_type__: ClassVar[str] = "model"

    project: str | None = Field(
        default=None,
        description="The URN of the project associated with this model entity.",
        max_length=constants.MAX_LENGTH_255,
    )
    description: str | None = Field(
        default=None,
        description="Optional description of the model.",
        max_length=1000,
    )
    spec: ModelSpec | None = Field(
        default=None,
        description="Detailed specification for the model.",
    )
    finetuning_type: FinetuningType | None = Field(
        default=None,
        description="Set for full weight finetuned models",
    )
    fileset: str | None = Field(
        default=None,
        description="A set of checkpoint files, configs, and other auxiliary info associated with this model - expected format {workspace}/{fileset_name}",
    )
    base_model: str | None = Field(
        default=None,
        description="Link to another model which is used as a base for the current model.",
    )
    api_endpoint: APIEndpointData | None = Field(
        default=None,
        description="Data about the inference endpoint for this model.",
    )
    backend_format: BackendFormat | None = Field(
        default=None,
        description="Inference API wire format expected by the backend. Defaults to OPENAI_CHAT at routing time.",
    )

    prompt: PromptData | None = Field(
        default=None,
        description="Configuration for prompt engineering.",
    )
    custom_fields: dict[str, Any] = Field(
        default_factory=dict,
        description="Custom fields for additional metadata.",
    )
    ownership: dict[str, Any] | None = Field(
        default=None,
        description="Ownership information for the model.",
    )
    model_providers: list[str] = Field(
        default_factory=list,
        description="List of ModelProvider workspace/name resource names that provide inference for this Model Entity.",
    )
    trust_remote_code: bool | None = Field(
        default=False,
        description="Whether to trust remote code for the checkpoint. Some models without support in certain libraries such as Transformers require additional custom Python code to execute. Due to security ramifications of running this code, only admins can enable this flag for a given Hugging Face organization.",
    )


class ModelProvider(EntityBase):
    """ModelProvider entity defining a reachable network endpoint for inference.

    Examples include OpenAI, NIMs, Bedrock, NVIDIA Build, etc.
    A ModelProvider may be provisioned automatically by Models Controller for
    ModelDeployments, or manually by an end user for external endpoints.
    """

    __entity_type__: ClassVar[str] = "model_provider"

    host_url: str = Field(
        description="The network endpoint URL for the model provider.",
        max_length=2048,
    )
    api_key_secret_name: str | None = Field(
        default=None,
        description="Reference to the API key stored in Secrets service.",
        max_length=constants.MAX_LENGTH_255,
    )
    served_models: list[ServedModelMapping] = Field(
        default_factory=list,
        description="List of models served by this provider with routing information for IGW.",
    )
    enabled_models: list[str] | None = Field(
        default=None,
        description="Optional list of specific models to enable from this provider.",
    )
    status: ModelProviderStatus = Field(
        default=ModelProviderStatus.UNKNOWN,
        description="Current status of the model provider, populated by models service.",
    )
    status_message: str | None = Field(
        default=None,
        description="Detailed status message, populated by models service.",
        max_length=1000,
    )
    model_deployment_id: str | None = Field(
        default=None,
        description="Optional reference to the ModelDeployment ID if this provider was auto-created for a deployment.",
        max_length=constants.MAX_LENGTH_255,
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
    project: str | None = Field(
        default=None,
        description="The URN of the project associated with this model provider.",
        max_length=constants.MAX_LENGTH_255,
    )
    description: str | None = Field(
        default=None,
        description="Optional description of the model provider.",
        max_length=1000,
    )
    auth_header_format: str | None = Field(
        default=None,
        description=(
            "Jinja2 template controlling how the API key is sent upstream. "
            "Must contain exactly one variable named `auth_secret`. "
            "Defaults to `'Authorization: Bearer {{ auth_secret }}'` when not set."
        ),
        max_length=1024,
    )

    # Auth context from the request that created this provider.
    # Used for delegated access when the controller accesses secrets.
    _auth_context: Optional[AuthContext] = PrivateAttr(default=None)

    @computed_field
    @property
    def auth_context(self) -> Optional[AuthContext]:
        return self._auth_context

    def with_auth_context(self, auth_context: AuthContext | None) -> Self:
        """Updates the provider (in-place) with auth context for delegated access."""
        self._auth_context = auth_context
        return self


class ModelDeployment(EntityBase):
    """ModelDeployment entity representing a deployed model instance.

    These entities are versioned - each update creates a new version.
    The entity name is formatted as '{base_name}-v{entity_version}' to ensure uniqueness.
    Status updates are mutable operations that don't create new versions.

    Uses hard deletes (no soft delete flag).
    """

    __entity_type__: ClassVar[str] = "model_deployment"

    # Base name (without version suffix) for grouping versions
    base_name: str = Field(
        description="The logical name of the deployment (without version suffix).",
        max_length=constants.MAX_LENGTH_255,
    )

    # Numeric version (1, 2, 3, ...)
    entity_version: int = Field(
        default=1,
        description="Version of this deployment. Automatically managed.",
        ge=1,
    )

    config: str = Field(
        description="Reference to the ModelDeploymentConfig base_name.",
        max_length=constants.MAX_LENGTH_255,
    )
    config_version: int = Field(
        description="Specific config version to use.",
        ge=1,
    )
    status: ModelDeploymentStatus = Field(
        default=ModelDeploymentStatus.UNKNOWN,
        description="Current status of the deployment, populated by models controller.",
    )
    status_message: str | None = Field(
        default=None,
        description="Detailed status message, populated by models controller.",
        max_length=1000,
    )
    status_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="History of status changes for this deployment.",
    )
    model_provider_id: str | None = Field(
        default=None,
        description="Optional reference to the auto-created ModelProvider workspace/name).",
        max_length=constants.MAX_LENGTH_255,
    )
    project: str | None = Field(
        default=None,
        description="The URN of the project associated with this deployment.",
        max_length=constants.MAX_LENGTH_255,
    )

    # Auth context from the request that created this deployment.
    # Used for delegated access when the controller accesses secrets/files.
    _auth_context: Optional[AuthContext] = PrivateAttr(default=None)

    @computed_field
    @property
    def auth_context(self) -> Optional[AuthContext]:
        return self._auth_context

    def with_auth_context(self, auth_context: AuthContext | None) -> Self:
        """Updates the deployment (in-place) with auth context for delegated access."""
        self._auth_context = auth_context
        return self


class ModelDeploymentConfig(EntityBase):
    """ModelDeploymentConfig entity storing configuration details for model deployment.

    These entities are versioned - each update creates a new version.
    The entity name is formatted as '{base_name}-v{entity_version}' to ensure uniqueness.

    Uses hard deletes (no soft delete flag).
    """

    __entity_type__: ClassVar[str] = "model_deployment_config"

    # Base name (without version suffix) for grouping versions
    base_name: str = Field(
        description="The logical name of the config (without version suffix).",
        max_length=constants.MAX_LENGTH_255,
    )

    # Numeric version (1, 2, 3, ...)
    entity_version: int = Field(
        default=1,
        description="Version of this deployment config. Automatically managed.",
        ge=1,
    )

    engine: Engine | None = Field(
        default=None,
        description="Inference engine selecting the compiler path (nim/vllm/generic).",
    )
    model_spec: ModelDeploymentConfigModelSpec | None = Field(
        default=None,
        description="What model to serve and how -- independent of the executor it runs on.",
    )
    executor_config: ContainerExecutorConfig | None = Field(
        default=None,
        description="Compute + container settings for the executor the deployment runs on.",
    )
    model_entity_id: str | None = Field(
        default=None,
        description="Optional reference to the base model entity ID for this deployment.",
        max_length=constants.MAX_LENGTH_255,
    )
    project: str | None = Field(
        default=None,
        description="The URN of the project associated with this deployment configuration.",
        max_length=constants.MAX_LENGTH_255,
    )
    description: str | None = Field(
        default=None,
        description="Optional description of the deployment configuration.",
        max_length=1000,
    )


class Prompt(EntityBase):
    """A reusable, stored chat prompt, addressed by workspace/name.

    Captures the messages, declared template variables, optional tool definitions,
    and default inference parameters needed to invoke a model through the
    Inference Gateway.
    """

    __entity_type__: ClassVar[str] = "prompt"

    project: str | None = Field(
        default=None,
        description="The URN of the project associated with this prompt.",
        max_length=constants.MAX_LENGTH_255,
    )
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
