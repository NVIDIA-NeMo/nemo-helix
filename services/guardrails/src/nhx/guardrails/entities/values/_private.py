# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private value objects for the Guardrails service.

These are internal configuration and response types used by the guardrails engine.
"""

import logging
import os
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import yaml
from nemoguardrails.colang import parse_colang_file, parse_flow_elements
from nemoguardrails.exceptions import InvalidRailsConfigurationError
from nemoguardrails.rails.llm.config import (
    JAILBREAK_FLOW_HEURISTICS,
    JAILBREAK_FLOW_MODEL,
    ContentSafetyConfig,
    ContextBloatDetectionConfig,
    CrowdStrikeAIDRRailConfig,
    GLiNERDetection,
    HFClassifierConfig,
    PolygrafDetection,
    RegexDetection,
    _join_config,
    _load_imported_paths,
    _load_path,
    _parse_colang_files_recursively,
    _unique_list_concat,
)

# Import helper functions from nemoguardrails for config loading
from nemoguardrails.rails.llm.config import (
    _default_config as _nemo_default_config,
)
from nhx.common.entities.utils import get_random_id
from nhx.common.entities.values import Value
from pydantic import ConfigDict, Field, SecretStr, model_validator

log = logging.getLogger(__name__)

_default_config = {
    "sample_conversation": 'user "Hello there!"\n  express greeting\nbot express greeting\n  "Hello! How can I assist you today?"\nuser "What can you do for me?"\n  ask about capabilities\nbot respond about capabilities\n  "As an AI assistant, I can help you with a wide range of tasks. This includes question answering on various topics, generating text for various purposes and providing suggestions based on your preferences."\nuser "Tell me a bit about the history of NVIDIA."\n  ask general question\nbot response for general question\n  "NVIDIA is a technology company that specializes in designing and manufacturing graphics processing units (GPUs) and other computer hardware. The company was founded in 1993 by Jen-Hsun Huang, Chris Malachowsky, and Curtis Priem."\nuser "tell me more"\n  request more information\nbot provide more information\n  "Initially, the company focused on developing 3D graphics processing technology for the PC gaming market. In 1999, NVIDIA released the GeForce 256, the world\'s first GPU, which was a major breakthrough for the gaming industry. The company continued to innovate in the GPU space, releasing new products and expanding into other markets such as professional graphics, mobile devices, and artificial intelligence."\nuser "thanks"\n  express appreciation\nbot express appreciation and offer additional help\n  "You\'re welcome. If you have any more questions or if there\'s anything else I can help you with, please don\'t hesitate to ask."\n',
    "instructions": [
        {
            "type": "general",
            "content": "Below is a conversation between a helpful AI assistant and a user. The bot is designed to generate human-like text based on the input that it receives. The bot is talkative and provides lots of specific details. If the bot does not know the answer to a question, it truthfully says it does not know.",
        }
    ],
    "prompting_mode": "standard",
}


def get_random_id_wrapper() -> str:
    return get_random_id(prefix="")


class LogAdapterConfig(Value):
    name: str = Field(default="FileSystem", description="The name of the adapter.")
    model_config = ConfigDict(extra="allow")


class SpanFormat(str, Enum):
    legacy = "legacy"
    opentelemetry = "opentelemetry"


class TracingConfig(Value):
    enabled: bool = False
    adapters: List[LogAdapterConfig] = Field(
        default_factory=lambda: [LogAdapterConfig()],
        description="The list of tracing adapters to use. If not specified, the default adapters are used.",
    )
    span_format: str = Field(
        default=SpanFormat.opentelemetry,
        description="The span format to use. Options are 'legacy' (simple metrics) or 'opentelemetry' (OpenTelemetry semantic conventions).",
    )
    enable_content_capture: bool = Field(
        default=False,
        description=(
            "Capture prompts and responses (user/assistant/tool message content) in tracing/telemetry events. "
            "Disabled by default for privacy and alignment with OpenTelemetry GenAI semantic conventions. "
            "WARNING: Enabling this may include PII and sensitive data in your telemetry backend."
        ),
    )


class CacheStatsConfig(Value):
    """Configuration for cache statistics tracking and logging."""

    enabled: bool = Field(
        default=False,
        description="Whether cache statistics tracking is enabled",
    )
    log_interval: Optional[float] = Field(
        default=None,
        description="Seconds between periodic cache stats logging to logs (None disables logging)",
    )


class ModelCacheConfig(Value):
    """Configuration for model caching."""

    enabled: bool = Field(
        default=False,
        description="Whether caching is enabled (default: False - no caching)",
    )
    maxsize: int = Field(default=50000, description="Maximum number of entries in the cache per model")
    stats: CacheStatsConfig = Field(
        default_factory=CacheStatsConfig,
        description="Configuration for cache statistics tracking and logging",
    )


class ModelParameters(Value):
    """Parameters for configuring how to interact with a model in a guardrails config."""

    # Allow additional fields to maintain compatibility with nemoguardrails, which allows
    # arbitrary fields in the model's `parameters` field.
    model_config = ConfigDict(extra="allow")

    base_url: Optional[str] = Field(default=None, description="The URL to use for inference with this model.")
    default_headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="Custom HTTP headers to include in requests to this model. Each key-value pair represents a header name (key) and its default value (value). You can override the default value for a header by populating it in the request headers.",
    )

    def __getitem__(self, key: str) -> Any:
        """Enable dict-style access: params['key']."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Enable dict-style assignment: params['key'] = value."""
        setattr(self, key, value)

    def __eq__(self, other: object) -> bool:
        """Compare against plain dicts using the public parameter values."""
        if isinstance(other, dict):
            return self.model_dump(exclude_none=True) == other
        return super().__eq__(other)

    def get(self, key: str, default: Any = None) -> Any:
        """Enable dict-style lookup with a default value."""
        return self.model_dump(exclude_none=True).get(key, default)

    def keys(self):
        """Support `**params` expansion without inheriting from Mapping."""
        return self.model_dump(exclude_none=True).keys()

    def items(self):
        """Enable dict-style iteration over parameter key/value pairs."""
        return self.model_dump(exclude_none=True).items()


# Model types that the user should not be able to configure in a guardrails config.
# Each key-value pair maps the model type to a user-facing validation error message.
# - `generate_user_intent`: The internal nemoguardrails task that invokes the main model
# for inference. NOTE: The task name is a bit misleading. From the Guardrails service's perspective,
# this action always takes the generation path (i.e. runs inference with the main model).
# If this model type is configured in a guardrail config, nemoguardrails bypasses our internal
# ChatNIM class in favor of ChatNVIDIA, which is not intended to be used with the main model.
_UNSUPPORTED_MODEL_TASKS: dict[str, str] = {
    "generate_user_intent": (
        "The 'generate_user_intent' task invokes the main model for inference, so it "
        "cannot be separately configured. The main model is always used for this task."
    ),
}


class Model(Value):
    """Configuration of a model used by the rails engine.

    If using Inference Gateway, the `model` field should be a Model Entity reference ('workspace/model_name').
    """

    model_config = {
        "arbitrary_types_allowed": True,
        "protected_namespaces": (),
        # This is needed to ensure we don't generate separate
        # -Input/-Output schemas for the child objects.
        "json_schema_mode_override": "validation",
    }

    type: str
    engine: str
    model: Optional[str] = Field(
        default=None,
        description=(
            "The model name. If using Inference Gateway, this should be the Model Entity reference ('workspace/model_name')."
        ),
    )
    parameters: Optional["ModelParameters"] = Field(
        default_factory=ModelParameters,
        description="Additional parameters to configure how to interact with the model.",
    )

    mode: Literal["chat", "text"] = Field(
        default="chat",
        description="Whether the mode is 'text' completion or 'chat' completion. Allowed values are 'chat' or 'text'.",
    )
    # Cache configuration specific to this model (for content safety models)
    cache: Optional["ModelCacheConfig"] = Field(
        default=None,
        description="Cache configuration for this specific model (primarily used for content safety models)",
    )

    @property
    def api_key_env_var(self) -> None:
        """This property exists for compatibility with the `nemoguardrails` library. The `@property` decorator
        ensures this field doesn't appear in the OpenAPI spec.

        Previously, `api_key_env_var` was used to reference an environment variable to use as the model's authentication credentials.
        This feature is no longer supported. Instead, users should use Inference Gateway to manage credentials via secrets.
        """
        return None

    @model_validator(mode="before")
    @classmethod
    def set_and_validate_model(cls, data: Any) -> Any:
        if isinstance(data, dict):
            parameters = data.get("parameters")
            if parameters is None:
                return data
            model_field = data.get("model")
            model_from_params = parameters.get("model_name") or parameters.get("model")

            if model_field and model_from_params:
                raise ValueError(
                    "Model name must be specified in exactly one place: either in the 'model' field or in parameters, not both."
                )
            if not model_field and model_from_params:
                data["model"] = model_from_params
                if "model_name" in parameters and parameters["model_name"] == model_from_params:
                    parameters.pop("model_name")
                elif "model" in parameters and parameters["model"] == model_from_params:
                    parameters.pop("model")
            return data

    @model_validator(mode="after")
    def model_must_be_none_empty(self) -> "Model":
        """Validate that a model name is present either directly or in parameters.

        Note: For "main" type models, the model name is optional because it gets
        populated at runtime using the incoming request's model field.
        """
        if self.type == "main":
            return self
        if not self.model or not self.model.strip():
            raise ValueError(
                "Model name must be specified either directly in the 'model' field or through 'model_name'/'model' in parameters"
            )
        return self

    @model_validator(mode="after")
    def validate_model_type_not_reserved(self) -> "Model":
        """Reject model types that correspond to nemoguardrails internal actions that should
        not be manually configured.
        """
        if self.type in _UNSUPPORTED_MODEL_TASKS:
            detail = _UNSUPPORTED_MODEL_TASKS[self.type]
            raise InvalidRailsConfigurationError(f"Model for task '{self.type}' cannot be manually set. {detail}")
        return self


class FiddlerGuardrails(Value):
    """Configuration for Fiddler Guardrails."""

    fiddler_endpoint: str = Field(
        default="http://localhost:8080/process/text",
        description="The global endpoint for Fiddler Guardrails requests.",
    )
    safety_threshold: float = Field(
        default=0.1,
        description="Fiddler Guardrails safety detection threshold.",
    )
    faithfulness_threshold: float = Field(
        default=0.05,
        description="Fiddler Guardrails faithfulness detection threshold.",
    )


class LLMCallSummary(Value):
    task: Optional[str] = Field(default=None, description="The internal task that made the call.")
    duration: Optional[float] = Field(default=None, description="The duration in seconds.")
    total_tokens: Optional[int] = Field(default=None, description="The total number of used tokens.")
    prompt_tokens: Optional[int] = Field(default=None, description="The number of input tokens.")
    completion_tokens: Optional[int] = Field(default=None, description="The number of output tokens.")
    started_at: Optional[float] = Field(default=0, description="The timestamp for when the LLM call started.")
    finished_at: Optional[float] = Field(default=0, description="The timestamp for when the LLM call finished.")


class LLMCallInfo(LLMCallSummary):
    id: Optional[str] = Field(default=None, description="The unique prompt identifier.")
    prompt: Optional[str] = Field(default=None, description="The prompt that was used for the LLM call.")
    completion: Optional[str] = Field(default=None, description="The completion generated by the LLM.")
    raw_response: Optional[dict] = Field(
        default=None,
        description="The raw response received from the LLM. May contain additional information, e.g. logprobs.",
    )
    llm_model_name: Optional[str] = Field(
        default="unknown",
        description="The name of the model use for the LLM call.",
    )


class ExecutedAction(Value):
    """Information about an action that was executed."""

    action_name: str = Field(description="The name of the action that was executed.")
    action_params: Dict[str, Any] = Field(default_factory=dict, description="The parameters for the action.")
    return_value: Any = Field(default=None, description="The value returned by the action.")
    llm_calls: List[LLMCallInfo] = Field(
        default_factory=list,
        description="Information about the LLM calls made by the action.",
    )
    started_at: Optional[float] = Field(default=None, description="Timestamp for when the action started.")
    finished_at: Optional[float] = Field(default=None, description="Timestamp for when the action finished.")
    duration: Optional[float] = Field(default=None, description="How long the action took to execute, in seconds.")


class ActivatedRail(Value):
    """A rail that was activated during the generation."""

    type: str = Field(description="The type of the rail that was activated, e.g., input, output, dialog.")
    name: str = Field(description="The name of the rail, i.e., the name of the flow implementing the rail.")
    decisions: List[str] = Field(
        default_factory=list,
        description="A sequence of decisions made by the rail, e.g., 'bot refuse to respond', 'stop', 'continue'.",
    )
    executed_actions: List[ExecutedAction] = Field(
        default_factory=list, description="The list of actions executed by the rail."
    )
    stop: bool = Field(
        default=False,
        description="Whether the rail decided to stop any further processing.",
    )
    additional_info: Optional[dict] = Field(default=None, description="Additional information coming from rail.")
    started_at: Optional[float] = Field(default=None, description="Timestamp for when the rail started.")
    finished_at: Optional[float] = Field(default=None, description="Timestamp for when the rail finished.")
    duration: Optional[float] = Field(
        default=None,
        description="The duration in seconds for applying the rail. "
        "Some rails are applied instantly, e.g., dialog rails, so they don't have a duration.",
    )


class GenerationStats(Value):
    """General stats about the generation."""

    input_rails_duration: Optional[float] = Field(
        default=None,
        description="The time in seconds spent in processing the input rails.",
    )
    dialog_rails_duration: Optional[float] = Field(
        default=None,
        description="The time in seconds spent in processing the dialog rails.",
    )
    generation_rails_duration: Optional[float] = Field(
        default=None,
        description="The time in seconds spent in generation rails.",
    )
    output_rails_duration: Optional[float] = Field(
        default=None,
        description="The time in seconds spent in processing the output rails.",
    )
    total_duration: Optional[float] = Field(default=None, description="The total time in seconds.")
    llm_calls_duration: Optional[float] = Field(default=0, description="The time in seconds spent in LLM calls.")
    llm_calls_count: Optional[int] = Field(default=0, description="The number of LLM calls in total.")
    llm_calls_total_prompt_tokens: Optional[int] = Field(default=0, description="The total number of prompt tokens.")
    llm_calls_total_completion_tokens: Optional[int] = Field(
        default=0, description="The total number of completion tokens."
    )
    llm_calls_total_tokens: Optional[int] = Field(default=0, description="The total number of tokens.")


class GenerationLog(Value):
    """Contains additional logging information associated with a generation call."""

    activated_rails: List[ActivatedRail] = Field(
        default_factory=list,
        description="The list of rails that were activated during generation.",
    )
    stats: GenerationStats = Field(
        default_factory=GenerationStats,
        description="General stats about the generation process.",
    )
    llm_calls: Optional[List[LLMCallInfo]] = Field(
        default=None,
        description="The list of LLM calls that have been made to fulfill the generation request. ",
    )
    internal_events: Optional[List[dict]] = Field(
        default=None, description="The complete sequence of internal events generated."
    )
    colang_history: Optional[str] = Field(
        default=None, description="The Colang history associated with the generation."
    )


class GenerationRailsOptions(Value):
    """Options for what rails should be used during the generation."""

    input: Union[bool, List[str]] = Field(
        default=True,
        description="Whether the input rails are enabled or not. "
        "If a list of names is specified, then only the specified input rails will be applied.",
    )
    output: Union[bool, List[str]] = Field(
        default=True,
        description="Whether the output rails are enabled or not. "
        "If a list of names is specified, then only the specified output rails will be applied.",
    )
    retrieval: Union[bool, List[str]] = Field(
        default=True,
        description="Whether the retrieval rails are enabled or not. "
        "If a list of names is specified, then only the specified retrieval rails will be applied.",
    )
    dialog: bool = Field(
        default=True,
        description="Whether the dialog rails are enabled or not.",
    )


class GenerationLogOptions(Value):
    """Options for what should be included in the generation log."""

    activated_rails: bool = Field(
        default=False,
        description="Include detailed information about the rails that were activated during generation.",
    )
    llm_calls: bool = Field(
        default=False,
        description="Include information about all the LLM calls that were made. "
        "This includes: prompt, completion, token usage, raw response, etc.",
    )
    internal_events: bool = Field(
        default=False,
        description="Include the array of internal generated events.",
    )
    colang_history: bool = Field(
        default=False,
        description="Include the history of the conversation in Colang format.",
    )
    stats: bool = Field(
        default=False,
        description="Include generation statistics — rail durations, LLM call counts, and token usage.",
    )


class GenerationOptions(Value):
    """A set of options that should be applied during a generation.

    The GenerationOptions control various things such as what rails are enabled,
    additional parameters for the main LLM, whether the rails should be enforced or
    ran in parallel, what to be included in the generation log, etc.
    """

    rails: GenerationRailsOptions = Field(
        default_factory=GenerationRailsOptions,
        description="Options for which rails should be applied for the generation. By default, all rails are enabled.",
    )
    llm_params: Optional[dict] = Field(
        default=None,
        description="Additional parameters that should be used for the LLM call",
    )
    llm_output: Optional[bool] = Field(
        default=False,
        description="Whether the response should also include any custom LLM output.",
    )
    output_vars: Optional[Union[bool, List[str]]] = Field(
        default=None,
        description="Whether additional context information should be returned. "
        "When True is specified, the whole context is returned. "
        "Otherwise, a list of key names can be specified.",
    )
    log: GenerationLogOptions = Field(
        default_factory=GenerationLogOptions,
        description="Options about what to include in the log. By default, nothing is included. ",
    )

    @model_validator(mode="before")
    def check_fields(cls, values):
        # Translate the `rails` generation option from List[str] to dict.
        if "rails" in values and isinstance(values["rails"], list):
            _rails = {
                "input": False,
                "dialog": False,
                "retrieval": False,
                "output": False,
            }
            for rail_type in values["rails"]:
                _rails[rail_type] = True
            values["rails"] = _rails

        return values


class ExceptionContent(Value):
    """RailException content"""

    type: str = Field(..., description="Type of the exception.")
    message: str = Field(..., description="Detailed message about the exception.")


class MessageTemplate(Value):
    """Template for a message structure."""

    type: str = Field(description="The type of message, e.g., 'assistant', 'user', 'system'.")
    content: str = Field(description="The content of the message.")


class TaskPrompt(Value):
    """Configuration for prompts that will be used for a specific task."""

    task: str = Field(description="The id of the task associated with this prompt.")
    content: Optional[str] = Field(default=None, description="The content of the prompt, if it's a string.")
    messages: Optional[List[Union[MessageTemplate, str]]] = Field(
        default=None,
        description="The list of messages included in the prompt. Used for chat models.",
    )
    models: Optional[List[str]] = Field(
        default=None,
        description="If specified, the prompt will be used only for the given LLM engines/models. "
        "The format is a list of strings with the format: <engine> or <engine>/<model>.",
    )
    output_parser: Optional[str] = Field(
        default=None,
        description="The name of the output parser to use for this prompt.",
    )
    max_length: Optional[int] = Field(
        default=16000,
        description="The maximum length of the prompt in number of characters.",
        ge=1,
    )
    mode: Optional[str] = Field(
        default=_default_config["prompting_mode"],
        description="Corresponds to the `prompting_mode` for which this prompt is fetched. Default is 'standard'.",
    )
    stop: Optional[List[str]] = Field(
        default=None,
        description="If specified, will be configure stop tokens for models that support this.",
    )

    max_tokens: Optional[int] = Field(
        default=None,
        description="The maximum number of tokens that can be generated in the chat completion.",
        ge=1,
    )

    @model_validator(mode="before")
    def check_fields(cls, values):
        if not values.get("content") and not values.get("messages"):
            raise ValueError("One of `content` or `messages` must be provided.")

        if values.get("content") and values.get("messages"):
            raise ValueError("Only one of `content` or `messages` must be provided.")

        return values


class Instruction(Value):
    """Configuration for instructions in natural language that should be passed to the LLM."""

    type: str
    content: str


class InputRails(Value):
    """Configuration of input rails."""

    parallel: Optional[bool] = Field(
        default=False,
        description="If True, the input rails are executed in parallel.",
    )

    flows: List[str] = Field(
        default_factory=list,
        description="The names of all the flows that implement input rails.",
    )


class OutputRailsStreamingConfig(Value):
    """Configuration for managing streaming output of LLM tokens."""

    enabled: bool = Field(default=True, description="Enables streaming mode when True.")
    chunk_size: int = Field(
        default=200,
        description="The number of tokens in each processing chunk. This is the size of the token block on which output rails are applied.",
    )
    context_size: int = Field(
        default=50,
        description="The number of tokens carried over from the previous chunk to provide context for continuity in processing.",
    )
    stream_first: bool = Field(
        default=True,
        description="If True, token chunks are streamed immediately before output rails are applied.",
    )


class OutputRails(Value):
    """Configuration of output rails."""

    parallel: Optional[bool] = Field(
        default=False,
        description="If True, the output rails are executed in parallel.",
    )

    flows: List[str] = Field(
        default_factory=list,
        description="The names of all the flows that implement output rails.",
    )

    streaming: OutputRailsStreamingConfig = Field(
        default_factory=OutputRailsStreamingConfig,
        description="Configuration for streaming output rails.",
    )

    apply_to_reasoning_traces: Optional[bool] = Field(
        default=False,
        description=(
            "If True, output rails will apply guardrails to both reasoning traces and output response. "
            "If False, output rails will only apply guardrails to the output response excluding the reasoning traces, "
            "thus keeping reasoning traces unaltered."
        ),
    )


class RetrievalRails(Value):
    """Configuration of retrieval rails."""

    flows: List[str] = Field(
        default_factory=list,
        description="The names of all the flows that implement retrieval rails.",
    )


class ActionRails(Value):
    """Configuration of action rails.

    Action rails control various options related to the execution of actions.
    Currently, only

    In the future multiple options will be added, e.g., what input validation should be
    performed per action, output validation, throttling, disabling, etc.
    """

    instant_actions: Optional[List[str]] = Field(
        default=None,
        description="The names of all actions which should finish instantly.",
    )


class ToolOutputRails(Value):
    """Configuration of tool output rails.
    Tool output rails are applied to tool calls before they are executed.
    They can validate tool names, parameters, and context to ensure safe tool usage.
    """

    flows: List[str] = Field(
        default_factory=list,
        description="The names of all the flows that implement tool output rails.",
    )
    parallel: Optional[bool] = Field(
        default=False,
        description="If True, the tool output rails are executed in parallel.",
    )


class ToolInputRails(Value):
    """Configuration of tool input rails.
    Tool input rails are applied to tool results before they are processed.
    They can validate, filter, or transform tool outputs for security and safety.
    """

    flows: List[str] = Field(
        default_factory=list,
        description="The names of all the flows that implement tool input rails.",
    )
    parallel: Optional[bool] = Field(
        default=False,
        description="If True, the tool input rails are executed in parallel.",
    )


class SingleCallConfig(Value):
    """Configuration for the single LLM call option for topical rails."""

    enabled: bool = False
    fallback_to_multiple_calls: bool = Field(
        default=True,
        description="Whether to fall back to multiple calls if a single call is not possible.",
    )


class UserMessagesConfig(Value):
    """Configuration for how the user messages are interpreted."""

    embeddings_only: bool = Field(
        default=False,
        description="Whether to use only embeddings for computing the user canonical form messages.",
    )
    embeddings_only_similarity_threshold: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="The similarity threshold to use when using only embeddings for computing the user canonical form messages.",
    )
    embeddings_only_fallback_intent: Optional[str] = Field(
        default=None,
        description="Defines the fallback intent when the similarity is below the threshold. If set to None, the user intent is computed normally using the LLM. If set to a string value, that string is used as the intent.",
    )


class DialogRails(Value):
    """Configuration of topical rails."""

    single_call: SingleCallConfig = Field(
        default_factory=SingleCallConfig,
        description="Configuration for the single LLM call option.",
    )
    user_messages: UserMessagesConfig = Field(
        default_factory=UserMessagesConfig,
        description="Configuration for how the user messages are interpreted.",
    )


class FactCheckingRailConfig(Value):
    """Configuration data for the fact-checking rail."""

    parameters: Dict[str, Any] = Field(default_factory=dict)
    fallback_to_self_check: bool = Field(
        default=False,
        description="Whether to fall back to self-check if another method fail.",
    )


class JailbreakDetectionConfig(Value):
    """Configuration data for jailbreak detection."""

    server_endpoint: Optional[str] = Field(
        default=None, description="The endpoint for the jailbreak detection heuristics/model container."
    )
    length_per_perplexity_threshold: float = Field(default=89.79, gt=0, description="The length/perplexity threshold.")
    prefix_suffix_perplexity_threshold: float = Field(
        default=1845.65, gt=0, description="The prefix/suffix perplexity threshold."
    )
    nim_base_url: Optional[str] = Field(
        default=None,
        description="Base URL for jailbreak detection model. Example: http://localhost:8000/v1",
    )
    nim_server_endpoint: Optional[str] = Field(
        default="classify",
        description="Classification path uri. Defaults to 'classify' for NemoGuard JailbreakDetect.",
    )
    api_key: Optional[SecretStr] = Field(
        default=None,
        description="Secret String with API key for use in Jailbreak requests. Takes precedence over api_key_env_var",
    )
    api_key_env_var: Optional[str] = Field(
        default=None,
        description="Environment variable containing API key for jailbreak detection model",
    )
    # Legacy fields, keep for backward compatibility with deprecation warnings
    nim_url: Optional[str] = Field(
        default=None,
        deprecated="Use 'nim_base_url' instead. This field will be removed in a future version.",
        description="DEPRECATED: Use nim_base_url instead",
    )
    nim_port: Optional[int] = Field(
        default=None,
        deprecated="Include port in 'nim_base_url' instead. This field will be removed in a future version.",
        description="DEPRECATED: Include port in nim_base_url instead",
    )
    embedding: Optional[str] = Field(
        default=None,
        deprecated="This field is no longer used.",
    )

    @model_validator(mode="after")
    def migrate_deprecated_fields(self) -> "JailbreakDetectionConfig":
        """Migrate deprecated nim_url/nim_port fields to nim_base_url format."""
        if self.nim_url and not self.nim_base_url:
            port = self.nim_port or 8000
            self.nim_base_url = f"http://{self.nim_url}:{port}/v1"
        return self

    @model_validator(mode="after")
    def validate_urls(self) -> "JailbreakDetectionConfig":
        """Validate URL formats for endpoints."""
        if self.nim_base_url and not self.nim_base_url.startswith(("http://", "https://")):
            raise ValueError(f"nim_base_url must start with 'http://' or 'https://', got '{self.nim_base_url}'")
        if self.server_endpoint and not self.server_endpoint.startswith(("http://", "https://")):
            raise ValueError(f"server_endpoint must start with 'http://' or 'https://', got '{self.server_endpoint}'")
        return self


class AutoAlignOptions(Value):
    """List of guardrails that are activated"""

    guardrails_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="The guardrails configuration that is passed to the AutoAlign endpoint",
    )


class AutoAlignRailConfig(Value):
    """Configuration data for the AutoAlign API"""

    parameters: Optional[Dict[str, Any]] = Field(default=None)
    input: Optional[AutoAlignOptions] = Field(
        default=None,
        description="Input configuration for AutoAlign guardrails",
    )
    output: Optional[AutoAlignOptions] = Field(
        default=None,
        description="Output configuration for AutoAlign guardrails",
    )


class PatronusEvaluationSuccessStrategy(str, Enum):
    """
    Strategy for determining whether a Patronus Evaluation API
    request should pass, especially when multiple evaluators
    are called in a single request.
    ALL_PASS requires all evaluators to pass for success.
    ANY_PASS requires only one evaluator to pass for success.
    """

    ALL_PASS = "all_pass"
    ANY_PASS = "any_pass"


class PatronusEvaluateApiParams(Value):
    """Config to parameterize the Patronus Evaluate API call"""

    success_strategy: Optional[PatronusEvaluationSuccessStrategy] = Field(
        default=PatronusEvaluationSuccessStrategy.ALL_PASS,
        description="Strategy to determine whether the Patronus Evaluate API Guardrail passes or not.",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters to the Patronus Evaluate API",
    )


class PatronusEvaluateConfig(Value):
    """Config for the Patronus Evaluate API call"""

    evaluate_config: PatronusEvaluateApiParams = Field(
        default_factory=PatronusEvaluateApiParams,
        description="Configuration passed to the Patronus Evaluate API",
    )


class PatronusRailConfig(Value):
    """Configuration data for the Patronus Evaluate API"""

    input: Optional[PatronusEvaluateConfig] = Field(
        default=None,
        description="Patronus Evaluate API configuration for an Input Guardrail",
    )
    output: Optional[PatronusEvaluateConfig] = Field(
        default=None,
        description="Patronus Evaluate API configuration for an Output Guardrail",
    )


class PrivateAIDetectionOptions(Value):
    """Configuration options for Private AI."""

    entities: List[str] = Field(
        default_factory=list,
        description="The list of entities that should be detected.",
    )


class PrivateAIDetection(Value):
    """Configuration for Private AI."""

    server_endpoint: Optional[str] = Field(
        default=None,
        description="The endpoint for the private AI detection server.",
    )
    input: Optional[PrivateAIDetectionOptions] = Field(
        default=None,
        description="Configuration of the entities to be detected on the user input.",
    )
    output: Optional[PrivateAIDetectionOptions] = Field(
        default=None,
        description="Configuration of the entities to be detected on the bot output.",
    )
    retrieval: Optional[PrivateAIDetectionOptions] = Field(
        default=None,
        description="Configuration of the entities to be detected on retrieved relevant chunks.",
    )


class ClavataRailOptions(Value):
    """Configuration data for the Clavata API"""

    policy: str = Field(
        description="The policy alias to use when evaluating inputs or outputs.",
    )

    labels: List[str] = Field(
        default_factory=list,
        description="""A list of labels to match against the policy.
        If no labels are provided, the overall policy result will be returned.
        If labels are provided, only hits on the provided labels will be considered a hit.""",
    )


class ClavataRailConfig(Value):
    """Configuration data for the Clavata API"""

    server_endpoint: str = Field(
        default="https://gateway.app.clavata.ai:8443",
        description="The endpoint for the Clavata API",
    )

    policies: Dict[str, str] = Field(
        default_factory=dict,
        description="A dictionary of policy aliases and their corresponding IDs.",
    )

    label_match_logic: Literal["ANY", "ALL"] = Field(
        default="ANY",
        description="""The logic to use when deciding whether the evaluation matched.
        If ANY, only one of the configured labels needs to be found in the input or output.
        If ALL, all configured labels must be found in the input or output.""",
    )

    input: Optional[ClavataRailOptions] = Field(
        default=None,
        description="Clavata configuration for an Input Guardrail",
    )
    output: Optional[ClavataRailOptions] = Field(
        default=None,
        description="Clavata configuration for an Output Guardrail",
    )


class InjectionDetection(Value):
    injections: List[str] = Field(
        default_factory=list,
        description="The list of injection types to detect. Options are 'sqli', 'template', 'code', 'xss'."
        "Currently, only SQL injection, template injection, code injection, "
        "and markdown cross-site scripting are supported. "
        "Custom rules can be added, provided they are in the `yara_path` and have a `.yara` file extension.",
    )
    action: str = Field(
        default="reject",
        pattern=r"^(reject|omit)$",
        description="Action to take. Options are 'reject' to offer a rejection message, "
        "'omit' to mask the offending content, and 'sanitize' to pass the content as-is in the safest way. "
        "These options are listed in descending order of relative safety. 'sanitize' is not implemented at this time.",
    )
    yara_rules: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Dictionary mapping rule names to YARA rule strings. If provided, these rules will be used "
        "instead of loading rules from yara_path. Each rule should be a valid YARA rule string.",
    )


class SensitiveDataDetectionOptions(Value):
    entities: List[str] = Field(
        default_factory=list,
        description="The list of entities that should be detected. "
        "Check out https://microsoft.github.io/presidio/supported_entities/ for"
        "the list of supported entities.",
    )
    # TODO: this is not currently in use.
    mask_token: str = Field(
        default="*",
        description="The token that should be used to mask the sensitive data.",
    )

    score_threshold: float = Field(
        default=0.2,
        description="The score threshold that should be used to detect the sensitive data.",
    )


class SensitiveDataDetection(Value):
    """Configuration of what sensitive data should be detected."""

    recognizers: Optional[List[dict]] = Field(
        default=None,
        description="Additional custom recognizers. "
        "Check out https://microsoft.github.io/presidio/tutorial/08_no_code/ for more details.",
    )
    input: Optional[SensitiveDataDetectionOptions] = Field(
        default=None,
        description="Configuration of the entities to be detected on the user input.",
    )
    output: Optional[SensitiveDataDetectionOptions] = Field(
        default=None,
        description="Configuration of the entities to be detected on the bot output.",
    )
    retrieval: Optional[SensitiveDataDetectionOptions] = Field(
        default=None,
        description="Configuration of the entities to be detected on retrieved relevant chunks.",
    )


class PangeaRailOptions(Value):
    """Configuration data for the Pangea AI Guard API"""

    recipe: str = Field(
        description="""Recipe key of a configuration of data types and settings defined in the Pangea User Console. It
        specifies the rules that are to be applied to the text, such as defang malicious URLs."""
    )


class PangeaRailConfig(Value):
    """Configuration data for the Pangea AI Guard API"""

    input: Optional[PangeaRailOptions] = Field(
        default=None,
        description="Pangea configuration for an Input Guardrail",
    )
    output: Optional[PangeaRailOptions] = Field(
        default=None,
        description="Pangea configuration for an Output Guardrail",
    )


class GuardrailsAIValidatorConfig(Value):
    """Configuration for a single Guardrails AI validator."""

    name: str = Field(
        description="Unique identifier or import path for the Guardrails AI validator (e.g., 'toxic_language', 'pii', 'regex_match', or 'guardrails/competitor_check')."
    )

    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters to pass to the validator during initialization (e.g., threshold, regex pattern).",
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata to pass to the validator during validation (e.g., valid_topics, context).",
    )


class GuardrailsAIRailConfig(Value):
    """Configuration data for Guardrails AI integration."""

    validators: List[GuardrailsAIValidatorConfig] = Field(
        default_factory=list,
        description="List of Guardrails AI validators to apply. Each validator can have its own parameters and metadata.",
    )

    def get_validator_config(self, name: str) -> Optional[GuardrailsAIValidatorConfig]:
        """Get a specific validator configuration by name."""
        for _validator in self.validators:
            if _validator.name == name:
                return _validator
        return None


class TrendMicroRailConfig(Value):
    """Configuration data for the Trend Micro AI Guard API"""

    v1_url: str = Field(
        default="https://api.xdr.trendmicro.com/v3.0/aiSecurity/applyGuardrails",
        description="The endpoint for the Trend Micro AI Guard API. For other regions, use: https://api.{region}.xdr.trendmicro.com/v3.0/aiSecurity/applyGuardrails where region is eu, jp, au, in, sg, or mea.",
    )

    api_key_env_var: Optional[str] = Field(
        default=None,
        description="Environment variable containing API key for Trend Micro AI Guard",
    )

    application_name: str = Field(
        default="nemo-guardrails",
        description="Application name for TMV1-Application-Name header (REQUIRED). Must contain only letters, numbers, hyphens, and underscores, with a maximum length of 64 characters.",
        pattern=r"^[a-zA-Z0-9_-]+$",
        max_length=64,
    )

    detailed_response: bool = Field(
        default=False,
        description="If True, returns detailed AI Guard results with confidence scores (Prefer: return=representation). If False, returns minimal response with only action and reasons (Prefer: return=minimal).",
    )

    def get_api_key(self) -> Optional[str]:
        """Helper to return an API key (if it exists) from a Trend Micro configuration.
        The `api_key_env_var` field, a string stored in this environment variable.
        If the environment variable is not found None is returned.
        """

        if self.api_key_env_var:
            v1_api_key = os.getenv(self.api_key_env_var)
            if v1_api_key:
                return v1_api_key

            log.warning(
                "Specified a value for Trend Micro config api_key_env var at %s but the environment variable was not set!"
                % self.api_key_env_var
            )

        return None


class AIDefenseRailConfig(Value):
    """Configuration data for the Cisco AI Defense API"""

    timeout: float = Field(
        default=30.0,
        description="Timeout in seconds for API requests to AI Defense service",
    )

    fail_open: bool = Field(
        default=False,
        description="If True, allow content when AI Defense API call fails (fail open). If False, block content when API call fails (fail closed). Does not affect missing configuration validation.",
    )


class RailsConfigData(Value):
    """Configuration data for specific rails that are supported out-of-the-box."""

    fact_checking: Optional[FactCheckingRailConfig] = Field(
        default=None,
        description="Configuration data for the fact-checking rail.",
    )

    autoalign: Optional[AutoAlignRailConfig] = Field(
        default=None,
        description="Configuration data for the AutoAlign guardrails API.",
    )

    patronus: Optional[PatronusRailConfig] = Field(
        default=None,
        description="Configuration data for the Patronus Evaluate API.",
    )

    sensitive_data_detection: Optional[SensitiveDataDetection] = Field(
        default=None,
        description="Configuration for detecting sensitive data.",
    )

    regex_detection: Optional[RegexDetection] = Field(
        default_factory=RegexDetection,
        description="Configuration for regex pattern detection.",
    )

    jailbreak_detection: Optional[JailbreakDetectionConfig] = Field(
        default=None,
        description="Configuration for jailbreak detection.",
    )
    injection_detection: Optional[InjectionDetection] = Field(
        default=None,
        description="Configuration for injection detection.",
    )
    privateai: Optional[PrivateAIDetection] = Field(
        default=None,
        description="Configuration for Private AI.",
    )

    gliner: Optional[GLiNERDetection] = Field(
        default_factory=GLiNERDetection,
        description="Configuration for GLiNER PII detection.",
    )

    polygraf: Optional[PolygrafDetection] = Field(
        default_factory=PolygrafDetection,
        description="Configuration for Polygraf PII detection.",
    )

    fiddler: Optional[FiddlerGuardrails] = Field(
        default=None,
        description="Configuration for Fiddler Guardrails.",
    )
    clavata: Optional[ClavataRailConfig] = Field(
        default=None,
        description="Configuration for Clavata.",
    )

    crowdstrike_aidr: Optional[CrowdStrikeAIDRRailConfig] = Field(
        default_factory=CrowdStrikeAIDRRailConfig,
        description="Configuration for CrowdStrike AIDR.",
    )

    pangea: Optional[PangeaRailConfig] = Field(
        default=None,
        description="Configuration for Pangea.",
    )

    guardrails_ai: Optional[GuardrailsAIRailConfig] = Field(
        default=None,
        description="Configuration for Guardrails AI validators.",
    )

    trend_micro: Optional[TrendMicroRailConfig] = Field(
        default=None,
        description="Configuration for Trend Micro.",
    )

    ai_defense: Optional[AIDefenseRailConfig] = Field(
        default=None,
        description="Configuration for Cisco AI Defense.",
    )

    content_safety: Optional[ContentSafetyConfig] = Field(
        default=None,
        description="Configuration for content safety rails.",
    )

    hf_classifier: Optional[Dict[str, HFClassifierConfig]] = Field(
        default=None,
        description="Named HF classifier configurations. Keys are classifier names referenced by flows.",
    )

    context_bloat_detection: Optional[ContextBloatDetectionConfig] = Field(
        default_factory=ContextBloatDetectionConfig,
        description="Configuration for context bloat / context manipulation detection.",
    )


class Rails(Value):
    """Configuration of specific rails."""

    # NOTE: it avoids unecessary yaml parsing as RailsConfigData
    # includes default_factory Objects
    #
    config: Optional[RailsConfigData] = Field(
        default=None,
        description="Configuration data for specific rails that are supported out-of-the-box.",
    )
    input: Optional[InputRails] = Field(default=None, description="Configuration of the input rails.")
    output: Optional[OutputRails] = Field(default=None, description="Configuration of the output rails.")
    retrieval: Optional[RetrievalRails] = Field(
        default=None,
        description="Configuration of the retrieval rails.",
    )
    dialog: Optional[DialogRails] = Field(default=None, description="Configuration of the dialog rails.")
    actions: Optional[ActionRails] = Field(default=None, description="Configuration of action rails.")
    tool_output: Optional[ToolOutputRails] = Field(
        default=None,
        description="Configuration of tool output rails.",
    )
    tool_input: Optional[ToolInputRails] = Field(
        default=None,
        description="Configuration of tool input rails.",
    )

    class Config:
        # NOTE (rdinu): disabling the top level `config` field as it is not valid in the OpenAPI spec.
        # json_schema_extra = {"config": {"data": {"type": "object", "example": {"key": "value"}}}}
        extra = "forbid"


# Prefix for model in a flow definition
MODEL_PREFIX = "$model="


def _normalize_flow_id(flow_id: str) -> str:
    """Normalize the flow id by removing the arguments from the id.

    Args:
        flow_id(str): The flow id.

    Example:

        flow_id = "flow_id_v1(arg1, arg2)"
        _normalize_flow_id(flow_id) -> "flow_id_v1"

    """
    flow_id = flow_id.strip()
    if "(" in flow_id:
        flow_id = flow_id.split("(")[0]

    elif "$" in flow_id:
        flow_id = flow_id.split("$")[0]

    return flow_id.strip()


def _get_flow_model(flow_text) -> Optional[str]:
    """Helper to return a model name from a flow definition"""
    if MODEL_PREFIX not in flow_text:
        return None
    return flow_text.split(MODEL_PREFIX)[-1].strip()


def _validate_rail_prompts(rails: list[str], prompts: list[Any], validation_rail: str) -> None:
    for rail in rails:
        flow_id = _normalize_flow_id(rail)
        flow_model = _get_flow_model(rail)
        if flow_id == validation_rail:
            prompt_flow_id = flow_id.replace(" ", "_")
            expected_prompt = f"{prompt_flow_id} $model={flow_model}"
            if expected_prompt not in prompts:
                raise InvalidRailsConfigurationError(
                    f"Missing a `{expected_prompt}` prompt template, which is required for the `{validation_rail}` rail."
                )


# NOTE: This maps to the `RailsConfig` class in `nemo-guardrails`.
# See https://github.com/NVIDIA/NeMo-Guardrails/blob/develop/nemoguardrails/rails/llm/config.py
class RailsConfig(Value):
    """Configuration object for the models and the rails."""

    # TODO: add typed config for user_messages, bot_messages, and flows.

    models: List[Model] = Field(default_factory=list, description="The list of models used by the rails configuration.")

    instructions: Optional[List[Instruction]] = Field(
        default=[Instruction.model_validate(obj) for obj in _default_config["instructions"]],
        description="List of instructions in natural language that the LLM should use.",
    )

    actions_server_url: Optional[str] = Field(
        default=None,
        description="The URL of the actions server that should be used for the rails.",
    )  # consider as conflict

    sample_conversation: Optional[str] = Field(
        default=_default_config["sample_conversation"],
        description="The sample conversation that should be used inside the prompts.",
    )

    prompts: Optional[List[TaskPrompt]] = Field(
        default=None,
        description="The prompts that should be used for the various LLM tasks.",
    )

    prompting_mode: Optional[str] = Field(
        default=_default_config["prompting_mode"],
        description="Allows choosing between different prompting strategies.",
    )

    # Some tasks need to be as deterministic as possible. The lowest possible temperature
    # will be used for those tasks. Models like dolly don't allow for a temperature of 0.0,
    # for example, in which case a custom one can be set.
    lowest_temperature: Optional[float] = Field(
        default=0.001,
        description="The lowest temperature that should be used for the LLM.",
    )

    # This should only be enabled for highly capable LLMs i.e. gpt-3.5-turbo-instruct or similar.
    enable_multi_step_generation: Optional[bool] = Field(
        default=False,
        description="Whether to enable multi-step generation for the LLM.",
    )

    colang_version: str = Field(default="1.0", description="The Colang version to use.")

    custom_data: Dict = Field(
        default_factory=dict,
        description="Any custom configuration data that might be needed.",
    )

    rails: Rails = Field(
        default_factory=Rails,
        description="Configuration for the various rails (input, output, etc.).",
    )

    enable_rails_exceptions: bool = Field(
        default=False,
        description="If set, the pre-defined guardrails raise exceptions instead of returning pre-defined messages.",
    )

    passthrough: Optional[bool] = Field(
        default=None,
        description="Whether the original prompt should pass through the guardrails configuration as is. "
        "This means it will not be altered in any way. ",
    )

    tracing: TracingConfig = Field(
        default_factory=TracingConfig,
        description="Configuration for tracing.",
    )

    @model_validator(mode="before")
    def check_model_exists_for_input_rails(cls, values):
        """Make sure we have a model for each input rail where one is provided using $model=<model_type>"""
        rails = values.get("rails", {})

        # Handle both dict (during initial parsing) and Rails object (when passed directly)
        if isinstance(rails, Rails):
            input_flows = getattr(rails.input, "flows", []) if rails.input else []
        else:
            input_flows = (rails.get("input") or {}).get("flows", []) if isinstance(rails, dict) else []

        # If no flows have a model, early-out
        input_flows_without_model = [_get_flow_model(flow) is None for flow in input_flows]
        if all(input_flows_without_model):
            return values

        models = values.get("models", []) or []
        model_types = {model.type if isinstance(model, Model) else model["type"] for model in models}

        for flow in input_flows:
            flow_model = _get_flow_model(flow)
            if not flow_model:
                continue
            if flow_model not in model_types:
                flow_id = _normalize_flow_id(flow)
                available_types = ", ".join(f"'{str(t)}'" for t in sorted(model_types)) if model_types else "none"
                raise InvalidRailsConfigurationError(
                    f"Input flow '{flow_id}' references model type '{flow_model}' that is not defined in the configuration. Detected model types: {available_types}."
                )
        return values

    @model_validator(mode="before")
    def check_model_exists_for_output_rails(cls, values):
        """Make sure we have a model for each output rail where one is provided using $model=<model_type>"""
        rails = values.get("rails", {})

        # Handle both dict (during initial parsing) and Rails object (when passed directly)
        if isinstance(rails, Rails):
            output_flows = getattr(rails.output, "flows", []) if rails.output else []
        else:
            output_flows = (rails.get("output") or {}).get("flows", []) if isinstance(rails, dict) else []

        # If no flows have a model, early-out
        output_flows_without_model = [_get_flow_model(flow) is None for flow in output_flows]
        if all(output_flows_without_model):
            return values

        models = values.get("models", []) or []
        model_types = {model.type if isinstance(model, Model) else model["type"] for model in models}

        for flow in output_flows:
            flow_model = _get_flow_model(flow)
            if not flow_model:
                continue
            if flow_model not in model_types:
                flow_id = _normalize_flow_id(flow)
                available_types = ", ".join(f"'{str(t)}'" for t in sorted(model_types)) if model_types else "none"
                raise InvalidRailsConfigurationError(
                    f"Output flow '{flow_id}' references model type '{flow_model}' that is not defined in the configuration. Detected model types: {available_types}."
                )
        return values

    class Config:
        # NOTE (rdinu): disabling the top level `config` field as it is not valid in the OpenAPI spec.
        # json_schema_extra = {"rails": {"data": {"type": "object", "example": {"key": "value"}}}}
        # Use "ignore" to allow extra fields from nemoguardrails that we don't model
        extra = "ignore"

    @model_validator(mode="before")
    def check_prompt_exist_for_self_check_rails(cls, values):
        rails = values.get("rails", {})
        prompts = values.get("prompts", []) or []

        # Handle both dict (during initial parsing) and Rails object (when passed directly)
        if isinstance(rails, Rails):
            enabled_input_rails = getattr(rails.input, "flows", []) if rails.input else []
            enabled_output_rails = getattr(rails.output, "flows", []) if rails.output else []
        else:
            enabled_input_rails = (rails.get("input") or {}).get("flows", []) if isinstance(rails, dict) else []
            enabled_output_rails = (rails.get("output") or {}).get("flows", []) if isinstance(rails, dict) else []
        provided_task_prompts = [prompt.task if hasattr(prompt, "task") else prompt.get("task") for prompt in prompts]

        # Input moderation prompt verification
        if "self check input" in enabled_input_rails and "self_check_input" not in provided_task_prompts:
            raise InvalidRailsConfigurationError(
                "Missing a `self_check_input` prompt template, which is required for the `self check input` rail."
            )
        if "llama guard check input" in enabled_input_rails and "llama_guard_check_input" not in provided_task_prompts:
            raise InvalidRailsConfigurationError(
                "Missing a `llama_guard_check_input` prompt template, which is required for the `llama guard check input` rail."
            )

        # Only content-safety and topic-safety include a $model reference in the rail flow text
        # Need to match rails with flow_id (excluding $model reference) and match prompts
        # on the full flow_id (including $model reference)
        _validate_rail_prompts(enabled_input_rails, provided_task_prompts, "content safety check input")
        _validate_rail_prompts(enabled_input_rails, provided_task_prompts, "topic safety check input")

        # Output moderation prompt verification
        if "self check output" in enabled_output_rails and "self_check_output" not in provided_task_prompts:
            raise InvalidRailsConfigurationError(
                "Missing a `self_check_output` prompt template, which is required for the `self check output` rail."
            )
        if (
            "llama guard check output" in enabled_output_rails
            and "llama_guard_check_output" not in provided_task_prompts
        ):
            raise InvalidRailsConfigurationError(
                "Missing a `llama_guard_check_output` prompt template, which is required for the `llama guard check output` rail."
            )
        if (
            "patronus lynx check output hallucination" in enabled_output_rails
            and "patronus_lynx_check_output_hallucination" not in provided_task_prompts
        ):
            raise InvalidRailsConfigurationError(
                "Missing a `patronus_lynx_check_output_hallucination` prompt template, which is required for the `patronus lynx check output hallucination` rail."
            )

        if "self check facts" in enabled_output_rails and "self_check_facts" not in provided_task_prompts:
            raise InvalidRailsConfigurationError(
                "Missing a `self_check_facts` prompt template, which is required for the `self check facts` rail."
            )

        # Only content-safety and topic-safety include a $model reference in the rail flow text
        # Need to match rails with flow_id (excluding $model reference) and match prompts
        # on the full flow_id (including $model reference)
        _validate_rail_prompts(enabled_output_rails, provided_task_prompts, "content safety check output")

        return values

    @model_validator(mode="before")
    def check_output_parser_exists(cls, values):
        tasks_requiring_output_parser = [
            "self_check_input",
            "self_check_facts",
            "self_check_output",
            # "content_safety_check input $model",
            # "content_safety_check output $model",
        ]
        prompts = values.get("prompts") or []
        tasks_missing_output_parser: list[str] = []
        for prompt in prompts:
            task = prompt.task if hasattr(prompt, "task") else prompt.get("task")
            output_parser = prompt.output_parser if hasattr(prompt, "output_parser") else prompt.get("output_parser")

            if any(task.startswith(task_prefix) for task_prefix in tasks_requiring_output_parser) and not output_parser:
                if task not in tasks_missing_output_parser:
                    tasks_missing_output_parser.append(task)

        for task in tasks_missing_output_parser:
            log.warning(
                f"Output parser is not registered for task '{task}'. "
                f"Register 'output_parser' in prompts.yml for this task. "
                "Using 'is_content_safe' as the default output parser. "
                "This behavior will be deprecated in future versions."
            )
        return values

    @model_validator(mode="before")
    def check_jailbreak_detection_config(cls, values):
        """Validate jailbreak detection configuration against enabled flows."""
        rails = values.get("rails") or {}

        # Handle both dict (during initial parsing) and Rails object (when passed directly)
        if isinstance(rails, Rails):
            config_data = rails.config.model_dump() if rails.config else {}
            input_flows = getattr(rails.input, "flows", []) if rails.input else []
        else:
            config_data = rails.get("config") or {}
            input_flows = (rails.get("input") or {}).get("flows") or []

        jailbreak_config = (config_data.get("jailbreak_detection") if isinstance(config_data, dict) else None) or {}
        has_model_flow = JAILBREAK_FLOW_MODEL in input_flows
        has_heuristics_flow = JAILBREAK_FLOW_HEURISTICS in input_flows
        has_any_jailbreak_flow = has_model_flow or has_heuristics_flow

        # Case A: Config present but no flow references it
        if jailbreak_config and not has_any_jailbreak_flow:
            log.warning(
                "Jailbreak detection configuration is present under "
                "rails.config.jailbreak_detection but no jailbreak detection flow "
                "is enabled. To use jailbreak detection, add 'jailbreak detection model' "
                "or 'jailbreak detection heuristics' to rails.input.flows."
            )

        # Case B: "jailbreak detection model" flow is enabled
        if has_model_flow:
            nim_base_url = jailbreak_config.get("nim_base_url")
            nim_url = jailbreak_config.get("nim_url")  # deprecated, migrated later
            server_endpoint = jailbreak_config.get("server_endpoint")
            nim_server_endpoint = jailbreak_config.get("nim_server_endpoint", "classify")

            if nim_base_url or nim_url:
                if not nim_server_endpoint:
                    raise InvalidRailsConfigurationError(
                        "nim_base_url is set for jailbreak detection model but "
                        "nim_server_endpoint is empty. Both must be configured "
                        "when using NIM-based jailbreak detection."
                    )
            elif not server_endpoint:
                log.warning(
                    "No endpoint configured for jailbreak detection model. "
                    "Will fall back to local in-process detection, which is "
                    "not recommended for production."
                )

        # Case C: "jailbreak detection heuristics" flow is enabled
        if has_heuristics_flow:
            server_endpoint = jailbreak_config.get("server_endpoint")
            if not server_endpoint:
                log.warning(
                    "No server_endpoint configured for jailbreak detection heuristics. "
                    "Will fall back to local in-process detection, which is "
                    "not recommended for production."
                )

        return values

    @classmethod
    def from_path(
        cls,
        config_path: str,
    ):
        """Loads a configuration from a given path.

        Supports loading a from a single file, or from a directory.
        """
        # If the config path is a file, we load the YAML content.
        # Otherwise, if it's a folder, we iterate through all files.
        if os.path.isfile(config_path) and config_path.endswith((".yaml", ".yml")):
            with open(config_path) as f:
                raw_config = yaml.safe_load(f.read())

        elif os.path.isdir(config_path):
            raw_config, colang_files = _load_path(config_path)

            # If we have import paths, we also need to load them.
            if raw_config.get("import_paths"):
                _load_imported_paths(raw_config, colang_files)

            # Parse the colang files after we know the colang version
            _parse_colang_files_recursively(raw_config, colang_files, parsed_colang_files=[])

        else:
            raise ValueError(f"Invalid config path {config_path}.")

        # If there are no instructions, we use the default ones.
        if len(raw_config.get("instructions", [])) == 0:
            raw_config["instructions"] = _nemo_default_config["instructions"]

        raw_config["config_path"] = config_path

        return cls.parse_object(raw_config)

    @classmethod
    def from_content(
        cls,
        colang_content: Optional[str] = None,
        yaml_content: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        """Loads a configuration from the provided colang/YAML content/config dict."""
        raw_config = {}

        if config:
            _join_config(raw_config, config)

        if yaml_content:
            _join_config(raw_config, yaml.safe_load(yaml_content))

        # Parse the colang files after we know the colang version
        colang_version = raw_config.get("colang_version", "1.0")

        # We start parsing the colang files one by one, and if we have
        # new import paths, we continue to update
        colang_files: List[Tuple[str, str]] = []
        parsed_colang_files: List[dict] = []

        # First, we parse the starting content.
        if colang_content:
            colang_files.append(("main.co", "main.co"))

            _parsed_config = parse_colang_file(
                "main.co",
                content=colang_content,
                version=colang_version,
            )

            # We join only the "import_paths" field in the config for now
            _join_config(
                raw_config,
                {"import_paths": _parsed_config.get("import_paths", [])},
            )

            parsed_colang_files.append(_parsed_config)

        # Load any new colang files potentially coming from imports
        if raw_config.get("import_paths"):
            _load_imported_paths(raw_config, colang_files)

        # Next, we parse any additional files recursively
        _parse_colang_files_recursively(raw_config, colang_files, parsed_colang_files)

        # If there are no instructions, we use the default ones.
        if len(raw_config.get("instructions", [])) == 0:
            raw_config["instructions"] = _nemo_default_config["instructions"]

        return cls.parse_object(raw_config)

    @classmethod
    def parse_object(cls, obj):
        """Parses a configuration object from a given dictionary."""
        # If we have flows, we need to process them further from CoYML to CIL, but only for
        # version 1.0.

        if obj.get("colang_version", "1.0") == "1.0":
            for flow_data in obj.get("flows", []):
                # If the first element in the flow does not have a "_type", we need to convert
                if flow_data.get("elements") and not flow_data["elements"][0].get("_type"):
                    flow_data["elements"] = parse_flow_elements(flow_data["elements"])

        return cls.parse_obj(obj)

    def __add__(self, other: "RailsConfig") -> "RailsConfig":
        """Adds two RailsConfig objects."""
        return _join_rails_configs(self, other)


def _join_rails_configs(base_rails_config: RailsConfig, updated_rails_config: RailsConfig) -> RailsConfig:
    """Helper to join two rails configuration."""

    config_old_types = {}
    for model_old in base_rails_config.models:
        config_old_types[model_old.type] = model_old

    for model_new in updated_rails_config.models:
        if model_new.type in config_old_types:
            if model_new.engine != config_old_types[model_new.type].engine:
                raise ValueError("Both config files should have the same engine for the same model type")
            if model_new.model != config_old_types[model_new.type].model:
                raise ValueError("Both config files should have the same model for the same model type")

    if base_rails_config.actions_server_url != updated_rails_config.actions_server_url:
        raise ValueError("Both config files should have the same actions_server_url")

    combined_rails_config_dict = _join_dict(base_rails_config.model_dump(), updated_rails_config.model_dump())
    # filter out empty strings to avoid leading/trailing commas
    config_paths = [
        base_rails_config.model_dump()["config_path"] or "",
        updated_rails_config.model_dump()["config_path"] or "",
    ]
    combined_rails_config_dict["config_path"] = ",".join(filter(None, config_paths))
    combined_rails_config = RailsConfig(**combined_rails_config_dict)
    return combined_rails_config


def _join_dict(dict1, dict2):
    """
    Joins two dictionaries recursively.
    - If values are dictionaries, it applies _join_dict recursively.
    - If values are lists, it concatenates them, ensuring unique elements.
    - For other types, values from dict2 overwrite dict1.
    """
    result = dict(dict1)  # Create a copy of dict1 to avoid modifying the original

    for key, value in dict2.items():
        # If key is in both dictionaries and both values are dictionaries, apply _join_dict recursively
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
            result[key] = _join_dict(dict1[key], value)
        # If key is in both dictionaries and both values are lists, concatenate unique elements
        elif key in dict1 and isinstance(dict1[key], list) and isinstance(value, list):
            # Since we want values from dict2 to take precedence, we concatenate dict2 first
            result[key] = _unique_list_concat(value, dict1[key])
        # Otherwise, simply overwrite the value from dict2
        else:
            result[key] = value

    return result
