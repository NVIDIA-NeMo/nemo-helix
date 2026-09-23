# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# =============================================================================
# Dataset Schemas for DPO Training
# =============================================================================
# Preference Dataset Schemas for DPO Training:
# - PreferenceDataset: Native format with context + ranked completions
# - BinaryPreferenceDataset: Simple prompt/chosen/rejected strings
# - HelpSteer3Dataset: NVIDIA HelpSteer3 format with preference scores
# - Tulu3PreferenceDataset: AllenAI Tulu3 format with message lists
#
# SFT Dataset Schemas:
# - SFTDatasetItemSchema: Standard prompt/completion format
from typing import Annotated, Any, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, model_validator

# Dataset class names from nhx.rl.tasks.training.backends.nemo_rl.preference_datasets
# These constants ensure consistency between the discriminator and Tag values
PREFERENCE_DATASET = "PreferenceDataset"
BINARY_PREFERENCE_DATASET = "BinaryPreferenceDataset"
HELPSTEER3_DATASET = "HelpSteer3"
TULU3_PREFERENCE_DATASET = "Tulu3Preference"


class ChatMessage(BaseModel):
    """A single message in a conversation."""

    role: str = Field(..., description="The role of the message sender (e.g., 'user', 'assistant', 'system')")
    content: str = Field(..., description="The content of the message")


class CompletionItem(BaseModel):
    """A ranked completion in a preference dataset."""

    rank: int = Field(..., description="Rank of this completion (0 = best/chosen, higher = worse)")
    completion: List[ChatMessage] = Field(..., description="The completion as a list of messages")


class PreferenceDatasetItemSchema(BaseModel):
    """Schema for native PreferenceDataset format.

    This is the canonical format used by nemo-rl's PreferenceDataset class.
    It supports multi-turn context and multiple ranked completions.

    Example:
        {
            "context": [{"role": "user", "content": "What is 2+2?"}],
            "completions": [
                {"rank": 0, "completion": [{"role": "assistant", "content": "4"}]},
                {"rank": 1, "completion": [{"role": "assistant", "content": "5"}]}
            ]
        }
    """

    context: List[ChatMessage] = Field(
        ..., description="The conversation context (prompt messages including previous turns)"
    )
    completions: List[CompletionItem] = Field(
        ..., description="List of ranked completions (rank 0 = preferred, rank 1 = rejected, etc.)"
    )

    model_config = ConfigDict(extra="allow")


class BinaryPreferenceDatasetItemSchema(BaseModel):
    """Schema for BinaryPreferenceDataset format.

    Simple format with prompt, chosen response, and rejected response as strings.
    The prompt can be either a string or a list of messages.

    Example:
        {
            "prompt": "What is the capital of France?",
            "chosen": "The capital of France is Paris.",
            "rejected": "The capital of France is London."
        }
    """

    prompt: Union[str, List[ChatMessage]] = Field(..., description="The input prompt (string or list of messages)")
    chosen: str = Field(..., description="The preferred/chosen response")
    rejected: str = Field(..., description="The rejected/non-preferred response")

    model_config = ConfigDict(extra="allow")


class HelpSteer3DatasetItemSchema(BaseModel):
    """Schema for NVIDIA HelpSteer3 preference dataset format.

    Uses numeric preference scores to indicate which response is preferred.
    - Negative overall_preference: response1 is preferred
    - Positive overall_preference: response2 is preferred
    - Zero overall_preference: tie (no preference)

    Example:
        {
            "context": "Explain quantum computing",
            "response1": "Quantum computing uses qubits...",
            "response2": "Quantum computing is magic...",
            "overall_preference": -2
        }
    """

    context: Union[str, List[ChatMessage]] = Field(..., description="The input context (string or list of messages)")
    response1: str = Field(..., description="First response option")
    response2: str = Field(..., description="Second response option")
    overall_preference: int = Field(
        ...,
        description="Preference score: negative=response1 preferred, positive=response2 preferred, 0=tie",
    )

    model_config = ConfigDict(extra="allow")


class Tulu3PreferenceDatasetItemSchema(BaseModel):
    """Schema for AllenAI Tulu3 preference dataset format.

    Contains full conversation histories for both chosen and rejected responses.
    The last message in each list must be from the assistant role.

    Example:
        {
            "chosen": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi! How can I help?"}
            ],
            "rejected": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Go away."}
            ]
        }
    """

    chosen: List[ChatMessage] = Field(
        ..., description="Full conversation with preferred response (last message must be assistant)"
    )
    rejected: List[ChatMessage] = Field(
        ..., description="Full conversation with rejected response (last message must be assistant)"
    )

    model_config = ConfigDict(extra="allow")


def get_preference_dataset_discriminator(v: Any) -> str:
    """Determine the preference dataset schema type based on field presence.

    This discriminator function examines the fields present in the data
    to determine which schema type it matches. Returns the NeMo RL dataset
    class name that corresponds to the detected format:
    - PreferenceDataset: Has 'context' and 'completions' fields (native format)
    - HelpSteer3: Has 'overall_preference' field (HelpSteer3 format)
    - Tulu3PreferenceDataset: Has 'chosen' and 'rejected' as lists of messages
    - BinaryPreferenceDataset: Has 'prompt', 'chosen', 'rejected'

    Args:
        v: The data to discriminate (dict or model instance)

    Returns:
        NeMo RL dataset class name identifying the schema type
    """
    if isinstance(v, dict):
        # Native PreferenceDataset format: context + completions
        if "completions" in v and "context" in v:
            return PREFERENCE_DATASET

        # HelpSteer3 format: has overall_preference score
        if "overall_preference" in v:
            return HELPSTEER3_DATASET

        # Tulu3 format: chosen/rejected are lists of messages (must check BEFORE BinaryPreferenceDataset)
        # Tulu3 data may also have 'prompt' field, so we differentiate by checking if chosen/rejected are lists
        if "chosen" in v and "rejected" in v:
            chosen = v.get("chosen")
            if isinstance(chosen, list) and len(chosen) > 0:
                # Check if it looks like a message list
                if isinstance(chosen[0], dict) and "role" in chosen[0]:
                    return TULU3_PREFERENCE_DATASET

        # BinaryPreferenceDataset format: prompt + chosen + rejected (as strings)
        if "prompt" in v and "chosen" in v and "rejected" in v:
            return BINARY_PREFERENCE_DATASET

    return PREFERENCE_DATASET  # Default fallback


# Union type for all preference dataset formats
DPOPreferenceDatasetSchemaType = Annotated[
    Union[
        Annotated[PreferenceDatasetItemSchema, Tag(PREFERENCE_DATASET)],
        Annotated[BinaryPreferenceDatasetItemSchema, Tag(BINARY_PREFERENCE_DATASET)],
        Annotated[HelpSteer3DatasetItemSchema, Tag(HELPSTEER3_DATASET)],
        Annotated[Tulu3PreferenceDatasetItemSchema, Tag(TULU3_PREFERENCE_DATASET)],
    ],
    Discriminator(get_preference_dataset_discriminator),
]


# =============================================================================
# SFT Dataset Schemas
# =============================================================================
class SFTPromptTemplateDatasetItemSchema(BaseModel):
    """Schema for standard SFT (Supervised Fine-Tuning) dataset format.

    The standard format has prompt and completion fields, but allows additional
    fields for custom templates (e.g., {input}, {output}, {instruction}, etc.).

    Example (standard format):
        {
            "prompt": "What is the capital of France?",
            "completion": "The capital of France is Paris."
        }

    Example (custom template format):
        {
            "instruction": "Answer the question",
            "input": "What is the capital of France?",
            "output": "The capital of France is Paris."
        }
    """

    model_config = ConfigDict(extra="allow")

    # Make all fields optional so custom templates can use any field names
    prompt: Optional[str] = Field(None, description="The input prompt (standard format)")
    completion: Optional[str] = Field(None, description="The expected completion/output (standard format)")


class FunctionCallDetails(BaseModel):
    """Details of a function call made by a tool call.

    Example:
        {
            "name": "get_weather",
            "arguments": {"location": "San Francisco"}
        }
    """

    name: str = Field(..., description="The name of the function to call")
    arguments: dict[str, Any] = Field(..., description="The arguments to pass to the function")
    content_type: Optional[str] = Field(None, description="Optional content type of the function response")


class ToolCall(BaseModel):
    """A tool call in a message."""

    type: Literal["function"] = Field(..., description="The type of tool call (must be 'function')")
    function: FunctionCallDetails = Field(..., description="Function call details including name and arguments")


class SFTChatMessage(BaseModel):
    """A single message in an SFT chat conversation.

    Each message must have a role and at least one of: content, thinking, or tool_calls.

    Important: content and thinking are mutually exclusive within a single message.
    If both are needed, they should be in separate messages (e.g., one message with
    thinking followed by another message with content).
    """

    role: str = Field(..., description="The role of the message sender (e.g., 'user', 'assistant', 'system')")
    content: str | None = Field(None, description="The content of the message")
    thinking: str | None = Field(None, description="Thinking/reasoning content")
    # min_length=1: an empty tool_calls list is non-None but renders a tool-call
    # block the templates never close, producing malformed prompts. A message with
    # no tool calls must use tool_calls=None, not [].
    tool_calls: list[ToolCall] | None = Field(None, min_length=1, description="Tool calls made in this message")

    @staticmethod
    def _schema_extra(schema: dict[str, Any]) -> None:
        """Add anyOf constraint requiring at least one of content, thinking, or tool_calls."""
        schema["anyOf"] = [
            {
                "required": ["content"],
                "properties": {"content": {"type": "string"}},
                "not": {"required": ["thinking"]},
            },
            {
                "required": ["thinking"],
                "properties": {"thinking": {"type": "string"}},
                "not": {"required": ["content"]},
            },
            {"required": ["tool_calls"], "properties": {"tool_calls": {"minItems": 1}}},
        ]

    model_config = ConfigDict(extra="forbid", json_schema_extra=_schema_extra)

    @model_validator(mode="after")
    def check_has_content_or_thinking_or_tool_calls(self) -> "SFTChatMessage":
        """Validate that message has at least one of content, thinking, or tool_calls.

        Also enforces that content and thinking are mutually exclusive - they cannot
        both be present in the same message.
        """
        if self.content is None and self.thinking is None and self.tool_calls is None:
            raise ValueError("Message must have at least one of: content, thinking, or tool_calls")

        if self.content is not None and self.thinking is not None:
            raise ValueError("Message cannot have both content and thinking - they are mutually exclusive")

        return self


class FunctionParameters(BaseModel):
    """Parameters schema for a function definition.

    Example:
        {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city name"}
            }
        }
    """

    type: Literal["object"] = Field(..., description="The type of parameters (must be 'object')")
    properties: dict[str, Any] = Field(..., description="The properties/arguments the function accepts")


class FunctionDefinitionDetails(BaseModel):
    """Details of a function definition for tool calling.

    Example:
        {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {"type": "object", "properties": {...}},
            "required": ["location"]
        }
    """

    name: str = Field(..., description="The name of the function")
    description: str = Field(..., description="A description of what the function does")
    parameters: FunctionParameters = Field(..., description="The parameters schema for the function")
    required: list[str] | None = Field(None, description="List of required parameter names")


class ToolDefinition(BaseModel):
    """A tool definition for function calling."""

    type: Literal["function"] = Field(..., description="The type of tool (must be 'function')")
    function: FunctionDefinitionDetails = Field(
        ..., description="Function definition with name, description, and parameters"
    )


class SFTChatDatasetItemSchema(BaseModel):
    """Schema for SFT chat format based on MESSAGES_SCHEMA.

    This format represents conversations with message lists and optional tool definitions.

    Example:
        {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"}
            ],
            "tools": [...]  # optional
        }
    """

    # min_length=1: templates index messages[0] immediately, so an empty list
    # would pass validation and then crash during rendering.
    messages: list[SFTChatMessage] = Field(..., min_length=1, description="List of messages in the conversation")
    tools: list[ToolDefinition] | None = Field(
        None, description="Optional tool definitions available in the conversation"
    )

    model_config = ConfigDict(extra="allow")


# Embedding Dataset Schemas
class EmbeddingDatasetItemSchema(BaseModel):
    """Schema for embedding dataset format.

    Example:
        {
            "query": "What is machine learning?",
            "pos_doc": "Machine learning is a branch of AI...",
            "neg_doc": ["Deep learning is...", "Neural networks are..."]
        }
    """

    query: str = Field(..., description="The query text")
    pos_doc: str = Field(..., description="The positive document")
    neg_doc: list[str] = Field(..., description="List of negative documents")

    model_config = ConfigDict(extra="allow")


def get_sft_dataset_discriminator(v: Any) -> str:
    """Determine the SFT dataset schema type based on field presence.

    This discriminator examines the fields to determine format:
    - "EmbeddingDatasetItemSchema": Has 'query', 'pos_doc', 'neg_doc' fields (embedding format)
    - "SFTChatDatasetItemSchema": Has 'messages' field (chat format)
    - "SFTPromptTemplateDatasetItemSchema": Has other fields (prompt template format)

    Args:
        v: The data to discriminate (dict or model instance)

    Returns:
        Schema type name identifying the format
    """
    if isinstance(v, dict):
        # Embedding format: has query, pos_doc, neg_doc fields
        if "query" in v and "pos_doc" in v and "neg_doc" in v:
            return "EmbeddingDatasetItemSchema"

        # Chat format: has messages array. The returned tag must match the
        # SFTChatDatasetItemSchema Tag on the union below.
        if "messages" in v:
            return "SFTChatDatasetItemSchema"

        # Prompt template format: has prompt/completion or custom fields
        return "SFTPromptTemplateDatasetItemSchema"

    return "SFTPromptTemplateDatasetItemSchema"  # Default fallback


# Union type for all SFT dataset formats
SFTDatasetSchemaType = Annotated[
    Union[
        Annotated[SFTPromptTemplateDatasetItemSchema, Tag(str(SFTPromptTemplateDatasetItemSchema.__name__))],
        Annotated[SFTChatDatasetItemSchema, Tag(str(SFTChatDatasetItemSchema.__name__))],
        Annotated[EmbeddingDatasetItemSchema, Tag(str(EmbeddingDatasetItemSchema.__name__))],
    ],
    Discriminator(get_sft_dataset_discriminator),
]
