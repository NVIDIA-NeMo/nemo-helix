# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metric types for evals SDK.

These types contain all metric configuration fields but do not require
workspace/name (they do not inherit from EntityBase). They can be used
directly for inline metric definitions in API requests.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from nhx_evals_sdk.dataset_schemas.common import empty_object_schema
from nhx_evals_sdk.dataset_schemas.compatibility import merge_metric_required_schemas
from nhx_evals_sdk.dataset_schemas.templates import infer_required_schema_from_template
from nhx_evals_sdk.enums import MetricType
from nhx_evals_sdk.values.common import SecretRef, SupportedJobTypes
from nhx_evals_sdk.values.dataset_schemas import InputSchema
from nhx_evals_sdk.values.llm_judge_defaults import (
    LLM_JUDGE_SCORES_CONTEXT_KEY,
    default_judge_prompt_template_for_model,
)
from nhx_evals_sdk.values.llm_judge_defaults import (
    default_judge_prompt_template_chat as default_judge_prompt_template_chat,
)
from nhx_evals_sdk.values.llm_judge_defaults import (
    default_judge_prompt_template_completions as default_judge_prompt_template_completions,
)
from nhx_evals_sdk.values.llm_judge_defaults import (
    is_chat_inference as is_chat_inference,
)
from nhx_evals_sdk.values.models import Model, ModelRef, ReasoningParams
from nhx_evals_sdk.values.params import InferenceParams
from nhx_evals_sdk.values.protocol import MetricTypeName
from nhx_evals_sdk.values.scores import RemoteScore, Score

# =============================================================================
# Prompt Template Constants and Helpers
# =============================================================================

# TODO: Align optional_fields with template path semantics.
# Keep support for dataset-relative nested paths (for example "reference.text")
# and runtime sample paths (for example "sample.output_text"), while avoiding
# dependence on the "item." alias form (normalize "item.foo" -> "foo").
OptionalFieldName = Annotated[str, Field(min_length=1)]


def _input_schema_from_template(
    template: str | dict | list,
    *,
    ignored_roots: set[str] | None = None,
    optional_fields: set[str] | None = None,
) -> InputSchema:
    return InputSchema(
        schema=infer_required_schema_from_template(
            template,
            ignored_roots=ignored_roots,
            optional_fields=optional_fields,
        )
    )


def _input_schema_from_templates(templates: Iterable[str | dict | list]) -> InputSchema:
    schemas = [infer_required_schema_from_template(template) for template in templates]
    if not schemas:
        return InputSchema(schema=empty_object_schema())

    merged_schema = merge_metric_required_schemas((f"template_{index}", schema) for index, schema in enumerate(schemas))
    return InputSchema(schema=merged_schema)


# =============================================================================
# Base Metric Type
# =============================================================================


class MetricBase(BaseModel):
    """Base class for inline metrics.

    Contains common fields shared by all metric types.
    """

    __entity_type__: ClassVar[str] = "metric"

    type: MetricTypeName = Field(description="The type of metric. Used as a discriminator for the metric type.")
    description: str | None = Field(default=None, description="Human-readable description of the metric.")
    labels: dict[str, str] = Field(
        default_factory=dict, description="Labels are key-value pairs that can be used for grouping and filtering."
    )
    supported_job_types: list[Literal[SupportedJobTypes.ONLINE, SupportedJobTypes.OFFLINE]] = Field(
        default=[SupportedJobTypes.ONLINE, SupportedJobTypes.OFFLINE],
        description="A metric can evaluate model outputs for online evaluations or pre-generated outputs for offline evaluations.",
    )

    def model_post_init(self, __context: Any) -> None:
        """Mark ``type`` as set so it is included when exclude_unset=True."""
        self.__pydantic_fields_set__.add("type")

    def input_schema(self) -> InputSchema:
        """Return the canonical evaluator input schema required by this metric."""
        return InputSchema(schema=empty_object_schema())


# =============================================================================
# Metric Types
# =============================================================================


class BLEU(MetricBase):
    """BLEU metric configuration."""

    type: Literal[MetricType.BLEU] = MetricType.BLEU
    references: list[str] = Field(
        description="The templates for the ground truth references to calculate BLEU metric with."
    )
    candidate: str | None = Field(
        default=None,
        description="The template for the candidate to calculate BLEU metric on. If not provided, the output text from the model is used.",
    )

    def input_schema(self) -> InputSchema:
        templates: list[str] = [*self.references]
        if self.candidate is not None:
            templates.append(self.candidate)
        return _input_schema_from_templates(templates)


class ExactMatch(MetricBase):
    """Exact Match metric configuration."""

    type: Literal[MetricType.EXACT_MATCH] = MetricType.EXACT_MATCH
    reference: str = Field(
        description="The template for the ground truth reference to calculate the exact match metric with.",
    )
    candidate: str | None = Field(
        default=None,
        description="The template for the candidate to evaluate the exact match metric on. If not provided, the output text from the model is used.",
    )

    def input_schema(self) -> InputSchema:
        templates = [self.reference]
        if self.candidate is not None:
            templates.append(self.candidate)
        return _input_schema_from_templates(templates)


class F1(MetricBase):
    """F1 metric configuration."""

    type: Literal[MetricType.F1] = MetricType.F1
    reference: str = Field(description="The template for the ground truth reference to calculate the F1 metric with.")
    candidate: str | None = Field(
        default=None,
        description="The template for the candidate to evaluate the F1 metric on. If not provided, the output text from the model is used.",
    )

    def input_schema(self) -> InputSchema:
        templates = [self.reference]
        if self.candidate is not None:
            templates.append(self.candidate)
        return _input_schema_from_templates(templates)


class RetrievalNDCG(MetricBase):
    """Normalized discounted cumulative gain over BEIR retrieval results."""

    type: Literal[MetricType.RETRIEVAL_NDCG] = MetricType.RETRIEVAL_NDCG
    k: list[int] = Field(default=[1, 5, 10, 100], min_length=1)

    @model_validator(mode="after")
    def validate_k(self) -> Self:
        """Require unique, positive retrieval cutoffs."""
        if any(cutoff < 1 for cutoff in self.k):
            raise ValueError("retrieval cutoffs must be positive")
        if len(set(self.k)) != len(self.k):
            raise ValueError("retrieval cutoffs must be unique")
        self.k.sort()
        return self


class RetrievalRecall(MetricBase):
    """Recall over BEIR retrieval results."""

    type: Literal[MetricType.RETRIEVAL_RECALL] = MetricType.RETRIEVAL_RECALL
    k: list[int] = Field(default=[1, 5, 10, 100], min_length=1)

    @model_validator(mode="after")
    def validate_k(self) -> Self:
        """Require unique, positive retrieval cutoffs."""
        if any(cutoff < 1 for cutoff in self.k):
            raise ValueError("retrieval cutoffs must be positive")
        if len(set(self.k)) != len(self.k):
            raise ValueError("retrieval cutoffs must be unique")
        self.k.sort()
        return self


class RetrievalPrecision(MetricBase):
    """Precision over BEIR retrieval results."""

    type: Literal[MetricType.RETRIEVAL_PRECISION] = MetricType.RETRIEVAL_PRECISION
    k: list[int] = Field(default=[1, 5, 10, 100], min_length=1)

    @model_validator(mode="after")
    def validate_k(self) -> Self:
        """Require unique, positive retrieval cutoffs."""
        if any(cutoff < 1 for cutoff in self.k):
            raise ValueError("retrieval cutoffs must be positive")
        if len(set(self.k)) != len(self.k):
            raise ValueError("retrieval cutoffs must be unique")
        self.k.sort()
        return self


class RetrievalMAP(MetricBase):
    """Mean average precision over BEIR retrieval results."""

    type: Literal[MetricType.RETRIEVAL_MAP] = MetricType.RETRIEVAL_MAP
    k: list[int] = Field(default=[1, 5, 10, 100], min_length=1)

    @model_validator(mode="after")
    def validate_k(self) -> Self:
        """Require unique, positive retrieval cutoffs."""
        if any(cutoff < 1 for cutoff in self.k):
            raise ValueError("retrieval cutoffs must be positive")
        if len(set(self.k)) != len(self.k):
            raise ValueError("retrieval cutoffs must be unique")
        self.k.sort()
        return self


class LLMJudge(MetricBase):
    """LLM-as-a-Judge metric configuration."""

    type: Literal[MetricType.LLM_JUDGE] = MetricType.LLM_JUDGE
    model: Model | ModelRef = Field(
        description="The judge model to use for the metric.",
        examples=[
            {
                "endpoint": "https://api.openai.com/v1",
                "name": "gpt-4o",
                "api_key_secret": "secret/my_openai_api_key",
            }
        ],
    )
    scores: list[Score] = Field(
        description="Definitions of scores that will be extracted from the judge's output.", min_length=1
    )
    prompt_template: str | dict | None = Field(
        default=None,
        description="The prompt template for the judge. Can be either a simple string or a structured object (e.g., OpenAI messages format). Use Jinja template variables like {{sample.output_text}} to use the model output within the template or {{item.xxx}} to reference input columns from the dataset.",
        examples=[
            {"type": "string", "content": "You are an expert judge evaluating the correctness of AI responses."},
            {
                "type": "object",
                "content": {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert judge evaluating the correctness of AI responses.",
                        },
                        {
                            "role": "user",
                            "content": "Question: {{item.prompt}}\nAnswer: {{sample.output_text}}\nReference: {{item.reference}}\nRate the correctness from 1-5.",
                        },
                    ]
                },
            },
        ],
    )
    optional_fields: list[OptionalFieldName] = Field(
        default_factory=list,
        description=(
            "Prompt template fields that should remain in the inferred input schema but not be required. "
            "Use this for fields like 'reference' when the metric can still run without them."
        ),
        examples=[["reference"]],
    )
    structured_output: dict | None = Field(
        default=None,
        description="JSON schema to apply structured output for the judge model evaluation. Structured output is derived from scores when omitted. Use this option if there are custom requirements for the output of the judge.",
    )
    inference: InferenceParams | None = Field(default=None, description="Inference parameters for the judge model.")
    system_prompt: str | None = Field(
        default=None,
        description="Initial instructions that define the judge model's role and behavior for the conversation. "
        "This is prepended to the messages as a system message.",
    )
    reasoning: ReasoningParams | None = Field(
        default=None,
        description="Custom settings that control the judge model's reasoning behavior. "
        "For reasoning models (e.g., Nemotron), use `end_token` to strip reasoning traces from the output.",
    )
    ignore_request_failure: bool = Field(
        default=False,
        description="If True, request failures will be ignored and the result will be marked as NaN. "
        "If False (default), request failures will raise an exception.",
    )

    @model_validator(mode="after")
    def unique_scores(self) -> Self:
        if not self.scores:
            return self

        scores = {score.name for score in self.scores}
        if len(scores) != len(self.scores):
            raise ValueError("score names must be unique")
        return self

    @model_validator(mode="after")
    def reject_reserved_prompt_template_keys(self) -> Self:
        """Fail fast on misplaced evaluator controls in `prompt_template`."""
        if self.prompt_template is None or not isinstance(self.prompt_template, dict):
            return self

        reserved_keys = {"system_prompt", "reasoning"}
        found_keys = sorted(reserved_keys.intersection(self.prompt_template.keys()))
        if found_keys:
            keys_str = ", ".join(found_keys)
            raise ValueError(
                f"prompt_template cannot include {keys_str}. "
                "Use top-level fields 'system_prompt' and 'reasoning' instead."
            )
        return self

    def input_schema(self) -> InputSchema:
        job_type = getattr(self, "job_type", SupportedJobTypes.ONLINE)
        prompt_template = (
            self.prompt_template
            if self.prompt_template is not None
            else default_judge_prompt_template_for_model(self.model, job_type)
        )
        return _input_schema_from_template(
            prompt_template,
            ignored_roots={LLM_JUDGE_SCORES_CONTEXT_KEY},
            optional_fields=set(self.optional_fields),
        )


NumberCheckOperation = Literal[
    "equals",
    "==",
    "!=",
    "<>",
    "not equals",
    ">=",
    "gte",
    "greater than or equal",
    ">",
    "gt",
    "greater than",
    "<=",
    "lte",
    "less than or equal",
    "<",
    "lt",
    "less than",
    "absolute difference",
]


class NumberCheck(MetricBase):
    """Number check metric configuration."""

    type: Literal[MetricType.NUMBER_CHECK] = MetricType.NUMBER_CHECK
    operation: NumberCheckOperation = Field(description="The operation to compute for the metric.")
    left_template: str = Field(
        description="The template to use for rendering the left value of the operator to compute the metric.",
        examples=["{{item.dataset_column_name}}"],
    )
    right_template: str = Field(
        description="The template to use for rendering the right value of the operator to compute the metric.",
        examples=["{{sample.output_text}}"],
    )
    epsilon: int | float | None = Field(
        default=None, description="Specify the tolerance for the absolute difference of values."
    )

    @model_validator(mode="after")
    def absolute_difference(self) -> Self:
        if self.operation == "absolute difference":
            if self.epsilon is None:
                raise ValueError(f"epsilon value is required with operation {self.operation}")
        elif self.epsilon:
            raise ValueError(f"epsilon value can only be used with absolute difference operation: {self.operation}")
        return self

    def input_schema(self) -> InputSchema:
        return _input_schema_from_templates([self.left_template, self.right_template])


class _RemoteBase(MetricBase):
    url: str = Field(description="The URL of the remote endpoint.")
    api_key_secret: SecretRef | None = Field(
        default=None,
        description="Optional secret reference of an API key for authentication. Format: workspace/secret_name or secret_name within the job workspace.",
    )
    timeout_seconds: float = Field(default=30.0, description="Request timeout in seconds.")
    max_retries: int = Field(default=3, description="Maximum number of retry attempts.")


class Remote(_RemoteBase):
    """Remote metric configuration."""

    type: Literal[MetricType.REMOTE] = MetricType.REMOTE
    body: dict[str, Any] = Field(description="Jinja template for request payload")
    scores: list[RemoteScore] = Field(description="List of scores to extract from the remote response")

    def input_schema(self) -> InputSchema:
        # NAT evaluators currently receive the entire dataset row as an opaque `item`
        # payload, and this metric config does not carry a machine-readable schema for
        # what that evaluator expects. Until we can discover or declare the NAT input
        # contract (for example from evaluator metadata), treat the accepted input as an
        # unconstrained object rather than over-specifying required fields here.
        return InputSchema(schema=empty_object_schema())


class NemoAgentToolkitRemote(_RemoteBase):
    """NeMo Agent Toolkit Remote metric configuration."""

    type: Literal[MetricType.NEMO_AGENT_TOOLKIT_REMOTE] = MetricType.NEMO_AGENT_TOOLKIT_REMOTE
    evaluator_name: str = Field(description="The name of the evaluator (also used as the score name).")

    def input_schema(self) -> InputSchema:
        # NAT evaluators currently receive the entire dataset row as an opaque `item`
        # payload, and this metric config does not carry a machine-readable schema for
        # what that evaluator expects. Until we can discover or declare the NAT input
        # contract (for example from evaluator metadata), treat the accepted input as an
        # unconstrained object rather than over-specifying required fields here.
        return InputSchema(schema=empty_object_schema())


class ROUGE(MetricBase):
    """ROUGE metric configuration."""

    type: Literal[MetricType.ROUGE] = MetricType.ROUGE
    reference: str = Field(description="The template for the ground truth reference to evaluate the ROUGE metric with.")
    candidate: str | None = Field(
        default=None,
        description="The template for the candidate to evaluate the ROUGE metric on. If not provided, the output text from the model is used.",
    )

    def input_schema(self) -> InputSchema:
        templates = [self.reference]
        if self.candidate is not None:
            templates.append(self.candidate)
        return _input_schema_from_templates(templates)


StringCheckOperation = Literal[
    "equals",
    "==",
    "!=",
    "<>",
    "not equals",
    "contains",
    "not contains",
    "startswith",
    "endswith",
]


class StringCheck(MetricBase):
    """String check metric configuration."""

    type: Literal[MetricType.STRING_CHECK] = MetricType.STRING_CHECK
    operation: StringCheckOperation = Field(description="The operation to compute for the metric.")
    left_template: str = Field(
        description="The template to use for rendering the left value of the operator to compute the metric.",
        examples=["{{item.dataset_column_name}}"],
    )
    right_template: str = Field(
        description="The template to use for rendering the right value of the operator to compute the metric.",
        examples=["{{sample.output_text | trim}}"],
    )

    def input_schema(self) -> InputSchema:
        return _input_schema_from_templates([self.left_template, self.right_template])


class ToolCalling(MetricBase):
    """Tool Calling metric configuration."""

    model_config = ConfigDict(validate_assignment=True)

    type: Literal[MetricType.TOOL_CALLING] = MetricType.TOOL_CALLING
    reference: str = Field(description="The template for the ground truth reference to evaluate tool calling accuracy.")

    def input_schema(self) -> InputSchema:
        return _input_schema_from_template(self.reference)


# =============================================================================
# Inline RAGAS Metric Types
# =============================================================================


class _RAGASJudgeConfig(BaseModel):
    """Configuration for the LLM judge used by RAGAS metrics."""

    judge_model: Model | ModelRef = Field(description="The LLM model to use as judge.")
    inference: InferenceParams = Field(
        default_factory=InferenceParams, description="Inference parameters for the judge."
    )
    ignore_request_failure: bool = Field(
        default=False,
        description="If True, request failures to the judge model are ignored and the metric result "
        "is marked as NaN. Parse/output formatting failures are always converted to NaN.",
    )


class _RAGASEmbeddingsConfig(BaseModel):
    """Configuration for embeddings used by RAGAS metrics."""

    embeddings_model: Model | ModelRef = Field(description="The embeddings model to use.")


class _RAGASBase(MetricBase):
    """Base class for inline RAGAS metrics."""

    input_template: dict[str, Any] | None = Field(
        default=None,
        description="Optional Jinja template for rendering the input payload for RAGAS evaluation.",
    )

    def input_schema(self) -> InputSchema:
        if self.input_template is None:
            return InputSchema(schema=empty_object_schema())
        return _input_schema_from_template(self.input_template)


class TopicAdherence(_RAGASBase, _RAGASJudgeConfig):
    """RAGAS metric for measuring topic adherence."""

    type: Literal[MetricType.TOPIC_ADHERENCE] = MetricType.TOPIC_ADHERENCE
    metric_mode: Literal["f1", "precision", "recall"] = Field(
        default="f1", description="The mode for computing topic adherence score."
    )


class ToolCallAccuracy(_RAGASBase):
    """RAGAS metric for measuring tool call accuracy."""

    type: Literal[MetricType.TOOL_CALL_ACCURACY] = MetricType.TOOL_CALL_ACCURACY


class AgentGoalAccuracy(_RAGASBase, _RAGASJudgeConfig):
    """RAGAS metric for measuring agent goal accuracy."""

    type: Literal[MetricType.AGENT_GOAL_ACCURACY] = MetricType.AGENT_GOAL_ACCURACY
    use_reference: bool = Field(default=True, description="Whether to use reference for goal accuracy evaluation.")


class AnswerAccuracy(_RAGASBase, _RAGASJudgeConfig):
    """RAGAS metric for measuring answer accuracy."""

    type: Literal[MetricType.ANSWER_ACCURACY] = MetricType.ANSWER_ACCURACY


class ContextRelevance(_RAGASBase, _RAGASJudgeConfig):
    """RAGAS metric for measuring context relevance."""

    type: Literal[MetricType.CONTEXT_RELEVANCE] = MetricType.CONTEXT_RELEVANCE


class ResponseGroundedness(_RAGASBase, _RAGASJudgeConfig):
    """RAGAS metric for measuring response groundedness."""

    type: Literal[MetricType.RESPONSE_GROUNDEDNESS] = MetricType.RESPONSE_GROUNDEDNESS


class ContextRecall(_RAGASBase, _RAGASJudgeConfig):
    """RAGAS metric for measuring context recall."""

    type: Literal[MetricType.CONTEXT_RECALL] = MetricType.CONTEXT_RECALL


class ContextPrecision(_RAGASBase, _RAGASJudgeConfig):
    """RAGAS metric for measuring context precision."""

    type: Literal[MetricType.CONTEXT_PRECISION] = MetricType.CONTEXT_PRECISION


class ContextEntityRecall(_RAGASBase, _RAGASJudgeConfig):
    """RAGAS metric for measuring context entity recall."""

    type: Literal[MetricType.CONTEXT_ENTITY_RECALL] = MetricType.CONTEXT_ENTITY_RECALL


class ResponseRelevancy(_RAGASBase, _RAGASJudgeConfig, _RAGASEmbeddingsConfig):
    """RAGAS metric for measuring response relevancy."""

    type: Literal[MetricType.RESPONSE_RELEVANCY] = MetricType.RESPONSE_RELEVANCY
    strictness: int = Field(
        default=1,
        description="Number of parallel questions generated. NIM can only generate 1.",
    )


class Faithfulness(_RAGASBase, _RAGASJudgeConfig):
    """RAGAS metric for measuring faithfulness."""

    type: Literal[MetricType.FAITHFULNESS] = MetricType.FAITHFULNESS


class NoiseSensitivity(_RAGASBase, _RAGASJudgeConfig):
    """RAGAS metric for measuring noise sensitivity."""

    type: Literal[MetricType.NOISE_SENSITIVITY] = MetricType.NOISE_SENSITIVITY


class TunableRagEvaluator(MetricBase):
    """Tunable RAG evaluator with customizable judge prompt and weighted sub-scores."""

    type: Literal[MetricType.TUNABLE_RAG_EVALUATOR] = MetricType.TUNABLE_RAG_EVALUATOR
    model: Model | ModelRef = Field(description="Judge model used to score generated answers.")
    judge_llm_prompt: str = Field(
        default="",
        description="Optional custom judge rubric. Ignored when default_scoring is true except as extra context.",
    )
    default_scoring: bool = Field(
        default=True,
        description="Use built-in coverage/correctness/relevance rubric and weighted composite.",
    )
    default_score_weights: dict[str, float] = Field(
        default_factory=lambda: {"coverage": 0.5, "correctness": 0.3, "relevance": 0.2},
        description="Weights for coverage/correctness/relevance when default_scoring is true.",
    )
    inference: InferenceParams | None = Field(
        default=None,
        description="Optional inference parameters for the judge model.",
    )

    def input_schema(self) -> InputSchema:
        return InputSchema(
            schema={
                "type": "object",
                "properties": {
                    "inputs": {
                        "type": "object",
                        "properties": {"instruction": {"type": "string"}},
                    },
                    "reference": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                    },
                },
            }
        )
