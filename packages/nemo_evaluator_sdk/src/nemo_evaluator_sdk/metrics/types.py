# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discriminated unions for SDK metric configuration models."""

from typing import Annotated, TypeAlias

from nemo_evaluator_sdk.agent_eval.metrics import (
    AgentPhaseSuccessMetric,
    EvidencePresenceMetric,
    SkillUsedMetric,
    ToolArgumentMatchesInputMetric,
    ToolCallCountMetric,
)
from nemo_evaluator_sdk.metrics.bleu import BLEUMetric
from nemo_evaluator_sdk.metrics.exact_match import ExactMatchMetric
from nemo_evaluator_sdk.metrics.f1 import F1Metric
from nemo_evaluator_sdk.metrics.llm_judge import LLMJudgeMetric
from nemo_evaluator_sdk.metrics.number_check import NumberCheckMetric
from nemo_evaluator_sdk.metrics.ragas.metrics import (
    AgentGoalAccuracyMetric,
    AnswerAccuracyMetric,
    ContextEntityRecallMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
    ContextRelevanceMetric,
    FaithfulnessMetric,
    NoiseSensitivityMetric,
    ResponseGroundednessMetric,
    ResponseRelevancyMetric,
    ToolCallAccuracyMetric,
    TopicAdherenceMetric,
)
from nemo_evaluator_sdk.metrics.remote import NemoAgentToolkitRemoteMetric, RemoteMetric
from nemo_evaluator_sdk.metrics.retrieval import (
    RetrievalMAPMetric,
    RetrievalNDCGMetric,
    RetrievalPrecisionMetric,
    RetrievalRecallMetric,
)
from nemo_evaluator_sdk.metrics.rouge import ROUGEMetric
from nemo_evaluator_sdk.metrics.runner_rewards import GymRewardMetric, HarborRewardMetric
from nemo_evaluator_sdk.metrics.string_check import StringCheckMetric
from nemo_evaluator_sdk.metrics.tool_calling import ToolCallingMetric
from nemo_evaluator_sdk.metrics.tunable_rag_evaluator import TunableRagEvaluatorMetric
from pydantic import Field

MetricVariants: TypeAlias = (
    BLEUMetric
    | ExactMatchMetric
    | F1Metric
    | LLMJudgeMetric
    | NumberCheckMetric
    | RemoteMetric
    | NemoAgentToolkitRemoteMetric
    | RetrievalNDCGMetric
    | RetrievalRecallMetric
    | RetrievalPrecisionMetric
    | RetrievalMAPMetric
    | ROUGEMetric
    | StringCheckMetric
    | ToolCallingMetric
    | TunableRagEvaluatorMetric
    | TopicAdherenceMetric
    | ToolCallAccuracyMetric
    | AgentGoalAccuracyMetric
    | AnswerAccuracyMetric
    | ContextRelevanceMetric
    | ResponseGroundednessMetric
    | ContextRecallMetric
    | ContextPrecisionMetric
    | ContextEntityRecallMetric
    | ResponseRelevancyMetric
    | FaithfulnessMetric
    | NoiseSensitivityMetric
    | GymRewardMetric
    | HarborRewardMetric
    | AgentPhaseSuccessMetric
    | EvidencePresenceMetric
    | SkillUsedMetric
    | ToolCallCountMetric
    | ToolArgumentMatchesInputMetric
)
"""Raw union of SDK metric configuration models, excluding service-only system metrics."""

MetricsUnion: TypeAlias = Annotated[MetricVariants, Field(discriminator="type")]
"""Discriminated union of SDK metric configuration models."""

__all__ = ["MetricVariants", "MetricsUnion"]
