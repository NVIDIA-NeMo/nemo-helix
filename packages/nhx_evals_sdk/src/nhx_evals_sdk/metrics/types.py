# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discriminated unions for SDK metric configuration models."""

from typing import Annotated, TypeAlias

from nhx_evals_sdk.agent_eval.metrics import (
    AgentPhaseSuccessMetric,
    EvidencePresenceMetric,
    SkillUsedMetric,
    ToolArgumentMatchesInputMetric,
    ToolCallCountMetric,
)
from nhx_evals_sdk.metrics.bleu import BLEUMetric
from nhx_evals_sdk.metrics.exact_match import ExactMatchMetric
from nhx_evals_sdk.metrics.f1 import F1Metric
from nhx_evals_sdk.metrics.harbor_skill_metrics import HarborSkillMetrics
from nhx_evals_sdk.metrics.llm_judge import LLMJudgeMetric
from nhx_evals_sdk.metrics.number_check import NumberCheckMetric
from nhx_evals_sdk.metrics.ragas.metrics import (
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
from nhx_evals_sdk.metrics.remote import NemoAgentToolkitRemoteMetric, RemoteMetric
from nhx_evals_sdk.metrics.retrieval import (
    RetrievalMAPMetric,
    RetrievalNDCGMetric,
    RetrievalPrecisionMetric,
    RetrievalRecallMetric,
)
from nhx_evals_sdk.metrics.rouge import ROUGEMetric
from nhx_evals_sdk.metrics.runner_rewards import GymRewardMetric, HarborRewardMetric
from nhx_evals_sdk.metrics.sequence_similarity import SequenceSimilarityMetric
from nhx_evals_sdk.metrics.skill_readiness import (
    SkillDescriptionQualityMetric,
    SkillReliabilityReadinessMetric,
    SkillScriptQualityMetric,
    SkillStaticCorrectnessMetric,
    SkillUnicodeSafetyMetric,
)
from nhx_evals_sdk.metrics.skill_scans import (
    AgentSafetyMetric,
    SkillBundledTestMetric,
    SkillCodeRiskMetric,
    SkillDependencyComplianceMetric,
    SkillLicenseMetric,
    SkillPackageConfigurationRiskMetric,
    SkillSecurityScanMetric,
    SkillSensitiveDataMetric,
)
from nhx_evals_sdk.metrics.skill_traces import (
    ComponentCoordinationMetric,
    ComponentUsageMetric,
    ErrorRecoveryMetric,
    ExpectedScriptInvocationMetric,
    SkillOveractivationMetric,
    SkillRoutingMetric,
    ToolArgumentComplianceMetric,
    ToolExecutionOutcomeMetric,
    ToolProductivityMetric,
    ToolSelectionMetric,
    WorkflowComplianceMetric,
)
from nhx_evals_sdk.metrics.skills import (
    SkillActivationMetric,
    SkillContextEfficiencyMetric,
    SkillReferenceIntegrityMetric,
    SkillStructureMetric,
    TokenEfficiencyMetric,
)
from nhx_evals_sdk.metrics.string_check import StringCheckMetric
from nhx_evals_sdk.metrics.tool_calling import ToolCallingMetric
from nhx_evals_sdk.metrics.tunable_rag_evaluator import TunableRagEvaluatorMetric
from pydantic import Field

MetricVariants: TypeAlias = (
    BLEUMetric
    | ExactMatchMetric
    | F1Metric
    | LLMJudgeMetric
    | HarborSkillMetrics
    | NumberCheckMetric
    | RemoteMetric
    | NemoAgentToolkitRemoteMetric
    | RetrievalNDCGMetric
    | RetrievalRecallMetric
    | RetrievalPrecisionMetric
    | RetrievalMAPMetric
    | ROUGEMetric
    | SequenceSimilarityMetric
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
    | SkillActivationMetric
    | SkillContextEfficiencyMetric
    | SkillReferenceIntegrityMetric
    | SkillStructureMetric
    | TokenEfficiencyMetric
    | SkillStaticCorrectnessMetric
    | SkillDescriptionQualityMetric
    | SkillReliabilityReadinessMetric
    | SkillScriptQualityMetric
    | SkillUnicodeSafetyMetric
    | SkillSecurityScanMetric
    | SkillSensitiveDataMetric
    | SkillCodeRiskMetric
    | SkillLicenseMetric
    | SkillDependencyComplianceMetric
    | SkillPackageConfigurationRiskMetric
    | AgentSafetyMetric
    | SkillBundledTestMetric
    | SkillOveractivationMetric
    | SkillRoutingMetric
    | ExpectedScriptInvocationMetric
    | WorkflowComplianceMetric
    | ErrorRecoveryMetric
    | ToolProductivityMetric
    | ToolSelectionMetric
    | ToolArgumentComplianceMetric
    | ToolExecutionOutcomeMetric
    | ComponentCoordinationMetric
    | ComponentUsageMetric
)
"""Raw union of SDK metric configuration models, excluding service-only system metrics."""

MetricsUnion: TypeAlias = Annotated[MetricVariants, Field(discriminator="type")]
"""Discriminated union of SDK metric configuration models."""

__all__ = ["MetricVariants", "MetricsUnion"]
