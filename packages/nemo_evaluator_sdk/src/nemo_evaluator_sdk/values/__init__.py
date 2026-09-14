# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public value types for evaluator SDK runtime.

The public interface resolves lazily (PEP 562), for the same reason the package root does:
every ``from nemo_evaluator_sdk.values.X import ...`` runs this barrel first, so
eagerly re-exporting all 97 names dragged ``.datasets``/``.results`` (pyarrow, numpy) and
``.metrics``/``.scores`` (jsonschema, jinja2) into ``agent_eval``, which uses none of them.
Measured: 485 modules and +57 MB RSS for ``import agent_eval.runtimes.harbor_runtime`` before,
300 modules and pydantic alone after.

Add a new re-export to ``_LAZY_ATTRS``, the ``TYPE_CHECKING`` block and ``__all__`` — never as a
module-level import. ``tests/test_lazy_public_api.py`` locks the boundary in.
"""

# ruff: noqa: I001 - the vendored SDK mirror uses different import-order settings.

from importlib import import_module as _import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_evaluator_sdk.values.agents import (
        Agent,
        AgentBase,
        GenericAgent,
        NatAgentConfig,
        NemoAgentToolkitAgent,
    )
    from nemo_evaluator_sdk.values.atif import (
        FinalMetrics,
        Metrics,
        Observation,
        ObservationResult,
        Step,
        ToolCall,
        Trajectory,
    )
    from nemo_evaluator_sdk.values.common import SecretRef, SupportedJobTypes
    from nemo_evaluator_sdk.values.dataset_args import DatasetArg
    from nemo_evaluator_sdk.values.dataset_schemas import (
        FieldMapping,
        InputSchema,
    )
    from nemo_evaluator_sdk.values.datasets import DatasetInput, DatasetRows
    from nemo_evaluator_sdk.values.evidence import (
        ATIFTraceHandle,
        CandidateEvidence,
        CommandResult,
        EvidenceDescriptor,
        FilesystemDiff,
        FilesystemEntry,
        LocalFilesystemEvidence,
        LogHandle,
        OTLPTraceHandle,
        TraceHandle,
        WellKnownEvidenceKey,
        parse_atif,
    )
    from nemo_evaluator_sdk.values.metrics import (
        BLEU,
        F1,
        ROUGE,
        AgentGoalAccuracy,
        AnswerAccuracy,
        ContextEntityRecall,
        ContextPrecision,
        ContextRecall,
        ContextRelevance,
        ExactMatch,
        Faithfulness,
        LLMJudge,
        MetricBase,
        NemoAgentToolkitRemote,
        NoiseSensitivity,
        NumberCheck,
        Remote,
        ResponseGroundedness,
        ResponseRelevancy,
        StringCheck,
        ToolCallAccuracy,
        ToolCalling,
        TopicAdherence,
        TunableRagEvaluator,
    )
    from nemo_evaluator_sdk.values.models import (
        InferenceConfig,
        Model,
        ModelRef,
        RankingInference,
        ReasoningParams,
    )
    from nemo_evaluator_sdk.values.retrieval import Retrieval
    from nemo_evaluator_sdk.values.targets import EvalTarget
    from nemo_evaluator_sdk.values.multi_metric_results import BenchmarkEvaluationResult
    from nemo_evaluator_sdk.values.params import (
        InferenceParams,
        RunConfig,
        RunConfigOnline,
        RunConfigOnlineModel,
    )
    from nemo_evaluator_sdk.values.protocol import (
        BooleanValue,
        CandidateOutput,
        ContinuousScore,
        DatasetRow,
        DiscreteScore,
        Label,
        MetricDescriptor,
        MetricDiagnostic,
        MetricInput,
        MetricOutput,
        MetricOutputSpec,
        MetricResult,
        MetricTypeName,
    )
    from nemo_evaluator_sdk.values.results import (
        AggregatedMetricResult,
        AggregateFieldName,
        AggregateRangeScore,
        AggregateRubricScore,
        AggregateScore,
        AggregateScoreBase,
        DefaultAggregateFieldName,
        EvaluationResult,
        Histogram,
        HistogramBin,
        MetricScore,
        Percentiles,
        RowScore,
        RubricScoreStat,
        RubricScoreValue,
        SampleResult,
        ScoreStats,
    )
    from nemo_evaluator_sdk.values.scores import (
        JSONScoreParser,
        RangeScore,
        RegexScoreParser,
        RemoteScore,
        Rubric,
        RubricScore,
        Score,
        score_discriminator,
    )


# Re-exported name -> the submodule that defines it, relative to this package. Relative on
# purpose: the vendoring tool mirrors this file into nemo_platform.beta.evaluator by rewriting
# module paths, and a relative name has nothing to rewrite, so the mirror is correct by
# construction. Mirrors the TYPE_CHECKING block above, in the same order.
_LAZY_ATTRS: dict[str, str] = {
    "Agent": ".agents",
    "AgentBase": ".agents",
    "GenericAgent": ".agents",
    "NatAgentConfig": ".agents",
    "NemoAgentToolkitAgent": ".agents",
    "FinalMetrics": ".atif",
    "Metrics": ".atif",
    "Observation": ".atif",
    "ObservationResult": ".atif",
    "Step": ".atif",
    "ToolCall": ".atif",
    "Trajectory": ".atif",
    "SecretRef": ".common",
    "SupportedJobTypes": ".common",
    "FieldMapping": ".dataset_schemas",
    "InputSchema": ".dataset_schemas",
    "DatasetArg": ".dataset_args",
    "DatasetInput": ".datasets",
    "DatasetRows": ".datasets",
    "ATIFTraceHandle": ".evidence",
    "CandidateEvidence": ".evidence",
    "CommandResult": ".evidence",
    "EvidenceDescriptor": ".evidence",
    "FilesystemDiff": ".evidence",
    "FilesystemEntry": ".evidence",
    "LocalFilesystemEvidence": ".evidence",
    "LogHandle": ".evidence",
    "OTLPTraceHandle": ".evidence",
    "TraceHandle": ".evidence",
    "WellKnownEvidenceKey": ".evidence",
    "parse_atif": ".evidence",
    "BLEU": ".metrics",
    "F1": ".metrics",
    "ROUGE": ".metrics",
    "AgentGoalAccuracy": ".metrics",
    "AnswerAccuracy": ".metrics",
    "ContextEntityRecall": ".metrics",
    "ContextPrecision": ".metrics",
    "ContextRecall": ".metrics",
    "ContextRelevance": ".metrics",
    "ExactMatch": ".metrics",
    "Faithfulness": ".metrics",
    "LLMJudge": ".metrics",
    "MetricBase": ".metrics",
    "NemoAgentToolkitRemote": ".metrics",
    "NoiseSensitivity": ".metrics",
    "NumberCheck": ".metrics",
    "Remote": ".metrics",
    "ResponseGroundedness": ".metrics",
    "ResponseRelevancy": ".metrics",
    "StringCheck": ".metrics",
    "ToolCallAccuracy": ".metrics",
    "ToolCalling": ".metrics",
    "TopicAdherence": ".metrics",
    "TunableRagEvaluator": ".metrics",
    "Model": ".models",
    "ModelRef": ".models",
    "InferenceConfig": ".models",
    "RankingInference": ".models",
    "ReasoningParams": ".models",
    "Retrieval": ".retrieval",
    "EvalTarget": ".targets",
    "BenchmarkEvaluationResult": ".multi_metric_results",
    "InferenceParams": ".params",
    "RunConfig": ".params",
    "RunConfigOnline": ".params",
    "RunConfigOnlineModel": ".params",
    "BooleanValue": ".protocol",
    "CandidateOutput": ".protocol",
    "ContinuousScore": ".protocol",
    "DatasetRow": ".protocol",
    "DiscreteScore": ".protocol",
    "Label": ".protocol",
    "MetricDescriptor": ".protocol",
    "MetricDiagnostic": ".protocol",
    "MetricInput": ".protocol",
    "MetricOutput": ".protocol",
    "MetricOutputSpec": ".protocol",
    "MetricResult": ".protocol",
    "MetricTypeName": ".protocol",
    "AggregatedMetricResult": ".results",
    "AggregateFieldName": ".results",
    "AggregateRangeScore": ".results",
    "AggregateRubricScore": ".results",
    "AggregateScore": ".results",
    "AggregateScoreBase": ".results",
    "DefaultAggregateFieldName": ".results",
    "EvaluationResult": ".results",
    "Histogram": ".results",
    "HistogramBin": ".results",
    "MetricScore": ".results",
    "Percentiles": ".results",
    "RowScore": ".results",
    "RubricScoreStat": ".results",
    "RubricScoreValue": ".results",
    "SampleResult": ".results",
    "ScoreStats": ".results",
    "JSONScoreParser": ".scores",
    "RangeScore": ".scores",
    "RegexScoreParser": ".scores",
    "RemoteScore": ".scores",
    "Rubric": ".scores",
    "RubricScore": ".scores",
    "Score": ".scores",
    "score_discriminator": ".scores",
}

__all__ = [
    "Agent",
    "AgentBase",
    "GenericAgent",
    "NatAgentConfig",
    "NemoAgentToolkitAgent",
    "AggregateFieldName",
    "AggregatedMetricResult",
    "AggregateRangeScore",
    "AggregateRubricScore",
    "AggregateScore",
    "AggregateScoreBase",
    "ATIFTraceHandle",
    "BenchmarkEvaluationResult",
    "BooleanValue",
    "CandidateEvidence",
    "CandidateOutput",
    "CommandResult",
    "ContinuousScore",
    "FilesystemDiff",
    "FilesystemEntry",
    "FinalMetrics",
    "LogHandle",
    "Metrics",
    "Observation",
    "ObservationResult",
    "OTLPTraceHandle",
    "Step",
    "ToolCall",
    "Trajectory",
    "TraceHandle",
    "WellKnownEvidenceKey",
    "parse_atif",
    "DatasetRow",
    "DatasetRows",
    "DefaultAggregateFieldName",
    "DiscreteScore",
    "RunConfig",
    "RunConfigOnline",
    "RunConfigOnlineModel",
    "FieldMapping",
    "Histogram",
    "HistogramBin",
    "InferenceParams",
    "JSONScoreParser",
    "Label",
    "LocalFilesystemEvidence",
    "MetricDescriptor",
    "MetricDiagnostic",
    "MetricInput",
    "MetricOutput",
    "MetricOutputSpec",
    "MetricResult",
    "MetricTypeName",
    "MetricScore",
    "Model",
    "ModelRef",
    "InferenceConfig",
    "RankingInference",
    "EvalTarget",
    "Retrieval",
    "DatasetArg",
    "DatasetInput",
    "EvaluationResult",
    "EvidenceDescriptor",
    "Percentiles",
    "RangeScore",
    "ReasoningParams",
    "InputSchema",
    "RegexScoreParser",
    "RemoteScore",
    "RowScore",
    "Rubric",
    "RubricScore",
    "RubricScoreStat",
    "RubricScoreValue",
    "SampleResult",
    "Score",
    "ScoreStats",
    "SecretRef",
    "SupportedJobTypes",
    "score_discriminator",
    # Metrics
    "AgentGoalAccuracy",
    "AnswerAccuracy",
    "BLEU",
    "ContextEntityRecall",
    "ContextPrecision",
    "ContextRecall",
    "ContextRelevance",
    "ExactMatch",
    "F1",
    "Faithfulness",
    "LLMJudge",
    "MetricBase",
    "NemoAgentToolkitRemote",
    "NoiseSensitivity",
    "NumberCheck",
    "Remote",
    "ResponseGroundedness",
    "ResponseRelevancy",
    "ROUGE",
    "StringCheck",
    "ToolCallAccuracy",
    "ToolCalling",
    "TopicAdherence",
    "TunableRagEvaluator",
]


def __getattr__(name: str) -> object:
    """Import the submodule that defines ``name`` on first access (PEP 562).

    An *unknown* name raises ``AttributeError``, which is required: ``from pkg import sub`` only
    falls back to importing a submodule when attribute lookup raises ``AttributeError``.

    A *known* name whose submodule fails to import propagates that ``ImportError`` unchanged, so
    the real cause is not hidden behind a bogus "no attribute". The consequence is that
    ``hasattr`` raises rather than returning ``False`` when a name's dependencies are missing;
    catch ``ImportError`` around the access, or test membership against ``__all__``.
    """
    submodule = _LAZY_ATTRS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module(submodule, __name__), name)
    globals()[name] = value  # cache, so later lookups skip __getattr__ entirely
    return value


def __dir__() -> list[str]:
    # The declared surface plus any submodule the caller has already imported. Machinery is
    # imported under a leading underscore so the filter keeps it out of autocomplete without a
    # denylist; ``TYPE_CHECKING`` is the one exception, unaliased so type checkers recognise it.
    public = {name for name in globals() if not name.startswith("_")} - {"TYPE_CHECKING"}
    return sorted(set(__all__) | public)
