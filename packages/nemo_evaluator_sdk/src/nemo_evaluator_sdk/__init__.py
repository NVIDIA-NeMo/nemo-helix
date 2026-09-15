# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo Evaluator SDK.

The public surface resolves lazily (PEP 562). Importing this package must not drag in the
execution/backend or metric stack: importing any submodule runs this module first, so eager
re-exports made ``import nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime`` — all the
optimizer needs — cost ~1400 modules (openai, sacrebleu, zstandard, ...) instead of ~485, and
turned every one of those transitive packages into an evaluation-time failure mode for the
SDK-backed evaluator.

Add a new re-export to ``_LAZY_ATTRS``, the ``TYPE_CHECKING`` block and ``__all__`` — never as a
module-level import. ``tests/test_lazy_public_api.py`` locks the boundary in.
"""

# ruff: noqa: I001 - the vendored SDK mirror uses different import-order settings.

from importlib import import_module as _import_module
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _package_version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Annotations and static analysis only; these must never execute at run time. Listing the
    # names in ``__all__`` is what marks them as re-exports for ruff and the type checkers.
    #
    # AGENTS.md ("Python Style notes") says not to import types under TYPE_CHECKING and to use a
    # regular import "when possible". A regular import is exactly what this module exists to
    # remove, so the exception is deliberate: these names are re-exports, not annotations, and
    # every one of them resolves for real through ``__getattr__`` below.
    from nemo_evaluator_sdk.agent_stream_translation import (
        AgentStreamTranslation,
        AgentStreamTranslationContext,
        AgentStreamTranslator,
        SseFrame,
    )
    from nemo_evaluator_sdk.datasets import DatasetLoadError, load_dataset, load_dataset_as_dicts
    from nemo_evaluator_sdk.execution.backends.local.backend import LocalBackend
    from nemo_evaluator_sdk.execution.evaluator import Evaluator
    from nemo_evaluator_sdk.execution.values import (
        EvaluationError,
        EvaluationPhase,
    )
    from nemo_evaluator_sdk.metrics.bleu import BLEUMetric
    from nemo_evaluator_sdk.metrics.exact_match import ExactMatchMetric
    from nemo_evaluator_sdk.metrics.f1 import F1Metric
    from nemo_evaluator_sdk.metrics.llm_judge import LLMJudgeMetric
    from nemo_evaluator_sdk.metrics.number_check import NumberCheckMetric
    from nemo_evaluator_sdk.metrics.retrieval import (
        RetrievalMAPMetric,
        RetrievalNDCGMetric,
        RetrievalPrecisionMetric,
        RetrievalRecallMetric,
    )
    from nemo_evaluator_sdk.metrics.protocol import (
        Metric,
        MetricTypeName,
        validate_metric_result,
    )
    from nemo_evaluator_sdk.metrics.remote import NemoAgentToolkitRemoteMetric, RemoteMetric
    from nemo_evaluator_sdk.metrics.rouge import ROUGEMetric
    from nemo_evaluator_sdk.metrics.string_check import StringCheckMetric
    from nemo_evaluator_sdk.metrics.tool_calling import ToolCallingMetric
    from nemo_evaluator_sdk.metrics.tunable_rag_evaluator import TunableRagEvaluatorMetric
    from nemo_evaluator_sdk.resolver_protocols import ModelResolver, SecretResolver
    from nemo_evaluator_sdk.resolvers import LocalModelResolver, LocalSecretResolver
    from nemo_evaluator_sdk.retrieval.beir import BeirDataset, BeirDatasetError
    from nemo_evaluator_sdk.structured_output import (
        InferenceFn,
        InferenceStructuredOutput,
        StructuredOutput,
        StructuredOutputMode,
        default_structured_output_mode,
        detect_structured_output_mode,
    )
    from nemo_evaluator_sdk.values import (
        Agent,
        AgentBase,
        BooleanValue,
        CandidateOutput,
        ContinuousScore,
        BenchmarkEvaluationResult,
        DatasetRow,
        DatasetRows,
        DiscreteScore,
        EvaluationResult,
        FieldMapping,
        InferenceParams,
        JSONScoreParser,
        Label,
        MetricDescriptor,
        MetricDiagnostic,
        MetricInput,
        MetricOutput,
        MetricOutputSpec,
        MetricResult,
        Model,
        ModelRef,
        GenericAgent,
        InferenceConfig,
        NatAgentConfig,
        NemoAgentToolkitAgent,
        RangeScore,
        RankingInference,
        ReasoningParams,
        RemoteScore,
        Retrieval,
        RubricScore,
        RunConfig,
        RunConfigOnline,
        RunConfigOnlineModel,
        SecretRef,
    )


def _resolve_version() -> str:
    """Report the version of whichever distribution actually shipped this code.

    ``nemo-evaluator-sdk`` is not published on its own — this package is also vendored into the
    ``nemo-platform`` wheel as ``nemo_platform.beta.evaluator``. There the SDK distribution does
    not exist, so resolving only that name reported ``"0.0.0"`` unconditionally and any telemetry
    or support log that read it got a useless constant.
    """
    for distribution in ("nemo-evaluator-sdk", "nemo-platform"):
        try:
            return _package_version(distribution)
        except _PackageNotFoundError:
            continue
    return "0.0.0"


version = _resolve_version()

# Re-exported name -> the submodule that defines it, relative to this package. Relative on
# purpose: the vendoring tool mirrors this file into nemo_platform.beta.evaluator by rewriting
# module paths, and a relative name has nothing to rewrite, so the mirror is correct by
# construction. Mirrors the TYPE_CHECKING block above, in the same order.
_LAZY_ATTRS: dict[str, str] = {
    "AgentStreamTranslation": ".agent_stream_translation",
    "AgentStreamTranslationContext": ".agent_stream_translation",
    "AgentStreamTranslator": ".agent_stream_translation",
    "SseFrame": ".agent_stream_translation",
    "DatasetLoadError": ".datasets",
    "load_dataset": ".datasets",
    "load_dataset_as_dicts": ".datasets",
    "LocalBackend": ".execution.backends.local.backend",
    "Evaluator": ".execution.evaluator",
    "EvaluationError": ".execution.values",
    "EvaluationPhase": ".execution.values",
    "BLEUMetric": ".metrics.bleu",
    "ExactMatchMetric": ".metrics.exact_match",
    "F1Metric": ".metrics.f1",
    "LLMJudgeMetric": ".metrics.llm_judge",
    "NumberCheckMetric": ".metrics.number_check",
    "RetrievalNDCGMetric": ".metrics.retrieval",
    "RetrievalRecallMetric": ".metrics.retrieval",
    "RetrievalPrecisionMetric": ".metrics.retrieval",
    "RetrievalMAPMetric": ".metrics.retrieval",
    "Metric": ".metrics.protocol",
    "MetricTypeName": ".metrics.protocol",
    "validate_metric_result": ".metrics.protocol",
    "NemoAgentToolkitRemoteMetric": ".metrics.remote",
    "RemoteMetric": ".metrics.remote",
    "ROUGEMetric": ".metrics.rouge",
    "StringCheckMetric": ".metrics.string_check",
    "ToolCallingMetric": ".metrics.tool_calling",
    "TunableRagEvaluatorMetric": ".metrics.tunable_rag_evaluator",
    "ModelResolver": ".resolver_protocols",
    "SecretResolver": ".resolver_protocols",
    "LocalModelResolver": ".resolvers",
    "LocalSecretResolver": ".resolvers",
    "BeirDataset": ".retrieval.beir",
    "BeirDatasetError": ".retrieval.beir",
    "InferenceFn": ".structured_output",
    "InferenceStructuredOutput": ".structured_output",
    "StructuredOutput": ".structured_output",
    "StructuredOutputMode": ".structured_output",
    "default_structured_output_mode": ".structured_output",
    "detect_structured_output_mode": ".structured_output",
    "Agent": ".values",
    "AgentBase": ".values",
    "BooleanValue": ".values",
    "CandidateOutput": ".values",
    "ContinuousScore": ".values",
    "BenchmarkEvaluationResult": ".values",
    "DatasetRow": ".values",
    "DatasetRows": ".values",
    "DiscreteScore": ".values",
    "EvaluationResult": ".values",
    "FieldMapping": ".values",
    "InferenceParams": ".values",
    "JSONScoreParser": ".values",
    "Label": ".values",
    "MetricDescriptor": ".values",
    "MetricDiagnostic": ".values",
    "MetricInput": ".values",
    "MetricOutput": ".values",
    "MetricOutputSpec": ".values",
    "MetricResult": ".values",
    "Model": ".values",
    "ModelRef": ".values",
    "GenericAgent": ".values",
    "InferenceConfig": ".values",
    "NatAgentConfig": ".values",
    "NemoAgentToolkitAgent": ".values",
    "RangeScore": ".values",
    "RankingInference": ".values",
    "ReasoningParams": ".values",
    "RemoteScore": ".values",
    "Retrieval": ".values",
    "RubricScore": ".values",
    "RunConfig": ".values",
    "RunConfigOnline": ".values",
    "RunConfigOnlineModel": ".values",
    "SecretRef": ".values",
}

__all__ = [
    "BLEUMetric",
    "Agent",
    "AgentBase",
    "EvaluationError",
    "EvaluationPhase",
    "DatasetLoadError",
    "DatasetRows",
    "RunConfig",
    "RunConfigOnline",
    "RunConfigOnlineModel",
    "BenchmarkEvaluationResult",
    "EvaluationResult",
    "Evaluator",
    "ExactMatchMetric",
    "F1Metric",
    "FieldMapping",
    "InferenceParams",
    "InferenceFn",
    "InferenceStructuredOutput",
    "JSONScoreParser",
    "Metric",
    "MetricTypeName",
    "MetricDescriptor",
    "MetricDiagnostic",
    "MetricInput",
    "MetricOutput",
    "MetricOutputSpec",
    "MetricResult",
    "LLMJudgeMetric",
    "BooleanValue",
    "CandidateOutput",
    "ContinuousScore",
    "DatasetRow",
    "DiscreteScore",
    "Label",
    "LocalBackend",
    "LocalModelResolver",
    "LocalSecretResolver",
    "Model",
    "ModelRef",
    "GenericAgent",
    "InferenceConfig",
    "ModelResolver",
    "NatAgentConfig",
    "NemoAgentToolkitAgent",
    "AgentStreamTranslation",
    "AgentStreamTranslationContext",
    "AgentStreamTranslator",
    "NemoAgentToolkitRemoteMetric",
    "NumberCheckMetric",
    "Retrieval",
    "RetrievalNDCGMetric",
    "RetrievalRecallMetric",
    "RetrievalPrecisionMetric",
    "RetrievalMAPMetric",
    "RangeScore",
    "RankingInference",
    "ReasoningParams",
    "RemoteMetric",
    "RemoteScore",
    "ROUGEMetric",
    "RubricScore",
    "SecretRef",
    "BeirDataset",
    "BeirDatasetError",
    "SecretResolver",
    "SseFrame",
    "StringCheckMetric",
    "StructuredOutput",
    "StructuredOutputMode",
    "ToolCallingMetric",
    "TunableRagEvaluatorMetric",
    "default_structured_output_mode",
    "detect_structured_output_mode",
    "load_dataset",
    "load_dataset_as_dicts",
    "validate_metric_result",
    "version",
]


def __getattr__(name: str) -> object:
    """Import the submodule that defines ``name`` on first access (PEP 562).

    An *unknown* name raises ``AttributeError``, which is required: ``from pkg import sub`` only
    falls back to importing a submodule when attribute lookup raises ``AttributeError``.

    A *known* name whose submodule fails to import propagates that ``ImportError`` unchanged, and
    that is deliberate — ``ModuleNotFoundError: No module named 'sacrebleu'`` is far more useful
    than an ``AttributeError`` claiming ``BLEUMetric`` does not exist. The consequence is that
    ``hasattr(nemo_evaluator_sdk, name)`` raises rather than returning ``False`` when a name's
    dependencies are not installed, since ``hasattr`` only swallows ``AttributeError``. To probe
    for an optional part of the surface, catch ``ImportError`` around the access instead of using
    ``hasattr``; to probe only for name membership, test against ``__all__``.
    """
    submodule = _LAZY_ATTRS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module(submodule, __name__), name)
    globals()[name] = value  # cache, so later lookups skip __getattr__ entirely
    return value


def __dir__() -> list[str]:
    # The declared surface plus any submodule the caller has already imported. Everything this
    # module needs for its own machinery is imported under a leading underscore so the filter
    # below keeps it out of autocomplete and inspect.getmembers without a name-by-name denylist;
    # ``TYPE_CHECKING`` is the one exception, kept unaliased so type checkers still recognise it.
    public = {name for name in globals() if not name.startswith("_")} - {"TYPE_CHECKING"}
    return sorted(set(__all__) | public)
