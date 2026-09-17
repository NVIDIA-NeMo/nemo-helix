# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BEIR corpus retrieval evaluation job."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar, Self

import httpx
from nemo_evaluator.filesets import (
    FilesetRef,
    download_dataset,
    download_dataset_sync,
    load_beir_dataset,
)
from nemo_evaluator.jobs.evaluate import (
    AGGREGATE_SCORES_RESULT_NAME,
    ARTIFACTS_RESULT_NAME,
    DEFAULT_RESULT_NAME,
    RESULT_IGNORE_PATTERNS,
    ROW_SCORES_RESULT_NAME,
    RUN_METADATA_RESULT_NAME,
    EvaluateJob,
)
from nemo_evaluator.jobs.metric_resolution import PlatformMetricModelResolver
from nemo_evaluator.jobs.secret_env import build_task_environment
from nemo_evaluator.jobs.utils import run_with_isolated_async_client
from nemo_evaluator_sdk import Evaluator
from nemo_evaluator_sdk.execution.metric_execution import run_sync as run_coro_sync
from nemo_evaluator_sdk.metrics.protocol import Metric
from nemo_evaluator_sdk.metrics.retrieval import (
    RetrievalMAPMetric,
    RetrievalNDCGMetric,
    RetrievalPrecisionMetric,
    RetrievalRecallMetric,
)
from nemo_evaluator_sdk.retrieval.beir import BeirDataset
from nemo_evaluator_sdk.retrieval.nim_ranking import NimRankingClient, NimRankingError
from nemo_evaluator_sdk.values.models import Model, ModelRef
from nemo_evaluator_sdk.values.multi_metric_results import BenchmarkEvaluationResult
from nemo_evaluator_sdk.values.retrieval import Retrieval, Truncation
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_platform_plugin.job import NemoJob
from nemo_platform_plugin.job_context import JobContext
from nemo_platform_plugin.jobs.api_factory import (
    ContainerSpec,
    CPUExecutionProviderSpec,
    PlatformJobSpec,
    PlatformJobStep,
)
from nemo_platform_plugin.jobs.image import get_qualified_image
from nemo_platform_plugin.models.client import AsyncModelsClient
from nemo_platform_plugin.sdk import AsyncNeMoPlatform
from pydantic import BaseModel, ConfigDict, Field, model_validator

EVAL_RESULTS_FILE_NAME = "eval_results.json"
EVAL_RESULTS_RESULT_NAME = "eval-results"
_PROJECTED_SUFFIXES = ("ndcg_cut_", "recall_", "P_", "map_cut_")


class RetrievalInputSpec(BaseModel):
    """Submitter-facing retrieval target before platform model refs are resolved."""

    model_config = ConfigDict(extra="forbid")

    embeddings: Model | ModelRef
    reranker: Model | ModelRef | None = None
    first_stage_k: int = Field(default=100, ge=1, description="Dense-search cutoff before reranking.")
    truncate_long_documents: Truncation | None = Field(
        default="end",
        description="How to cap passages at 65535 characters: keep the start ('end'), the tail ('start'), or error (null).",
        json_schema_extra={"nullable": True},
    )
    batch_size: int = Field(default=32, ge=1, description="Embedding HTTP batch size.")
    embedding_in_flight: int = Field(
        default=2,
        ge=1,
        description="Concurrent embedding POSTs to one NIM so the GPU is not idle between batches.",
    )
    embedding_dimensions: int | None = Field(
        default=None,
        gt=0,
        description="Expected embedding width. Omit to accept the model's native width.",
    )


class RetrieveEvalInputSpec(BaseModel):
    """Submitter-facing BEIR retrieval evaluation spec."""

    model_config = ConfigDict(extra="forbid")

    dataset: FilesetRef = Field(description="Fileset containing a BEIR test split.")
    target: RetrievalInputSpec | Model | ModelRef = Field(
        description="Retrieval pipeline, embedding NIM, or platform model reference."
    )
    baseline: RetrievalInputSpec | Model | ModelRef | None = Field(
        default=None,
        description="Optional baseline retrieval pipeline used for relative nDCG and recall at 10.",
    )
    k: list[int] = Field(default=[1, 5, 10, 100], min_length=1)

    @model_validator(mode="after")
    def validate_k(self) -> Self:
        """Require unique positive cutoffs."""
        if any(cutoff < 1 for cutoff in self.k):
            raise ValueError("retrieval cutoffs must be positive")
        if len(set(self.k)) != len(self.k):
            raise ValueError("retrieval cutoffs must be unique")
        self.k.sort()
        return self


class RetrieveEvalSpec(BaseModel):
    """Canonical BEIR retrieval evaluation spec."""

    model_config = ConfigDict(extra="forbid")

    dataset: FilesetRef
    target: Retrieval
    baseline: Retrieval | None = None
    k: list[int] = Field(default=[1, 5, 10, 100], min_length=1)

    @model_validator(mode="after")
    def validate_k(self) -> Self:
        """Require unique positive cutoffs in canonical task payloads."""
        if any(cutoff < 1 for cutoff in self.k):
            raise ValueError("retrieval cutoffs must be positive")
        if len(set(self.k)) != len(self.k):
            raise ValueError("retrieval cutoffs must be unique")
        self.k.sort()
        return self


class _RetrieveEvalJobBase(NemoJob):
    """Score a BEIR fileset with a deployed embedding NIM and optional reranker."""

    name: ClassVar[str] = "retrieve-eval"
    description: ClassVar[str] = "Evaluate dense retrieval over a BEIR corpus with nDCG, recall, precision, and MAP."
    container: ClassVar[str] = "cpu-tasks"
    input_spec_schema: ClassVar[type[BaseModel] | None] = RetrieveEvalInputSpec
    spec_schema: ClassVar[type[BaseModel] | None] = RetrieveEvalSpec
    job_collection_path: ClassVar[str | None] = "/retrieve-eval/jobs"
    generate_legacy_verbs: ClassVar[bool] = False

    @classmethod
    async def to_spec(
        cls,
        input_spec: BaseModel,
        workspace: str,
        entity_client: object,
        async_sdk: AsyncNeMoPlatform,
        is_local: bool,
    ) -> BaseModel:
        """Resolve a platform model reference before the job is compiled."""
        del workspace, entity_client, is_local
        submit_spec = RetrieveEvalInputSpec.model_validate(input_spec.model_dump())
        return RetrieveEvalSpec(
            dataset=submit_spec.dataset,
            target=await _resolve_retrieval(submit_spec.target, async_sdk),
            baseline=await _resolve_retrieval(submit_spec.baseline, async_sdk) if submit_spec.baseline else None,
            k=submit_spec.k,
        )

    @classmethod
    async def compile(
        cls,
        workspace: str,
        spec: BaseModel,
        entity_client: object,
        job_name: str | None,
        async_sdk: AsyncNeMoPlatform,
        profile: str | None = None,
        options: dict | None = None,
    ) -> PlatformJobSpec:
        """Compile a CPU task that calls the embedding target through IGW."""
        del workspace, entity_client, job_name, async_sdk, options
        canonical = RetrieveEvalSpec.model_validate(spec.model_dump())
        secret_refs = [
            (model.api_key_env, model.api_key_secret.root)
            for retrieval in (canonical.target, canonical.baseline)
            if retrieval is not None
            for model in (retrieval.embeddings, retrieval.reranker)
            if model is not None and model.api_key_secret is not None and model.api_key_env
        ]
        environment = build_task_environment(secret_refs)
        return PlatformJobSpec(
            steps=[
                PlatformJobStep(
                    name="retrieve-eval",
                    executor=CPUExecutionProviderSpec(
                        profile=profile or "default",
                        provider="cpu",
                        container=ContainerSpec(
                            image=get_qualified_image("nmp-cpu-tasks"),
                            entrypoint=["python", "-m"],
                            command=["nemo_evaluator.tasks.retrieve_eval"],
                        ),
                    ),
                    config=canonical.model_dump(mode="json"),
                    environment=environment,
                )
            ]
        )

    def _score_dataset(self, spec: RetrieveEvalSpec, ctx: JobContext, dataset_path: Path) -> dict:
        dataset = load_beir_dataset(dataset_path)
        cutoffs = _metric_cutoffs(spec)
        metrics = [
            RetrievalNDCGMetric(k=cutoffs),
            RetrievalRecallMetric(k=cutoffs),
            RetrievalPrecisionMetric(k=cutoffs),
            RetrievalMAPMetric(k=cutoffs),
        ]
        evaluator = Evaluator()
        started_at = datetime.now(UTC)
        result, baseline_result = _run_pipelines(evaluator, dataset, spec, metrics)
        result_files = EvaluateJob._write_result_files(
            result,
            ctx.storage.persistent,
            run_id=ctx.job_id,
            started_at=started_at,
        )
        artifact = ctx.results.save(DEFAULT_RESULT_NAME, result_files.full_result)
        ctx.results.save(AGGREGATE_SCORES_RESULT_NAME, result_files.aggregate_scores)
        ctx.results.save(ROW_SCORES_RESULT_NAME, result_files.row_scores)
        ctx.results.save(RUN_METADATA_RESULT_NAME, result_files.run_metadata)
        ctx.results.save(
            ARTIFACTS_RESULT_NAME,
            result_files.artifacts_dir,
            ignore_patterns=RESULT_IGNORE_PATTERNS,
        )

        eval_results = _project_eval_results(result)
        eval_results_path = Path(ctx.storage.persistent) / EVAL_RESULTS_FILE_NAME
        eval_results_path.write_text(json.dumps(eval_results, indent=2), encoding="utf-8")
        ctx.results.save(EVAL_RESULTS_RESULT_NAME, eval_results_path)
        output: dict[str, object] = {
            "status": "completed",
            "artifact": artifact.model_dump(),
            "eval_results": eval_results,
        }
        if baseline_result is not None:
            baseline_scores = _project_eval_results(baseline_result)
            relative = {
                name: _relative_change(eval_results.get(name), baseline_scores.get(name))
                for name in ("ndcg_cut_10", "recall_10")
                if name in eval_results and name in baseline_scores
            }
            output["baseline_eval_results"] = baseline_scores
            output["relative"] = relative
        return output


def _run_pipelines(
    evaluator: Evaluator,
    dataset: BeirDataset,
    spec: RetrieveEvalSpec,
    metrics: Sequence[Metric],
) -> tuple[BenchmarkEvaluationResult, BenchmarkEvaluationResult | None]:
    """Score target and optional baseline against the same BEIR split.

    Both pipelines share the downloaded fileset. When a baseline is present they
    hit distinct NIM endpoints, so they run concurrently instead of doubling
    wall time.
    """

    async def _score() -> tuple[BenchmarkEvaluationResult, BenchmarkEvaluationResult | None]:
        target_coro = evaluator.run(metrics=metrics, dataset=dataset, target=spec.target)
        if spec.baseline is None:
            return await target_coro, None
        target_result, baseline_eval = await asyncio.gather(
            target_coro,
            evaluator.run(metrics=metrics, dataset=dataset, target=spec.baseline),
        )
        return target_result, baseline_eval

    return run_coro_sync(_score)


class RetrieveEvalJob(_RetrieveEvalJobBase):
    """Public/local retrieval-evaluation job that downloads through sync typed clients."""

    def run(
        self,
        config: dict,
        *,
        ctx: JobContext,
        client: NemoClient,
    ) -> dict:
        """Download, validate, score, and persist a BEIR retrieval result."""
        spec = RetrieveEvalSpec.model_validate(config)
        dataset_path = download_dataset_sync(
            client=client,
            dataset=spec.dataset,
            destination=str(ctx.storage.persistent / "dataset"),
        )
        return self._score_dataset(spec, ctx, dataset_path)


class AsyncRetrieveEvalJob(_RetrieveEvalJobBase):
    """Task-container variant that downloads retrieval datasets through async typed clients."""

    def run(
        self,
        config: dict,
        *,
        ctx: JobContext,
        async_client: AsyncNemoClient,
    ) -> dict:
        """Download, validate, score, and persist a BEIR retrieval result."""
        spec = RetrieveEvalSpec.model_validate(config)
        dataset_path = run_with_isolated_async_client(
            async_client,
            lambda isolated_client: download_dataset(
                client=isolated_client,
                dataset=spec.dataset,
                destination=str(ctx.storage.persistent / "dataset"),
            ),
        )
        return self._score_dataset(spec, ctx, dataset_path)


async def _resolve_retrieval(
    value: RetrievalInputSpec | Model | ModelRef,
    async_sdk: AsyncNeMoPlatform,
) -> Retrieval:
    if isinstance(value, ModelRef):
        models_client = client_from_platform(async_sdk, AsyncModelsClient)
        return Retrieval(embeddings=await PlatformMetricModelResolver(models_client).resolve_model(value))
    if isinstance(value, Model):
        return Retrieval(embeddings=value)
    embeddings = value.embeddings
    reranker = value.reranker
    if isinstance(embeddings, ModelRef):
        models_client = client_from_platform(async_sdk, AsyncModelsClient)
        embeddings = await PlatformMetricModelResolver(models_client).resolve_model(embeddings)
    if isinstance(reranker, ModelRef):
        reranker_ref = reranker.root
        models_client = client_from_platform(async_sdk, AsyncModelsClient)
        reranker = await PlatformMetricModelResolver(models_client).resolve_model(reranker)
        try:
            async with asyncio.timeout(15.0):
                reranker = await NimRankingClient(model=reranker, max_retries=0, timeout=15.0).preflight()
        except TimeoutError as error:
            raise ValueError(
                f"Reranker ModelRef '{reranker_ref}' compatibility preflight exceeded the 15-second deadline"
            ) from error
        except (httpx.HTTPError, NimRankingError) as error:
            raise ValueError(
                f"Reranker ModelRef '{reranker_ref}' has no compatible ranking endpoint: {error}"
            ) from error
    return Retrieval(
        embeddings=embeddings,
        reranker=reranker,
        first_stage_k=value.first_stage_k,
        truncate_long_documents=value.truncate_long_documents,
        batch_size=value.batch_size,
        embedding_in_flight=value.embedding_in_flight,
        embedding_dimensions=value.embedding_dimensions,
    )


def _metric_cutoffs(spec: RetrieveEvalSpec) -> list[int]:
    """Include cutoff 10 whenever a baseline comparison will project nDCG and recall at 10."""
    cutoffs = list(spec.k)
    if spec.baseline is not None and 10 not in cutoffs:
        cutoffs.append(10)
        cutoffs.sort()
    return cutoffs


def _project_eval_results(result: BenchmarkEvaluationResult) -> dict[str, float]:
    projected: dict[str, float] = {}
    for score in result.aggregate_scores.scores:
        short = score.name.rsplit(".", 1)[-1]
        if short.startswith(_PROJECTED_SUFFIXES) and score.mean is not None:
            projected[short] = score.mean
    return projected


def _relative_change(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline in (None, 0):
        return None
    return (current - baseline) / baseline
