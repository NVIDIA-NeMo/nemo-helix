# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK-backed evaluator job for the evaluator plugin scaffold."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, ClassVar, Self, TypeAlias, cast

# Imported for their registration side effects: each module registers its
# payload kind in the bundle registry so MetricBundle payloads validate.
import nemo_evaluator.shared.metric_bundles.cloudpickle  # noqa: F401
import nemo_evaluator.shared.metric_bundles.inline  # noqa: F401
from nemo_evaluator.api.schemas import MetricInline
from nemo_evaluator.filesets import FilesetRef, download_dataset, download_dataset_sync
from nemo_evaluator.jobs.agent_spec import target_agent_identity
from nemo_evaluator.jobs.metric_resolution import (
    resolve_metrics_to_inline,
    to_runtime_bundle,
    unresolved_model_refs,
)
from nemo_evaluator.jobs.publication import publish_row_eval_result
from nemo_evaluator.jobs.publication_spec import RowPublicationSpec
from nemo_evaluator.jobs.result_persistence import persist_evaluate_result
from nemo_evaluator.jobs.utils import as_async_nemo_client, as_nemo_client, run_with_isolated_async_client
from nemo_evaluator.metric_refs import MetricRefOrInline
from nemo_evaluator.shared.metric_bundles.bundles import unbundle_metric
from nemo_evaluator_sdk import Evaluator
from nemo_evaluator_sdk.execution.config import resolve_params
from nemo_evaluator_sdk.values import (
    Agent,
    AgentBase,
    FieldMapping,
    Model,
    RunConfig,
    RunConfigOnline,
    RunConfigOnlineModel,
)
from nemo_evaluator_sdk.values.multi_metric_results import BenchmarkEvaluationResult
from nemo_evaluator_sdk.values.results import EvaluationResult
from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_platform_plugin.entities import EntityClient
from nemo_platform_plugin.intake.client import AsyncIntakeClient
from nemo_platform_plugin.job import NemoJob
from nemo_platform_plugin.job_context import JobContext
from nemo_platform_plugin.jobs.api_factory import PlatformJobSpec
from nemo_platform_plugin.sdk import AsyncNeMoPlatform, NeMoPlatform
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

TargetSpec = Model | Agent
MetricSpec: TypeAlias = Annotated[list[MetricRefOrInline], Field(min_length=1)]
# Canonical spec carries inline metrics only (refs resolved) — still the wire DTO,
# so the runtime MetricBundle never surfaces as a public schema.
ResolvedMetricSpec: TypeAlias = Annotated[list[MetricInline], Field(min_length=1)]
EvaluationArtifactResult: TypeAlias = EvaluationResult | BenchmarkEvaluationResult
InlineDataset: TypeAlias = Annotated[list[dict[str, object]], Field(min_length=1)]
DatasetSpec: TypeAlias = InlineDataset | FilesetRef

DEFAULT_RESULT_NAME = "evaluation-results"
DEFAULT_FILE_NAME = "evaluation-results.json"
ARTIFACTS_RESULT_NAME = "artifacts"
AGGREGATE_SCORES_RESULT_NAME = "aggregate-scores"
ROW_SCORES_RESULT_NAME = "row-scores"
RUN_METADATA_RESULT_NAME = "run-metadata"
AGGREGATE_SCORES_FILE_NAME = "aggregate-scores.json"
ROW_SCORES_FILE_NAME = "row-scores.jsonl"
RUN_METADATA_FILE_NAME = "run-metadata.json"
RESULT_IGNORE_PATTERNS = ["cache.db", "cache/"]


@dataclass(frozen=True)
class EvaluationResultFiles:
    """Filesystem layout for an evaluator SDK result."""

    full_result: Path
    aggregate_scores: Path
    row_scores: Path
    run_metadata: Path
    artifacts_dir: Path


def _resolve_run_dataset(
    dataset: DatasetSpec,
    *,
    ctx: JobContext,
    client: NemoClient | None = None,
    async_client: AsyncNemoClient | None = None,
) -> InlineDataset | Path:
    """Resolve an evaluator plugin dataset for local SDK execution."""
    if not isinstance(dataset, FilesetRef):
        return dataset

    destination = str(ctx.storage.persistent / "dataset")
    # Prefer sync when available; async path isolates httpx so later run_sync calls
    # (result persistence) can reuse the injected async client.
    if client is not None:
        return download_dataset_sync(
            client=client,
            dataset=dataset,
            destination=destination,
        )
    if async_client is not None:
        return run_with_isolated_async_client(
            async_client,
            lambda client: download_dataset(client=client, dataset=dataset, destination=destination),
        )
    raise ValueError("FilesetRef datasets require an SDK client for local evaluator job execution.")


class _EvaluateSpecCommon(BaseModel):
    """Fields shared by the submitter input and the canonical (resolved) spec.

    ``EvaluateInputSpec`` and ``EvaluateSpec`` are siblings rather than a
    subtype pair: they differ only in their ``metrics`` field (refs allowed vs.
    fully resolved), and a mutable field can't be narrowed across inheritance
    without violating invariance.
    """

    model_config = ConfigDict(extra="forbid")

    dataset: DatasetSpec = Field(
        description="Inline dataset rows or a persisted FilesetRef dataset source to evaluate.",
    )
    params: RunConfig | RunConfigOnline | RunConfigOnlineModel | None = Field(
        default=None, description="Optional evaluator SDK execution parameters."
    )
    target: TargetSpec | None = Field(default=None, description="Optional model or agent target for online evaluation.")
    prompt_template: str | dict[str, Any] | None = Field(
        default=None, description="Optional prompt template for online target generation."
    )
    field_mapping: FieldMapping | None = Field(
        default=None, description="Optional mapping from canonical evaluator fields to dataset columns."
    )
    publication: RowPublicationSpec | None = Field(
        default=None,
        description="Where the completed run publishes its results, beyond its own result artifacts. "
        "Omit to publish nowhere.",
    )

    @model_validator(mode="after")
    def validate_params_for_target(self) -> Self:
        self.params = resolve_params(self.params, self.target)
        return self

    @model_validator(mode="after")
    def _require_resolvable_publication_identity(self) -> Self:
        # Publishing needs an agent name and only some targets carry one. Rejecting here makes it a
        # 422 on submit rather than a failure discovered after the evaluation has already run — and
        # without it a target that names nothing publishes every trajectory under an empty name.
        intake = self.publication.intake if self.publication is not None else None
        if intake is None or intake.agent_name is not None:
            return self
        if target_agent_identity(self.target)[0] is None:
            source = "an offline evaluation" if self.target is None else f"a {type(self.target).__name__} target"
            raise ValueError(
                f"`publication.intake.agent_name` is required: it cannot be derived from {source}. "
                "Supply the name the published trajectories should be recorded under."
            )
        return self


class EvaluateInputSpec(_EvaluateSpecCommon):
    """Submitter-facing SDK evaluation input for the evaluator plugin job."""

    metrics: MetricSpec = Field(
        description="Metrics to evaluate, given as inline metrics and/or references to stored metrics.",
    )


class EvaluateSpec(_EvaluateSpecCommon):
    """Canonical SDK evaluation spec with platform model and metric references resolved."""

    metrics: ResolvedMetricSpec = Field(description="Inline metrics with all references resolved.")

    @model_validator(mode="after")
    def reject_unresolved_metric_model_refs(self) -> Self:
        unresolved_refs = unresolved_model_refs([unbundle_metric(to_runtime_bundle(metric)) for metric in self.metrics])
        if unresolved_refs:
            raise ValueError(
                "EvaluateSpec metric models must be resolved before compile/run: " + ", ".join(unresolved_refs)
            )
        return self


class EvaluateJob(NemoJob):
    """Run evaluator SDK metrics against inline rows or FilesetRef datasets."""

    name: ClassVar[str] = "evaluate"
    description: ClassVar[str] = "Run evaluator SDK metrics against inline rows or FilesetRef datasets."
    container: ClassVar[str] = "cpu-tasks"
    input_spec_schema: ClassVar[type[BaseModel] | None] = EvaluateInputSpec
    spec_schema: ClassVar[type[BaseModel] | None] = EvaluateSpec
    job_collection_path: ClassVar[str | None] = "/evaluate/jobs"

    @classmethod
    async def compile(
        cls,
        *,
        workspace: str,
        spec: BaseModel,
        entity_client: object,
        job_name: str | None,
        async_sdk: AsyncNeMoPlatform | None,
        profile: str | None = None,
        options: dict | None = None,
    ) -> PlatformJobSpec:
        """Compile canonical spec to a plugin-native evaluator job."""
        del workspace, entity_client, job_name, async_sdk, options
        from nemo_evaluator.jobs.compiler import compile_evaluate_job

        canonical_spec = spec if isinstance(spec, EvaluateSpec) else EvaluateSpec.model_validate(spec.model_dump())
        canonical_spec.params = resolve_params(canonical_spec.params, canonical_spec.target)
        return compile_evaluate_job(canonical_spec, profile=profile)

    @staticmethod
    def _write_result_files(
        result: EvaluationArtifactResult, persistent_dir: Path, *, run_id: str | None, started_at: datetime
    ) -> EvaluationResultFiles:
        """Write full, aggregate, row-level and run-metadata evaluator artifacts."""
        result_payload = result.model_dump(mode="json")
        full_result_path = persistent_dir / DEFAULT_FILE_NAME
        full_result_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")

        artifacts_dir = persistent_dir / ARTIFACTS_RESULT_NAME
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        aggregate_path = artifacts_dir / AGGREGATE_SCORES_FILE_NAME
        aggregate_path.write_text(result.aggregate_scores.model_dump_json(indent=2), encoding="utf-8")
        row_scores_path = artifacts_dir / ROW_SCORES_FILE_NAME
        with row_scores_path.open("w", encoding="utf-8") as f:
            for row_score in result.row_scores:
                f.write(row_score.model_dump_json() + "\n")

        # `EvaluationResult` has nowhere to carry timings, so the run identity Intake publishes
        # under is written beside the scores. A re-publish must reuse these: `session_id` is
        # `{run_id}:{trial id}` and the span key includes `start_time`, so minting either afresh
        # writes a second trajectory instead of replacing the first.
        run_metadata_path = artifacts_dir / RUN_METADATA_FILE_NAME
        run_metadata_path.write_text(
            json.dumps({"run_id": run_id, "started_at": started_at.isoformat()}, indent=2),
            encoding="utf-8",
        )

        return EvaluationResultFiles(
            full_result=full_result_path,
            aggregate_scores=aggregate_path,
            row_scores=row_scores_path,
            run_metadata=run_metadata_path,
            artifacts_dir=artifacts_dir,
        )

    @classmethod
    async def to_spec(
        cls,
        input_spec: BaseModel,
        *,
        workspace: str,
        entity_client: object,
        async_sdk: AsyncNeMoPlatform | None,
        is_local: bool,
    ) -> BaseModel:
        """Resolve submitter-facing model and metric references into the canonical evaluation spec."""
        del is_local
        submit_spec = (
            input_spec.model_copy(deep=True)
            if isinstance(input_spec, EvaluateInputSpec)
            else EvaluateInputSpec.model_validate_json(input_spec.model_dump_json())
        )
        entity_client = cast(EntityClient | None, entity_client)
        metrics = await resolve_metrics_to_inline(
            submit_spec.metrics,
            workspace=workspace,
            entity_client=entity_client,
            async_sdk=async_sdk,
        )
        return EvaluateSpec(
            metrics=metrics,
            dataset=submit_spec.dataset,
            params=resolve_params(submit_spec.params, submit_spec.target),
            target=submit_spec.target,
            prompt_template=submit_spec.prompt_template,
            field_mapping=submit_spec.field_mapping,
            publication=submit_spec.publication,
        )

    def run(
        self,
        config: dict,
        *,
        ctx: JobContext,
        sdk: NemoClient | NeMoPlatform | None = None,
        async_sdk: AsyncNemoClient | AsyncNeMoPlatform | None = None,
    ) -> dict:
        """Run the evaluator job locally and persist its result artifact."""
        client = as_nemo_client(sdk)
        async_client = as_async_nemo_client(async_sdk)
        spec = EvaluateSpec.model_validate(config)
        # Stamped here because the row evaluator records no timing at all and `EvaluationResult` has
        # nowhere to put it. Publication needs a start time that is a function of the run, not of
        # when it was published, or re-ingest duplicates spans instead of replacing them.
        started_at = datetime.now(UTC)
        evaluator = Evaluator()
        params = resolve_params(spec.params, spec.target)
        metrics = [unbundle_metric(to_runtime_bundle(metric)) for metric in spec.metrics]
        dataset = _resolve_run_dataset(
            spec.dataset,
            ctx=ctx,
            client=client,
            async_client=async_client,
        )
        if isinstance(spec.target, Model):
            if not isinstance(params, RunConfigOnlineModel):
                raise TypeError("model target requires RunConfigOnlineModel")
            result = evaluator.run_sync(
                metrics=metrics,
                dataset=dataset,
                config=params,
                target=spec.target,
                field_mapping=spec.field_mapping,
                prompt_template=spec.prompt_template,
            )
        elif isinstance(spec.target, AgentBase):
            if type(params) is not RunConfigOnline:
                raise TypeError("agent target requires RunConfigOnline")
            if spec.prompt_template is None:
                raise ValueError("agent target requires prompt_template")
            result = evaluator.run_sync(
                metrics=metrics,
                dataset=dataset,
                config=params,
                target=spec.target,
                field_mapping=spec.field_mapping,
                prompt_template=spec.prompt_template,
            )
        else:
            if type(params) is not RunConfig:
                raise TypeError("offline evaluation requires RunConfig")
            result = evaluator.run_sync(
                metrics=metrics,
                dataset=dataset,
                config=params,
                target=None,
                field_mapping=spec.field_mapping,
                prompt_template=None,
            )
        result_files = self._write_result_files(
            result, ctx.storage.persistent, run_id=ctx.job_id, started_at=started_at
        )
        artifact = ctx.results.save(DEFAULT_RESULT_NAME, result_files.full_result)
        ctx.results.save(AGGREGATE_SCORES_RESULT_NAME, result_files.aggregate_scores)
        ctx.results.save(ROW_SCORES_RESULT_NAME, result_files.row_scores)
        ctx.results.save(RUN_METADATA_RESULT_NAME, result_files.run_metadata)
        ctx.results.save(ARTIFACTS_RESULT_NAME, result_files.artifacts_dir, ignore_patterns=RESULT_IGNORE_PATTERNS)

        # Persist the queryable result record (aggregate scores); per-row detail lives in the fileset
        # bundle referenced by `artifact`. Best-effort: the authoritative output (result artifacts) is
        # already saved above, so a persistence failure must not fail an otherwise-successful eval —
        # log and continue.
        try:
            persist_evaluate_result(
                result,
                target=spec.target,
                dataset_ref=spec.dataset.root if isinstance(spec.dataset, FilesetRef) else None,
                metric_types=[metric.type for metric in metrics],
                ctx=ctx,
                bundle_ref=artifact.artifact_url,
                async_sdk=async_client,
            )
        except Exception:
            logger.warning(
                "Failed to persist evaluate result record; the result artifacts are unaffected",
                exc_info=True,
            )

        # TODO: Implement progress reporting hook in SDK - AALGO-149
        # self.report_progress(
        #     ctx,
        #     work_done=1,
        #     work_total=1,
        #     status="completed",
        # )

        output = {
            "status": "completed",
            "artifact": artifact.model_dump(),
        }

        # Publication runs last, after the artifacts and the queryable record are both durable, so a
        # failed publish costs a re-publish rather than a re-run. It is also the only step here that
        # can fail the job (when `required`).
        publication = spec.publication.intake if spec.publication is not None else None
        if publication is not None:
            intake = AsyncIntakeClient.from_client(async_client) if async_client is not None else None
            outcome = publish_row_eval_result(
                result,
                spec=publication,
                target=spec.target,
                run_id=ctx.job_id,
                started_at=started_at,
                workspace=ctx.workspace,
                intake=intake,
            )
            output["publication"] = outcome.model_dump(exclude_none=True)

        return output
