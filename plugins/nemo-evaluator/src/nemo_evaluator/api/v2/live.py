# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct in-process evaluation under /apis/evaluator/v2/workspaces/{workspace}/evaluate/live.

One dataset row, scored inline: no job record, no result artifacts, no Intake publish. This is the
interactive path behind Studio's Live Test panel.

Sibling of ``evaluate/jobs``: the same row evaluation, run in-process instead of as a job.
Every other long-running evaluation in this plugin is offloaded to a Job worker. This route
deliberately breaks that pattern, so three constraints keep it from reintroducing the blocking that
pattern exists to avoid:

* ``await Evaluator().run(...)``, never ``run_sync``, which joins a worker thread and would pin
  this worker's event loop for the whole of generation plus judge inference.
* A module-level semaphore, so simultaneous requests cannot fan out without bound.
* An outer timeout covering generation and judging together.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Self

from fastapi import APIRouter, Depends, HTTPException, status
from nemo_evaluator.api.schemas import MetricRefOrInline
from nemo_evaluator.authz import scope
from nemo_evaluator.jobs.metric_resolution import (
    HelixMetricSecretResolver,
    resolve_metrics_to_inline,
    to_runtime_bundle,
)
from nemo_evaluator.shared.metric_bundles.bundles import unbundle_metric
from nemo_evaluator_sdk.execution.backends.local.backend import LocalBackend
from nemo_evaluator_sdk.execution.config import resolve_params
from nemo_evaluator_sdk.execution.evaluator import Evaluator
from nemo_evaluator_sdk.execution.utils import unique_metric_keys
from nemo_evaluator_sdk.metrics.protocol import Metric, MetricWithModels
from nemo_evaluator_sdk.metrics.utils import metric_type_name
from nemo_evaluator_sdk.resolver_protocols import SecretResolver
from nemo_evaluator_sdk.values import FieldMapping, Model
from nemo_evaluator_sdk.values.multi_metric_results import BenchmarkEvaluationResult
from nemo_evaluator_sdk.values.params import RunConfig, RunConfigOnlineModel
from nemo_evaluator_sdk.values.results import AggregateScore
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.dependencies import get_nemo_client
from nemo_helix_plugin.entities import EntityClient
from nemo_helix_plugin.entity_client import get_entity_client
from nemo_helix_plugin.secrets.client import AsyncSecretsClient
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

#: Deliberately the same permission the ``evaluate/jobs`` submit route derives. Scoring one row
#: in-process is a weaker capability than submitting a job, so a caller who can do the latter
#: needs no separate grant for this.
_CREATE_EVALUATION = scope.permission("create", description="Create an evaluation")

#: Ceiling on the whole handler, matching what the Live Test panel allows.
LIVE_TIMEOUT_S = 120

#: Per-upstream-call timeout; two of these must still fit inside LIVE_TIMEOUT_S.
LIVE_REQUEST_TIMEOUT_S = 55

#: Keeps /live from being used to run a full evaluation past the Jobs service.
MAX_METRICS = 24

#: Bounds outbound inference: each model-backed metric costs an upstream call.
MAX_MODEL_BACKED_METRICS = 4

#: Simultaneous in-flight scoring calls per process.
MAX_CONCURRENT = 4

_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT)


def get_live_semaphore() -> asyncio.Semaphore:
    """Provide the process-wide cap on concurrent live evaluations.

    A dependency rather than a direct module lookup so a caller -- a test, most of all -- can
    substitute its own cap through ``app.dependency_overrides`` instead of rebinding this module's
    attribute, which is shared by every test in the process.
    """
    return _SEMAPHORE


class LiveScoreRequest(BaseModel):
    """One row, the metrics to score it with, and an optional model to generate it.

    Deliberately not a subclass of the job's ``EvaluateInputSpec``: this route persists nothing, so
    the publication field and its identity validator have no meaning here, and a ``FilesetRef``
    dataset would reintroduce the file I/O the route exists to avoid.
    """

    model_config = ConfigDict(extra="forbid")

    dataset: list[dict[str, Any]] = Field(
        min_length=1,
        max_length=1,
        description="The single row to evaluate. Offline requests carry the model output on the row.",
    )
    metrics: list[MetricRefOrInline] = Field(
        min_length=1,
        max_length=MAX_METRICS,
        description="Metrics to score the row with, given as inline definitions and/or references "
        "to stored metrics (`workspace/metric-name`).",
    )
    target: Model | None = Field(
        default=None,
        description="Optional model to generate the row's output before scoring. Omit to score an "
        "output already present on the row. Agent targets are not supported on this route.",
    )
    prompt_template: str | dict[str, Any] | None = Field(
        default=None, description="Optional prompt template applied to online target generation."
    )
    field_mapping: FieldMapping | None = Field(
        default=None, description="Optional mapping from canonical evaluator fields to dataset columns."
    )
    params: RunConfig | RunConfigOnlineModel | None = Field(
        default=None,
        description="Optional execution parameters. Single-row, no-retry and timeout settings are "
        "pinned by the route and cannot be overridden.",
    )

    @model_validator(mode="after")
    def _require_prompt_template_for_target(self) -> Self:
        """Require a prompt template alongside a target, as the job compiler does.

        Without it the template renderer is handed a ``None`` deep inside generation, so the caller
        gets a 502 about the model instead of a 422 about their request.
        """
        if self.target is not None and self.prompt_template is None:
            raise ValueError("prompt_template is required when target is set")
        return self


class LiveMetricResult(BaseModel):
    """One requested metric's outcome: its scores, or why it produced none."""

    model_config = ConfigDict(extra="forbid")

    metric: str = Field(description="The metric's runtime type name, matching the request order.")
    scores: list[AggregateScore] = Field(
        default_factory=list, description="Aggregated scores for this metric. Empty when it failed."
    )
    error: str | None = Field(default=None, description="Why this metric produced no scores. Null when it succeeded.")


class LiveScoreResponse(BaseModel):
    """Per-metric outcomes for the row, plus whatever text was scored.

    Every requested metric appears, in request order, whether or not it succeeded. A failing metric
    reports its own error and leaves its siblings' scores intact, so one broken metric out of eight
    is identifiable rather than collapsing the whole response into a single error.
    """

    model_config = ConfigDict(extra="forbid")

    output: str | None = Field(
        description="The candidate text that was scored. Generated server-side for an online "
        "target; echoed from the row for an offline request."
    )
    metrics: list[LiveMetricResult] = Field(description="One entry per requested metric, in request order.")


router = APIRouter()


@router.post(
    "/evaluate/live",
    summary="Run Live Evaluation",
    response_description="Score one row inline, without creating a job",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_429_TOO_MANY_REQUESTS: {"description": "Too many concurrent live evaluations"},
        status.HTTP_504_GATEWAY_TIMEOUT: {"description": "Live evaluation did not finish in time"},
    },
)
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[_CREATE_EVALUATION])
async def run_live_evaluation(
    workspace: str,
    request: LiveScoreRequest,
    entity_client: EntityClient = Depends(get_entity_client),
    async_client: AsyncNemoClient = Depends(get_nemo_client),
    semaphore: asyncio.Semaphore = Depends(get_live_semaphore),
) -> LiveScoreResponse:
    """Run a synchronous evaluation on a single dataset row.

    Scores the given row with the given metrics and returns the results immediately. Use this for
    quick, interactive checks of an evaluation config before saving it. For anything larger, use
    the async job-based evaluation endpoints.

    Creates no job and stores no result: nothing is persisted and nothing is published to Intake.

    The dataset must be provided inline and hold exactly one row. Each metric is either an inline
    definition or a reference to a stored metric (`workspace/metric-name`). Supply at most 24
    metrics, of which at most 4 may be model-backed (an LLM judge or a RAGAS metric), since each
    model-backed metric costs an upstream inference call.

    By default the row carries the model output already, and is scored offline. Supplying a
    `target` instead makes the server generate that output first, in which case `prompt_template`
    is required.

    **Per-metric results:**
    Every requested metric appears in the response, in request order, with either its scores or
    the error that stopped it -- so one failing metric out of several is identifiable rather than
    collapsing the whole response into a single error. A request in which no metric scored at all
    fails with 502.

    Past the concurrency cap the request is rejected with 429 rather than queued; past the overall
    time limit it fails with 504.
    """
    if semaphore.locked():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many live evaluations are already running. Retry in a moment.",
        )
    async with semaphore:
        try:
            # Reference resolution reaches the entity store, Files and Models, so it is bounded by
            # the same timeout as scoring; outside it a slow lookup would hold a slot indefinitely.
            async with asyncio.timeout(LIVE_TIMEOUT_S):
                _reject_target_secret(request.target)
                metrics = await _resolve_metrics(
                    request.metrics, workspace=workspace, entity_client=entity_client, async_client=async_client
                )
                params = _live_params(request.params, request.target)
                secret_resolver = HelixMetricSecretResolver(
                    client_from_platform(async_client, AsyncSecretsClient), workspace=workspace
                )
                output, results = await _score(metrics, request=request, params=params, secret_resolver=secret_resolver)
        except HTTPException:
            raise
        except TimeoutError:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Live evaluation did not finish within {LIVE_TIMEOUT_S}s.",
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc))
        except Exception as exc:
            # The SDK's own wording is returned deliberately: naming the failing metric and the
            # upstream status is why scoring runs through the evaluator at all. The caller is
            # authenticated, workspace-scoped, and supplied the metrics and models named back.
            logger.exception("Live evaluation failed")
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))

    # Nothing scored is a total failure, not a partial one.
    if results and all(entry.error is not None for entry in results):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="; ".join(f"{entry.metric}: {entry.error}" for entry in results),
        )
    return LiveScoreResponse(output=output, metrics=results)


def _reject_target_secret(target: Model | None) -> None:
    """Reject a target carrying an ``api_key_secret``, which this route cannot resolve.

    ``Model.api_key`` reads the process environment, which a job container populates and a request
    handler cannot. The SDK's generation path takes no resolved key, so there is nowhere to put one.
    """
    if target is not None and target.api_key_secret is not None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="target.api_key_secret is not supported on /live. Point the target at an "
            "Inference Gateway URL that needs no caller-supplied key, or submit an evaluate job "
            "instead.",
        )


async def _resolve_metrics(
    metrics: list[MetricRefOrInline],
    *,
    workspace: str,
    entity_client: EntityClient,
    async_client: AsyncNemoClient,
) -> list[Metric]:
    """Resolve any judge model references and return executable runtime metrics.

    Resolution is shared with the job path so a judge ``ModelRef`` resolves the same way here: the
    SDK's default ``LocalModelResolver`` starts empty and would reject every reference. Any secrets
    those metrics carry are resolved separately, by :class:`HelixMetricSecretResolver`.
    """
    resolved = await resolve_metrics_to_inline(
        list(metrics),
        workspace=workspace,
        entity_client=entity_client if isinstance(entity_client, EntityClient) else None,
        async_client=async_client,
    )
    runtime = [unbundle_metric(to_runtime_bundle(metric)) for metric in resolved]
    model_backed = [metric for metric in runtime if isinstance(metric, MetricWithModels)]
    if len(model_backed) > MAX_MODEL_BACKED_METRICS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"{len(model_backed)} model-backed metrics requested; /live allows at most "
            f"{MAX_MODEL_BACKED_METRICS}. Each one costs an upstream inference call. Submit an "
            "evaluate job to score more than that at once.",
        )
    return runtime


def _live_params(
    params: RunConfig | RunConfigOnlineModel | None,
    target: Model | None,
) -> RunConfig | RunConfigOnlineModel:
    """Return execution params with the interactive settings pinned.

    The SDK's defaults are built for batch jobs: retries with backoff hide the upstream status and
    an unbounded request timeout outlives the handler. ``ignore_request_failure`` is on so one
    failing metric does not abort its siblings; a failed *generation* is caught separately and never
    reported as a score. A model target is widened to ``RunConfigOnlineModel``, which carries those.
    """
    if target is not None and not isinstance(params, RunConfigOnlineModel):
        params = RunConfigOnlineModel.model_validate(params.model_dump() if params is not None else {})
    resolved = resolve_params(params, target)
    pinned: dict[str, Any] = {"parallelism": 1, "limit_samples": 1}
    if isinstance(resolved, RunConfigOnlineModel):
        pinned |= {
            "max_retries": 0,
            "request_timeout": LIVE_REQUEST_TIMEOUT_S,
            "ignore_request_failure": True,
        }
    return resolved.model_copy(update=pinned)


async def _score(
    metrics: list[Metric],
    *,
    request: LiveScoreRequest,
    params: RunConfig | RunConfigOnlineModel,
    secret_resolver: SecretResolver,
) -> tuple[str | None, list[LiveMetricResult]]:
    """Score the row and report each metric's outcome separately.

    Offline, each metric runs on its own, so a failure names the metric that caused it and leaves
    the others' scores intact: the SDK aborts a whole run on the first failure, and the offline
    ``RunConfig`` cannot tolerate failures. Online the run stays single, because the target must
    generate exactly once, and per-metric errors come from the run itself, which tolerates them.
    """
    if isinstance(request.target, Model):
        result = await _run(metrics, request=request, params=params, secret_resolver=secret_resolver)
        row = result.row_scores[0] if result.row_scores else None
        # A row the target never answered for has nothing to score, so it fails the request rather
        # than every metric individually.
        generation_error = row.sample.get("inference_error") if row is not None else None
        if generation_error:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"target generation failed: {generation_error}",
            )
        # Regroup the namespaced scores so the response matches the offline path.
        keys = unique_metric_keys(metrics)
        errors = (row.metric_errors or {}) if row is not None else {}
        by_key: dict[str, list[AggregateScore]] = {key: [] for key in keys}
        for score in result.aggregate_scores.scores:
            by_key.setdefault(score.name.split(".", 1)[0], []).append(score)
        return _output_of(result), [
            LiveMetricResult(metric=key, scores=[] if key in errors else by_key[key], error=errors.get(key))
            for key in keys
        ]

    output: str | None = None
    results: list[LiveMetricResult] = []
    for metric in metrics:
        try:
            result = await _run([metric], request=request, params=params, secret_resolver=secret_resolver)
        except Exception as exc:
            results.append(LiveMetricResult(metric=metric_type_name(metric), error=str(exc)))
            continue
        if output is None:
            output = _output_of(result)
        results.append(LiveMetricResult(metric=metric_type_name(metric), scores=list(result.aggregate_scores.scores)))
    return output, results


def _output_of(result: BenchmarkEvaluationResult) -> str | None:
    """Return the candidate text that was scored, when the row carries one."""
    row = result.row_scores[0] if result.row_scores else None
    output = row.sample.get("output_text") if row is not None else None
    return output if isinstance(output, str) else None


def _inline_rows(dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the request's rows as plain inline data.

    The evaluator's ``dataset`` argument is polymorphic -- inline rows, a file path, or a glob --
    and this route only ever supplies rows. Rebuilding them here means a path cannot reach the
    dataset loader even if the request field's declared type is later widened.
    """
    return [dict(row) for row in dataset]


async def _run(
    metrics: list[Metric],
    *,
    request: LiveScoreRequest,
    params: RunConfig | RunConfigOnlineModel,
    secret_resolver: SecretResolver,
) -> BenchmarkEvaluationResult:
    """Dispatch on the target, mirroring ``EvaluateJob._run_evaluator`` minus the job plumbing.

    The backend is constructed per request so its secret resolver carries that caller's
    credentials, and is never shared between callers.
    """
    backend = LocalBackend()
    backend.secret_resolver = secret_resolver
    evaluator = Evaluator(client=backend)
    if isinstance(request.target, Model):
        if not isinstance(params, RunConfigOnlineModel):
            raise TypeError("model target requires RunConfigOnlineModel")
        return await evaluator.run(
            metrics=metrics,
            dataset=_inline_rows(request.dataset),
            config=params,
            target=request.target,
            field_mapping=request.field_mapping,
            prompt_template=request.prompt_template,
        )
    if type(params) is not RunConfig:
        raise TypeError("offline evaluation requires RunConfig")
    return await evaluator.run(
        metrics=metrics,
        dataset=_inline_rows(request.dataset),
        config=params,
        target=None,
        field_mapping=request.field_mapping,
        prompt_template=None,
    )
