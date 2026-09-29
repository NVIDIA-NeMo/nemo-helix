# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct in-process evaluation under /apis/evaluator/v2/workspaces/{workspace}/live.

One dataset row, generated and scored inline: no job record, no result artifacts, no Intake
publish. This is the interactive path behind Studio's Live Test panel, where a user checks an
evaluation config against a single row before saving it.

Every other long-running evaluation in this plugin is offloaded to a Job worker, and this route
deliberately breaks that pattern. Three constraints keep it from reintroducing the blocking that
pattern exists to avoid, and none of them are optional:

* ``await Evaluator().run(...)``, never ``run_sync``. ``run_sync`` detects the running loop, spawns
  a worker thread and joins it, which would block this worker's event loop for the whole of
  generation plus judge inference. ``EvaluateJob._run_evaluator`` is a sync method and calls
  ``run_sync`` correctly; the structure there is worth copying, the call form is not.
* A module-level semaphore, so N simultaneous requests cannot fan out to N generations plus N judge
  calls. Excess is rejected rather than queued.
* An outer timeout covering generation and judging together, bounding the handler even when a
  per-request ``request_timeout`` does not fire.

Two smaller shapes of the handler are load-bearing too. Reference resolution runs *inside* the
error mapping rather than ahead of it: an unresolvable judge ``ModelRef`` is the most common real
failure -- Studio's own notes record unavailable models surfacing as a 502 from an upstream 404 --
and outside the mapping it would reach the caller as a bare 500 carrying no detail. And the
concurrency check reads ``locked()`` immediately before ``acquire`` with no await between them, so
nothing can slip in between and turn a rejection into a queued request.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Self

from fastapi import APIRouter, Depends, HTTPException, status
from nemo_evaluator.api.schemas import MetricRefOrInline
from nemo_evaluator.authz import scope
from nemo_evaluator.jobs.metric_resolution import resolve_metrics_to_inline, to_runtime_bundle
from nemo_evaluator.shared.metric_bundles.bundles import unbundle_metric
from nemo_evaluator_sdk.execution.backends.local.backend import LocalBackend
from nemo_evaluator_sdk.execution.config import resolve_params
from nemo_evaluator_sdk.execution.evaluator import Evaluator
from nemo_evaluator_sdk.execution.utils import unique_metric_keys
from nemo_evaluator_sdk.metrics.protocol import Metric, MetricWithModels
from nemo_evaluator_sdk.metrics.utils import metric_type_name
from nemo_evaluator_sdk.resolver_protocols import SecretResolver
from nemo_evaluator_sdk.values import FieldMapping, Model
from nemo_evaluator_sdk.values.common import SecretRef
from nemo_evaluator_sdk.values.multi_metric_results import BenchmarkEvaluationResult
from nemo_evaluator_sdk.values.params import RunConfig, RunConfigOnlineModel
from nemo_evaluator_sdk.values.results import AggregateScore
from nemo_helix_plugin.authz import CallerKind, PermissionSet, path_rule, perm
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.dependencies import get_nemo_client
from nemo_helix_plugin.entities import EntityClient
from nemo_helix_plugin.entity_client import get_entity_client
from nemo_helix_plugin.refs import parse_entity_ref
from nemo_helix_plugin.secrets.client import AsyncSecretsClient
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

#: Ceiling on the whole handler: generation plus judging, not judging alone. Matches the 120s the
#: Live Test panel already allows, so the route never outlives the client waiting on it.
LIVE_TIMEOUT_S = 120

#: Per-upstream-call timeout. Generation and judging run sequentially on a single row, so two of
#: these must still fit inside LIVE_TIMEOUT_S.
LIVE_REQUEST_TIMEOUT_S = 55

#: Cap on metrics per request, so /live cannot be used to run a full evaluation past the Jobs
#: service. The row cap is enforced by the dataset field and reinforced by ``limit_samples=1``.
#: Sized to fit every metric type that scores without external infrastructure in one call.
MAX_METRICS = 24

#: Cap on *model-backed* metrics, which is where the cost actually is. Measured on one row: seven
#: deterministic metrics score in ~11ms, while adding a single llm-judge takes ~3.7s, because the
#: expense is an upstream inference call per metric and not the metric count. With MAX_CONCURRENT
#: this bounds a worker's outbound fan-out at MAX_JUDGE_METRICS * MAX_CONCURRENT calls.
MAX_MODEL_BACKED_METRICS = 4

#: Simultaneous in-flight scoring calls *per process*. Each one can hold a generation and a judge
#: call, so this bounds outbound inference, not just handler count.
MAX_CONCURRENT = 4

_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT)


class _HelixSecretResolver:
    """Resolve a metric's secret references through the Secrets service, as the calling principal.

    The SDK's default ``LocalSecretResolver`` reads ``os.environ``, which a job populates through
    ``build_task_environment`` and a request handler has no way to. This is the in-process
    equivalent, and it deliberately uses the *request-scoped* client: a service-privileged one
    would let any caller read another workspace's key and have a judge at a URL of their choosing
    receive it. An unqualified ref resolves in the request's own workspace.
    """

    def __init__(self, secrets_client: AsyncSecretsClient, *, workspace: str) -> None:
        """Bind the resolver to one caller's credentials and workspace."""
        self._secrets_client = secrets_client
        self._workspace = workspace

    async def resolve_secret(self, secret_ref: SecretRef) -> str | None:
        """Return the secret's value, or ``None`` when the caller cannot see one by that name."""
        ref = parse_entity_ref(secret_ref.root, self._workspace)
        try:
            response = await self._secrets_client.access_secret(name=ref.name, workspace=ref.workspace)
        except NotFoundError:
            return None
        return response.data().value


class LivePerms(PermissionSet, namespace="evaluator.live"):
    """Permissions for the direct in-process evaluation route."""

    RUN = perm("Run a single-row evaluation directly, without creating a job")


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
    "/live",
    summary="Run Live Evaluation",
    response_description="Score one row inline, without creating a job",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_429_TOO_MANY_REQUESTS: {"description": "Too many concurrent live evaluations"},
        status.HTTP_504_GATEWAY_TIMEOUT: {"description": "Live evaluation did not finish in time"},
    },
)
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[LivePerms.RUN])
async def run_live_evaluation(
    workspace: str,
    request: LiveScoreRequest,
    entity_client: EntityClient = Depends(get_entity_client),
    async_client: AsyncNemoClient = Depends(get_nemo_client),
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
    if _SEMAPHORE.locked():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many live evaluations are already running. Retry in a moment.",
        )
    async with _SEMAPHORE:
        try:
            _reject_target_secret(request.target)
            metrics = await _resolve_metrics(
                request.metrics, workspace=workspace, entity_client=entity_client, async_client=async_client
            )
            params = _live_params(request.params, request.target)
            secret_resolver = _HelixSecretResolver(
                client_from_platform(async_client, AsyncSecretsClient), workspace=workspace
            )
            async with asyncio.timeout(LIVE_TIMEOUT_S):
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
            logger.warning(f"Live evaluation failed: {type(exc).__name__}")
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))

    # Per-metric errors are how a partial failure stays diagnosable, but a run where *nothing*
    # scored did not partially succeed -- returning 200 there would let a client that reads only
    # `scores` mistake a total failure for an empty result.
    if results and all(entry.error is not None for entry in results):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="; ".join(f"{entry.metric}: {entry.error}" for entry in results),
        )
    return LiveScoreResponse(output=output, metrics=results)


def _reject_target_secret(target: Model | None) -> None:
    """Reject a target carrying an ``api_key_secret``, which this route cannot resolve.

    A job reaches its target's key because the compiler turns ``api_key_secret`` into a
    ``from_secret`` task environment variable and ``Model.api_key`` reads it back out of the
    process environment. In-process there is nothing to read: the only ways to supply it would be
    mutating ``os.environ`` per request, which leaks one caller's credential into another's
    request, or threading a resolved key through the SDK's generation path, which does not take
    one. Rejecting is the honest answer until the SDK accepts a key.
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
    those metrics carry are resolved separately, by :class:`_HelixSecretResolver`.
    """
    resolved = await resolve_metrics_to_inline(
        list(metrics),
        workspace=workspace,
        entity_client=entity_client if isinstance(entity_client, EntityClient) else None,
        async_sdk=async_client,
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

    The SDK's defaults are built for batch jobs. Three of them are actively wrong for a live test:
    three retries with backoff turn a 502 into a long wait before an opaque failure, an unbounded
    request timeout lets one call outlive the handler, and ``ignore_request_failure`` would return
    a NaN score on a request that never succeeded. Pinning ``max_retries=0`` here is what removes
    the original reason for generating client-side.

    A model target requires ``RunConfigOnlineModel``, so a bare ``RunConfig`` is widened rather than
    making the caller restate settings this function is about to pin anyway.
    """
    if target is not None and not isinstance(params, RunConfigOnlineModel):
        params = RunConfigOnlineModel.model_validate(params.model_dump() if params is not None else {})
    resolved = resolve_params(params, target)
    pinned: dict[str, Any] = {"parallelism": 1, "limit_samples": 1}
    if isinstance(resolved, RunConfigOnlineModel):
        pinned |= {
            "max_retries": 0,
            "request_timeout": LIVE_REQUEST_TIMEOUT_S,
            "ignore_request_failure": False,
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

    Offline, each metric runs on its own so a failure names the metric that caused it and leaves
    the others' scores intact. That costs nothing worth counting -- offline scoring is pure CPU
    over a single row -- and it is the only way to get per-metric attribution, since the SDK
    aborts a whole run on the first failure unless failures are tolerated, and tolerating them is
    not available to the offline ``RunConfig``.

    Online, the run stays single: the target must generate exactly once, and re-running per metric
    would re-generate per metric. A metric failure there still fails the request as a whole.
    """
    if isinstance(request.target, Model):
        result = await _run(metrics, request=request, params=params, secret_resolver=secret_resolver)
        # One run, so scores arrive namespaced under the keys the SDK assigned. Regrouping by those
        # keys keeps the response shape identical to the offline path.
        keys = unique_metric_keys(metrics)
        by_key: dict[str, list[AggregateScore]] = {key: [] for key in keys}
        for score in result.aggregate_scores.scores:
            key = score.name.split(".", 1)[0]
            by_key.setdefault(key, []).append(score)
        return _output_of(result), [LiveMetricResult(metric=key, scores=by_key[key]) for key in keys]

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
            dataset=request.dataset,
            config=params,
            target=request.target,
            field_mapping=request.field_mapping,
            prompt_template=request.prompt_template,
        )
    if type(params) is not RunConfig:
        raise TypeError("offline evaluation requires RunConfig")
    return await evaluator.run(
        metrics=metrics,
        dataset=request.dataset,
        config=params,
        target=None,
        field_mapping=request.field_mapping,
        prompt_template=None,
    )
