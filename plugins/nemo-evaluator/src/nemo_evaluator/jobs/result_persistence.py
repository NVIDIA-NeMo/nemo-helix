# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persist eval-run results as queryable entities.

Both evaluator jobs persist the *full* result bundle (rows/trials) to the job's fileset via
``ctx.results.save``. This module adds the other half the legacy service had: a concise, queryable
**result entity** (aggregated scores + traits to filter on), with ``bundle_ref`` pointing back at the
fileset bundle. The entity is the evaluator's source of truth.

The task-container job variants use async typed clients for entity-store writes and drive the write
with ``run_sync``. An in-process run without an async task client simply skips persistence.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from nemo_evaluator.entities import AgentEvalResultEntity, EvaluateResultEntity
from nemo_evaluator.jobs.agent_spec import (
    AgentTarget,
    FabricRunnerTarget,
    GymRunnerTarget,
    HarborRunnerTarget,
    ModelTarget,
    Target,
)
from nemo_evaluator.jobs.utils import run_with_isolated_async_client
from nemo_evaluator_sdk.agent_eval.results import AgentEvalResult
from nemo_evaluator_sdk.values import Agent, AgentBase, Model
from nemo_evaluator_sdk.values.multi_metric_results import BenchmarkEvaluationResult
from nemo_evaluator_sdk.values.results import EvaluationResult
from nemo_platform_plugin.client.client import AsyncNemoClient
from nemo_platform_plugin.entities import EntityBase, EntityClient
from nemo_platform_plugin.entities.client import AsyncEntitiesClient
from nemo_platform_plugin.job_context import JobContext

logger = logging.getLogger(__name__)


def _entity_client(async_client: AsyncNemoClient | None) -> EntityClient | None:
    """The standard async ``EntityClient`` for the job's async task SDK, or ``None`` if absent.

    ``None`` means no async task client was injected, so persistence is skipped.
    """
    if async_client is None:
        return None
    return EntityClient(AsyncEntitiesClient.from_client(async_client))


#: Query-parameter keys whose values are redacted before a target URL is persisted/returned.
_SENSITIVE_QUERY_KEY = re.compile(r"token|key|secret|password|passwd|pwd|auth|credential|sig", re.IGNORECASE)


def _safe_target_url(url: object) -> str | None:
    """Render a target endpoint URL for persistence with credentials stripped.

    ``target_url`` is stored on the result entity and returned by the read APIs, so any userinfo
    (``user:pass@``) or sensitive query values (api keys/tokens) the endpoint URL carries would leak.
    Drop userinfo, redact sensitive query values, and omit the URL entirely when it has no host (i.e.
    can't be safely normalized).
    """
    if url is None:
        return None
    try:
        parts = urlsplit(str(url))
    except ValueError:
        return None
    if not parts.hostname:
        return None
    netloc = parts.hostname if parts.port is None else f"{parts.hostname}:{parts.port}"
    query = urlencode(
        [
            (key, "REDACTED" if _SENSITIVE_QUERY_KEY.search(key) else value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
        ]
    )
    return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))


def _agent_target_fields(target: Target | None) -> tuple[str | None, str | None, str | None]:
    """(kind, name, url) flat target traits for an agent-eval target."""
    if isinstance(target, ModelTarget):
        return "model", target.model.name, _safe_target_url(target.model.url)
    if isinstance(target, AgentTarget):
        return "agent", target.agent.name, _safe_target_url(target.agent.url)
    if isinstance(target, FabricRunnerTarget):
        return "fabric", target.model, None
    if isinstance(target, GymRunnerTarget):
        return "gym", target.agent, None
    if isinstance(target, HarborRunnerTarget):
        return "harbor", target.agent_import_path or target.agent_name, None
    return None, None, None


def _row_target_fields(target: Model | Agent | None) -> tuple[str | None, str | None, str | None]:
    """(kind, name, url) flat target traits for a row-eval target."""
    if isinstance(target, Model):
        return "model", target.name, _safe_target_url(target.url)
    if isinstance(target, AgentBase):
        return "agent", target.name, _safe_target_url(target.url)
    return None, None, None


def _persist(entity: EntityBase, *, async_client: AsyncNemoClient | None) -> None:
    if async_client is None:
        logger.info("No async task client injected; skipping result-entity persistence.")
        return
    # Best-effort: the eval has already succeeded and the full bundle is saved, so a transient
    # entity-store error must not fail the job — the record is re-derivable from the bundle. Log
    # loudly and move on. (`save` is create-or-update, so a re-run of the same job id is idempotent.)
    try:

        async def _save(client: AsyncNemoClient) -> EntityBase:
            entity_client = _entity_client(client)
            assert entity_client is not None
            return await entity_client.save(entity)

        run_with_isolated_async_client(async_client, _save)
    except Exception:
        logger.exception(
            "Failed to persist result entity %r in workspace %r; the result bundle is still saved",
            entity.name,
            entity.workspace,
        )


def persist_agent_eval_result(
    result: AgentEvalResult,
    *,
    target: Target | None,
    ctx: JobContext,
    bundle_ref: str,
    async_client: AsyncNemoClient | None,
) -> None:
    """Persist an ``AgentEvalJob`` run as an :class:`AgentEvalResultEntity` (aggregate scores rollup)."""
    if ctx.job_id is None:
        logger.info("No job id; skipping result-entity persistence.")
        return
    target_kind, target_name, target_url = _agent_target_fields(target)
    entity = AgentEvalResultEntity(
        name=ctx.job_id,
        workspace=ctx.workspace,
        job_id=ctx.job_id,
        target_kind=target_kind,
        target_name=target_name,
        target_url=target_url,
        scores=result.summary.scores,
        bundle_ref=bundle_ref,
    )
    _persist(entity, async_client=async_client)


def persist_evaluate_result(
    result: EvaluationResult | BenchmarkEvaluationResult,
    *,
    target: Model | Agent | None,
    dataset_ref: str | None,
    metric_types: list[str],
    ctx: JobContext,
    bundle_ref: str,
    async_client: AsyncNemoClient | None,
) -> None:
    """Persist an ``EvaluateJob`` (row-eval) run as an :class:`EvaluateResultEntity` (aggregates)."""
    if ctx.job_id is None:
        logger.info("No job id; skipping result-entity persistence.")
        return
    target_kind, target_name, target_url = _row_target_fields(target)
    entity = EvaluateResultEntity(
        name=ctx.job_id,
        workspace=ctx.workspace,
        job_id=ctx.job_id,
        target_kind=target_kind,
        target_name=target_name,
        target_url=target_url,
        scores=result.aggregate_scores,
        bundle_ref=bundle_ref,
        dataset_ref=dataset_ref,
        metric_types=metric_types,
    )
    _persist(entity, async_client=async_client)
