# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read service for persisted eval-result entities.

Results are written by the jobs themselves (see ``jobs.result_persistence``); this service only
exposes them for list/get/delete. Stored entities are mapped to API DTOs (``AgentEvalResult`` /
``EvaluateResult``) — like ``MetricService`` maps ``MetricBundleEntity`` to ``Metric`` — so the wire
contract round-trips cleanly (an ``EntityBase``'s ``id``/``created_at`` are computed and don't
deserialize from the entity's own serialized form). Each result type has its own concretely-typed
methods so the API contract (and generated SDK) sees the real DTO.
"""

from __future__ import annotations

from datetime import datetime
from typing import TypeVar

from nemo_evaluator.api.schemas import AgentEvalResult, EvaluateResult
from nemo_evaluator.entities import AgentEvalResultEntity, EvaluateResultEntity
from nemo_helix_plugin.entities import PaginationInfo
from nemo_helix_plugin.entity_client import NemoAnyEntityDeleteClientProtocol, NemoEntityNotFoundError
from nemo_helix_plugin.filter_ops import FilterOperation
from nemo_helix_plugin.schema import Page, PaginationData


def _timestamps(entity: AgentEvalResultEntity | EvaluateResultEntity) -> tuple[datetime, datetime]:
    """The entity's persistence timestamps, guarded — a stored result must have them."""
    created_at = entity.created_at
    updated_at = entity.updated_at
    if created_at is None or updated_at is None:
        raise ValueError(f"Stored result '{entity.workspace}/{entity.name}' is missing persistence timestamps")
    return created_at, updated_at


def _to_agent_eval(entity: AgentEvalResultEntity) -> AgentEvalResult:
    created_at, updated_at = _timestamps(entity)
    return AgentEvalResult(
        id=entity.id,
        name=entity.name,
        workspace=entity.workspace,
        project=entity.project,
        job_id=entity.job_id,
        target_kind=entity.target_kind,
        target_name=entity.target_name,
        target_url=entity.target_url,
        scores=entity.scores,
        bundle_ref=entity.bundle_ref,
        created_at=created_at,
        updated_at=updated_at,
    )


def _to_evaluate(entity: EvaluateResultEntity) -> EvaluateResult:
    created_at, updated_at = _timestamps(entity)
    return EvaluateResult(
        id=entity.id,
        name=entity.name,
        workspace=entity.workspace,
        project=entity.project,
        job_id=entity.job_id,
        target_kind=entity.target_kind,
        target_name=entity.target_name,
        target_url=entity.target_url,
        scores=entity.scores,
        bundle_ref=entity.bundle_ref,
        created_at=created_at,
        updated_at=updated_at,
        dataset_ref=entity.dataset_ref,
        metric_types=entity.metric_types,
    )


def _pagination(src: PaginationInfo, current_page_size: int) -> PaginationData:
    """Carry the entity-store pagination counts into the API ``Page`` envelope."""
    return PaginationData(
        page=src.page,
        page_size=src.page_size,
        current_page_size=current_page_size,
        total_pages=src.total_pages,
        total_results=src.total_results,
    )


_ResultEntityT = TypeVar("_ResultEntityT", AgentEvalResultEntity, EvaluateResultEntity)


class ResultService:
    """List/get/delete for persisted eval-result entities, exposed as API DTOs."""

    def __init__(self, entity_client: NemoAnyEntityDeleteClientProtocol):
        self.entity_client = entity_client

    # --- agent-eval results --------------------------------------------------

    async def list_agent_eval_results(
        self,
        *,
        workspace: str,
        page: int = 1,
        page_size: int = 100,
        sort: str | None = None,
        filter_operation: FilterOperation | None = None,
    ) -> Page[AgentEvalResult]:
        result = await self.entity_client.list(
            AgentEvalResultEntity,
            workspace=workspace,
            filter_operation=filter_operation,
            sort=sort,
            page=page,
            page_size=page_size,
        )
        data = [_to_agent_eval(e) for e in result.data]
        return Page(data=data, pagination=_pagination(result.pagination, len(data)), sort=sort, filter=None)

    async def get_agent_eval_result(self, workspace: str, name: str) -> AgentEvalResult | None:
        try:
            entity = await self.entity_client.get(AgentEvalResultEntity, workspace=workspace, name=name)
        except NemoEntityNotFoundError:
            return None
        return _to_agent_eval(entity)

    async def delete_agent_eval_result(self, workspace: str, name: str) -> bool:
        return await self._delete(AgentEvalResultEntity, workspace, name)

    # --- (row) eval results --------------------------------------------------

    async def list_eval_results(
        self,
        *,
        workspace: str,
        page: int = 1,
        page_size: int = 100,
        sort: str | None = None,
        filter_operation: FilterOperation | None = None,
    ) -> Page[EvaluateResult]:
        result = await self.entity_client.list(
            EvaluateResultEntity,
            workspace=workspace,
            filter_operation=filter_operation,
            sort=sort,
            page=page,
            page_size=page_size,
        )
        data = [_to_evaluate(e) for e in result.data]
        return Page(data=data, pagination=_pagination(result.pagination, len(data)), sort=sort, filter=None)

    async def get_eval_result(self, workspace: str, name: str) -> EvaluateResult | None:
        try:
            entity = await self.entity_client.get(EvaluateResultEntity, workspace=workspace, name=name)
        except NemoEntityNotFoundError:
            return None
        return _to_evaluate(entity)

    async def delete_eval_result(self, workspace: str, name: str) -> bool:
        return await self._delete(EvaluateResultEntity, workspace, name)

    async def _delete(self, entity_cls: type[_ResultEntityT], workspace: str, name: str) -> bool:
        """Delete by workspace/name; ``False`` if absent. Type-agnostic (delete takes no body)."""
        try:
            await self.entity_client.delete(entity_cls, name, workspace=workspace)
        except NemoEntityNotFoundError:
            return False
        return True
