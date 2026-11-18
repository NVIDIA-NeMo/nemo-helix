# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tick every pending, running, or cancelling distillation workflow.

The entity is saved before a start or cancel runs, so a crash finds the node
``running`` with the names it already chose and does not create a second resource.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import ClassVar

from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client_provider import get_async_nemo_client
from nemo_helix_plugin.controller import NemoController
from nemo_helix_plugin.entities import EntityClient, EntityConflictError
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.log_utils import sanitize_for_log

from nemo_distillation_plugin.executor import Ports, tick
from nemo_distillation_plugin.models import Workflow
from nemo_distillation_plugin.platform import Platform

logger = logging.getLogger(__name__)

ClientFactory = Callable[[], AbstractAsyncContextManager[AsyncNemoClient]]
_ACTIVE = ("pending", "running", "cancelling")


@asynccontextmanager
async def _client() -> AsyncGenerator[AsyncNemoClient]:
    async with get_async_nemo_client(as_service="distillation", internal=True) as client:
        yield client


class DistillationWorkflowController(NemoController):
    """Reconcile distillation workflows in every workspace."""

    name: ClassVar[str] = "distillation-workflows"
    dependencies: ClassVar[list[str]] = ["entities", "jobs", "files", "models"]

    def __init__(
        self,
        client_factory: ClientFactory = _client,
        *,
        entities: EntityClient | None = None,
        ports: Ports | None = None,
    ) -> None:
        super().__init__()
        self._client_factory = client_factory
        self._entities = entities
        self._ports = ports
        self._context: AbstractAsyncContextManager[AsyncNemoClient] | None = None

    @property
    def interval_seconds(self) -> float:
        return 10.0

    async def on_startup(self) -> None:
        if self._entities is not None and self._ports is not None:
            return
        self._context = self._client_factory()
        client = await self._context.__aenter__()
        self._entities = EntityClient(AsyncEntitiesClient.from_client(client))
        self._ports = Platform(client)

    async def on_shutdown(self) -> None:
        if self._context is not None:
            await self._context.__aexit__(None, None, None)

    async def list_objects(self) -> list:
        return []

    async def reconcile_one(self, obj: object) -> None:
        raise NotImplementedError

    async def reconcile(self) -> None:
        if self._entities is None or self._ports is None:
            return
        for status in _ACTIVE:
            if self.stop_requested():
                return
            try:
                workflows = await self._every(status)
            except Exception:
                logger.exception("could not list %s distillation workflows", status)
                continue
            for workflow in workflows:
                if self.stop_requested():
                    return
                await self._tick(workflow)

    async def _tick(self, workflow: Workflow) -> None:
        assert self._entities is not None and self._ports is not None
        try:
            actions = await tick(workflow, self._ports, time.time())
            await self._entities.update(workflow)
        except EntityConflictError:
            return
        except Exception:
            logger.exception(
                "could not tick workflow %s/%s",
                sanitize_for_log(workflow.workspace),
                sanitize_for_log(workflow.name),
            )
            return
        for action in actions:
            await self._act(workflow, action)

    async def _act(self, workflow: Workflow, action) -> None:
        assert self._entities is not None and self._ports is not None
        spec = workflow.graph[action.node]
        adapter = self._ports.adapter(spec.kind)
        try:
            if action.type == "start":
                await adapter.start(workflow, action.node)
            else:
                await adapter.cancel(workflow, action.node)
        except Exception as exc:
            logger.exception(
                "could not %s node %s of %s/%s",
                action.type,
                sanitize_for_log(action.node),
                sanitize_for_log(workflow.workspace),
                sanitize_for_log(workflow.name),
            )
            await self._fail(workflow, action.node, str(exc))

    async def _fail(self, workflow: Workflow, node_key: str, error: str) -> None:
        assert self._entities is not None
        try:
            fresh = await self._entities.get(Workflow, name=workflow.name, workspace=workflow.workspace)
        except Exception:
            logger.exception("could not reload workflow %s to record a node failure", sanitize_for_log(workflow.name))
            return
        node = fresh.nodes.get(node_key)
        if node is None:
            return
        node.status = "failed"
        node.error = error
        try:
            await self._entities.update(fresh)
        except EntityConflictError:
            return

    async def _every(self, status: str) -> list[Workflow]:
        assert self._entities is not None
        found: list[Workflow] = []
        page = 1
        while True:
            result = await self._entities.list(
                Workflow,
                workspace="-",
                filter_obj={"status": status},
                sort="created_at",
                page=page,
            )
            found.extend(result.data)
            if not result.data or page >= result.pagination.total_pages:
                return found
            page += 1
