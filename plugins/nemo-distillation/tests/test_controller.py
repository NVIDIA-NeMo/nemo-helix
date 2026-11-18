# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

from nemo_distillation_plugin.controller import DistillationWorkflowController
from nemo_distillation_plugin.executor import Poll, Ports
from nemo_distillation_plugin.models import Workflow
from nemo_distillation_plugin.template import assemble
from nemo_helix_plugin.entities import EntityClient, EntityConflictError
from samples import inputs


class _Page:
    def __init__(self, data: list[Workflow]) -> None:
        self.data = data
        self.pagination = type("Pagination", (), {"total_pages": 1})()


class Entities:
    def __init__(self, workflow: Workflow, *, conflict: bool = False, events: list[str] | None = None) -> None:
        self.workflow = workflow
        self.conflict = conflict
        self.events = events if events is not None else []

    async def list(self, entity_type, *, workspace, filter_obj=None, sort=None, page=1, page_size=100):
        del entity_type, workspace, sort, page, page_size
        status = (filter_obj or {}).get("status")
        data = [self.workflow] if status == self.workflow.status else []
        return _Page(data)

    async def update(self, entity: Workflow) -> Workflow:
        self.events.append("save")
        if self.conflict:
            raise EntityConflictError("stale")
        self.workflow = entity
        return entity

    async def get(self, entity_type, name: str, *, workspace: str) -> Workflow:
        del entity_type, name, workspace
        return self.workflow


class _Ports:
    def __init__(self, events: list[str] | None = None) -> None:
        self.started: list[str] = []
        self.events = events if events is not None else []

    def adapter(self, kind: str) -> "_Ports":
        del kind
        return self

    async def start(self, workflow: Workflow, node_key: str) -> None:
        del workflow
        self.events.append("start")
        self.started.append(node_key)

    async def poll(self, workflow: Workflow, node_key: str) -> Poll:
        del workflow, node_key
        return Poll(phase="running")

    async def cancel(self, workflow: Workflow, node_key: str) -> None:
        del workflow

    async def read_output(self, workflow: Workflow, node_key: str):
        del workflow, node_key
        return None

    async def write_output(self, workflow: Workflow, node_key: str, payload) -> None:
        del workflow, node_key, payload


async def test_a_tick_is_saved_before_the_node_starts() -> None:
    workflow = assemble("distill", "default", inputs(student_model="student"))
    events: list[str] = []
    entities = Entities(workflow, events=events)
    ports = _Ports(events)
    controller = DistillationWorkflowController(entities=cast(EntityClient, entities), ports=cast(Ports, ports))

    await controller.reconcile()

    assert events[0] == "save"
    assert ports.started == ["experiment", "deploy_teacher"]
    assert entities.workflow.nodes["experiment"].status == "running"
    assert entities.workflow.nodes["prepare"].status == "pending"


async def test_a_version_conflict_does_not_start_the_node() -> None:
    workflow = assemble("distill", "default", inputs(student_model="student"))
    entities = Entities(workflow, conflict=True, events=[])
    ports = _Ports([])
    controller = DistillationWorkflowController(entities=cast(EntityClient, entities), ports=cast(Ports, ports))

    await controller.reconcile()

    assert ports.started == []
