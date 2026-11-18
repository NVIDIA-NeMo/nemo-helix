# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

from nemo_distillation_plugin.executor import Action, Poll, tick
from nemo_distillation_plugin.models import GraphNode, RuntimeNode, Workflow


class Ports:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.cancelled: list[str] = []
        self.phase: Literal["running", "completed", "failed", "cancelled", "missing"] = "running"
        self.outputs: dict[str, Any] = {"ok": True}
        self.files: dict[str, dict[str, Any] | None] = {}
        self.written: dict[str, dict[str, Any]] = {}

    def adapter(self, kind: str) -> "Ports":
        del kind
        return self

    async def start(self, workflow: Workflow, node_key: str) -> None:
        del workflow
        self.started.append(node_key)

    async def poll(self, workflow: Workflow, node_key: str) -> Poll:
        del workflow, node_key
        return Poll(phase=self.phase, outputs=dict(self.outputs))

    async def cancel(self, workflow: Workflow, node_key: str) -> None:
        del workflow
        self.cancelled.append(node_key)

    async def read_output(self, workflow: Workflow, node_key: str) -> dict[str, Any] | None:
        del workflow
        return self.files.get(node_key)

    async def write_output(self, workflow: Workflow, node_key: str, payload: dict[str, Any]) -> None:
        del workflow
        self.written[node_key] = payload


def _workflow(*nodes: tuple[str, str, list[str], bool]) -> Workflow:
    return Workflow(
        name="distill",
        workspace="default",
        fileset="distill-data",
        graph={key: GraphNode(kind=kind, depends_on=deps, always_run=always) for key, kind, deps, always in nodes},
        nodes={
            key: RuntimeNode(external={"job": f"distill-{key}", "deployment": f"dep-{key}", "model": f"model-{key}"})
            for key, *_ in nodes
        },
    )


async def test_a_ready_node_is_marked_running_before_it_starts() -> None:
    workflow = _workflow(("prepare", "convert_atif", [], False))
    ports = Ports()

    actions = await tick(workflow, ports, now=0)

    assert actions == [Action("start", "prepare")]
    assert workflow.nodes["prepare"].status == "running"
    assert workflow.status == "running"
    assert ports.started == []


async def test_a_missing_resource_restarts_without_a_new_name() -> None:
    workflow = _workflow(("prepare", "convert_atif", [], False))
    workflow.nodes["prepare"].status = "running"
    external = dict(workflow.nodes["prepare"].external)
    ports = Ports()
    ports.phase = "missing"

    actions = await tick(workflow, ports, now=0)

    assert actions == [Action("start", "prepare")]
    assert workflow.nodes["prepare"].external == external
    assert workflow.nodes["prepare"].status == "running"


async def test_own_output_waits_for_the_file_then_fails_after_the_grace() -> None:
    workflow = _workflow(("prepare", "convert_atif", [], False))
    workflow.nodes["prepare"].status = "running"
    ports = Ports()
    ports.phase = "completed"

    await tick(workflow, ports, now=0)
    assert workflow.nodes["prepare"].status == "running"
    assert workflow.nodes["prepare"].confirm_since == 0

    await tick(workflow, ports, now=60)
    assert workflow.nodes["prepare"].status == "failed"
    assert workflow.status == "failed"


async def test_a_confirmed_file_is_copied_onto_the_entity() -> None:
    workflow = _workflow(("prepare", "convert_atif", [], False))
    workflow.nodes["prepare"].status = "running"
    ports = Ports()
    ports.phase = "completed"
    ports.files["prepare"] = {"dataset": "default/distill-data#nodes/prepare/", "tasks": [{"id": "t"}]}

    await tick(workflow, ports, now=0)

    assert workflow.nodes["prepare"].status == "succeeded"
    assert workflow.nodes["prepare"].outputs["tasks"] == [{"id": "t"}]
    assert ports.written == {}


async def test_other_kinds_write_the_confirmed_output() -> None:
    workflow = _workflow(("train", "customization_job", [], False))
    workflow.nodes["train"].status = "running"
    ports = Ports()
    ports.phase = "completed"
    ports.outputs = {"job": "distill-train", "model": "default/distill-train"}

    await tick(workflow, ports, now=0)

    assert workflow.nodes["train"].status == "succeeded"
    assert ports.written["train"]["model"] == "default/distill-train"


async def test_a_failed_dependency_skips_the_downstream_node() -> None:
    workflow = _workflow(
        ("prepare", "convert_atif", [], False),
        ("train", "customization_job", ["prepare"], False),
    )
    workflow.nodes["prepare"].status = "failed"

    await tick(workflow, Ports(), now=0)

    assert workflow.nodes["train"].status == "skipped"
    assert workflow.status == "failed"


async def test_cancel_stops_running_work_and_still_starts_cleanup() -> None:
    workflow = _workflow(
        ("deploy", "model_deployment", [], False),
        ("eval", "agent_eval", ["deploy"], False),
        ("undeploy", "delete_deployment", ["eval"], True),
    )
    workflow.status = "cancelling"
    workflow.nodes["deploy"].status = "running"
    ports = Ports()

    actions = await tick(workflow, ports, now=0)

    assert workflow.nodes["eval"].status == "cancelled"
    assert workflow.nodes["undeploy"].status == "running"
    assert Action("start", "undeploy") in actions
    assert Action("cancel", "deploy") in actions
    assert workflow.status == "cancelling"
