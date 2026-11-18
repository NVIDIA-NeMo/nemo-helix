# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Walk the default graph against a fake platform.

The live dev-stack run waits on a rebuilt ``nhx-customizer-tasks`` image. This is the
check that can run here: the stub's output feeds training, the experiment exists
before any eval is submitted, and cancel deletes every deployment the run created.
"""

from __future__ import annotations

import json
from typing import Any

from nemo_customizer.jobs.atif_convert import fileset_prefix, stub_output
from nemo_distillation_plugin.executor import tick
from nemo_distillation_plugin.models import DistillInputs, Workflow
from nemo_distillation_plugin.nodes.adapters import build_adapters
from nemo_distillation_plugin.runtime import Missing
from nemo_distillation_plugin.template import assemble

INPUTS = DistillInputs(source_agent="default/teacher", student_model="default/student")


class FakePlatform:
    def __init__(self, files: dict[tuple[str, str], bytes]) -> None:
        self.files = files
        self.events: list[tuple[str, str]] = []
        self.jobs: dict[str, str] = {}
        self.deployments: dict[str, str] = {}
        self.created: list[tuple[str, dict[str, Any]]] = []
        self.deleted: list[str] = []
        self.experiments: dict[str, dict[str, Any]] = {}
        self.evals: dict[str, dict[str, Any]] = {}
        self.agents: list[dict[str, Any]] = []
        self.customizations: list[dict[str, Any]] = []

    async def submit_convert(self, *, workspace: str, name: str, fileset: str, node_key: str) -> None:
        self.events.append(("convert", name))
        self.jobs[name] = "completed"
        prefix = fileset_prefix(node_key)
        output = stub_output(workspace=workspace, fileset=fileset, node_key=node_key)
        self.files[(fileset, f"{prefix}/output.json")] = json.dumps(output).encode()

    async def submit_customization(self, *, workspace: str, name: str, spec: dict[str, Any]) -> None:
        del workspace
        self.events.append(("train", name))
        self.customizations.append(spec)
        self.jobs[name] = "completed"

    async def job_status(self, *, workspace: str, name: str) -> str:
        del workspace
        if name not in self.jobs:
            raise Missing(name)
        return self.jobs[name]

    async def cancel_job(self, *, workspace: str, name: str) -> None:
        del workspace
        if name in self.jobs:
            self.jobs[name] = "cancelled"

    async def create_deployment(
        self, *, workspace: str, config_name: str, deployment_name: str, inputs: dict[str, Any]
    ) -> None:
        del workspace, config_name
        self.events.append(("deploy", deployment_name))
        self.created.append((deployment_name, dict(inputs)))
        self.deployments[deployment_name] = "READY"

    async def deployment_status(self, *, workspace: str, name: str) -> str:
        del workspace
        if name not in self.deployments:
            raise Missing(name)
        return self.deployments[name]

    async def delete_deployment(self, *, workspace: str, name: str) -> None:
        del workspace
        self.deleted.append(name)
        self.deployments.pop(name, None)

    async def prepare_agent(
        self,
        *,
        workspace: str,
        agent_name: str,
        source_agent: str,
        model: str | None,
        workflow_name: str,
        arm: str,
    ) -> None:
        del workspace, source_agent, workflow_name
        self.agents.append({"name": agent_name, "model": model, "arm": arm})

    async def delete_agent(self, *, workspace: str, agent_name: str) -> None:
        del workspace
        self.events.append(("delete-agent", agent_name))

    async def submit_eval(self, *, workspace: str, name: str, spec: dict[str, Any]) -> None:
        del workspace
        self.events.append(("eval", name))
        self.evals[name] = spec

    async def eval_status(self, *, workspace: str, name: str) -> str:
        del workspace
        if name not in self.evals:
            raise Missing(name)
        return "completed"

    async def cancel_eval(self, *, workspace: str, name: str) -> None:
        del workspace
        self.evals.pop(name, None)

    async def ensure_experiment(
        self, *, workspace: str, name: str, evaluations: dict[str, str], dataset_name: str
    ) -> str:
        del dataset_name
        self.events.append(("experiment", name))
        self.experiments[name] = {"id": "exp-id", "evaluations": dict(evaluations)}
        return "exp-id"

    async def get_experiment(self, *, workspace: str, name: str) -> dict[str, Any]:
        del workspace
        if name not in self.experiments:
            raise Missing(name)
        return self.experiments[name]


class Harness:
    def __init__(self) -> None:
        self.files: dict[tuple[str, str], bytes] = {}
        self.platform = FakePlatform(self.files)
        self.adapters = build_adapters(self.platform)

    def adapter(self, kind: str):
        return self.adapters[kind]

    async def read_output(self, workflow: Workflow, node_key: str) -> dict[str, Any] | None:
        blob = self.files.get((workflow.fileset, f"{fileset_prefix(node_key)}/output.json"))
        if blob is None:
            return None
        payload = json.loads(blob)
        return payload if isinstance(payload, dict) else None

    async def write_output(self, workflow: Workflow, node_key: str, payload: dict[str, Any]) -> None:
        self.files[(workflow.fileset, f"{fileset_prefix(node_key)}/output.json")] = json.dumps(payload).encode()


async def _apply(workflow: Workflow, harness: Harness) -> None:
    for action in await tick(workflow, harness, now=0):
        adapter = harness.adapter(workflow.graph[action.node].kind)
        if action.type == "start":
            await adapter.start(workflow, action.node)
        else:
            await adapter.cancel(workflow, action.node)


async def test_the_stub_flows_through_train_deploy_and_eval() -> None:
    workflow = assemble("distill", "default", INPUTS)
    harness = Harness()

    for _ in range(40):
        await _apply(workflow, harness)
        if workflow.status in {"completed", "failed", "cancelled"}:
            break

    assert workflow.status == "completed", _stuck(workflow)
    kinds = [kind for kind, _ in harness.platform.events]
    assert kinds.index("experiment") < kinds.index("eval")
    train = harness.platform.customizations[0]
    assert "deployment_config" not in train
    assert train["output"]["name"] == workflow.nodes["train[lora]"].external["model"]
    assert train["dataset"]["training"] == "default/distill-data#nodes/prepare/"
    assert train["training"]["finetuning_type"] == "lora"
    agents = {agent["arm"]: agent for agent in harness.platform.agents}
    assert agents["original"]["model"] is None
    assert agents["base"]["model"] == "default/student"
    adapter = workflow.nodes["train[lora]"].external["model"]
    assert agents["lora"]["model"] == f"default/student&adapters/default/{adapter}"
    original_job = workflow.nodes["eval_original"].external["job"]
    original_eval = harness.platform.evals[original_job]
    assert (
        original_eval["publication"]["intake"]["evaluation_id"]
        == (workflow.nodes["experiment"].external["evaluation_original"])
    )
    assert original_eval["tasks"] == [{"id": "stub-task-1", "intent": "Stub task for distillation plumbing."}]
    assert {name for name, _ in harness.platform.created} <= set(harness.platform.deleted)
    assert harness.platform.deployments == {}
    assert workflow.results is not None
    assert set(workflow.results) >= {"original", "base", "distilled"}
    assert workflow.stages[0]["status"] == "completed"


async def test_cancel_leaves_no_deployments() -> None:
    workflow = assemble("distill", "default", INPUTS)
    harness = Harness()
    saw_deployment = False

    for _ in range(40):
        await _apply(workflow, harness)
        if harness.platform.created and not saw_deployment:
            workflow.status = "cancelling"
            saw_deployment = True
        if workflow.status in {"cancelled", "completed", "failed"} and saw_deployment:
            break

    assert saw_deployment
    assert workflow.status == "cancelled", _stuck(workflow)
    assert workflow.nodes["eval_distilled[lora]"].status == "cancelled"
    assert harness.platform.deployments == {}
    assert {name for name, _ in harness.platform.created} <= set(harness.platform.deleted)


def _stuck(workflow: Workflow) -> str:
    states = {key: node.status for key, node in workflow.nodes.items()}
    return f"{workflow.status} {states}"
