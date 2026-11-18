# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapters from a graph kind to the platform calls that kind makes.

``start`` is idempotent: the external names were stored before the first call,
and a missing resource on poll asks for ``start`` again.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

from nemo_distillation_plugin.executor import Poll
from nemo_distillation_plugin.graph import resolve
from nemo_distillation_plugin.models import Workflow
from nemo_distillation_plugin.nodes.gateway import gateway_model
from nemo_distillation_plugin.runtime import Missing

_JobPhase = Literal["completed", "failed", "cancelled"]
_JOB_DONE: dict[str, _JobPhase] = {
    "completed": "completed",
    "error": "failed",
    "cancelled": "cancelled",
}
_DEPLOY_FAILED = frozenset({"ERROR", "LOST", "UNKNOWN"})


class PlatformPort(Protocol):
    async def submit_convert(self, *, workspace: str, name: str, fileset: str, node_key: str) -> None: ...

    async def submit_customization(self, *, workspace: str, name: str, spec: dict[str, Any]) -> None: ...

    async def job_status(self, *, workspace: str, name: str) -> str: ...

    async def cancel_job(self, *, workspace: str, name: str) -> None: ...

    async def create_deployment(
        self, *, workspace: str, config_name: str, deployment_name: str, inputs: dict[str, Any]
    ) -> None: ...

    async def deployment_status(self, *, workspace: str, name: str) -> str: ...

    async def delete_deployment(self, *, workspace: str, name: str) -> None: ...

    async def prepare_agent(
        self,
        *,
        workspace: str,
        agent_name: str,
        source_agent: str,
        model: str | None,
        workflow_name: str,
        arm: str,
    ) -> None: ...

    async def delete_agent(self, *, workspace: str, agent_name: str) -> None: ...

    async def submit_eval(self, *, workspace: str, name: str, spec: dict[str, Any]) -> None: ...

    async def eval_status(self, *, workspace: str, name: str) -> str: ...

    async def cancel_eval(self, *, workspace: str, name: str) -> None: ...

    async def ensure_experiment(
        self,
        *,
        workspace: str,
        name: str,
        evaluations: dict[str, str],
        dataset_name: str,
    ) -> str: ...

    async def get_experiment(self, *, workspace: str, name: str) -> dict[str, Any]: ...


class _Adapter:
    def __init__(self, platform: PlatformPort) -> None:
        self.platform = platform


class ConvertAdapter(_Adapter):
    async def start(self, workflow: Workflow, node_key: str) -> None:
        external = workflow.nodes[node_key].external
        await self.platform.submit_convert(
            workspace=workflow.workspace,
            name=external["job"],
            fileset=workflow.fileset,
            node_key=node_key,
        )

    async def poll(self, workflow: Workflow, node_key: str) -> Poll:
        return await _job_poll(self.platform, workflow, workflow.nodes[node_key].external["job"])

    async def cancel(self, workflow: Workflow, node_key: str) -> None:
        await self.platform.cancel_job(workspace=workflow.workspace, name=workflow.nodes[node_key].external["job"])


class CustomizationAdapter(_Adapter):
    async def start(self, workflow: Workflow, node_key: str) -> None:
        external = workflow.nodes[node_key].external
        inputs = resolve(workflow.graph[node_key].inputs, workflow)
        spec = {
            "model": inputs["student_model"],
            "dataset": {"training": inputs["dataset"]},
            "training": {"finetuning_type": inputs["finetuning_type"]},
            "output": {"name": external["model"]},
        }
        await self.platform.submit_customization(workspace=workflow.workspace, name=external["job"], spec=spec)

    async def poll(self, workflow: Workflow, node_key: str) -> Poll:
        external = workflow.nodes[node_key].external
        poll = await _job_poll(self.platform, workflow, external["job"])
        if poll.phase == "completed":
            poll.outputs = {"job": external["job"], "model": f"{workflow.workspace}/{external['model']}"}
        return poll

    async def cancel(self, workflow: Workflow, node_key: str) -> None:
        await self.platform.cancel_job(workspace=workflow.workspace, name=workflow.nodes[node_key].external["job"])


class DeploymentAdapter(_Adapter):
    async def start(self, workflow: Workflow, node_key: str) -> None:
        external = workflow.nodes[node_key].external
        inputs = resolve(workflow.graph[node_key].inputs, workflow)
        await self.platform.create_deployment(
            workspace=workflow.workspace,
            config_name=external["config"],
            deployment_name=external["deployment"],
            inputs=inputs,
        )

    async def poll(self, workflow: Workflow, node_key: str) -> Poll:
        external = workflow.nodes[node_key].external
        try:
            status = await self.platform.deployment_status(workspace=workflow.workspace, name=external["deployment"])
        except Missing:
            return Poll(phase="missing")
        if status == "READY":
            inputs = resolve(workflow.graph[node_key].inputs, workflow)
            return Poll(
                phase="completed",
                outputs={
                    "deployment": external["deployment"],
                    "config": external["config"],
                    "model": gateway_model(workflow.workspace, inputs),
                },
            )
        if status == "DELETED":
            return Poll(phase="cancelled")
        if status in _DEPLOY_FAILED:
            return Poll(phase="failed", error=f"deployment {external['deployment']} is {status}")
        return Poll(phase="running")

    async def cancel(self, workflow: Workflow, node_key: str) -> None:
        await self.platform.delete_deployment(
            workspace=workflow.workspace, name=workflow.nodes[node_key].external["deployment"]
        )


class DeleteDeploymentAdapter(_Adapter):
    async def start(self, workflow: Workflow, node_key: str) -> None:
        inputs = resolve(workflow.graph[node_key].inputs, workflow)
        await self.platform.delete_deployment(workspace=workflow.workspace, name=str(inputs["deployment"]))

    async def poll(self, workflow: Workflow, node_key: str) -> Poll:
        name = str(resolve(workflow.graph[node_key].inputs, workflow)["deployment"])
        try:
            status = await self.platform.deployment_status(workspace=workflow.workspace, name=name)
        except Missing:
            return Poll(phase="completed", outputs={"deployment": name, "deleted": True})
        if status == "DELETED":
            return Poll(phase="completed", outputs={"deployment": name, "deleted": True})
        if status in _DEPLOY_FAILED:
            return Poll(phase="failed", error=f"deployment {name} is {status}")
        return Poll(phase="running")

    async def cancel(self, workflow: Workflow, node_key: str) -> None:
        del workflow, node_key


class AgentEvalAdapter(_Adapter):
    async def start(self, workflow: Workflow, node_key: str) -> None:
        external = workflow.nodes[node_key].external
        inputs = resolve(workflow.graph[node_key].inputs, workflow)
        model = inputs.get("model")
        await self.platform.prepare_agent(
            workspace=workflow.workspace,
            agent_name=external["agent"],
            source_agent=str(inputs["source_agent"]),
            model=str(model) if model else None,
            workflow_name=workflow.name,
            arm=str(inputs["arm"]),
        )
        tasks = inputs.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise ValueError(f"{node_key} has no tasks to evaluate")
        await self.platform.submit_eval(
            workspace=workflow.workspace,
            name=external["job"],
            spec={
                "tasks": tasks,
                "target": {
                    "kind": "fabric",
                    "source": {"agent": f"{workflow.workspace}/{external['agent']}"},
                },
                "publication": {"intake": {"evaluation_id": inputs["evaluation_id"]}},
            },
        )

    async def poll(self, workflow: Workflow, node_key: str) -> Poll:
        external = workflow.nodes[node_key].external
        try:
            status = await self.platform.eval_status(workspace=workflow.workspace, name=external["job"])
        except Missing:
            return Poll(phase="missing")
        phase = _JOB_DONE.get(status)
        if phase is None:
            return Poll(phase="running")
        await self.platform.delete_agent(workspace=workflow.workspace, agent_name=external["agent"])
        if phase == "completed":
            return Poll(
                phase="completed",
                outputs={
                    "score": None,
                    "input_tokens_per_task": None,
                    "output_tokens_per_task": None,
                    "median_latency_sec": None,
                    "estimated_cost_per_1k_tasks": None,
                    "eval_job": external["job"],
                    "agent": external["agent"],
                },
            )
        if phase == "cancelled":
            return Poll(phase="cancelled")
        return Poll(phase="failed", error=f"eval job {external['job']} is {status}")

    async def cancel(self, workflow: Workflow, node_key: str) -> None:
        external = workflow.nodes[node_key].external
        await self.platform.cancel_eval(workspace=workflow.workspace, name=external["job"])
        await self.platform.delete_agent(workspace=workflow.workspace, agent_name=external["agent"])


class ExperimentAdapter(_Adapter):
    async def start(self, workflow: Workflow, node_key: str) -> None:
        external = workflow.nodes[node_key].external
        arms = resolve(workflow.graph[node_key].inputs, workflow)["arms"]
        await self.platform.ensure_experiment(
            workspace=workflow.workspace,
            name=external["experiment"],
            evaluations={arm: external[f"evaluation_{arm}"] for arm in arms},
            dataset_name=f"{workflow.workspace}/{workflow.fileset}",
        )

    async def poll(self, workflow: Workflow, node_key: str) -> Poll:
        external = workflow.nodes[node_key].external
        try:
            experiment = await self.platform.get_experiment(workspace=workflow.workspace, name=external["experiment"])
        except Missing:
            return Poll(phase="missing")
        arms = workflow.graph[node_key].inputs.get("arms") or []
        return Poll(
            phase="completed",
            outputs={
                "experiment": external["experiment"],
                "experiment_id": experiment.get("id"),
                "evaluations": {arm: external[f"evaluation_{arm}"] for arm in arms},
            },
        )

    async def cancel(self, workflow: Workflow, node_key: str) -> None:
        del workflow, node_key


def build_adapters(platform: PlatformPort) -> dict[str, _Adapter]:
    return {
        "convert_atif": ConvertAdapter(platform),
        "customization_job": CustomizationAdapter(platform),
        "model_deployment": DeploymentAdapter(platform),
        "delete_deployment": DeleteDeploymentAdapter(platform),
        "agent_eval": AgentEvalAdapter(platform),
        "experiment": ExperimentAdapter(platform),
    }


async def _job_poll(platform: PlatformPort, workflow: Workflow, name: str) -> Poll:
    try:
        status = await platform.job_status(workspace=workflow.workspace, name=name)
    except Missing:
        return Poll(phase="missing")
    phase: Literal["running", "completed", "failed", "cancelled"] = _JOB_DONE.get(status, "running")
    if phase == "failed":
        return Poll(phase="failed", error=f"job {name} is {status}")
    if phase == "cancelled":
        return Poll(phase="cancelled")
    return Poll(phase=phase)
