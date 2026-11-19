# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapters from a graph kind to the platform calls that kind makes.

``start`` is idempotent: the external names were stored before the first call,
and a missing resource on poll asks for ``start`` again.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

from nemo_helix_plugin.jobs.types import derive_job_fileset_name

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
    async def submit_convert(
        self,
        *,
        workspace: str,
        name: str,
        fileset: str,
        node_key: str,
        trajectory_filesets: list[str],
        pass_filter: dict[str, Any],
        student_model: str | None,
        max_tokens: dict[str, Any] | None,
    ) -> None: ...

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

    async def read_eval_config(self, workspace: str, ref: str) -> dict[str, Any]: ...

    async def submit_eval(self, *, workspace: str, name: str, spec: dict[str, Any]) -> None: ...

    async def eval_status(self, *, workspace: str, name: str) -> str: ...

    async def cancel_eval(self, *, workspace: str, name: str) -> None: ...

    async def submit_dataset_eval(
        self,
        *,
        workspace: str,
        name: str,
        agent_name: str,
        deployment_name: str,
        spec: dict[str, Any],
        evaluation_id: str,
    ) -> None: ...

    async def create_agent_deployment(
        self,
        *,
        workspace: str,
        name: str,
        agent_name: str,
        mode: str,
        image: str,
        use_image_entrypoint: bool,
    ) -> None: ...

    async def agent_deployment_status(self, *, workspace: str, name: str) -> str: ...

    async def delete_agent_deployment(self, *, workspace: str, name: str) -> None: ...

    async def dataset_eval_status(self, *, workspace: str, name: str) -> str: ...

    async def cancel_dataset_eval(self, *, workspace: str, name: str) -> None: ...

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
        inputs = resolve(workflow.graph[node_key].inputs, workflow)
        filesets = inputs.get("trajectory_filesets")
        if not isinstance(filesets, list) or not filesets:
            raise ValueError(f"{node_key} has no evaluation output filesets")
        if not all(isinstance(item, str) and item for item in filesets):
            raise ValueError(f"{node_key} has no evaluation output filesets")
        await self.platform.submit_convert(
            workspace=workflow.workspace,
            name=external["job"],
            fileset=workflow.fileset,
            node_key=node_key,
            trajectory_filesets=list(filesets),
            pass_filter=dict(inputs.get("pass_filter") or {}),
            student_model=inputs.get("student_model"),
            max_tokens=inputs.get("max_tokens"),
        )

    async def poll(self, workflow: Workflow, node_key: str) -> Poll:
        return await _job_poll(self.platform, workflow, workflow.nodes[node_key].external["job"])

    async def cancel(self, workflow: Workflow, node_key: str) -> None:
        await self.platform.cancel_job(workspace=workflow.workspace, name=workflow.nodes[node_key].external["job"])


class CustomizationAdapter(_Adapter):
    async def start(self, workflow: Workflow, node_key: str) -> None:
        external = workflow.nodes[node_key].external
        inputs = resolve(workflow.graph[node_key].inputs, workflow)
        # Shape of the automodel job spec (AutomodelJobInput): the dataset is an object, LoRA is a
        # finetuning_type, and parallelism, schedule, batch and optimizer are siblings of training.
        # The candidate's hyperparameters carry only what it set; the rest stays the customizer's default.
        hyperparameters = dict(inputs.get("hyperparameters") or {})
        training: dict[str, Any] = {"training_type": "sft", "finetuning_type": inputs["finetuning_type"]}
        for key in ("max_seq_length", "lora"):
            if key in hyperparameters:
                training[key] = hyperparameters[key]
        spec: dict[str, Any] = {
            "model": inputs["student_model"],
            "dataset": {"training": str(inputs["dataset"])},
            "training": training,
            "parallelism": {"num_gpus_per_node": int(inputs.get("gpu") or 1)},
            "output": {"name": external["model"]},
        }
        for section in ("schedule", "batch", "optimizer"):
            if hyperparameters.get(section):
                spec[section] = hyperparameters[section]
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
        workspace, name = _deployment_address(workflow, node_key)
        await self.platform.delete_deployment(workspace=workspace, name=name)

    async def poll(self, workflow: Workflow, node_key: str) -> Poll:
        workspace, name = _deployment_address(workflow, node_key)
        try:
            status = await self.platform.deployment_status(workspace=workspace, name=name)
        except Missing:
            return Poll(phase="completed", outputs={"deployment": name, "deleted": True})
        if status == "DELETED":
            return Poll(phase="completed", outputs={"deployment": name, "deleted": True})
        if status in _DEPLOY_FAILED:
            return Poll(phase="failed", error=f"deployment {name} is {status}")
        return Poll(phase="running")

    async def cancel(self, workflow: Workflow, node_key: str) -> None:
        del workflow, node_key


class AgentDeploymentAdapter(_Adapter):
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
        await self.platform.create_agent_deployment(
            workspace=workflow.workspace,
            name=external["deployment"],
            agent_name=external["agent"],
            mode=str(inputs["mode"]),
            image=str(inputs.get("image") or ""),
            use_image_entrypoint=bool(inputs.get("use_image_entrypoint")),
        )

    async def poll(self, workflow: Workflow, node_key: str) -> Poll:
        external = workflow.nodes[node_key].external
        try:
            status = await self.platform.agent_deployment_status(
                workspace=workflow.workspace, name=external["deployment"]
            )
        except Missing:
            return Poll(phase="missing")
        if status == "running":
            return Poll(
                phase="completed",
                outputs={"deployment": external["deployment"], "agent": external["agent"]},
            )
        if status == "failed":
            return Poll(phase="failed", error=f"agent deployment {external['deployment']} is {status}")
        if status == "deleting":
            return Poll(phase="cancelled")
        return Poll(phase="running")

    async def cancel(self, workflow: Workflow, node_key: str) -> None:
        external = workflow.nodes[node_key].external
        await self.platform.delete_agent_deployment(workspace=workflow.workspace, name=external["deployment"])
        await self.platform.delete_agent(workspace=workflow.workspace, agent_name=external["agent"])


class DeleteAgentDeploymentAdapter(_Adapter):
    async def start(self, workflow: Workflow, node_key: str) -> None:
        inputs = resolve(workflow.graph[node_key].inputs, workflow)
        await self.platform.delete_agent_deployment(workspace=workflow.workspace, name=str(inputs["deployment"]))
        await self.platform.delete_agent(workspace=workflow.workspace, agent_name=str(inputs["agent"]))

    async def poll(self, workflow: Workflow, node_key: str) -> Poll:
        name = str(resolve(workflow.graph[node_key].inputs, workflow)["deployment"])
        try:
            status = await self.platform.agent_deployment_status(workspace=workflow.workspace, name=name)
        except Missing:
            return Poll(phase="completed", outputs={"deployment": name, "deleted": True})
        if status == "failed":
            return Poll(phase="completed", outputs={"deployment": name, "deleted": True})
        return Poll(phase="running")

    async def cancel(self, workflow: Workflow, node_key: str) -> None:
        del workflow, node_key


class AgentEvalAdapter(_Adapter):
    async def start(self, workflow: Workflow, node_key: str) -> None:
        external = workflow.nodes[node_key].external
        inputs = resolve(workflow.graph[node_key].inputs, workflow)
        agent_name = str(inputs["agent_name"])
        config = await self.platform.read_eval_config(workflow.workspace, str(inputs["config"]))
        tasks = config.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise ValueError(f"{node_key}: {inputs['config']} has no tasks to evaluate")
        submitted: dict[str, Any] = {
            "tasks": tasks,
            "target": {
                "kind": "fabric",
                "source": {"agent": f"{workflow.workspace}/{agent_name}"},
            },
            "publication": {"intake": {"evaluation_id": inputs["evaluation_id"]}},
        }
        if config.get("max_concurrent_tasks") is not None:
            submitted["max_concurrent_tasks"] = config["max_concurrent_tasks"]
        await self.platform.submit_eval(workspace=workflow.workspace, name=external["job"], spec=submitted)

    async def poll(self, workflow: Workflow, node_key: str) -> Poll:
        external = workflow.nodes[node_key].external
        inputs = resolve(workflow.graph[node_key].inputs, workflow)
        try:
            status = await self.platform.eval_status(workspace=workflow.workspace, name=external["job"])
        except Missing:
            return Poll(phase="missing")
        phase = _JOB_DONE.get(status)
        if phase is None:
            return Poll(phase="running")
        if phase == "completed":
            return Poll(
                phase="completed",
                outputs=_eval_outputs(workflow.workspace, external["job"], str(inputs["agent_name"])),
            )
        if phase == "cancelled":
            return Poll(phase="cancelled")
        return Poll(phase="failed", error=f"eval job {external['job']} is {status}")

    async def cancel(self, workflow: Workflow, node_key: str) -> None:
        await self.platform.cancel_eval(workspace=workflow.workspace, name=workflow.nodes[node_key].external["job"])


class DatasetEvalAdapter(_Adapter):
    async def start(self, workflow: Workflow, node_key: str) -> None:
        external = workflow.nodes[node_key].external
        inputs = resolve(workflow.graph[node_key].inputs, workflow)
        spec = dict(await self.platform.read_eval_config(workflow.workspace, str(inputs["config"])))
        if inputs.get("dataset"):
            spec["dataset"] = str(inputs["dataset"])
        dataset = spec.get("dataset")
        if not isinstance(dataset, str) or not dataset or dataset.startswith("<"):
            raise ValueError(
                f"{node_key}: {inputs['config']} has no usable dataset reference (set the eval's `dataset`)"
            )
        if not spec.get("metrics"):
            raise ValueError(f"{node_key}: {inputs['config']} has no metrics")
        await self.platform.submit_dataset_eval(
            workspace=workflow.workspace,
            name=external["job"],
            agent_name=str(inputs["agent_name"]),
            deployment_name=str(inputs["agent_deployment"]),
            spec=spec,
            evaluation_id=str(inputs["evaluation_id"]),
        )

    async def poll(self, workflow: Workflow, node_key: str) -> Poll:
        external = workflow.nodes[node_key].external
        inputs = resolve(workflow.graph[node_key].inputs, workflow)
        try:
            status = await self.platform.dataset_eval_status(workspace=workflow.workspace, name=external["job"])
        except Missing:
            return Poll(phase="missing")
        phase = _JOB_DONE.get(status)
        if phase is None:
            return Poll(phase="running")
        if phase == "completed":
            return Poll(
                phase="completed",
                outputs=_eval_outputs(workflow.workspace, external["job"], str(inputs["agent_name"])),
            )
        if phase == "cancelled":
            return Poll(phase="cancelled")
        return Poll(phase="failed", error=f"eval job {external['job']} is {status}")

    async def cancel(self, workflow: Workflow, node_key: str) -> None:
        await self.platform.cancel_dataset_eval(
            workspace=workflow.workspace, name=workflow.nodes[node_key].external["job"]
        )


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
        "agent_deployment": AgentDeploymentAdapter(platform),
        "delete_agent_deployment": DeleteAgentDeploymentAdapter(platform),
        "agent_eval": AgentEvalAdapter(platform),
        "dataset_eval": DatasetEvalAdapter(platform),
        "experiment": ExperimentAdapter(platform),
    }


def _deployment_address(workflow: Workflow, node_key: str) -> tuple[str, str]:
    """A ``workspace/name`` deployment ref, or a name in the workflow workspace."""
    value = str(resolve(workflow.graph[node_key].inputs, workflow)["deployment"])
    if "/" in value:
        workspace, name = value.split("/", 1)
        if workspace and name:
            return workspace, name
    return workflow.workspace, value


def _eval_outputs(workspace: str, job: str, agent: str) -> dict[str, Any]:
    return {
        "score": None,
        "input_tokens_per_task": None,
        "output_tokens_per_task": None,
        "median_latency_sec": None,
        "estimated_cost_per_1k_tasks": None,
        "eval_job": job,
        "agent": agent,
        "fileset": f"{workspace}/{derive_job_fileset_name(job, workspace)}",
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
