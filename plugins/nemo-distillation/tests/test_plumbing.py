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
from nemo_distillation_plugin.models import Workflow
from nemo_distillation_plugin.nodes.adapters import build_adapters
from nemo_distillation_plugin.runtime import Missing
from nemo_distillation_plugin.template import assemble
from samples import DATASET_REF, EVAL_CONFIGS, TASKS_REF, TOOL_ARGS, inputs


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
        self.trajectory_filesets: list[str] = []
        self.dataset_evals: dict[str, dict[str, Any]] = {}
        self.agent_deployments: dict[str, str] = {}
        self.config_reads: list[str] = []
        self.convert_requests: list[dict[str, Any]] = []

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
    ) -> None:
        self.convert_requests.append(
            {"pass_filter": pass_filter, "student_model": student_model, "max_tokens": max_tokens}
        )
        self.events.append(("convert", name))
        self.trajectory_filesets = list(trajectory_filesets)
        self.jobs[name] = "completed"
        prefix = fileset_prefix(node_key)
        output = stub_output(
            workspace=workspace, fileset=fileset, node_key=node_key, trajectory_filesets=trajectory_filesets
        )
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
        self.events.append(("delete", name))
        self.deleted.append(name)
        self.deployments.pop(name, None)

    async def create_agent_deployment(
        self,
        *,
        workspace: str,
        name: str,
        agent_name: str,
        mode: str,
        image: str,
        use_image_entrypoint: bool,
    ) -> None:
        del workspace, agent_name, mode, image, use_image_entrypoint
        self.events.append(("agent-deploy", name))
        self.agent_deployments[name] = "running"

    async def agent_deployment_status(self, *, workspace: str, name: str) -> str:
        del workspace
        if name not in self.agent_deployments:
            raise Missing(name)
        return self.agent_deployments[name]

    async def delete_agent_deployment(self, *, workspace: str, name: str) -> None:
        del workspace
        self.events.append(("delete-agent-deployment", name))
        self.agent_deployments.pop(name, None)

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

    async def read_eval_config(self, workspace: str, ref: str) -> dict[str, Any]:
        del workspace
        self.config_reads.append(ref)
        return dict(EVAL_CONFIGS[ref])

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

    async def submit_dataset_eval(
        self,
        *,
        workspace: str,
        name: str,
        agent_name: str,
        deployment_name: str,
        spec: dict[str, Any],
        evaluation_id: str,
    ) -> None:
        del workspace
        self.events.append(("eval", name))
        self.dataset_evals[name] = {
            "agent": agent_name,
            "deployment": deployment_name,
            "spec": spec,
            "evaluation_id": evaluation_id,
        }

    async def dataset_eval_status(self, *, workspace: str, name: str) -> str:
        del workspace
        if name not in self.dataset_evals:
            raise Missing(name)
        return "completed"

    async def cancel_dataset_eval(self, *, workspace: str, name: str) -> None:
        del workspace
        self.dataset_evals.pop(name, None)

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
    workflow = assemble("distill", "default", inputs())
    harness = Harness()

    for _ in range(80):
        await _apply(workflow, harness)
        if workflow.status in {"completed", "failed", "cancelled"}:
            break

    assert workflow.status == "completed", _stuck(workflow)
    events = harness.platform.events
    kinds = [kind for kind, _ in events]
    assert kinds.index("experiment") < kinds.index("eval") < kinds.index("convert")
    teacher = workflow.nodes["deploy_teacher"].external["deployment"]
    teacher_agent = workflow.nodes["deploy_agent_original"].external["deployment"]
    base = workflow.nodes["deploy_base"].external["deployment"]
    assert events.index(("deploy", teacher)) < events.index(("agent-deploy", teacher_agent))
    assert events.index(("agent-deploy", teacher_agent)) < events.index(
        ("eval", workflow.nodes["eval_original[tasks]"].external["job"])
    )
    assert events.index(("delete", teacher)) < events.index(("deploy", base))
    train = harness.platform.customizations[0]
    assert "deployment_config" not in train
    assert train["output"]["name"] == workflow.nodes["train[lora]"].external["model"]
    assert train["dataset"] == {"training": "default/distill-data#nodes/prepare/"}
    assert train["training"] == {"training_type": "sft", "finetuning_type": "lora"}
    assert train["parallelism"] == {"num_gpus_per_node": 1}
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for agent in harness.platform.agents:
        by_arm.setdefault(agent["arm"], []).append(agent)
    assert all(agent["model"] == "default/qwen36" for agent in by_arm["original"])
    assert all(agent["model"] == "default/student" for agent in by_arm["base"])
    adapter = workflow.nodes["train[lora]"].external["model"]
    assert all(agent["model"] == f"default/student&adapters/default/{adapter}" for agent in by_arm["lora"])
    original_job = workflow.nodes["eval_original[tasks]"].external["job"]
    original_eval = harness.platform.evals[original_job]
    assert (
        original_eval["publication"]["intake"]["evaluation_id"]
        == (workflow.nodes["experiment"].external["evaluation_original"])
    )
    assert original_eval["tasks"][0]["id"] == "classify-0"
    dataset_job = workflow.nodes["eval_original[dataset]"].external["job"]
    assert harness.platform.dataset_evals[dataset_job]["spec"]["dataset"] == "default/esec#dataset.jsonl"
    assert harness.platform.dataset_evals[dataset_job]["spec"]["metrics"] == EVAL_CONFIGS[DATASET_REF]["metrics"]
    assert original_eval["max_concurrent_tasks"] == 1
    assert set(harness.platform.config_reads) == {TASKS_REF, DATASET_REF}
    assert "classify-0" not in workflow.model_dump_json()
    assert harness.platform.created and all(
        deploy_inputs["additional_args"] == TOOL_ARGS for _, deploy_inputs in harness.platform.created
    )
    assert harness.platform.dataset_evals[dataset_job]["deployment"] == teacher_agent
    assert harness.platform.trajectory_filesets == [
        workflow.nodes["eval_original[tasks]"].outputs["fileset"],
        workflow.nodes["eval_original[dataset]"].outputs["fileset"],
    ]
    assert {name for name, _ in harness.platform.created} <= set(harness.platform.deleted)
    assert harness.platform.deployments == {}
    assert harness.platform.agent_deployments == {}
    assert workflow.results is not None
    assert set(workflow.results) >= {"original", "base", "distilled"}
    assert workflow.stages[0]["status"] == "completed"


async def test_each_candidate_trains_with_its_own_hyperparameters() -> None:
    candidates = [
        {
            "name": "small",
            "hyperparameters": {
                "max_seq_length": 4096,
                "lora": {"rank": 8, "alpha": 16},
                "schedule": {"epochs": 3},
                "batch": {"global_batch_size": 2, "micro_batch_size": 1},
                "optimizer": {"learning_rate": 0.0001, "lr_decay_style": "linear"},
            },
        },
        {
            "name": "wide",
            "hyperparameters": {
                "lora": {"rank": 64, "alpha": 128, "exclude_modules": ["*.out_proj"]},
                "schedule": {"epochs": 1},
            },
        },
        {"name": "plain"},
        {"name": "full", "finetuning_type": "all_weights", "hyperparameters": {"optimizer": {"learning_rate": 1e-5}}},
    ]
    workflow = assemble("distill", "default", inputs(candidates=candidates, gpu_budget=4))
    harness = Harness()

    for _ in range(120):
        await _apply(workflow, harness)
        if workflow.status in {"completed", "failed", "cancelled"}:
            break

    assert workflow.status == "completed", _stuck(workflow)
    by_output = {job["output"]["name"]: job for job in harness.platform.customizations}
    job = {
        name: by_output[workflow.nodes[f"train[{name}]"].external["model"]]
        for name in ("small", "wide", "plain", "full")
    }
    assert job["small"]["training"] == {
        "training_type": "sft",
        "finetuning_type": "lora",
        "max_seq_length": 4096,
        "lora": {"rank": 8, "alpha": 16},
    }
    assert job["small"]["schedule"] == {"epochs": 3}
    assert job["small"]["batch"] == {"global_batch_size": 2, "micro_batch_size": 1}
    assert job["small"]["optimizer"] == {"learning_rate": 0.0001, "lr_decay_style": "linear"}
    assert job["wide"]["training"]["lora"] == {"rank": 64, "alpha": 128, "exclude_modules": ["*.out_proj"]}
    assert job["wide"]["schedule"] == {"epochs": 1}
    assert "batch" not in job["wide"] and "optimizer" not in job["wide"]
    # A candidate that sets nothing sends nothing, so every default stays the customizer's.
    assert job["plain"]["training"] == {"training_type": "sft", "finetuning_type": "lora"}
    assert not {"schedule", "batch", "optimizer"} & set(job["plain"])
    assert job["full"]["training"] == {"training_type": "sft", "finetuning_type": "all_weights"}
    assert job["full"]["optimizer"] == {"learning_rate": 1e-5}


async def test_one_gpu_runs_each_job_to_the_end_before_the_next() -> None:
    candidates = [{"name": "lora-r16", "finetuning_type": "lora"}, {"name": "full", "finetuning_type": "all_weights"}]
    workflow = assemble("distill", "default", inputs(candidates=candidates, gpu_budget=1))
    harness = Harness()

    for _ in range(120):
        await _apply(workflow, harness)
        if workflow.status in {"completed", "failed", "cancelled"}:
            break

    assert workflow.status == "completed", _stuck(workflow)
    events = harness.platform.events
    external = {key: node.external for key, node in workflow.nodes.items()}
    teacher = external["deploy_teacher"]["deployment"]
    base = external["deploy_base"]["deployment"]
    first = external["deploy[lora-r16]"]["deployment"]
    second = external["deploy[full]"]["deployment"]
    order = [
        ("deploy", teacher),
        ("eval", external["eval_original[tasks]"]["job"]),
        ("delete", teacher),
        ("deploy", base),
        ("eval", external["eval_base[tasks]"]["job"]),
        ("delete", base),
        ("train", external["train[lora-r16]"]["job"]),
        ("deploy", first),
        ("eval", external["eval_distilled_tasks[lora-r16]"]["job"]),
        ("delete", first),
        ("train", external["train[full]"]["job"]),
        ("deploy", second),
        ("eval", external["eval_distilled_tasks[full]"]["job"]),
        ("delete", second),
    ]
    positions = [events.index(event) for event in order]
    assert positions == sorted(positions)


async def test_existing_trajectories_start_at_conversion() -> None:
    workflow = assemble("distill", "default", inputs(teacher=None, trajectory_filesets=["default/runs-a"]))
    harness = Harness()

    for _ in range(80):
        await _apply(workflow, harness)
        if workflow.status in {"completed", "failed", "cancelled"}:
            break

    assert workflow.status == "completed", _stuck(workflow)
    assert harness.platform.trajectory_filesets == ["default/runs-a"]
    assert not [key for key in workflow.nodes if "teacher" in key or "original" in key]
    events = harness.platform.events
    assert events.index(("deploy", workflow.nodes["deploy_base"].external["deployment"])) < events.index(
        ("train", workflow.nodes["train[lora]"].external["job"])
    )
    assert workflow.results is not None
    assert workflow.results["original"] is None
    assert [stage["id"] for stage in workflow.stages][:2] == ["prepare", "customization"]
    assert "eval-original" not in [stage["id"] for stage in workflow.stages]


async def test_the_convert_step_receives_the_student_and_the_length_limit() -> None:
    limit = {"enabled": True, "behavior": "truncate-earlier-turns", "tokens": "32k"}
    workflow = assemble(
        "distill",
        "default",
        inputs(teacher=None, trajectory_filesets=["default/runs-a"], max_tokens=limit),
    )
    harness = Harness()

    for _ in range(80):
        await _apply(workflow, harness)
        if workflow.status in {"completed", "failed", "cancelled"}:
            break

    assert workflow.status == "completed", _stuck(workflow)
    [request] = harness.platform.convert_requests
    assert request["student_model"] == workflow.graph["train[lora]"].inputs["student_model"]
    assert request["max_tokens"] == {"enabled": True, "behavior": "truncate-earlier-turns", "tokens": 32768}


async def test_the_convert_step_has_no_limit_unless_one_is_set() -> None:
    workflow = assemble("distill", "default", inputs(teacher=None, trajectory_filesets=["default/runs-a"]))
    harness = Harness()

    for _ in range(80):
        await _apply(workflow, harness)
        if workflow.status in {"completed", "failed", "cancelled"}:
            break

    [request] = harness.platform.convert_requests
    assert request["max_tokens"] is None
    assert request["student_model"]


async def test_cancel_leaves_no_deployments() -> None:
    workflow = assemble("distill", "default", inputs())
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
    assert workflow.nodes["eval_distilled_tasks[lora]"].status == "cancelled"
    assert workflow.nodes["eval_distilled_dataset[lora]"].status == "cancelled"
    assert harness.platform.deployments == {}
    assert harness.platform.agent_deployments == {}
    assert {name for name, _ in harness.platform.created} <= set(harness.platform.deleted)


def _stuck(workflow: Workflow) -> str:
    states = {key: node.status for key, node in workflow.nodes.items()}
    return f"{workflow.status} {states}"
