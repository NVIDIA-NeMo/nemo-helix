# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The distillation template and the one-time wiring of names into it.

The workflow owns the teacher model deployment and the agent deployment. It
deploys the teacher, deploys the agent against that model, runs the original
evals, then removes both before any student deploy or training. ``prepare``
takes the original evals' output filesets. Each later arm deploys its own agent
from the same agent-deployment definition. Cleanup nodes are ``always_run``.
"""

from __future__ import annotations

from typing import Any

from nemo_distillation_plugin.graph import candidate_of, expand, validate
from nemo_distillation_plugin.models import (
    CandidateSpec,
    DistillInputs,
    EvalInput,
    GraphError,
    GraphNode,
    RuntimeNode,
    Workflow,
)
from nemo_distillation_plugin.names import allocate, check_candidate_name, check_workflow_name
from nemo_distillation_plugin.nodes.gateway import gateway_model
from nemo_distillation_plugin.projection import project

DEFAULT_CANDIDATE = CandidateSpec(name="lora", finetuning_type="lora")


def candidates_of(inputs: DistillInputs) -> list[CandidateSpec]:
    chosen = list(inputs.candidates) or [DEFAULT_CANDIDATE]
    seen: set[str] = set()
    for candidate in chosen:
        check_candidate_name(candidate.name)
        if candidate.name in seen:
            raise GraphError(f"candidate name {candidate.name!r} is repeated")
        seen.add(candidate.name)
    return chosen


def evals_of(inputs: DistillInputs) -> list[EvalInput]:
    seen: set[str] = set()
    for spec in inputs.evals:
        check_candidate_name(spec.name)
        if spec.name in seen:
            raise GraphError(f"eval name {spec.name!r} is repeated")
        seen.add(spec.name)
    return list(inputs.evals)


def render_graph(inputs: DistillInputs) -> dict[str, GraphNode]:
    """The template before name allocation. Matrix nodes fan out in :func:`expand`."""
    chosen = candidates_of(inputs)
    evals = evals_of(inputs)
    deploy = {
        "engine": inputs.deployment.engine,
        "gpu": inputs.deployment.gpu,
        "tool_call_config": inputs.deployment.tool_call_config,
        "additional_args": list(inputs.deployment.additional_args),
    }
    teacher = inputs.teacher
    arms = ["base", *[candidate.name for candidate in chosen]]
    if teacher is not None:
        arms.insert(0, "original")
    agent = _agent_inputs(inputs)
    nodes: dict[str, GraphNode] = {"experiment": GraphNode(kind="experiment", inputs={"arms": arms})}
    original: list[str] = []
    if teacher is not None:
        nodes["deploy_teacher"] = GraphNode(
            kind="model_deployment",
            inputs={
                "engine": teacher.engine,
                "gpu": teacher.gpu,
                "tool_call_config": teacher.tool_call_config,
                "additional_args": list(teacher.additional_args),
                "role": "teacher",
                "lora_enabled": False,
                "model": teacher.model,
            },
        )
        nodes["deploy_agent_original"] = GraphNode(
            kind="agent_deployment",
            depends_on=["deploy_teacher"],
            inputs={**agent, "source_agent": inputs.source_agent, "arm": "original"},
        )
        for spec in evals:
            nodes[f"eval_original[{spec.name}]"] = GraphNode(
                kind=_eval_kind(spec),
                depends_on=["experiment", "deploy_agent_original"],
                inputs=_eval_inputs(inputs, spec, arm="original"),
            )
        original = [f"eval_original[{spec.name}]" for spec in evals]
        trajectories: list[str] = [f"${{nodes.{key}.outputs.fileset}}" for key in original]
    else:
        trajectories = list(inputs.trajectory_filesets)
    nodes["prepare"] = GraphNode(
        kind="convert_atif",
        depends_on=list(original),
        inputs={"trajectory_filesets": trajectories, "pass_filter": dict(inputs.pass_filter)},
    )
    after_teacher: list[str] = []
    if teacher is not None:
        nodes["undeploy_agent_original"] = GraphNode(
            kind="delete_agent_deployment",
            depends_on=list(original),
            always_run=True,
        )
        nodes["undeploy_teacher"] = GraphNode(
            kind="delete_deployment",
            depends_on=list(original),
            always_run=True,
            inputs={"role": "teacher"},
        )
        after_teacher = ["undeploy_teacher"]
    nodes["deploy_base"] = GraphNode(
        kind="model_deployment",
        depends_on=[*after_teacher, *original],
        inputs={**deploy, "role": "base", "lora_enabled": True, "model": inputs.student_model},
    )
    nodes["deploy_agent_base"] = GraphNode(
        kind="agent_deployment",
        depends_on=["deploy_base"],
        inputs={**agent, "source_agent": inputs.source_agent, "arm": "base"},
    )
    for spec in evals:
        nodes[f"eval_base[{spec.name}]"] = GraphNode(
            kind=_eval_kind(spec),
            depends_on=["experiment", "deploy_agent_base"],
            inputs=_eval_inputs(inputs, spec, arm="base"),
        )
    nodes["train"] = GraphNode(
        kind="customization_job",
        depends_on=["prepare", *after_teacher],
        matrix={"candidate": [candidate.name for candidate in chosen]},
        inputs={
            "student_model": inputs.student_model,
            "dataset": "${nodes.prepare.outputs.dataset}",
        },
    )
    nodes["deploy"] = GraphNode(
        kind="model_deployment",
        depends_on=["train"],
        matrix="inherit",
        inputs={**deploy, "student_model": inputs.student_model},
    )
    nodes["deploy_agent"] = GraphNode(
        kind="agent_deployment",
        depends_on=["deploy"],
        matrix="inherit",
        inputs={**agent, "source_agent": inputs.source_agent},
    )
    for spec in evals:
        nodes[f"eval_distilled_{spec.name}"] = GraphNode(
            kind=_eval_kind(spec),
            depends_on=["experiment", "deploy_agent"],
            matrix="inherit",
            inputs=_eval_inputs(inputs, spec, arm=None),
        )
    nodes["undeploy_agent_base"] = GraphNode(
        kind="delete_agent_deployment",
        depends_on=[f"eval_base[{spec.name}]" for spec in evals],
        always_run=True,
    )
    nodes["undeploy_base"] = GraphNode(
        kind="delete_deployment",
        depends_on=[f"eval_base[{spec.name}]" for spec in evals],
        always_run=True,
        inputs={"deployment": "${nodes.deploy_base.outputs.deployment}"},
    )
    nodes["undeploy_agent"] = GraphNode(
        kind="delete_agent_deployment",
        depends_on=[f"eval_distilled_{spec.name}" for spec in evals],
        matrix="inherit",
        always_run=True,
    )
    nodes["undeploy"] = GraphNode(
        kind="delete_deployment",
        depends_on=[f"eval_distilled_{spec.name}" for spec in evals],
        matrix="inherit",
        always_run=True,
        inputs={"deployment": "${nodes.deploy.outputs.deployment}"},
    )
    return nodes


def _agent_inputs(inputs: DistillInputs) -> dict[str, Any]:
    spec = inputs.agent_deployment
    return {"mode": spec.mode, "image": spec.image, "use_image_entrypoint": spec.use_image_entrypoint}


def _eval_kind(spec: EvalInput) -> str:
    return "agent_eval" if spec.kind == "agent" else "dataset_eval"


def _eval_inputs(inputs: DistillInputs, spec: EvalInput, arm: str | None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "source_agent": inputs.source_agent,
        "eval": spec.name,
        "config": spec.config,
    }
    if spec.dataset is not None:
        body["dataset"] = spec.dataset
    if arm is not None:
        body["arm"] = arm
    return body


def build_graph(inputs: DistillInputs) -> dict[str, GraphNode]:
    """The template expanded over its candidates, then ordered into GPU lanes."""
    chosen = candidates_of(inputs)
    graph = expand(render_graph(inputs))
    _schedule(graph, inputs, chosen)
    validate(graph)
    return graph


def _schedule(graph: dict[str, GraphNode], inputs: DistillInputs, chosen: list[CandidateSpec]) -> None:
    """Chain jobs that share a GPU lane.

    A job is the base deploy-and-eval, or one candidate's train, deploy, eval. The budget
    allows ``gpu_budget // job_gpus`` jobs at once. Jobs are dealt to lanes in order, base
    first, and a job's first node waits for the cleanup nodes of the job before it in its
    lane. Those are ``always_run``, so a failed job still frees its lane.
    """
    jobs = [("deploy_base", ["undeploy_agent_base", "undeploy_base"])]
    jobs.extend(
        (f"train[{candidate.name}]", [f"undeploy_agent[{candidate.name}]", f"undeploy[{candidate.name}]"])
        for candidate in chosen
    )
    tails_by_lane: dict[int, list[str]] = {}
    lanes = inputs.parallel_jobs
    for index, (head, tails) in enumerate(jobs):
        for key in (head, *tails):
            if key not in graph:
                raise GraphError(f"cannot schedule: {key} is not in the graph")
        lane = index % lanes
        for dep in tails_by_lane.get(lane, []):
            if dep not in graph[head].depends_on:
                graph[head].depends_on.append(dep)
        tails_by_lane[lane] = tails


def render(inputs: DistillInputs) -> dict[str, Any]:
    """Expanded graph and counts. No names are allocated and nothing is written."""
    chosen = candidates_of(inputs)
    graph = build_graph(inputs)
    teacher_gpu = inputs.teacher.gpu if inputs.teacher is not None else 0
    return {
        "graph": graph,
        "counts": {
            "nodes": len(graph),
            "candidates": len(chosen),
            "passing_trials": 0,
            "estimated_tokens": 0,
            "gpu_hours": teacher_gpu
            + inputs.deployment.gpu * (1 + len(chosen))
            + sum(candidate.gpu for candidate in chosen),
            "parallel_jobs": inputs.parallel_jobs,
        },
    }


def assemble(name: str, workspace: str, inputs: DistillInputs) -> Workflow:
    """Build the entity, including the fileset name and every external name, before any I/O."""
    check_workflow_name(name)
    chosen = candidates_of(inputs)
    graph = build_graph(inputs)
    external = allocate(name, graph)
    _wire(workspace, graph, external, inputs, chosen)
    experiment_key = next(key for key, node in graph.items() if node.kind == "experiment")
    workflow = Workflow(
        name=name,
        workspace=workspace,
        template="distillation",
        status="pending",
        source_agent=inputs.source_agent,
        inputs=inputs.model_dump(mode="json"),
        fileset=f"{name}-data",
        graph=graph,
        nodes={key: RuntimeNode(external=dict(external[key])) for key in graph},
        experiment=external[experiment_key]["experiment"],
    )
    project(workflow)
    return workflow


def _wire(
    workspace: str,
    graph: dict[str, GraphNode],
    external: dict[str, dict[str, str]],
    inputs: DistillInputs,
    chosen: list[CandidateSpec],
) -> None:
    """Bake the names that are known at creation. Live bindings stay for prepare's outputs."""
    by_name = {candidate.name: candidate for candidate in chosen}
    experiment_key = next(key for key, node in graph.items() if node.kind == "experiment")
    evaluations = external[experiment_key]
    for key, node in graph.items():
        if node.kind == "customization_job":
            candidate = by_name[candidate_of(key) or ""]
            node.inputs["finetuning_type"] = candidate.finetuning_type
            node.inputs["gpu"] = candidate.gpu
            node.inputs["hyperparameters"] = candidate.hyperparameters.to_job()
        elif node.kind == "agent_deployment":
            arm = node.inputs.get("arm") or candidate_of(key)
            if not isinstance(arm, str) or not arm:
                raise GraphError(f"{key} has no agent arm")
            model_deploy = _dependency_kind(graph, node, "model_deployment")
            node.inputs["arm"] = arm
            node.inputs["model"] = gateway_model(workspace, graph[model_deploy].inputs)
        elif node.kind == "delete_agent_deployment":
            agent_deploy = _kind_target(graph, node, "agent_deployment")
            node.inputs["deployment"] = external[agent_deploy]["deployment"]
            node.inputs["agent"] = external[agent_deploy]["agent"]
        elif node.kind == "model_deployment" and node.inputs.get("role") not in {"base", "teacher"}:
            candidate = by_name[candidate_of(key) or ""]
            train_key = _dependency_kind(graph, node, "customization_job")
            node.inputs["finetuning_type"] = candidate.finetuning_type
            node.inputs["lora_enabled"] = candidate.finetuning_type == "lora"
            if candidate.finetuning_type == "lora":
                node.inputs["model"] = inputs.student_model
                node.inputs["adapter"] = external[train_key]["model"]
            else:
                node.inputs["model"] = f"{workspace}/{external[train_key]['model']}"
        elif node.kind == "delete_deployment":
            if node.inputs.get("role") == "teacher":
                teacher = next(
                    name
                    for name, item in graph.items()
                    if item.kind == "model_deployment" and item.inputs.get("role") == "teacher"
                )
                node.inputs["deployment"] = external[teacher]["deployment"]
                continue
            current = node.inputs.get("deployment")
            if isinstance(current, str) and not current.startswith("${"):
                continue
            deploy_key = _bound_deploy(current) or _deployment_target(graph, node)
            node.inputs["deployment"] = external[deploy_key]["deployment"]
        elif node.kind in {"agent_eval", "dataset_eval"}:
            arm = node.inputs.get("arm") or candidate_of(key)
            if not isinstance(arm, str) or not arm:
                raise GraphError(f"{key} has no eval arm")
            agent_deploy = _kind_target(graph, node, "agent_deployment")
            node.inputs["arm"] = arm
            node.inputs["agent_name"] = external[agent_deploy]["agent"]
            node.inputs["agent_deployment"] = external[agent_deploy]["deployment"]
            evaluation = evaluations.get(f"evaluation_{arm}")
            if evaluation is None:
                raise GraphError(f"{key} has no Evaluation for arm {arm!r}")
            node.inputs["evaluation_id"] = evaluation


def _dependency_kind(graph: dict[str, GraphNode], node: GraphNode, kind: str) -> str:
    matches = [dep for dep in node.depends_on if graph[dep].kind == kind]
    if len(matches) != 1:
        raise GraphError(f"expected one {kind} dependency, found {matches}")
    return matches[0]


def _kind_target(graph: dict[str, GraphNode], node: GraphNode, kind: str) -> str:
    """The node of ``kind`` this node acts on, directly or through the eval it follows."""
    direct = [dep for dep in node.depends_on if graph[dep].kind == kind]
    if len(direct) == 1:
        return direct[0]
    for dep in node.depends_on:
        nested = [child for child in graph[dep].depends_on if graph[child].kind == kind]
        if len(nested) == 1:
            return nested[0]
    raise GraphError(f"node has no {kind} among {node.depends_on}")


def _bound_deploy(value: object) -> str | None:
    """``${nodes.KEY.outputs.deployment}`` names the deploy node. KEY may contain brackets."""
    if not isinstance(value, str) or not value.startswith("${nodes.") or not value.endswith(".outputs.deployment}"):
        return None
    return value.removeprefix("${nodes.").removesuffix(".outputs.deployment}")


def _deployment_target(graph: dict[str, GraphNode], node: GraphNode) -> str:
    """The deployment a cleanup node deletes. It depends on the eval, which depends on the deploy."""
    direct = [dep for dep in node.depends_on if graph[dep].kind == "model_deployment"]
    if len(direct) == 1:
        return direct[0]
    for dep in node.depends_on:
        nested = [child for child in graph[dep].depends_on if graph[child].kind == "model_deployment"]
        if len(nested) == 1:
            return nested[0]
    raise GraphError(f"cleanup node has no deployment to delete among {node.depends_on}")
