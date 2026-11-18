# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The distillation template and the one-time wiring of names into it.

``experiment`` depends only on ``prepare``. The three evals depend on it, so the
Evaluation exists before any eval is submitted. ``eval_original`` and ``eval_base``
run beside training. Cleanup nodes are ``always_run``.
"""

from __future__ import annotations

from typing import Any

from nemo_distillation_plugin.graph import candidate_of, expand
from nemo_distillation_plugin.models import (
    CandidateSpec,
    DistillInputs,
    GraphError,
    GraphNode,
    RuntimeNode,
    Workflow,
)
from nemo_distillation_plugin.names import allocate, check_candidate_name, check_workflow_name
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


def render_graph(inputs: DistillInputs) -> dict[str, GraphNode]:
    """The template before name allocation. Matrix nodes fan out in :func:`expand`."""
    chosen = candidates_of(inputs)
    deploy = {
        "engine": inputs.deployment.engine,
        "gpu": inputs.deployment.gpu,
        "tool_call_config": inputs.deployment.tool_call_config,
    }
    arms = ["original", "base", *[candidate.name for candidate in chosen]]
    return {
        "prepare": GraphNode(
            kind="convert_atif",
            inputs={"eval_results": list(inputs.eval_results), "pass_filter": dict(inputs.pass_filter)},
        ),
        "experiment": GraphNode(kind="experiment", depends_on=["prepare"], inputs={"arms": arms}),
        "deploy_base": GraphNode(
            kind="model_deployment",
            depends_on=["prepare"],
            inputs={**deploy, "role": "base", "lora_enabled": True, "model": inputs.student_model},
        ),
        "eval_original": GraphNode(
            kind="agent_eval",
            depends_on=["experiment"],
            inputs={
                "source_agent": inputs.source_agent,
                "arm": "original",
                "tasks": "${nodes.prepare.outputs.tasks}",
            },
        ),
        "eval_base": GraphNode(
            kind="agent_eval",
            depends_on=["experiment", "deploy_base"],
            inputs={
                "source_agent": inputs.source_agent,
                "arm": "base",
                "model": "${nodes.deploy_base.outputs.model}",
                "tasks": "${nodes.prepare.outputs.tasks}",
            },
        ),
        "train": GraphNode(
            kind="customization_job",
            depends_on=["prepare"],
            matrix={"candidate": [candidate.name for candidate in chosen]},
            inputs={
                "student_model": inputs.student_model,
                "dataset": "${nodes.prepare.outputs.dataset}",
            },
        ),
        "deploy": GraphNode(
            kind="model_deployment",
            depends_on=["train"],
            matrix="inherit",
            inputs={**deploy, "student_model": inputs.student_model},
        ),
        "eval_distilled": GraphNode(
            kind="agent_eval",
            depends_on=["experiment", "deploy"],
            matrix="inherit",
            inputs={
                "source_agent": inputs.source_agent,
                "tasks": "${nodes.prepare.outputs.tasks}",
                "model": "${nodes.deploy.outputs.model}",
            },
        ),
        "undeploy_base": GraphNode(
            kind="delete_deployment",
            depends_on=["eval_base"],
            always_run=True,
            inputs={"deployment": "${nodes.deploy_base.outputs.deployment}"},
        ),
        "undeploy": GraphNode(
            kind="delete_deployment",
            depends_on=["eval_distilled"],
            matrix="inherit",
            always_run=True,
            inputs={"deployment": "${nodes.deploy.outputs.deployment}"},
        ),
    }


def render(inputs: DistillInputs) -> dict[str, Any]:
    """Expanded graph and counts. No names are allocated and nothing is written."""
    chosen = candidates_of(inputs)
    graph = expand(render_graph(inputs))
    deploys = sum(1 for node in graph.values() if node.kind == "model_deployment")
    return {
        "graph": graph,
        "counts": {
            "nodes": len(graph),
            "candidates": len(chosen),
            "passing_trials": 0,
            "estimated_tokens": 0,
            "gpu_hours": inputs.deployment.gpu * deploys,
        },
    }


def assemble(name: str, workspace: str, inputs: DistillInputs) -> Workflow:
    """Build the entity, including the fileset name and every external name, before any I/O."""
    check_workflow_name(name)
    chosen = candidates_of(inputs)
    graph = expand(render_graph(inputs))
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
        elif node.kind == "model_deployment" and node.inputs.get("role") != "base":
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
            deploy_key = _deployment_target(graph, node)
            node.inputs["deployment"] = external[deploy_key]["deployment"]
        elif node.kind == "agent_eval":
            arm = node.inputs.get("arm") or candidate_of(key)
            if not isinstance(arm, str) or not arm:
                raise GraphError(f"{key} has no eval arm")
            node.inputs["arm"] = arm
            evaluation = evaluations.get(f"evaluation_{arm}")
            if evaluation is None:
                raise GraphError(f"{key} has no Evaluation for arm {arm!r}")
            node.inputs["evaluation_id"] = evaluation


def _dependency_kind(graph: dict[str, GraphNode], node: GraphNode, kind: str) -> str:
    matches = [dep for dep in node.depends_on if graph[dep].kind == kind]
    if len(matches) != 1:
        raise GraphError(f"expected one {kind} dependency, found {matches}")
    return matches[0]


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
