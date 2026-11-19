# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nemo_distillation_plugin.graph import expand, validate
from nemo_distillation_plugin.models import GraphError, GraphNode
from nemo_distillation_plugin.template import build_graph, render
from pydantic import ValidationError
from samples import inputs


@pytest.mark.parametrize(
    "candidate",
    [
        {"name": "a", "hyperparameters": {"epoch": 3}},
        {"name": "a", "hyperparameters": {"schedule": {"epoch": 3}}},
        {"name": "a", "hyperparameters": {"schedule": {"epochs": 0}}},
        {"name": "a", "hyperparameters": {"optimizer": {"learning_rate": 0}}},
        {"name": "a", "hyperparameters": {"optimizer": {"lr_decay_style": "step"}}},
        {"name": "a", "hyperparameters": {"lora": {"rank": 0}}},
        {"name": "a", "hyperparameters": {"lora": {"target_modules": ["q_proj"], "exclude_modules": ["*.out_proj"]}}},
        {"name": "a", "hyperparameters": {"batch": {"global_batch_size": -1}}},
        {"name": "a", "finetuning_type": "all_weights", "hyperparameters": {"lora": {"rank": 8}}},
    ],
)
def test_bad_hyperparameters_fail_when_the_workflow_is_created(candidate: dict) -> None:
    with pytest.raises(ValidationError):
        inputs(candidates=[candidate])


def test_hyperparameters_default_to_nothing_set() -> None:
    candidate = inputs(candidates=[{"name": "a"}]).candidates[0]

    assert candidate.hyperparameters.to_job() == {}


def test_default_template_feeds_both_eval_filesets_into_prepare() -> None:
    graph = build_graph(inputs())

    assert set(graph) == {
        "experiment",
        "deploy_teacher",
        "deploy_agent_original",
        "eval_original[tasks]",
        "eval_original[dataset]",
        "prepare",
        "undeploy_agent_original",
        "undeploy_teacher",
        "deploy_base",
        "deploy_agent_base",
        "eval_base[tasks]",
        "eval_base[dataset]",
        "train[lora]",
        "deploy[lora]",
        "deploy_agent[lora]",
        "eval_distilled_tasks[lora]",
        "eval_distilled_dataset[lora]",
        "undeploy_agent_base",
        "undeploy_base",
        "undeploy_agent[lora]",
        "undeploy[lora]",
    }
    assert graph["experiment"].depends_on == []
    assert graph["deploy_teacher"].depends_on == []
    assert graph["deploy_teacher"].inputs["model"] == "default/qwen36"
    assert graph["deploy_teacher"].inputs["gpu"] == 1
    assert graph["deploy_agent_original"].depends_on == ["deploy_teacher"]
    assert graph["deploy_agent_original"].inputs["image"] == "example.invalid/agent:1"
    assert graph["eval_original[tasks]"].depends_on == ["experiment", "deploy_agent_original"]
    assert graph["eval_original[dataset]"].kind == "dataset_eval"
    assert graph["prepare"].depends_on == ["eval_original[tasks]", "eval_original[dataset]"]
    assert graph["undeploy_teacher"].depends_on == ["eval_original[tasks]", "eval_original[dataset]"]
    assert graph["undeploy_teacher"].always_run is True
    assert graph["undeploy_teacher"].inputs["role"] == "teacher"
    assert graph["undeploy_agent_original"].always_run is True
    assert graph["deploy_base"].depends_on == [
        "undeploy_teacher",
        "eval_original[tasks]",
        "eval_original[dataset]",
    ]
    assert graph["train[lora]"].depends_on == [
        "prepare",
        "undeploy_teacher",
        "undeploy_agent_base",
        "undeploy_base",
    ]
    assert graph["prepare"].inputs["trajectory_filesets"] == [
        "${nodes.eval_original[tasks].outputs.fileset}",
        "${nodes.eval_original[dataset].outputs.fileset}",
    ]
    assert "tasks" not in graph["eval_base[tasks]"].inputs
    assert graph["eval_base[tasks]"].depends_on == ["experiment", "deploy_agent_base"]
    assert graph["train[lora]"].inputs["dataset"] == "${nodes.prepare.outputs.dataset}"
    assert graph["deploy[lora]"].depends_on == ["train[lora]"]
    assert graph["deploy_agent[lora]"].depends_on == ["deploy[lora]"]
    assert graph["eval_distilled_tasks[lora]"].depends_on == ["experiment", "deploy_agent[lora]"]
    assert "tasks" not in graph["eval_distilled_tasks[lora]"].inputs
    assert graph["undeploy[lora]"].depends_on == ["eval_distilled_tasks[lora]", "eval_distilled_dataset[lora]"]
    assert graph["undeploy[lora]"].always_run is True
    assert graph["undeploy_agent[lora]"].always_run is True


TWO = [
    {"name": "lora-r16", "finetuning_type": "lora"},
    {"name": "full", "finetuning_type": "all_weights"},
]


def test_two_candidates_fan_out_and_keep_their_own_deps() -> None:
    graph = build_graph(inputs(candidates=TWO))

    assert graph["deploy[lora-r16]"].depends_on == ["train[lora-r16]"]
    assert graph["deploy[full]"].depends_on == ["train[full]"]
    assert graph["eval_distilled_tasks[full]"].depends_on == ["experiment", "deploy_agent[full]"]
    assert "evaluation_lora-r16" not in graph["experiment"].inputs["arms"]
    assert graph["experiment"].inputs["arms"] == ["original", "base", "lora-r16", "full"]


def test_one_gpu_runs_base_then_each_candidate_in_turn() -> None:
    graph = build_graph(inputs(candidates=TWO, gpu_budget=1))

    assert graph["train[lora-r16]"].depends_on == [
        "prepare",
        "undeploy_teacher",
        "undeploy_agent_base",
        "undeploy_base",
    ]
    assert graph["train[full]"].depends_on == [
        "prepare",
        "undeploy_teacher",
        "undeploy_agent[lora-r16]",
        "undeploy[lora-r16]",
    ]


def test_the_budget_sets_how_many_jobs_share_the_gpus() -> None:
    wide = [{"name": "a", "gpu": 2}, {"name": "b", "gpu": 2}, {"name": "c", "gpu": 2}]

    four = inputs(candidates=wide, gpu_budget=4, teacher={"model": "default/t", "gpu": 2})
    graph = build_graph(four)
    assert four.parallel_jobs == 2
    assert graph["deploy_base"].depends_on == ["undeploy_teacher", "eval_original[tasks]", "eval_original[dataset]"]
    assert graph["train[a]"].depends_on == ["prepare", "undeploy_teacher"]
    assert graph["train[b]"].depends_on[-2:] == ["undeploy_agent_base", "undeploy_base"]
    assert graph["train[c]"].depends_on[-2:] == ["undeploy_agent[a]", "undeploy[a]"]

    two = inputs(candidates=wide[:2], gpu_budget=2, teacher={"model": "default/t", "gpu": 2})
    assert two.parallel_jobs == 1
    assert build_graph(two)["train[a]"].depends_on[-2:] == ["undeploy_agent_base", "undeploy_base"]


def test_trajectory_filesets_skip_the_teacher_stage() -> None:
    graph = build_graph(inputs(teacher=None, trajectory_filesets=["default/runs-a", "default/runs-b"]))

    assert not [key for key in graph if "teacher" in key or "original" in key]
    assert graph["prepare"].depends_on == []
    assert graph["prepare"].inputs["trajectory_filesets"] == ["default/runs-a", "default/runs-b"]
    assert graph["deploy_base"].depends_on == []
    assert graph["train[lora]"].depends_on == ["prepare", "undeploy_agent_base", "undeploy_base"]
    assert graph["experiment"].inputs["arms"] == ["base", "lora"]


def test_exactly_one_entry_point_and_a_budget_that_fits() -> None:
    with pytest.raises(ValidationError, match="either teacher or trajectory_filesets"):
        inputs(teacher=None)
    with pytest.raises(ValidationError, match="either teacher or trajectory_filesets"):
        inputs(trajectory_filesets=["default/runs"])
    with pytest.raises(ValidationError, match="exceeds gpu_budget"):
        inputs(gpu_budget=1, teacher={"model": "default/t", "gpu": 2})
    with pytest.raises(ValidationError, match="exceeds gpu_budget"):
        inputs(gpu_budget=1, candidates=[{"name": "a", "gpu": 2}])


def test_render_reports_the_parallel_jobs() -> None:
    counts = render(inputs(candidates=TWO, gpu_budget=2))["counts"]

    assert counts["parallel_jobs"] == 2
    assert counts["gpu_hours"] == 1 + 1 * 3 + 2


def test_unknown_kind_bad_binding_and_cycle_are_rejected() -> None:
    with pytest.raises(GraphError, match="unknown kind"):
        validate({"a": GraphNode(kind="promote")})
    with pytest.raises(GraphError, match="bad binding"):
        validate({"a": GraphNode(kind="convert_atif", inputs={"x": "${nope}"})})
    with pytest.raises(GraphError, match="unknown node"):
        validate({"a": GraphNode(kind="convert_atif", inputs={"x": "${nodes.missing.outputs.y}"})})
    with pytest.raises(GraphError, match="cycle"):
        validate(
            {
                "a": GraphNode(kind="convert_atif", depends_on=["b"]),
                "b": GraphNode(kind="delete_deployment", depends_on=["a"]),
            }
        )


def test_a_cycle_cannot_be_expanded() -> None:
    with pytest.raises(GraphError, match="not expanded yet"):
        expand(
            {
                "a": GraphNode(kind="convert_atif", depends_on=["b"]),
                "b": GraphNode(kind="delete_deployment", depends_on=["a"]),
            }
        )


def test_eval_inputs_are_fileset_references() -> None:
    import pytest
    from nemo_distillation_plugin.models import EvalInput
    from pydantic import ValidationError
    from samples import DATASET_REF, TASKS_REF

    graph = build_graph(inputs())
    assert graph["eval_base[tasks]"].inputs["config"] == TASKS_REF
    assert graph["eval_base[dataset]"].inputs["config"] == DATASET_REF
    assert graph["eval_base[dataset]"].inputs["dataset"] == "default/esec#dataset.jsonl"
    assert "spec" not in graph["eval_base[tasks]"].inputs

    with pytest.raises(ValidationError, match="fileset file reference"):
        EvalInput(name="tasks", kind="agent", config="no-hash-here")
    with pytest.raises(ValidationError, match="only applies to kind 'dataset'"):
        EvalInput(name="tasks", kind="agent", config=TASKS_REF, dataset="default/esec#dataset.jsonl")
    with pytest.raises(ValidationError):
        EvalInput.model_validate({"name": "tasks", "kind": "agent", "config": TASKS_REF, "tasks": [{"id": "inline"}]})
