# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nemo_distillation_plugin.graph import expand, validate
from nemo_distillation_plugin.models import DistillInputs, GraphError, GraphNode
from nemo_distillation_plugin.template import render_graph

INPUTS = DistillInputs(source_agent="default/teacher", student_model="default/student")


def test_default_template_expands_one_lora_and_orders_the_experiment_first() -> None:
    graph = expand(render_graph(INPUTS))

    assert set(graph) == {
        "prepare",
        "experiment",
        "deploy_base",
        "eval_original",
        "eval_base",
        "train[lora]",
        "deploy[lora]",
        "eval_distilled[lora]",
        "undeploy_base",
        "undeploy[lora]",
    }
    assert graph["experiment"].depends_on == ["prepare"]
    assert graph["eval_original"].depends_on == ["experiment"]
    assert "train[lora]" not in graph["eval_original"].depends_on
    assert graph["eval_base"].depends_on == ["experiment", "deploy_base"]
    assert graph["deploy[lora]"].depends_on == ["train[lora]"]
    assert graph["eval_distilled[lora]"].depends_on == ["experiment", "deploy[lora]"]
    assert graph["eval_distilled[lora]"].inputs["model"] == "${nodes.deploy[lora].outputs.model}"
    assert graph["eval_distilled[lora]"].inputs["tasks"] == "${nodes.prepare.outputs.tasks}"
    assert graph["undeploy[lora]"].always_run is True


def test_two_candidates_fan_out_and_keep_their_own_deps() -> None:
    inputs = DistillInputs(
        source_agent="default/teacher",
        student_model="default/student",
        candidates=[
            {"name": "lora-r16", "finetuning_type": "lora"},
            {"name": "full", "finetuning_type": "all_weights"},
        ],
    )
    graph = expand(render_graph(inputs))

    assert graph["deploy[lora-r16]"].depends_on == ["train[lora-r16]"]
    assert graph["deploy[full]"].depends_on == ["train[full]"]
    assert graph["eval_distilled[full]"].depends_on == ["experiment", "deploy[full]"]
    assert "evaluation_lora-r16" not in graph["experiment"].inputs["arms"]
    assert graph["experiment"].inputs["arms"] == ["original", "base", "lora-r16", "full"]


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
