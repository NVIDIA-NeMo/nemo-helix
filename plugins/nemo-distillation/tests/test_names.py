# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nemo_distillation_plugin.models import MAX_RESOURCE_NAME_LEN, GraphError, GraphNode
from nemo_distillation_plugin.names import allocate
from nemo_distillation_plugin.template import assemble
from samples import inputs


def test_names_are_assigned_once_in_graph_order() -> None:
    workflow = assemble("distill", "default", inputs())
    names = [value for node in workflow.nodes.values() for value in node.external.values()]

    assert names == [f"distill-t{number}" for number in range(1, len(names) + 1)]
    assert workflow.nodes["experiment"].external["experiment"] == "distill-t1"
    assert workflow.nodes["train[lora]"].external["model"] != workflow.nodes["train[lora]"].external["job"]
    assert workflow.experiment == workflow.nodes["experiment"].external["experiment"]
    assert workflow.fileset == "distill-data"
    assert workflow.graph["train[lora]"].inputs["finetuning_type"] == "lora"
    assert workflow.graph["deploy[lora]"].inputs["lora_enabled"] is True
    assert workflow.graph["deploy[lora]"].inputs["adapter"] == workflow.nodes["train[lora]"].external["model"]
    assert (
        workflow.graph["undeploy[lora]"].inputs["deployment"] == workflow.nodes["deploy[lora]"].external["deployment"]
    )
    assert (
        workflow.graph["undeploy_teacher"].inputs["deployment"]
        == workflow.nodes["deploy_teacher"].external["deployment"]
    )
    assert workflow.graph["deploy_agent_original"].inputs["model"] == workflow.graph["deploy_teacher"].inputs["model"]
    assert (
        workflow.graph["eval_original[tasks]"].inputs["agent_name"]
        == workflow.nodes["deploy_agent_original"].external["agent"]
    )
    assert (
        workflow.graph["eval_original[tasks]"].inputs["evaluation_id"]
        == (workflow.nodes["experiment"].external["evaluation_original"])
    )
    assert (
        workflow.graph["eval_original[dataset]"].inputs["evaluation_id"]
        == workflow.graph["eval_original[tasks]"].inputs["evaluation_id"]
    )


def test_full_weights_deploys_the_output_model() -> None:
    workflow = assemble("distill", "default", inputs(candidates=[{"name": "full", "finetuning_type": "all_weights"}]))

    output = workflow.nodes["train[full]"].external["model"]
    assert workflow.graph["deploy[full]"].inputs["lora_enabled"] is False
    assert workflow.graph["deploy[full]"].inputs["model"] == f"default/{output}"


def test_a_name_that_cannot_fit_a_suffix_is_rejected() -> None:
    with pytest.raises(GraphError, match="at most"):
        allocate("a" * 53, {"prepare": GraphNode(kind="convert_atif")})
    assert len("a" * 53 + "-t1") > MAX_RESOURCE_NAME_LEN

    with pytest.raises(GraphError, match="workflow name"):
        assemble("a" * 40, "default", inputs())
