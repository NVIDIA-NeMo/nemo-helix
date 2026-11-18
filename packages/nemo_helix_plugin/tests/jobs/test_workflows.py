# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nemo_helix_plugin.jobs.spec import HelixJobSpec
from nemo_helix_plugin.jobs.workflows import workflow_order
from pydantic import ValidationError


def test_sequential_wire_shape_omits_workflow_fields() -> None:
    spec = HelixJobSpec.model_validate(
        {"steps": [{"name": "work", "executor": {"provider": "cpu", "container": {"image": "test"}}}]}
    )
    wire = spec.model_dump(mode="json")
    assert "workflow" not in wire
    assert "inputs" not in wire["steps"][0]
    assert "depends_on" not in wire["steps"][0]


@pytest.mark.parametrize(
    "extra",
    [
        {"depends_on": None},
        {"inputs": None, "depends_on": []},
        {"outputs": []},
    ],
)
def test_unsupported_or_null_workflow_fields_fail(extra: dict) -> None:
    with pytest.raises(ValidationError):
        HelixJobSpec.model_validate(
            {
                "workflow": {"mode": "continuous"},
                "steps": [{"name": "work", "executor": {"provider": "cpu", "container": {"image": "test"}}, **extra}],
            }
        )


@pytest.mark.parametrize("config", [{"mode": "forever"}, {"input_change_policy": "interrupt"}, {"unknown": True}])
def test_workflow_configuration_is_closed(config: dict) -> None:
    with pytest.raises(ValidationError):
        HelixJobSpec.model_validate(
            {
                "workflow": config,
                "steps": [
                    {"name": "work", "depends_on": [], "executor": {"provider": "cpu", "container": {"image": "test"}}}
                ],
            }
        )


def test_cycle_error_contains_a_closed_dependency_chain() -> None:
    with pytest.raises(ValueError, match="alpha -> beta -> gamma -> alpha"):
        workflow_order({"root": [], "alpha": ["beta"], "beta": ["gamma"], "gamma": ["alpha"]})


def test_deep_workflow_does_not_depend_on_python_recursion_limit() -> None:
    dependencies = {f"node-{i}": [f"node-{i - 1}"] if i else [] for i in range(2000)}
    assert workflow_order(dependencies) == list(dependencies)
