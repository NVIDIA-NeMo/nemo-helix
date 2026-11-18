# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
from nemo_distillation_plugin.models import GraphError
from nemo_distillation_plugin.nodes.clone import clone_config, require_fabric
from nemo_distillation_plugin.nodes.gateway import gateway_model

CONFIG: dict[str, Any] = {
    "config_format": "nemo-agents-spec-v1",
    "name": "teacher",
    "models": {"main": {"model": "default/teacher", "base_url": "http://keep", "provider": "openai"}},
    "harnesses": {"h": {"model": {"model": "default/teacher", "settings": {"temperature": 0}}}},
}


def test_clone_swaps_model_strings_and_leaves_the_source_alone() -> None:
    require_fabric("nemo-agents-spec-v1", CONFIG)
    cloned = clone_config(CONFIG, name="temp", model="default/student&adapters/default/ada")

    assert cloned["name"] == "temp"
    assert cloned["models"]["main"]["model"] == "default/student&adapters/default/ada"
    assert cloned["models"]["main"]["base_url"] == "http://keep"
    assert cloned["models"]["main"]["provider"] == "openai"
    assert cloned["harnesses"]["h"]["model"]["model"] == "default/student&adapters/default/ada"
    assert cloned["harnesses"]["h"]["model"]["settings"] == {"temperature": 0}
    assert CONFIG["models"]["main"]["model"] == "default/teacher"


def test_clone_without_a_model_keeps_the_source_model() -> None:
    cloned = clone_config(CONFIG, name="temp", model=None)
    assert cloned["models"]["main"]["model"] == "default/teacher"
    assert cloned["name"] == "temp"


def test_other_config_formats_are_refused() -> None:
    with pytest.raises(GraphError, match="nemo-agents-spec-v1"):
        require_fabric("nat-workflow-v1", {"config_format": "nat-workflow-v1"})


def test_gateway_names() -> None:
    assert gateway_model("default", {"role": "base", "lora_enabled": True, "model": "default/student"}) == (
        "default/student"
    )
    assert (
        gateway_model("default", {"lora_enabled": True, "model": "default/student", "adapter": "ada"})
        == "default/student&adapters/default/ada"
    )
    assert gateway_model("default", {"lora_enabled": False, "model": "default/out"}) == "default/out"
    assert gateway_model("default", {"lora_enabled": False, "model": "out"}) == "default/out"
