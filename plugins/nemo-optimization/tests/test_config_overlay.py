# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from nemo_optimization.config_overlay import (
    apply_suggestions,
    nest_dotted_paths,
    suggestions_to_profile_overlay,
)


def test_nest_dotted_paths() -> None:
    nested = nest_dotted_paths(
        {
            "models.default.temperature": 0.4,
            "models.default.top_p": 0.85,
            "harness.settings.workflow.max_tool_calls": 5,
        }
    )
    assert nested["models"]["default"]["temperature"] == 0.4
    assert nested["models"]["default"]["top_p"] == 0.85
    assert nested["harness"]["settings"]["workflow"]["max_tool_calls"] == 5


def test_apply_suggestions_strips_optimizer_metadata() -> None:
    base = {
        "schema_version": "fabric.agent/v1alpha1",
        "models": {"default": {"temperature": 0.0}},
        "optimizer": {
            "numeric": {"enabled": True},
            "search_space": {"temperature": {"low": 0.0, "high": 0.8}},
        },
        "optimizable_params": {"legacy": True},
    }
    trial = apply_suggestions(base, {"models.default.temperature": 0.6})
    assert trial["models"]["default"]["temperature"] == 0.6
    assert "optimizer" not in trial
    assert "optimizable_params" not in trial


def test_suggestions_to_profile_overlay_names_trial() -> None:
    overlay = suggestions_to_profile_overlay({"models.default.temperature": 0.2}, 7)
    assert overlay["metadata"]["name"] == "trial-007"
    assert overlay["models"]["default"]["temperature"] == 0.2


def test_nest_rejects_conflicting_intermediate_types() -> None:
    with pytest.raises(KeyError):
        nest_dotted_paths({"a": 2, "a.b": 1})
