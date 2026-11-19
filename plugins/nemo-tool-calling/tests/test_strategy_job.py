# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from nemo_agent_optimization_plugin.discovery import discover_strategy_jobs
from nemo_helix_plugin.errors import LocalRunError
from nemo_tool_calling_plugin.jobs.optimize import ToolCallingOptimizeJob, parse_prompts
from nemo_tool_calling_plugin.schemas.optimize import ToolCallingOptimizeSpec
from pydantic import ValidationError


def test_discovery_finds_the_tool_calling_strategy() -> None:
    discover_strategy_jobs.cache_clear()

    assert discover_strategy_jobs()["tool-calling"] is ToolCallingOptimizeJob


def test_spec_takes_the_dataset_from_the_router_bundle_fields() -> None:
    spec = ToolCallingOptimizeSpec.model_validate(
        {
            "agent": "default/data-designer-toolcall",
            "optimize_config": "prompts.jsonl",
            "optimize_config_fileset": "default/dd-toolcall-prompts",
            "workspace": "default",
        }
    )

    assert spec.runs_per_prompt == 3
    assert spec.max_turns == 30
    assert spec.call_budget == 3
    assert spec.analyzer_model == "default/nemotron3-5-super"
    assert spec.verify == ["config-fix", "schema-fix"]


def test_spec_accepts_inline_prompts() -> None:
    spec = ToolCallingOptimizeSpec.model_validate({"agent": "a", "prompts": [{"id": "p", "prompt": "x"}]})

    assert spec.prompts is not None and spec.prompts[0].id == "p"


@pytest.mark.parametrize(
    "payload",
    [
        {"agent": "a"},
        {"agent": "a", "optimize_config": "prompts.jsonl"},
        {"agent": "a", "prompts": [{"id": "p", "prompt": "x"}], "optimize_config": "p.jsonl"},
        {"agent": "a", "prompts": [{"id": "p", "prompt": "x"}], "output": "./out"},
    ],
)
def test_spec_rejects_missing_or_ambiguous_datasets(payload: dict) -> None:
    with pytest.raises(ValidationError):
        ToolCallingOptimizeSpec.model_validate(payload)


def test_parse_prompts_reads_the_eval_jsonl_shape() -> None:
    prompts = parse_prompts(['{"id": "product-reviews", "prompt": "100 reviews"}', "", '{"input": "orders"}'])

    assert [(prompt.id, prompt.prompt) for prompt in prompts] == [
        ("product-reviews", "100 reviews"),
        ("prompt-3", "orders"),
    ]


def test_parse_prompts_rejects_rows_without_a_prompt() -> None:
    with pytest.raises(LocalRunError):
        parse_prompts(['{"id": "x"}'])
