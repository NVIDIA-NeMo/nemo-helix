# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for evaluator_agent_eval.task_config."""

from pathlib import Path

import pytest
from evaluator_agent_eval.task_config import load_agentic_use_task_config


def test_task_config_parses_evaluator_extension(evaluator_task_dir: Path):
    config = load_agentic_use_task_config(evaluator_task_dir)

    assert config.suite_id == "custom_suite"
    assert config.suite_version == "v-test"
    assert config.evaluator.surface.constraint == "standalone_sdk"
    assert config.evaluator.expected.required_terms == [
        "packages/nhx_evals_sdk",
        "Evaluator",
        "ExactMatchMetric",
        "2+2?",
        "Capital of France?",
    ]


def test_task_config_rejects_missing_allowed_surfaces(tmp_path: Path):
    (tmp_path / "task.toml").write_text(
        """
version = "1.0"

[evaluator.surface]
constraint = "standalone_sdk"
allowed = []
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="allowed"):
        load_agentic_use_task_config(tmp_path)


def test_manifest_tasks_have_evaluator_config_and_verifier():
    agentic_use_dir = Path(__file__).parents[2]
    manifest_path = agentic_use_dir / "manifests" / "evaluator_agent_benchmark_mvp.txt"
    task_names = [
        stripped
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        for stripped in [line.strip()]
        if stripped and not stripped.startswith("#")
    ]

    assert task_names == [
        "evaluator-standalone-sdk-surface-discovery",
        "evaluator-standalone-sdk-simple-exact-match",
        "evaluator-standalone-sdk-agent-target",
        "evaluator-standalone-sdk-surface-adherence-metric",
    ]
    for task_name in task_names:
        task_dir = agentic_use_dir / task_name
        config = load_agentic_use_task_config(task_dir)

        assert config.evaluator.surface.constraint == "standalone_sdk"
        assert config.evaluator.surface.allowed == ["standalone_sdk"]
        assert (task_dir / "instruction.md").exists()
        assert (task_dir / "tests" / "test.sh").exists()
        assert (task_dir / "tests" / "test_outputs.py").exists()
        assert (task_dir / "tests" / "task_metrics.py").exists()
