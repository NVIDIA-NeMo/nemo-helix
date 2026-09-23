# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for evaluator_agent_eval unit tests."""

from pathlib import Path

import pytest


@pytest.fixture()
def evaluator_task_dir(tmp_path: Path) -> Path:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(
        """
version = "1.0"

[metadata]
author_name = "NeMo Helix Team"
suite_id = "custom_suite"
suite_version = "v-test"

[evaluator.surface]
constraint = "standalone_sdk"
allowed = ["standalone_sdk"]
forbidden = ["cli", "plugin_sdk", "legacy_service"]

[evaluator.expected]
required_terms = ["packages/nemo_evaluator_sdk", "Evaluator", "ExactMatchMetric", "2+2?", "Capital of France?"]

[evaluator]
forbidden_patterns = ["services/", "nemo evaluation", "plugin SDK"]
""".strip(),
        encoding="utf-8",
    )
    return task_dir


@pytest.fixture()
def agent_log_dir(tmp_path: Path) -> Path:
    path = tmp_path / "agent"
    path.mkdir()
    return path


@pytest.fixture()
def atif_payload() -> dict[str, object]:
    return {
        "schema_version": "ATIF-v1.6",
        "session_id": "session-1",
        "agent": {"name": "codex", "version": "test", "model_name": "gpt-test"},
        "steps": [
            {
                "step_id": 1,
                "source": "user",
                "message": "Use the standalone SDK.",
            },
            {
                "step_id": 2,
                "source": "agent",
                "message": "Run a search.",
                "tool_calls": [
                    {"tool_call_id": "call-1", "function_name": "shell", "arguments": {"command": "rg Evaluator"}},
                    {"tool_call_id": "call-2", "function_name": "shell", "arguments": {"command": "pytest"}},
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "call-2",
                            "content": "pytest failed with exit code 1",
                        }
                    ]
                },
            },
            {
                "step_id": 3,
                "source": "agent",
                "reasoning_content": "Retry after the failed command and corrected the assertion.",
            },
        ],
    }
