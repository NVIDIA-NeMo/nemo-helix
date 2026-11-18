# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from nemo_distillation_plugin.models import DistillInputs

EVALS = [
    {
        "name": "tasks",
        "kind": "agent",
        "spec": {
            "tasks": [
                {
                    "id": "classify-0",
                    "intent": "Classify.",
                    "inputs": {"instruction": "hi"},
                    "metrics": [{"type": "string-check"}],
                }
            ]
        },
    },
    {
        "name": "dataset",
        "kind": "dataset",
        "spec": {
            "dataset": "default/esec#dataset.jsonl",
            "metrics": [{"name": "accuracy"}],
            "prompt_template": "{{ item.user_message }}",
        },
    },
]

INPUTS: dict[str, Any] = {
    "source_agent": "default/teacher",
    "student_model": "default/student",
    "teacher": {"model": "default/qwen36", "gpu": 1},
    "agent_deployment": {"mode": "k8s", "image": "example.invalid/agent:1"},
    "evals": EVALS,
}


def inputs(**overrides: Any) -> DistillInputs:
    data: dict[str, Any] = dict(INPUTS)
    data.update(overrides)
    return DistillInputs(**data)
