# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from nemo_distillation_plugin.models import DistillInputs

TASKS_REF = "default/esec-eval-config#eval-config.task-driven.json"
DATASET_REF = "default/esec-eval-config#eval-config.dataset-driven.yml"

#: What the eval config files in the fileset contain. The workflow carries only the references.
EVAL_CONFIGS: dict[str, dict[str, Any]] = {
    TASKS_REF: {
        "tasks": [
            {
                "id": "classify-0",
                "intent": "Classify.",
                "inputs": {"instruction": "hi"},
                "metrics": [{"type": "string-check"}],
            }
        ],
        "max_concurrent_tasks": 1,
    },
    DATASET_REF: {
        "dataset": "<workspace>/<fileset>#dataset.jsonl",
        "metrics": [{"name": "accuracy"}],
        "prompt_template": "{{ item.user_message }}",
    },
}

EVALS = [
    {"name": "tasks", "kind": "agent", "config": TASKS_REF},
    {"name": "dataset", "kind": "dataset", "config": DATASET_REF, "dataset": "default/esec#dataset.jsonl"},
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
