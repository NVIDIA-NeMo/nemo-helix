# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the packaged Email Security Triage sample assets."""

import json

import yaml
from email_security_triage.resources import sample_file


def test_agent_config_is_packaged_and_valid() -> None:
    content = sample_file("agent.yaml").read_text(encoding="utf-8")
    config = yaml.safe_load(content)

    assert config["config_format"] == "nemo-agents-spec-v1"
    assert config["name"] == "email-security-triage"
    assert config["default_harness"] == "deepagents"


def test_dataset_is_packaged_and_valid() -> None:
    content = sample_file("dataset.jsonl").read_text(encoding="utf-8")
    records = [json.loads(line) for line in content.splitlines() if line]

    assert records
    assert all("user_message" in record and "emails" in record for record in records)


def test_evaluation_configs_are_packaged_and_valid() -> None:
    dataset_driven = yaml.safe_load(sample_file("eval-config.dataset-driven.yml").read_text(encoding="utf-8"))
    task_driven = json.loads(sample_file("eval-config.task-driven.json").read_text(encoding="utf-8"))

    assert isinstance(dataset_driven, dict)
    assert isinstance(task_driven, dict)
    assert dataset_driven
    assert task_driven
