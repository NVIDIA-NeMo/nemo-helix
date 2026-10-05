# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A task's secret-sourced environment."""

from __future__ import annotations

import pytest
from nemo_evaluator.jobs.secret_env import JobEnvSecretSource, build_task_environment
from nemo_evaluator_sdk.resolver_protocols import MissingSecretError
from nemo_evaluator_sdk.values.common import SecretRef


def test_two_secrets_under_one_env_name_are_refused_naming_both() -> None:
    with pytest.raises(ValueError) as excinfo:
        build_task_environment([("OPENAI_API_KEY", "team-a/openai-key"), ("OPENAI_API_KEY", "team-b/openai-key")])
    assert str(excinfo.value) == (
        "conflicting secret references for environment variable 'OPENAI_API_KEY': "
        "'team-a/openai-key' and 'team-b/openai-key'"
    )


def test_one_secret_under_several_consumers_is_emitted_once() -> None:
    environment = build_task_environment([("OPENAI_API_KEY", "openai-key"), ("OPENAI_API_KEY", "openai-key")])
    assert [variable.name for variable in environment].count("OPENAI_API_KEY") == 1


def test_sandbox_plan_cannot_be_sourced_from_a_secret() -> None:
    with pytest.raises(ValueError, match="NEMO_EVALUATOR_GYM_SANDBOX_PLAN.*reserved"):
        build_task_environment([("NEMO_EVALUATOR_GYM_SANDBOX_PLAN", "ws/plan")])


def test_job_resolver_reads_only_injected_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    ref = SecretRef(root="my-workspace/openai-key")
    resolver = JobEnvSecretSource(workspace="dev")
    monkeypatch.setenv("MY_WORKSPACE_OPENAI_KEY", "never-read")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("LLM_API_KEY", "injected")
    with pytest.raises(MissingSecretError, match="nemo secrets get openai-key --workspace my-workspace"):
        resolver.env_var_for(ref, "OPENAI_API_KEY")
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(MissingSecretError, match="nemo secrets get openai-key --workspace my-workspace"):
        resolver.env_var_for(ref, "OPENAI_API_KEY")
    assert resolver.env_var_for(ref, "LLM_API_KEY") == "LLM_API_KEY"
