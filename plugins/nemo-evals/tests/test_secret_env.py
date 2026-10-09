# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A task's secret-sourced environment."""

from __future__ import annotations

import pytest
from nemo_evals.jobs.secret_env import JobEnvSecretSource, build_task_environment
from nemo_helix_plugin.jobs.exceptions import HelixJobCompilationError
from nhx_evals_sdk.resolver_protocols import MissingSecretError
from nhx_evals_sdk.values.common import SecretRef


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
    with pytest.raises(ValueError, match="NEMO_EVALS_GYM_SANDBOX_PLAN.*reserved"):
        build_task_environment([("NEMO_EVALS_GYM_SANDBOX_PLAN", "ws/plan")])


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


@pytest.mark.parametrize(
    "env_name",
    [
        "NEMO_EVALS_ALLOW_INSECURE_CLOUDPICKLE_METRICS",
        "nemo_evals_allow_insecure_cloudpickle_metrics",
        "NHX_CONFIG_FILE_PATH",
    ],
)
def test_submitter_secrets_cannot_set_evals_or_platform_config(env_name: str) -> None:
    """Config reads env case-insensitively, and a worker would honour any of these as operator settings."""
    with pytest.raises(ValueError, match="reserved"):
        build_task_environment([(env_name, "ws/attacker-controlled")])


def test_other_plugins_env_names_stay_available_to_secrets() -> None:
    environment = build_task_environment([("NEMO_AGENTS_IGW_API_KEY", "igw-key")])
    assert "NEMO_AGENTS_IGW_API_KEY" in [variable.name for variable in environment]


def test_secret_env_errors_reject_the_submission_rather_than_failing_the_server() -> None:
    """The job API maps HelixJobCompilationError to 422; any other compile error becomes an opaque 500."""
    with pytest.raises(HelixJobCompilationError):
        build_task_environment([("nemo_evals_allow_insecure_cloudpickle_metrics", "ws/self-grant")])
    with pytest.raises(HelixJobCompilationError):
        build_task_environment([("OPENAI_API_KEY", "team-a/key"), ("OPENAI_API_KEY", "team-b/key")])
