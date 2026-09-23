# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from nemo_helix_ext.cli.telemetry.events import (
    CommandInvokedEvent,
    JobRunEvent,
    OnboardingStepEvent,
    TaskStatusEnum,
    is_ci_environment,
)
from pydantic import ValidationError


class TestCommandInvokedEvent:
    def test_defaults_and_aliases(self):
        e = CommandInvokedEvent(command="jobs create", task_status=TaskStatusEnum.COMPLETED, duration_sec=1.25)
        d = e.model_dump(by_alias=True, mode="json")
        assert d["nemoSource"] == "platform"
        assert d["command"] == "jobs create"
        assert d["taskStatus"] == "completed"
        assert d["durationSec"] == 1.25
        assert d["agentMode"] is False
        assert d["isCi"] is False
        assert e._event_name == "command_invoked"
        assert e._schema_version == "1.10"

    def test_no_free_text_fields_beyond_known(self):
        # privacy guard: the event cannot carry arbitrary payloads
        with pytest.raises(ValidationError):
            CommandInvokedEvent(command="x", task_status=TaskStatusEnum.COMPLETED, duration_sec=0, prompt="secret")


class TestOnboardingStepEvent:
    def test_fields(self):
        e = OnboardingStepEvent(step="provider_connected", task_status=TaskStatusEnum.COMPLETED, provider_type="openai")
        d = e.model_dump(by_alias=True, mode="json")
        assert d["step"] == "provider_connected"
        assert d["providerType"] == "openai"
        assert d["modelsDiscoveredBucket"] == "undefined"
        assert d["skillsTarget"] == "undefined"
        assert d["agentDeployed"] is False
        assert e._event_name == "onboarding_step"


class TestJobRunEvent:
    def test_token_defaults_are_minus_one(self):
        e = JobRunEvent(job_type="auditor.audit", task_status=TaskStatusEnum.ERROR, duration_sec=10.0)
        d = e.model_dump(by_alias=True, mode="json")
        assert d["jobType"] == "auditor.audit"
        assert d["inputTokens"] == -1
        assert d["outputTokens"] == -1
        assert d["model"] == "undefined"
        assert d["plugins"] == []
        assert e._event_name == "job_run"


class TestCiDetection:
    @pytest.mark.parametrize("var", ["CI", "GITLAB_CI", "GITHUB_ACTIONS", "BUILDKITE", "JENKINS_URL"])
    def test_ci_env_detected(self, monkeypatch, var):
        monkeypatch.setenv(var, "true")
        assert is_ci_environment() is True

    def test_ci_env_presence_value_detected(self, monkeypatch):
        monkeypatch.setenv("TEAMCITY_VERSION", "2026.07")
        assert is_ci_environment() is True

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
    def test_ci_env_falsey_value_is_not_detected(self, monkeypatch, value):
        monkeypatch.setenv("CI", value)
        assert is_ci_environment() is False

    def test_no_ci_env(self, monkeypatch):
        for var in ("CI", "GITLAB_CI", "GITHUB_ACTIONS", "BUILDKITE", "JENKINS_URL"):
            monkeypatch.delenv(var, raising=False)
        assert is_ci_environment() is False
