# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nemo_helix_plugin.integrations import IntegrationsSpec, WandbIntegration
from nemo_helix_plugin.schema import SecretRef
from nhx.customization_common.integrations import collect_integration_secret_envs, warn_incomplete_integrations


class TestCollectIntegrationSecretEnvs:
    def test_no_integrations(self) -> None:
        assert collect_integration_secret_envs(None) == []

    def test_wandb_without_secret(self) -> None:
        integrations = IntegrationsSpec.model_validate({"wandb": {"project": "my-project"}})
        assert collect_integration_secret_envs(integrations) == []

    def test_wandb_with_secret(self) -> None:
        integrations = IntegrationsSpec(
            wandb=WandbIntegration(
                project="my-project",
                api_key_secret=SecretRef("my-wandb-secret"),
            ),
        )
        result = collect_integration_secret_envs(integrations)
        assert len(result) == 1
        assert result[0] == {
            "name": "WANDB_API_KEY",
            "from_secret": {"name": "my-wandb-secret"},
        }

    def test_wandb_with_workspace_qualified_secret(self) -> None:
        integrations = IntegrationsSpec(
            wandb=WandbIntegration(api_key_secret=SecretRef("my-workspace/my-wandb-secret")),
        )
        result = collect_integration_secret_envs(integrations)
        assert result[0] == {
            "name": "WANDB_API_KEY",
            "from_secret": {"name": "my-workspace/my-wandb-secret"},
        }

    def test_mlflow_only_no_secrets(self) -> None:
        integrations = IntegrationsSpec.model_validate({"mlflow": {"experiment_name": "my-experiment"}})
        assert collect_integration_secret_envs(integrations) == []


class TestWarnIncompleteIntegrations:
    def test_warns_when_wandb_missing_secret(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        integrations = IntegrationsSpec.model_validate({"wandb": {"project": "my-project"}})
        caplog.set_level("WARNING")

        warn_incomplete_integrations(integrations)

        assert "api_key_secret is missing" in caplog.text

    def test_warns_when_mlflow_missing_tracking_uri(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        integrations = IntegrationsSpec.model_validate({"mlflow": {"experiment_name": "exp"}})
        caplog.set_level("WARNING")

        warn_incomplete_integrations(integrations)

        assert "tracking_uri is missing" in caplog.text
