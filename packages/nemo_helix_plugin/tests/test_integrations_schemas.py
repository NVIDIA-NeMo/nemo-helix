# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

import pytest
from nemo_helix_plugin.integrations import IntegrationsSpec, MlflowIntegration, WandbIntegration
from pydantic import ValidationError


class TestWandbIntegration:
    def test_accepts_full_config(self) -> None:
        wandb = WandbIntegration.model_validate(
            {
                "project": "proj",
                "name": "run-1",
                "entity": "team",
                "tags": ["sft"],
                "notes": "notes",
                "base_url": "https://wandb.example.com",
                "api_key_secret": "default/wandb-key",
            },
        )
        assert wandb.project == "proj"
        assert wandb.name == "run-1"
        assert wandb.api_key_secret is not None
        assert wandb.api_key_secret.root == "default/wandb-key"

    def test_run_name_deprecated_shim_maps_to_name(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            wandb = WandbIntegration.model_validate({"run_name": "legacy-run"})
        assert wandb.name == "legacy-run"
        assert len(caught) == 1
        assert issubclass(caught[0].category, DeprecationWarning)
        assert "run_name" in str(caught[0].message)

    def test_rejects_enabled_flag(self) -> None:
        with pytest.raises(ValidationError, match="enabled"):
            WandbIntegration.model_validate({"enabled": True, "project": "p"})

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            WandbIntegration.model_validate({"project": "p", "unknown": True})


class TestMlflowIntegration:
    def test_accepts_full_config(self) -> None:
        mlflow = MlflowIntegration.model_validate(
            {
                "experiment_name": "exp",
                "name": "run-1",
                "tracking_uri": "http://mlflow:5000",
                "tags": {"team": "nlp"},
                "description": "desc",
            },
        )
        assert mlflow.experiment_name == "exp"
        assert mlflow.name == "run-1"
        assert mlflow.tags == {"team": "nlp"}

    def test_run_name_deprecated_shim_maps_to_name(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mlflow = MlflowIntegration.model_validate({"run_name": "legacy-run"})
        assert mlflow.name == "legacy-run"
        assert len(caught) == 1
        assert issubclass(caught[0].category, DeprecationWarning)
        assert "run_name" in str(caught[0].message)


class TestIntegrationsSpec:
    def test_presence_enables_integrations(self) -> None:
        spec = IntegrationsSpec.model_validate(
            {
                "wandb": {"project": "p"},
                "mlflow": {"tracking_uri": "http://mlflow:5000"},
            },
        )
        assert spec.wandb is not None
        assert spec.mlflow is not None

    def test_rejects_empty_wandb_block(self) -> None:
        with pytest.raises(ValidationError, match="integrations.wandb"):
            IntegrationsSpec.model_validate({"wandb": {}})

    def test_rejects_empty_mlflow_block(self) -> None:
        with pytest.raises(ValidationError, match="integrations.mlflow"):
            IntegrationsSpec.model_validate({"mlflow": {}})

    def test_rejects_report_to_on_input(self) -> None:
        with pytest.raises(ValidationError, match="report_to"):
            IntegrationsSpec.model_validate({"report_to": ["wandb"]})

    def test_rejects_enabled_on_input(self) -> None:
        with pytest.raises(ValidationError, match="enabled"):
            IntegrationsSpec.model_validate({"wandb": {"enabled": False}})
