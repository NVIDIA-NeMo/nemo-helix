# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for adapters sidecar startup and platform client configuration."""

import logging
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
from nemo_helix_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR
from nhx.common.config import Configuration
from nhx.core.models.sidecars.adapters import main as adapters_main
from respx import MockRouter


@pytest.mark.parametrize("workload_identity", [False, True])
def test_adapter_clients_use_the_deployment_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: MockRouter,
    workload_identity: bool,
) -> None:
    scheme = "https" if workload_identity else "http"
    connection_url = f"{scheme}://nemo-helix-api:8080"
    advertised_url = "https://gateway.example"
    monkeypatch.setenv("NHX_CONFIG_FILE_PATH", str(tmp_path / "absent-config.yaml"))
    monkeypatch.setenv("NHX_BASE_URL", connection_url)
    monkeypatch.setenv("NHX_ADVERTISED_BASE_URL", advertised_url)
    monkeypatch.setenv("NHX_MODELS_URL", advertised_url)
    monkeypatch.setenv("NHX_FILES_URL", advertised_url)
    monkeypatch.setenv("NIM_PEFT_SOURCE", str(tmp_path / "adapters"))
    monkeypatch.setenv("NHX_MODEL_ENTITY_WORKSPACE", "default")
    monkeypatch.setenv("NHX_MODEL_ENTITY_NAME", "base-model")
    if workload_identity:
        monkeypatch.setenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, str(tmp_path / "subject-token"))
    else:
        monkeypatch.delenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, raising=False)
    Configuration.clear_cache()
    model = respx_mock.get(f"{connection_url}/apis/models/v2/workspaces/default/models/base-model").respond(
        200,
        json={
            "id": "model-id",
            "name": "base-model",
            "workspace": "default",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        },
    )
    fileset = respx_mock.get(f"{connection_url}/apis/files/v2/workspaces/default/filesets/adapter/files").respond(
        200, json={"data": []}
    )

    try:
        with patch("nemo_helix_plugin.client.oidc_factory.resolve_workload_exchange_provider") as exchange:
            exchange.return_value.get_access_token.return_value = "workload-token"
            controller = adapters_main.AdaptersController()
            try:
                controller._models.get_model(name=controller.model_name, workspace=controller.workspace)
                controller._files.list_files(name="adapter", workspace=controller.workspace)
            finally:
                controller._client.close()
                controller._loop.close()
            if workload_identity:
                exchange.assert_called_once_with(base_url=connection_url, subject_token_file=tmp_path / "subject-token")
            else:
                exchange.assert_not_called()
    finally:
        Configuration.clear_cache()

    for route in (model, fileset):
        headers = route.calls.last.request.headers
        if workload_identity:
            assert headers["Authorization"] == "Bearer workload-token"
            assert "X-NHX-Principal-Id" not in headers
        else:
            assert headers["X-NHX-Principal-Id"] == "service:models"
            assert "Authorization" not in headers


def test_run_without_nim_peft_source_is_a_graceful_noop(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """With NIM_PEFT_SOURCE unset, run() must warn and return instead of raising."""
    monkeypatch.delenv("NIM_PEFT_SOURCE", raising=False)
    stop_signal = threading.Event()

    with (
        patch.object(adapters_main, "get_platform_config"),
        patch.object(adapters_main, "Loop") as loop_cls,
        caplog.at_level(logging.WARNING, logger=adapters_main.logger.name),
    ):
        adapters_main.run(parent_stop_signal=stop_signal)

    loop_cls.assert_not_called()
    assert "NIM_PEFT_SOURCE is not set" in caplog.text
