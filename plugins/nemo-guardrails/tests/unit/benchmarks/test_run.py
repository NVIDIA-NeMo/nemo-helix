# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import httpx
from nemo_guardrails_plugin.benchmarks.paths import RunPaths, build_run_paths
from nemo_guardrails_plugin.benchmarks.run import _build_nhx_process, _smoke_test
from nemo_guardrails_plugin.benchmarks.seeding import SeededResources
from nemo_helix import NeMoHelix


def _run_paths(tmp_path: Path) -> RunPaths:
    return build_run_paths(
        nhx_repo_root=tmp_path / "nemo-helix",
        nemoguardrails_repo_root=tmp_path / "NeMo-Guardrails",
        run_id="test-run",
    )


def _seeded_resources() -> SeededResources:
    return SeededResources(
        workspace="benchmark",
        app_provider_name="app-provider",
        cs_provider_name="cs-provider",
        app_model_entity="benchmark/app-model",
        cs_model_entity="benchmark/cs-model",
        guardrail_config_name="content-safety",
        vm_name="guardrails-vm",
        no_guardrails_vm_name="control-vm",
    )


def test_nhx_process_uses_benchmark_service_subset(tmp_path: Path) -> None:
    process = _build_nhx_process(_run_paths(tmp_path))

    assert process.cmd == [
        sys.executable,
        "-m",
        "nemo_guardrails_plugin.benchmarks.platform_runner",
    ]


def test_smoke_test_retries_until_virtual_model_route_is_ready(monkeypatch) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(404, request=request, json={"error": "not ready"})
        return httpx.Response(200, request=request, json={"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("nemo_guardrails_plugin.benchmarks.run.time.sleep", lambda _seconds: None)
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        sdk = NeMoHelix(base_url="http://platform.test", http_client=http_client, max_retries=0)

        _smoke_test(sdk, _seeded_resources())

    assert attempts == 2
