# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nooa tracing adapter tests."""

from typing import cast

import pytest
from nemo_compass_plugin.analysis import observability


def test_setup_maps_intake_endpoint_auth_resource_and_session(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}
    exporter = object()

    def fake_otlp(*, endpoint: str, headers: dict[str, str] | None) -> object:
        seen["otlp"] = (endpoint, headers)
        return exporter

    def fake_enable_tracing(*, exporters: list[object], extra_resource_attrs: dict[str, str]) -> None:
        seen["enable"] = (exporters, extra_resource_attrs)

    monkeypatch.setattr(observability.exporters, "otlp", fake_otlp)
    monkeypatch.setattr(observability, "enable_tracing", fake_enable_tracing)
    monkeypatch.setattr(observability, "set_session", lambda session_id: seen.setdefault("session", session_id))
    monkeypatch.setattr(observability, "_otlp_auth_headers", lambda base_url: {"Authorization": "Bearer test"})

    configured = observability.setup_compass_observability(
        base_url="https://platform.example/",
        workspace="workspace",
        target_agent="target-agent",
        evaluation_context=observability.CompassEvaluationContext(
            evaluation_name="analyst-eval-1",
            test_case_name="smoke/g1",
        ),
    )

    assert configured.endpoint == ("https://platform.example/apis/intake/v2/workspaces/workspace/ingest/otlp/v1/traces")
    assert seen["otlp"] == (configured.endpoint, {"Authorization": "Bearer test"})
    enabled_exporters, attributes = cast(tuple[list[object], dict[str, str]], seen["enable"])
    assert enabled_exporters == [exporter]
    assert attributes["gen_ai.agent.name"] == observability.COMPASS_OBSERVABILITY_AGENT_NAME
    assert attributes["nemo.insights.target_agent"] == "target-agent"
    assert attributes["nemo.evaluation.name"] == "analyst-eval-1"
    assert attributes["nemo.test_case.name"] == "smoke/g1"
    assert seen["session"] == configured.session_id


def test_shutdown_flushes_nooa_traces(monkeypatch: pytest.MonkeyPatch) -> None:
    flushed: list[bool] = []
    monkeypatch.setattr(observability, "flush_traces", lambda: flushed.append(True))

    observability.CompassObservability(endpoint="endpoint", session_id="session").shutdown()

    assert flushed == [True]


def test_remote_http_export_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_otlp(*args: object, **kwargs: object) -> object:
        raise AssertionError("remote HTTP must be rejected before configuring the exporter")

    monkeypatch.setattr(observability.exporters, "otlp", unexpected_otlp)

    with pytest.raises(ValueError, match="must use HTTPS"):
        observability.setup_compass_observability(
            base_url="http://platform.example",
            workspace="workspace",
            target_agent="target-agent",
        )


@pytest.mark.parametrize("base_url", ["http://localhost:8080", "http://127.0.0.1:8080", "http://[::1]:8080"])
def test_loopback_http_export_is_allowed(base_url: str) -> None:
    assert observability.build_intake_otlp_traces_endpoint(base_url=base_url, workspace="workspace") == (
        f"{base_url}/apis/intake/v2/workspaces/workspace/ingest/otlp/v1/traces"
    )
