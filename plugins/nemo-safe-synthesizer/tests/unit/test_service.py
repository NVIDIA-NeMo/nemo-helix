# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_safe_synthesizer_plugin.service import SafeSynthesizerService


def test_service_metadata_preserves_public_name():
    service = SafeSynthesizerService()

    assert service.name == "safe-synthesizer"
    assert service.dependencies == ["entities", "auth", "jobs", "secrets", "files"]


def _unresolved_refs(schemas: dict, root: str) -> set[str]:
    seen: set[str] = set()
    unresolved: set[str] = set()
    pending = [root]
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        if name not in schemas:
            unresolved.add(name)
            continue
        pending.extend(re.findall(r'"#/components/schemas/([^"]+)"', json.dumps(schemas[name])))
    return unresolved


def test_service_routes_include_safe_synthesizer_jobs_path():
    pytest.importorskip("nemo_safe_synthesizer.config.job")

    service = SafeSynthesizerService()
    app = FastAPI()
    for spec in service.get_routers():
        app.include_router(spec.router, prefix=spec.prefix)

    client = TestClient(app)
    spec = client.get("/openapi.json").json()

    assert "/v2/workspaces/{workspace}/jobs" in spec["paths"]
    assert "/v2/workspaces/{workspace}/jobs/{job}/results/adapter/download" in spec["paths"]
    report_data_download = spec["paths"][
        "/v2/workspaces/{workspace}/jobs/{job}/results/evaluation-report-data/download"
    ]
    assert report_data_download["get"]["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/EvaluationReportData"
    }
    assert _unresolved_refs(spec["components"]["schemas"], "EvaluationReportData") == set()
    assert "GenerateJobRequest" in spec["components"]["schemas"]
    request_properties = spec["components"]["schemas"]["GenerateJobRequest"]["properties"]
    assert "profile" in request_properties
    assert "options" in request_properties


def test_service_authz_derives_from_job_routes():
    """Authz is derived from the ``@path_rule``-stamped job-factory routes
    (``AuthzScope("safe-synthesizer")``); there is no ``get_authz_contribution``.

    Doubles as the Phase-0 derivation gate: the service must derive with no
    problems and no fail-closed DENY bindings.
    """
    pytest.importorskip("nemo_safe_synthesizer.config.job")
    from nemo_helix_plugin.authz_discovery import _derive_service_contribution

    contribution, problems, _warnings = _derive_service_contribution(SafeSynthesizerService())

    assert problems == []
    assert not any(spec.deny for methods in contribution.endpoints.values() for spec in methods.values())
    for verb in ("create", "list", "read", "delete", "cancel"):
        assert f"safe-synthesizer.{verb}" in contribution.permissions

    jobs_path = "/apis/safe-synthesizer/v2/workspaces/{workspace}/jobs"
    assert contribution.endpoints[jobs_path]["get"].permissions == ["safe-synthesizer.list"]
    assert contribution.endpoints[jobs_path]["post"].scopes == ["safe-synthesizer:write", "platform:write"]
    assert contribution.endpoints[f"{jobs_path}/{{name}}/cancel"]["post"].permissions == ["safe-synthesizer.cancel"]
    assert contribution.endpoints[f"{jobs_path}/{{job}}/results/synthetic-data/download"]["get"].permissions == [
        "safe-synthesizer.read"
    ]
    report_data_download = f"{jobs_path}/{{job}}/results/evaluation-report-data/download"
    assert contribution.endpoints[report_data_download]["get"].permissions == ["safe-synthesizer.read"]
