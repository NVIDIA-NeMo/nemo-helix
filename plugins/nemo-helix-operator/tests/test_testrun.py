# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import io
import json
import tarfile
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_helix_operator import deploy, testrun
from nemo_helix_operator.service import HelixOperatorService
from nemo_helix_plugin.dependencies import get_nemo_client

PASSED = '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"><testcase/></testsuite></testsuites>'
FAILED = (
    '<testsuites><testsuite tests="2" failures="1" errors="0" skipped="1">'
    '<testcase classname="e2e.test_hello_world" name="test_job_lifecycle"><failure/></testcase>'
    "<testcase/></testsuite></testsuites>"
)


def _archive(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, text in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(text)
            tar.addfile(info, io.BytesIO(text.encode()))
    return buffer.getvalue()


def test_create_test_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POD_NAMESPACE", "ns")
    monkeypatch.setattr(testrun, "_runs", {})
    monkeypatch.setattr(deploy, "_upgrades", {})
    release = {"status": "deployed"}
    gate = threading.Event()
    calls: list[tuple[set[str], str, str]] = []

    async def helm(*args: str) -> tuple[int, str]:
        return 0, json.dumps({"version": 4, "info": release})

    async def pytest_run(work: Path, api_url: str) -> tuple[int, str]:
        calls.append(({str(p.relative_to(work)) for p in work.rglob("*.py")}, testrun._pythonpath(work), api_url))
        await asyncio.to_thread(gate.wait)
        return 0, PASSED

    archive = _archive({"e2e/test_hello_world.py": "", "packages/a/src/a.py": ""})
    download = MagicMock(read=AsyncMock(return_value=archive))
    files = MagicMock(download_file=AsyncMock(return_value=download))
    app = FastAPI()
    for spec in HelixOperatorService().get_routers():
        app.include_router(spec.router, prefix=spec.prefix)
    app.dependency_overrides[get_nemo_client] = lambda: MagicMock()
    body = {"source": {"fileset": "helix-src"}}

    with (
        patch.object(deploy, "_helm", side_effect=helm),
        patch.object(testrun, "_pytest", side_effect=pytest_run),
        patch.object(testrun.AsyncFilesClient, "from_client", return_value=files),
        TestClient(app) as client,
    ):
        response = client.post("/v2/workspaces/default/test-runs", json=body)
        assert response.status_code == 202, response.text
        run = response.json()
        assert (run["status"], run["tests"], run["helm_revision"]) == ("running", ["e2e/test_hello_world.py"], 4)
        assert client.post("/v2/workspaces/default/test-runs", json=body).status_code == 409

        gate.set()
        deadline = time.monotonic() + 5
        while (result := client.get(f"/v2/workspaces/default/test-runs/{run['id']}").json())["status"] == "running":
            assert time.monotonic() < deadline
            time.sleep(0.05)
        assert (result["status"], result["passed"], result["failed"]) == ("passed", 1, 0)
        ((extracted, pythonpath, api_url),) = calls
        assert extracted == {"e2e/test_hello_world.py", "packages/a/src/a.py"}
        assert pythonpath.endswith("packages/a/src") and api_url == "http://nemo-helix-api:8080"

        download.read.return_value = _archive({"e2e/other.py": ""})
        assert client.post("/v2/workspaces/default/test-runs", json=body).status_code == 422
        release["status"] = "pending-upgrade"
        assert client.post("/v2/workspaces/default/test-runs", json=body).status_code == 409
        assert client.get("/v2/workspaces/default/test-runs/nope").status_code == 404

    failed = testrun._result("x", 4, 1, FAILED)
    assert (failed.status, failed.passed, failed.failed, failed.skipped) == ("failed", 0, 1, 1)
    assert failed.detail == "e2e.test_hello_world::test_job_lifecycle"
    crashed = testrun._result("x", 4, 4, "ImportError while loading conftest")
    assert (crashed.status, crashed.detail) == ("failed", "ImportError while loading conftest")
