# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import tarfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_helix_operator import deploy
from nemo_helix_operator.service import HelixOperatorService
from nemo_helix_plugin.dependencies import get_nemo_client


def _archive(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, text in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(text)
            tar.addfile(info, io.BytesIO(text.encode()))
    return buffer.getvalue()


def test_create_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POD_NAMESPACE", "ns")
    monkeypatch.setattr(deploy, "_upgrades", {})
    release = {"status": "deployed"}
    upgrades: list[tuple[str, ...]] = []
    seen_chart: list[set[str]] = []

    async def helm(*args: str) -> tuple[int, str]:
        if args[0] == "status":
            return 0, json.dumps({"version": 4, "info": release})
        upgrades.append(args)
        seen_chart.append({str(p.relative_to(args[2])) for p in Path(args[2]).rglob("*")})
        return 0, "upgraded"

    archive = _archive({"k8s/helm/Chart.yaml": "name: x\n", "k8s/helm/templates/a.yaml": "", "services/a.py": ""})
    download = MagicMock(read=AsyncMock(return_value=archive))
    files = MagicMock(download_file=AsyncMock(return_value=download))
    app = FastAPI()
    for spec in HelixOperatorService().get_routers():
        app.include_router(spec.router, prefix=spec.prefix)
    app.dependency_overrides[get_nemo_client] = lambda: MagicMock()
    body = {"source": {"fileset": "helix-src"}, "api_image": "reg:5005/grp/nhx-api:abc-r3", "rollback_on_failure": True}

    with (
        patch.object(deploy, "_helm", side_effect=helm),
        patch.object(deploy.AsyncFilesClient, "from_client", return_value=files),
        TestClient(app) as client,
    ):
        response = client.post("/v2/workspaces/default/deployments", json=body)
        deadline = time.monotonic() + 5
        while not upgrades and time.monotonic() < deadline:
            time.sleep(0.05)

        assert response.status_code == 202, response.text
        assert response.json() == {
            "release": "nemo-helix",
            "revision": 5,
            "status": "pending-upgrade",
            "api_image": "reg:5005/grp/nhx-api:abc-r3",
            "core_image": "reg:5005/grp/nhx-api:abc-r3",
            "detail": None,
        }
        (args,) = upgrades
        assert args[:2] == ("upgrade", "nemo-helix") and args[3:6] == ("-n", "ns", "--reset-then-reuse-values")
        assert "api.image.repository=reg:5005/grp/nhx-api" in args and "core.image.tag=abc-r3" in args
        assert args[-1] == "--atomic"
        assert seen_chart == [{"Chart.yaml", "templates", "templates/a.yaml"}]

        assert (
            client.post(
                "/v2/workspaces/default/deployments", json=body | {"api_image": "reg:5005/grp/nhx-api"}
            ).status_code
            == 422
        )
        download.read.return_value = _archive({"k8s/helm/Chart.yaml": "", "k8s/helm/../../../evil": "x"})
        assert client.post("/v2/workspaces/default/deployments", json=body).status_code == 422
        assert len(upgrades) == 1
        release["status"] = "pending-upgrade"
        assert client.post("/v2/workspaces/default/deployments", json=body).status_code == 409
