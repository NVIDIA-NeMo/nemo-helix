# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import re
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_builder_plugin.entities import ContainerImage, Provenance
from nemo_builder_plugin.schema import SubmitBuildResponse
from nemo_helix_operator.service import HelixOperatorService
from nemo_helix_operator.source import DOCKERFILE_PATH, pack
from nemo_helix_plugin.dependencies import get_nemo_client


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo, check=True)


def test_pack(tmp_path: Path) -> None:
    for name, text in {
        "docker-bake.hcl": "",
        ".gitignore": ".venv/\n",
        "services/a/src/nhx/a.py": "A = 1\n",
        "services/gone/src/nhx/gone.py": "",
    }.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "init")
    (tmp_path / "services/gone/src/nhx/gone.py").unlink()
    (tmp_path / "packages/b/src/nhx").mkdir(parents=True)
    (tmp_path / "packages/b/src/nhx/b.py").write_text("B = 2\n")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv/x").write_text("")

    archive, metadata = pack(tmp_path, "ghcr.io/x/nhx-api:t")

    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        names = set(tar.getnames())
        member = tar.extractfile(DOCKERFILE_PATH)
        assert member is not None
        dockerfile = member.read().decode()
    assert names == {
        ".gitignore",
        "docker-bake.hcl",
        "services/a/src/nhx/a.py",
        "packages/b/src/nhx/b.py",
        DOCKERFILE_PATH,
    }
    assert re.fullmatch(r"[0-9a-f]{12}", str(metadata["commit"])) and metadata["dirty"] is True
    assert dockerfile.startswith("FROM ghcr.io/x/nhx-api:t\n")
    assert "COPY --chown=1000:1000 services/*/src/ packages/*/src/ plugins/*/src/ " in dockerfile
    assert f"ENV NHX_CODE_REVISION={metadata['commit']}-dirty\n" in dockerfile


def _image(revision: int, spec: str) -> ContainerImage:
    image = ContainerImage(
        name=f"helix-{revision}.{spec}",
        workspace="default",
        registry="reg:5005",
        repository="grp/proj/default/nhx-api",
        provenance=Provenance(
            build_set="helix",
            revision=revision,
            spec=spec,
            job=f"default/builder-helix-{revision}",
            request_digest="sha256:" + "0" * 64,
        ),
    )
    image._id = "id"
    image._created_at = datetime.now(timezone.utc)
    return image


def test_create_build() -> None:
    commit = "0123456789ab"
    builder = MagicMock()
    builder.list_container_images = AsyncMock(return_value=SimpleNamespace(data=lambda: [_image(2, "aaaaaaaaaaaa")]))
    builder.submit_build = AsyncMock(
        return_value=SimpleNamespace(
            data=lambda: SubmitBuildResponse(job="builder-helix-3", images=[_image(3, commit)])
        )
    )
    download = MagicMock(read=AsyncMock(return_value=json.dumps({"commit": commit, "dirty": False}).encode()))
    files = MagicMock(download_file=AsyncMock(return_value=download))

    app = FastAPI()
    for spec in HelixOperatorService().get_routers():
        app.include_router(spec.router, prefix=spec.prefix)
    app.dependency_overrides[get_nemo_client] = lambda: MagicMock()
    with (
        patch("nemo_helix_operator.service.AsyncBuilderClient.from_client", return_value=builder),
        patch("nemo_helix_operator.service.AsyncFilesClient.from_client", return_value=files),
    ):
        response = TestClient(app).post("/v2/workspaces/default/builds", json={"source": {"fileset": "helix-src"}})

    assert response.status_code == 201, response.text
    assert response.json()["tag"] == f"{commit}-r3"
    build_set = builder.submit_build.call_args.kwargs["body"]
    assert (build_set.name, build_set.revision) == ("helix", 3)
    (spec,) = build_set.build_specs
    assert spec.dockerfile == DOCKERFILE_PATH
    assert (spec.source.fileset, spec.source.archive) == ("helix-src", "source.tar.gz")
    assert (spec.output.repository, spec.output.tag) == ("nhx-api", f"{commit}-r3")
