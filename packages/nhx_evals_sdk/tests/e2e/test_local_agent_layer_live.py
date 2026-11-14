# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A local agent layer, built by real Docker, starts the way its environment image does."""

import json
import shutil
import subprocess
import uuid
from collections.abc import Iterator
from typing import Any

import pytest

pytest.importorskip("harbor", reason="local agent layers build on Harbor's Docker environment")

from harbor.models.trial.config import AgentConfig
from nhx_evals_sdk.agent_eval.runtimes.harbor import local_agent_layer
from nhx_evals_sdk.agent_eval.runtimes.harbor.agent_layer import AgentInstall
from nhx_evals_sdk.agent_eval.runtimes.harbor.local_agent_layer import build_agent_layer

pytestmark = [pytest.mark.e2e, pytest.mark.slow, pytest.mark.skip_in_ci]

_BASE_IMAGE = "python:3.12-slim"
_INSTALL = AgentInstall(agent="harbor:probe", inputs={}, harbor_version="0.24.0")
_STARTED = "test -f /installed && echo started"


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


def _docker(*args: str, stdin: str | None = None) -> str:
    return subprocess.run(["docker", *args], input=stdin, capture_output=True, text=True, check=True).stdout


def _start(image: str) -> dict[str, Any]:
    return json.loads(_docker("image", "inspect", "--format", "{{json .Config}}", image))


class _ProbeAgent:
    async def setup(self, environment: Any) -> None:
        result = await environment.exec("touch /installed", user="root")
        assert result.return_code == 0, result.stderr


@pytest.fixture
def images() -> Iterator[list[str]]:
    if not _docker_available():
        pytest.skip("Docker daemon is required to build an agent layer")
    _docker("pull", "--quiet", _BASE_IMAGE)
    created: list[str] = []
    yield created
    subprocess.run(["docker", "image", "rm", "--force", *created], capture_output=True)


@pytest.mark.parametrize(
    ("entrypoint", "command"),
    [
        pytest.param(None, ["sh", "-c", _STARTED], id="image-without-entrypoint"),
        pytest.param(["sh", "-c"], [_STARTED], id="image-with-entrypoint"),
    ],
)
async def test_a_layer_keeps_its_images_entrypoint(
    images: list[str], monkeypatch: pytest.MonkeyPatch, entrypoint: list[str] | None, command: list[str]
) -> None:
    """Harbor starts a trial's container with the image's own start; a layer that changes it never comes up."""
    suffix = uuid.uuid4().hex[:12]
    image = _BASE_IMAGE
    if entrypoint is not None:
        image = f"nemo-agent-layer-e2e-base:{suffix}"
        _docker(
            "build", "--quiet", "--tag", image, "-", stdin=f"FROM {_BASE_IMAGE}\nENTRYPOINT {json.dumps(entrypoint)}\n"
        )
        images.append(image)
    tag = f"nemo-agent-layer-e2e:{suffix}"
    images.append(tag)
    monkeypatch.setattr(local_agent_layer.AgentFactory, "create_agent_from_config", lambda *a, **k: _ProbeAgent())

    await build_agent_layer(image, tag, _INSTALL, AgentConfig(name="probe"), user=None)

    built, base = _start(tag), _start(image)
    assert (built.get("Entrypoint"), built.get("Cmd")) == (base.get("Entrypoint"), base.get("Cmd"))
    assert _docker("run", "--rm", tag, *command).strip() == "started"
