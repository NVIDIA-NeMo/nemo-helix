# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every customization contributor's fixtures compile, and each compiled step
authenticates as a service principal the auth service accepts."""

import json
import re
import sys
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import yaml
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.config import Runtime
from nemo_helix_plugin.discovery import discover_customization_contributors, discover_jobs
from nemo_helix_plugin.files.metadata import FilesetMetadata
from nemo_helix_plugin.files.storage_config import LocalStorageConfig
from nemo_helix_plugin.files.types import FilesetOutput, FilesetPurpose
from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.models.types import ModelEntity
from nhx.core.auth.app.account_resolution import _available_service_names
from nhx.core.auth.config import AuthServiceConfig
from nhx.customization_common.contributor import jobs as contributor_jobs

WORKSPACE = "default"
_NOW = datetime(2026, 1, 1)
_DATASET_FILES = ("training.jsonl", "validation.jsonl")
_ENV_CONFIG = "responses_api_models/vllm_model/configs/policy_model.yaml"
_ENV_MANIFEST = "nemo-environment.yaml"


def _contributor_fixtures() -> Iterator[tuple[str, Path]]:
    for name, contributor in sorted(discover_customization_contributors().items()):
        plugin_root = Path(sys.modules[type(contributor).__module__].__file__ or "").parents[2]
        fixtures = sorted((plugin_root / "tests" / "fixtures").glob("*.json"))
        if not fixtures:
            yield name, plugin_root / "tests" / "fixtures" / "<missing>"
        for fixture in fixtures:
            yield name, fixture


CONTRIBUTOR_FIXTURES = list(_contributor_fixtures())


def _json_response(payload: Any) -> httpx.Response:
    body = payload.model_dump_json() if hasattr(payload, "model_dump_json") else json.dumps(payload)
    return httpx.Response(200, content=body, headers={"content-type": "application/json"})


def _file_listing(workspace: str, name: str, paths: tuple[str, ...]) -> httpx.Response:
    return _json_response(
        {
            "data": [
                {
                    "file_ref": f"{workspace}/{name}#{path}",
                    "file_url": f"http://fake/{workspace}/{name}/{path}",
                    "path": path,
                    "size": 1,
                }
                for path in paths
            ]
        }
    )


class _FakeHelixApi:
    """Answers compile-time lookups; environment filesets hold a minimal ``native-v1`` package."""

    def __init__(self, environment_filesets: set[str]) -> None:
        self._environment_filesets = environment_filesets

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method != "GET":
            return httpx.Response(405, json={"detail": f"fake platform: unexpected {request.method} {path}"})

        if match := re.fullmatch(r"/apis/models/v2/workspaces/([^/]+)/models/([^/]+)", path):
            workspace, name = match.groups()
            return _json_response(
                ModelEntity(
                    id=f"model-{name}",
                    workspace=workspace,
                    name=name,
                    fileset=f"{workspace}/{name}-weights",
                    created_at=_NOW,
                    updated_at=_NOW,
                )
            )
        if re.fullmatch(r"/apis/models/v2/workspaces/[^/]+/adapters/[^/]+", path):
            return httpx.Response(404, json={"detail": "adapter not found"})
        if re.fullmatch(r"/apis/jobs/v2/workspaces/[^/]+/jobs", path):
            # No running jobs, so no output-name conflicts.
            return _json_response(
                {
                    "data": [],
                    "pagination": {
                        "page": 1,
                        "page_size": 10,
                        "current_page_size": 0,
                        "total_pages": 0,
                        "total_results": 0,
                    },
                }
            )
        if match := re.fullmatch(r"/apis/files/v2/workspaces/([^/]+)/filesets/([^/]+)/-/(.+)", path):
            _, name, file_path = match.groups()
            if name in self._environment_filesets and file_path == _ENV_MANIFEST:
                manifest = {"format": "native-v1", "config_paths": [_ENV_CONFIG], "metadata": {"name": name}}
                return httpx.Response(200, content=yaml.safe_dump(manifest).encode())
            return httpx.Response(404, json={"detail": f"fake platform: no file {file_path}"})
        if match := re.fullmatch(r"/apis/files/v2/workspaces/([^/]+)/filesets/([^/]+)/files", path):
            workspace, name = match.groups()
            if name in self._environment_filesets:
                return _file_listing(workspace, name, (_ENV_MANIFEST, _ENV_CONFIG))
            return _file_listing(workspace, name, _DATASET_FILES)
        if match := re.fullmatch(r"/apis/files/v2/workspaces/([^/]+)/filesets/([^/]+)", path):
            workspace, name = match.groups()
            return _json_response(
                FilesetOutput(
                    id=f"fileset-{name}",
                    workspace=workspace,
                    name=name,
                    description="",
                    purpose=FilesetPurpose.GENERIC,
                    storage=LocalStorageConfig(path=f"/data/{name}"),
                    metadata=FilesetMetadata(),
                    custom_fields={},
                    project="",
                    created_at=_NOW.isoformat(),
                    updated_at=_NOW.isoformat(),
                )
            )
        return httpx.Response(404, json={"detail": f"fake platform: no route for GET {path}"})


def _environment_filesets(fixture_json: dict[str, Any]) -> set[str]:
    environment = fixture_json.get("environment")
    return {environment.rsplit("/", 1)[-1]} if isinstance(environment, str) else set()


def _fake_client(fixture_json: dict[str, Any]) -> AsyncNemoClient:
    transport = httpx.MockTransport(_FakeHelixApi(_environment_filesets(fixture_json)))
    return AsyncNemoClient(
        base_url="http://fake-platform",
        workspace=WORKSPACE,
        http_client=httpx.AsyncClient(transport=transport, base_url="http://fake-platform"),
    )


@pytest.fixture(autouse=True)
def _kubernetes_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compile as a Kubernetes platform, which every backend supports."""
    monkeypatch.setattr(
        contributor_jobs.NemoHelixConfig,
        "get",
        classmethod(lambda cls: SimpleNamespace(runtime=Runtime.KUBERNETES)),
    )


def _configure_sandboxed_gym_cluster(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable the cluster settings GRPO requires."""
    from nhx.rl import config as rl_config

    monkeypatch.setattr(rl_config.platform_config, "sandbox_cluster_capable", True)
    monkeypatch.setattr(rl_config.config, "job_storage_pvc_claim", "job-storage")


def _job_for(contributor_name: str) -> type[NemoJob]:
    key = f"customization.{contributor_name}.jobs"
    jobs = discover_jobs()
    assert key in jobs, f"contributor {contributor_name!r} registers no {key!r} job entry point"
    return jobs[key]


async def _compile(job_cls: type[NemoJob], fixture_json: dict[str, Any]) -> Any:
    assert job_cls.input_spec_schema is not None
    async_client = _fake_client(fixture_json)
    input_spec = job_cls.input_spec_schema.model_validate(fixture_json)
    spec = await job_cls.to_spec(
        input_spec, workspace=WORKSPACE, entity_client=None, async_sdk=async_client, is_local=False
    )
    return await job_cls.compile(
        workspace=WORKSPACE, spec=spec, entity_client=None, job_name="contract-test", async_sdk=async_client
    )


def _step_service_names(job_spec: Any) -> dict[str, str]:
    """Map each step that authenticates with ``--service-name`` to the name it presents."""
    names: dict[str, str] = {}
    for step in job_spec.steps:
        container = getattr(step.executor, "container", None)
        command = list(getattr(container, "command", None) or [])
        if "--service-name" in command:
            names[step.name] = command[command.index("--service-name") + 1]
    return names


def test_every_contributor_ships_fixtures() -> None:
    missing = [name for name, fixture in CONTRIBUTOR_FIXTURES if fixture.name == "<missing>"]
    assert not missing, f"customization contributors without tests/fixtures/*.json: {missing}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("contributor_name", "fixture"),
    [pytest.param(name, fixture, id=f"{name}-{fixture.stem}") for name, fixture in CONTRIBUTOR_FIXTURES],
)
async def test_fixture_compiles_and_steps_use_allowed_service_names(
    contributor_name: str, fixture: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if fixture.name == "<missing>":
        pytest.skip("covered by test_every_contributor_ships_fixtures")
    fixture_json = json.loads(fixture.read_text())
    if _environment_filesets(fixture_json):
        _configure_sandboxed_gym_cluster(monkeypatch)

    job_spec = await _compile(_job_for(contributor_name), fixture_json)

    step_names = _step_service_names(job_spec)
    assert step_names, f"{contributor_name}: no compiled step passes --service-name; the contract cannot be checked"
    allowed = _available_service_names(AuthServiceConfig())
    rejected = {step: name for step, name in step_names.items() if name not in allowed}
    assert not rejected, (
        f"{contributor_name} ({fixture.name}): steps authenticate as service principals the auth service "
        f"rejects, so their platform calls fail with HTTP 502 once auth is enabled: {rejected}"
    )
