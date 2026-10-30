# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent layers: requesting one, and the Job's work of building its context and having Builder build it."""

from __future__ import annotations

import io
import json
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from nemo_builder_plugin.entities import ContainerImage, ImageRuntimeConfig, Provenance
from nemo_builder_plugin.schema import FileSetSource, ImageSource, SubmitBuildResponse
from nemo_evals.agent_layers.models import AgentLayer, AgentLayerEntity, AgentLayerRequest, layer_entity_name
from nemo_evals.agent_layers.runner import (
    AgentLayerBuildSpec,
    layer_builder_name,
    relative_link,
    run_agent_layer_build,
)
from nemo_evals.agent_layers.service import AgentLayerService, AgentNotLayerableError, agent_layer_key, newest_layer
from nemo_evals.entities import TaskEntity
from nemo_evals.image_builds.service import BuildJob
from nemo_evals.jobs.agent_layer_build import INSTALLER_ARCHIVE, agent_layer_build_job_request, is_job_for
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.jobs.constants import PERSISTENT_JOB_STORAGE_PATH_ENVVAR
from nemo_helix_plugin.jobs.providers import CPUExecutionProvider
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest

_CODEX = AgentLayerRequest(name="codex", kwargs={"version": "0.162.1"})
_TTL = timedelta(hours=1)


async def _request(service: AgentLayerService, name: str, digest: str, agent: AgentLayerRequest = _CODEX, **at: Any):
    """``service.request_layer`` for a run with a one-hour timeout, now unless ``now`` says otherwise."""
    now = at.get("now", datetime.now(timezone.utc))
    return await service.request_layer("default", name, digest, agent, ttl=_TTL, now=now)


class _Jobs:
    def __init__(self) -> None:
        self.created: list[CreateHelixJobRequest] = []

    async def create(self, request: CreateHelixJobRequest) -> None:
        self.created.append(request)

    async def get(self, name: str) -> BuildJob | None:
        return None


async def _allow(workspace: str) -> None:
    pass


def _ready(name: str, created_at: datetime | None = None) -> ContainerImage:
    image = ContainerImage(
        name=name,
        workspace="default",
        status="ready",
        registry="registry.test",
        repository=f"default/{name}",
        digest="sha256:" + "d" * 64,
        runtime_config=ImageRuntimeConfig(entrypoint=None, working_dir="/app"),
        provenance=Provenance(
            build_set=name.rsplit("-", 1)[0],
            revision=int(name.rsplit("-", 1)[1]),
            job=f"default/builder-{name}",
            request_digest="sha256:" + "a" * 64,
        ),
    )
    image._created_at = created_at or datetime.now(timezone.utc)
    return image


class _Builder:
    """Builder whose builds are ready once submitted; records each submit's source and Dockerfile."""

    def __init__(self) -> None:
        self.rows: dict[str, ContainerImage] = {}
        self.submits: list[tuple[str, FileSetSource | ImageSource, str | None]] = []

    async def get_image(self, name: str) -> ContainerImage | None:
        return self.rows.get(name)

    async def submit(self, *, name: str, revision: int, source: Any, dockerfile: str | None, platform: str | None):
        self.submits.append((name, source, dockerfile))
        self.rows[f"{name}-{revision}"] = _ready(f"{name}-{revision}")
        return SubmitBuildResponse(job=f"builder-{name}-{revision}", images=[self.rows[f"{name}-{revision}"]])

    async def job_status(self, job: str) -> str | None:
        return "completed"

    async def image_platforms(self, image: str) -> list[str]:
        return []


def _serve_archive(monkeypatch: pytest.MonkeyPatch, files: dict[str, bytes]) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        root = tarfile.TarInfo("sort-task")
        root.type, root.mode = tarfile.DIRTYPE, 0o755
        archive.addfile(root)
        for name, content in files.items():
            info = tarfile.TarInfo(f"sort-task/{name}")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))

    async def download(files_client, ref, output, **kwargs) -> None:
        output.write(buffer.getvalue())

    monkeypatch.setattr("nemo_evals.image_builds.runner.download_verified_async", download)


def _installer(tmp_path: Path) -> Path:
    """An installer archive shaped like the image's: ``nhx-agent-installer/venv/bin/python``."""
    path = tmp_path / INSTALLER_ARCHIVE
    with tarfile.open(path, "w:gz") as archive:
        content = b"#!/bin/sh\n"
        info = tarfile.TarInfo("nhx-agent-installer/python/cpython-3.12.15-linux-x86_64-gnu/bin/python3")
        info.size, info.mode = len(content), 0o755
        archive.addfile(info, io.BytesIO(content))
        # The real installer's links, absolute under the mount: uv's minor-version alias and the venv's interpreter.
        for name, target in [
            ("nhx-agent-installer/python/cpython-3.12-linux-x86_64-gnu", "python/cpython-3.12.15-linux-x86_64-gnu"),
            ("nhx-agent-installer/venv/bin/python", "python/cpython-3.12.15-linux-x86_64-gnu/bin/python3"),
        ]:
            link = tarfile.TarInfo(name)
            link.type, link.linkname = tarfile.SYMTYPE, f"/opt/nhx-agent-installer/{target}"
            archive.addfile(link)
    return path


class _Files:
    """The Files calls the runner makes: one fileset per build, holding the context archive."""

    def __init__(self) -> None:
        self.create_fileset = AsyncMock()
        self.uploads: dict[tuple[str, str], bytes] = {}
        self.upload_calls = 0

    async def upload_file(self, *, workspace: str, name: str, path: str, content: bytes) -> Mock:
        self.uploads[(name, path)] = content
        self.upload_calls += 1
        return Mock()


def _links(upload: bytes) -> dict[str, str]:
    with tarfile.open(fileobj=io.BytesIO(upload), mode="r:gz") as archive:
        return {member.name: member.linkname for member in archive.getmembers() if member.issym()}


def _context(upload: bytes) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(upload), mode="r:gz") as archive:
        for member in archive.getmembers():
            handle = archive.extractfile(member) if member.isfile() else None
            if handle is not None:
                files[member.name] = handle.read()
    return files


_ENV = {
    "task.toml": b'[agent]\nuser = "agent"\n',
    "environment/Dockerfile": b"FROM ubuntu:24.04\nCOPY setup.sh /setup.sh\nUSER agent\n",
    "environment/setup.sh": b"echo hi\n",
}


async def _requested(entity_store, store_task, name: str = "sort-task") -> tuple[AgentLayerBuildSpec, _Jobs]:
    jobs = _Jobs()
    digest = await store_task(name, config={"agent": {"user": "agent"}})
    layer = await _request(AgentLayerService(entity_store, jobs, _allow), name, digest)
    assert layer is not None
    spec = AgentLayerBuildSpec(
        workspace="default", task=name, revision_digest=digest, agent_key=layer.agent_key, attempt=1
    )
    return spec, jobs


async def _run(entity_store, spec: AgentLayerBuildSpec, builder: _Builder, files: _Files, tmp_path: Path):
    return await run_agent_layer_build(
        spec,
        entities=entity_store,
        files=cast(AsyncFilesClient, files),
        builder=builder,
        installer=_installer(tmp_path),
        sleep=AsyncMock(),
    )


async def test_the_layer_builds_the_environment_with_the_agent_installed(
    entity_store, store_task, monkeypatch, tmp_path
) -> None:
    _serve_archive(monkeypatch, {**_ENV, "instruction.md": b"Sort.\n"})
    spec, _ = await _requested(entity_store, store_task)
    builder, files = _Builder(), _Files()

    layer = await _run(entity_store, spec, builder, files, tmp_path)

    assert layer.status == "ready" and layer.image is not None
    assert layer.image.image_ref is not None and layer.image.image_ref.endswith("@sha256:" + "d" * 64)
    ((name, source, dockerfile),) = builder.submits
    assert name.startswith("harbor-agent-") and dockerfile == "Dockerfile.nhx-agent-layer"
    assert isinstance(source, FileSetSource) and source.fileset == f"default/{name}"
    context = _context(files.uploads[(name, "context.tar.gz")])
    # The environment's own files, the generated Dockerfile, the agent and the installer; nothing else of the task.
    assert context["context/setup.sh"] == b"echo hi\n"
    assert context["context/Dockerfile.nhx-agent-layer"].decode().splitlines()[-1] == "USER agent"
    agent = json.loads(context["context/nhx-agent-layer/agent.json"])
    assert agent == {"name": "codex", "import_path": None, "kwargs": {"version": "0.162.1"}, "user": "agent"}
    assert "context/nhx-agent-installer/python/cpython-3.12.15-linux-x86_64-gnu/bin/python3" in context
    # Builder refuses a link to an absolute path: the installer's links are relative, to the same files.
    links = _links(files.uploads[(name, "context.tar.gz")])
    assert links == {
        "context/nhx-agent-installer/python/cpython-3.12-linux-x86_64-gnu": "cpython-3.12.15-linux-x86_64-gnu",
        "context/nhx-agent-installer/venv/bin/python": "../../python/cpython-3.12.15-linux-x86_64-gnu/bin/python3",
    }
    assert not any("instruction.md" in path or "task.toml" in path for path in context)


async def test_tasks_whose_environments_hold_the_same_files_share_one_build(
    entity_store, store_task, monkeypatch, tmp_path
) -> None:
    builder, files = _Builder(), _Files()
    _serve_archive(monkeypatch, {**_ENV, "instruction.md": b"Sort.\n"})
    first, _ = await _requested(entity_store, store_task, "sort-a")
    await _run(entity_store, first, builder, files, tmp_path)
    _serve_archive(monkeypatch, {**_ENV, "instruction.md": b"Reverse.\n"})
    second, _ = await _requested(entity_store, store_task, "sort-b")

    layer = await _run(entity_store, second, builder, files, tmp_path)

    assert layer.status == "ready"
    assert len(builder.submits) == 1
    assert files.upload_calls == 1  # the second task found the build and uploaded no context


async def test_a_compose_task_gets_no_layer(entity_store, store_task, monkeypatch, tmp_path) -> None:
    _serve_archive(monkeypatch, {**_ENV, "environment/docker-compose.yaml": b"services:\n  main:\n    build: .\n"})
    spec, _ = await _requested(entity_store, store_task)

    layer = await _run(entity_store, spec, _Builder(), _Files(), tmp_path)

    assert layer.status == "failed" and "compose" in (layer.error or "")


async def test_a_request_starts_the_layers_job_once(entity_store, store_task) -> None:
    jobs = _Jobs()
    digest = await store_task(config={})
    service = AgentLayerService(entity_store, jobs, _allow)

    first = await _request(service, "sort-task", digest)
    again = await _request(service, "sort-task", digest)

    assert first is not None and again is not None and first.id == again.id == "1"
    assert [request.name for request in jobs.created] == [first.job, first.job]  # the second confirms the same Job
    assert first.agent_key == agent_layer_key(_CODEX)


async def _make_ready(entity_store, digest: str, attempt: int = 1) -> AgentLayerEntity:
    head = await entity_store.get(TaskEntity, name="sort-task", workspace="default")
    stored = await entity_store.get(
        AgentLayerEntity,
        name=layer_entity_name(digest, agent_layer_key(_CODEX), attempt),
        workspace="default",
        parent=head.id,
    )
    stored.status = "ready"
    return await entity_store.update(stored)


async def test_a_ready_layer_is_returned_without_a_job(entity_store, store_task) -> None:
    jobs = _Jobs()
    digest = await store_task(config={})
    service = AgentLayerService(entity_store, jobs, _allow)
    await _request(service, "sort-task", digest)
    await _make_ready(entity_store, digest)
    jobs.created.clear()

    again = await _request(service, "sort-task", digest)
    newest = await newest_layer(entity_store, "default", "sort-task", digest, agent_layer_key(_CODEX))

    assert again is not None and again.status == "ready" and again.id == "1"
    assert newest is not None and newest.status == "ready" and newest.attempt == 1
    assert jobs.created == []


async def test_a_layer_expires_after_its_runs_timeout_and_the_next_run_starts_another(entity_store, store_task) -> None:
    jobs = _Jobs()
    digest = await store_task(config={})
    service = AgentLayerService(entity_store, jobs, _allow)
    start = datetime(2026, 10, 30, 12, tzinfo=timezone.utc)
    first = await _request(service, "sort-task", digest, now=start)
    await _make_ready(entity_store, digest)

    assert first is not None and first.expires_at == start + _TTL
    before = await _request(service, "sort-task", digest, now=start + _TTL - timedelta(seconds=1))
    after = await _request(service, "sort-task", digest, now=start + _TTL)

    assert before is not None and before.id == "1"
    assert after is not None and after.id == "2" and after.status == "pending"
    second = await newest_layer(entity_store, "default", "sort-task", digest, agent_layer_key(_CODEX))
    # The new attempt reuses no Builder build older than one TTL.
    assert second is not None and second.built_after == start


async def test_a_builder_build_older_than_the_layers_cutoff_is_rebuilt(
    entity_store, store_task, monkeypatch, tmp_path
) -> None:
    _serve_archive(monkeypatch, {**_ENV, "instruction.md": b"Sort.\n"})
    spec, _ = await _requested(entity_store, store_task)
    builder, files = _Builder(), _Files()
    await _run(entity_store, spec, builder, files, tmp_path)
    ((name, _, _),) = builder.submits
    builder.rows[f"{name}-1"] = _ready(f"{name}-1", created_at=datetime.now(timezone.utc) - 2 * _TTL)
    await _make_ready(entity_store, spec.revision_digest)
    later, _ = await _requested_again(entity_store, spec, now=datetime.now(timezone.utc) + _TTL)

    layer = await _run(entity_store, later, builder, files, tmp_path)

    assert layer.status == "ready" and layer.image is not None and layer.image.builder_image == f"{name}-2"
    assert [submitted for submitted, _, _ in builder.submits] == [name, name]


async def _requested_again(
    entity_store, spec: AgentLayerBuildSpec, *, now: datetime
) -> tuple[AgentLayerBuildSpec, AgentLayer | None]:
    layer = await _request(AgentLayerService(entity_store, _Jobs(), _allow), spec.task, spec.revision_digest, now=now)
    assert layer is not None
    return spec.model_copy(update={"attempt": int(layer.id)}), layer


async def test_an_agent_whose_setup_isnt_an_install_is_refused(entity_store, store_task) -> None:
    digest = await store_task(config={})
    service = AgentLayerService(entity_store, _Jobs(), _allow)

    with pytest.raises(AgentNotLayerableError):
        await _request(service, "sort-task", digest, AgentLayerRequest(name="oracle"))


def test_the_job_fetches_the_installer_onto_storage_its_build_step_reads() -> None:
    spec = AgentLayerBuildSpec(workspace="default", task="t", revision_digest="e" * 64, agent_key="k", attempt=1)

    request = agent_layer_build_job_request(spec, name="agent-layer-build-x-1")

    assert request.platform_spec is not None
    fetch, build = request.platform_spec.steps
    assert isinstance(fetch.executor, CPUExecutionProvider) and fetch.executor.container is not None
    assert (fetch.executor.container.image or "").rpartition("/")[2].startswith("nhx-agent-installer")
    assert fetch.executor.container.entrypoint == ["/bin/cp"]
    for step in (fetch, build):
        assert [variable.name for variable in step.environment or []] == [PERSISTENT_JOB_STORAGE_PATH_ENVVAR]
    assert build.config == spec.model_dump(mode="json")
    assert is_job_for(request.custom_fields, spec)
    assert not is_job_for(request.custom_fields, spec.model_copy(update={"attempt": 2}))


def test_the_layers_builder_name_covers_everything_the_layer_holds() -> None:
    name = layer_builder_name(environment="e" * 64, agent_key="k", user="agent", platform=None)

    assert name.startswith("harbor-agent-")
    assert name == layer_builder_name(environment="e" * 64, agent_key="k", user="agent", platform=None)
    others = [
        layer_builder_name(environment="f" * 64, agent_key="k", user="agent", platform=None),
        layer_builder_name(environment="e" * 64, agent_key="k2", user="agent", platform=None),
        layer_builder_name(environment="e" * 64, agent_key="k", user="root", platform=None),
        layer_builder_name(environment="e" * 64, agent_key="k", user="agent", platform="linux/arm64"),
    ]
    assert name not in others and len(set(others)) == len(others)


def test_an_installer_link_outside_its_mount_is_refused() -> None:
    with pytest.raises(ValueError, match="links outside /opt/nhx-agent-installer"):
        relative_link("venv/bin/python", "/usr/bin/python3")
