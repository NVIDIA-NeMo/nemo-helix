# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Building a manifest's victim image with the builder plugin and recording it on the manifest."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

import httpx
import pytest
from _doubles import make_job_context
from nemo_agent_hardener_plugin.entities import AgentHardenerManifest
from nemo_agent_hardener_plugin.jobs import victim_build, victim_context
from nemo_agent_hardener_plugin.jobs.errors import (
    CATEGORY_PROVISIONING,
    CATEGORY_VICTIM_UNAVAILABLE,
    AgentHardenerRunError,
)
from nemo_builder_plugin.entities import ContainerImage, Provenance
from nemo_builder_plugin.schema import BuildMount, BuildSet, BuildSpec, FileSetSource, ImageSource, SubmitBuildResponse
from nemo_helix_plugin.client.errors import ConflictError
from nemo_helix_plugin.entities import EntityClient
from nemo_helix_plugin.job_results import ResultRef

_CONTEXT_REF = "default/job-fileset-victim-context-abc#results/attempt-1/victim-context"
_DIGEST = "sha256:" + "d" * 64


def _image(name: str, status: Literal["pending", "ready", "failed"], detail: str | None = None) -> ContainerImage:
    return ContainerImage(
        name=name,
        workspace="default",
        status=status,
        status_detail=detail,
        registry="registry.test",
        repository=f"nhx/default/{name.rsplit('-', 1)[0]}",
        digest=_DIGEST if status == "ready" else None,
        provenance=Provenance(
            build_set=name.rsplit("-", 1)[0], revision=1, job=f"default/builder-{name}", request_digest=_DIGEST
        ),
    )


class _Builder:
    """Builder whose one submit settles as *outcome* on the next read; or refuses with *refusal* (a 409)."""

    def __init__(self, outcome: Literal["ready", "failed"] = "ready", *, refusal: str | None = None) -> None:
        self.outcome = outcome
        self.refusal = refusal
        self.rows: dict[str, ContainerImage] = {}
        self.sources: list[FileSetSource | ImageSource] = []

    async def get_image(self, name: str) -> ContainerImage | None:
        row = self.rows.get(name)
        if row is None or row.status != "pending":
            return row
        detail = "kaniko: COPY failed: no such file" if self.outcome == "failed" else None
        self.rows[name] = _image(name, self.outcome, detail)
        return self.rows[name]

    async def submit(
        self,
        *,
        name: str,
        revision: int,
        source: FileSetSource | ImageSource,
        dockerfile: str | None,
        platform: str | None,
        mounts: Sequence[BuildMount] = (),
    ) -> SubmitBuildResponse:
        if self.refusal is not None:
            raise ConflictError(httpx.Response(409, json={"detail": self.refusal}))
        self.sources.append(source)
        row = _image(f"{name}-{revision}", "pending")
        self.rows[row.name] = row
        return SubmitBuildResponse(job=f"builder-{name}-{revision}", images=[row])

    async def image_platforms(self, image: str) -> list[str]:
        return []

    async def job_status(self, job: str) -> str | None:
        return "running"


class _Manifests:
    def __init__(self) -> None:
        self.manifest = AgentHardenerManifest(name="finance", workspace="default")
        self.updates: list[str] = []

    async def get(self, entity_type: type[AgentHardenerManifest], *, name: str, workspace: str) -> Any:
        return self.manifest.model_copy()

    async def update(self, entity: AgentHardenerManifest) -> AgentHardenerManifest:
        self.updates.append(entity.victim_image)
        self.manifest = entity
        return entity


async def _build(builder: _Builder, manifests: _Manifests, context_ref: str = _CONTEXT_REF) -> str:
    return await victim_build.build_victim_image(
        builder,
        cast(EntityClient, manifests),
        workspace="default",
        manifest_id="finance",
        context_ref=context_ref,
        poll_interval=0,
    )


async def test_a_ready_build_is_recorded_as_the_manifests_digest_pinned_victim_image() -> None:
    builder, manifests = _Builder(), _Manifests()

    image_ref = await _build(builder, manifests)

    assert image_ref == f"registry.test/nhx/default/ah-victim-finance@{_DIGEST}"
    assert manifests.updates == [image_ref]
    # The staged archive is the whole context: the builder unpacks it and builds its root.
    assert builder.sources == [
        FileSetSource(fileset="default/job-fileset-victim-context-abc", archive="results/attempt-1/victim-context")
    ]


async def test_a_failed_build_reports_the_builders_cause_and_leaves_the_manifest_alone() -> None:
    manifests = _Manifests()

    with pytest.raises(AgentHardenerRunError) as raised:
        await _build(_Builder("failed"), manifests)

    assert raised.value.category == CATEGORY_VICTIM_UNAVAILABLE
    assert "kaniko: COPY failed: no such file" in str(raised.value)
    assert "builder-ah-victim-finance-1" in str(raised.value)
    assert manifests.updates == []


async def test_a_builder_that_cannot_build_says_how_to_enable_it() -> None:
    manifests = _Manifests()

    with pytest.raises(AgentHardenerRunError) as raised:
        await _build(_Builder(refusal="registry is not configured"), manifests)

    assert raised.value.category == CATEGORY_PROVISIONING
    assert "registry is not configured" in str(raised.value)
    assert raised.value.remediation == victim_build.BUILDER_REMEDIATION
    assert manifests.updates == []


async def test_a_locally_staged_context_is_not_sent_to_the_builder() -> None:
    builder = _Builder()

    with pytest.raises(AgentHardenerRunError):
        await _build(builder, _Manifests(), context_ref="file:///tmp/results/victim-context")

    assert builder.sources == []


@pytest.mark.parametrize(
    "manifest_id",
    ["finance", "Finance_Bot.v2", "a" * 63, "x--y", "@@@", "finance-bot-with-a-rather-long-descriptive-name"],
)
def test_a_victim_build_name_is_one_the_builder_accepts(manifest_id: str) -> None:
    name = victim_build.victim_build_name(manifest_id)

    # The builder validates the name, and every name composed from it, at its largest revision we allow.
    BuildSet(name=name, revision=9999, build_specs=[BuildSpec(source=FileSetSource(fileset="context"))])
    assert re.fullmatch(r"ah-victim-[a-z0-9-]+", name)


def test_victim_build_names_keep_manifests_apart() -> None:
    assert victim_build.victim_build_name("finance") == "ah-victim-finance"
    assert victim_build.victim_build_name("Finance") != victim_build.victim_build_name("finance")
    long = "finance-bot-with-a-rather-long-descriptive-name"
    assert victim_build.victim_build_name(long + "-1") != victim_build.victim_build_name(long + "-2")


def test_the_job_builds_from_the_saved_archive_only_when_asked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = tmp_path / "finance-victim-context.tar.gz"
    archive.touch()
    monkeypatch.setattr(victim_context.AgentHardenerConfig, "get", lambda: None)
    monkeypatch.setattr(victim_context._common, "require_provisioned", lambda _config: None)
    monkeypatch.setattr(victim_context, "_materialize_manifest", lambda *_a: "m.yaml")
    monkeypatch.setattr(victim_context, "_stage", lambda *_a: tmp_path)
    monkeypatch.setattr(victim_context, "_archive_context", lambda _dir, _archive: None)
    builds: list[dict[str, str]] = []

    async def _build_in_cluster(**kwargs: str) -> str:
        builds.append(kwargs)
        return "registry.test/victim@" + _DIGEST

    monkeypatch.setattr(victim_context, "build_in_cluster", _build_in_cluster)
    ctx = make_job_context(tmp_path, on_save=lambda name, _path: ResultRef(name=name, artifact_url=_CONTEXT_REF))
    job = victim_context.AgentHardenerVictimContextJob()

    staged = job.run({"manifest_id": "finance"}, ctx=ctx)
    built = job.run({"manifest_id": "finance", "build": True}, ctx=ctx)

    assert "victim_image" not in staged
    assert built["victim_image"] == "registry.test/victim@" + _DIGEST
    assert builds == [{"workspace": "default", "manifest_id": "finance", "context_ref": _CONTEXT_REF}]
