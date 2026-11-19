# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stored skills through the real Agents, Files and Entities services.

Drives the routes with the typed ``AgentsClient`` (and the ``nemo agents skills`` CLI) so a route
that drifts from its typed endpoint fails here.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import httpx
import pytest
from nemo_agents_plugin.cli import AgentsCLI
from nemo_agents_plugin.service import AgentsService
from nemo_agents_plugin.skill_storage import SKILLS_FILESET_NAME
from nemo_helix_plugin.agents.client import AgentsClient
from nemo_helix_plugin.agents.skills import ListSkillsQueryParams, SkillInput, SkillSource
from nemo_helix_plugin.auth import RuntimePrincipal
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.errors import (
    ConflictError,
    NotFoundError,
    PermissionDeniedError,
    UnprocessableEntityError,
)
from nemo_helix_plugin.client_provider import DefaultNemoClientProvider, _build_headers, set_nemo_client_provider
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceQueryParams, CreateWorkspaceRequest
from nhx.core.files.service import FilesService
from nhx.platform_runner.plugin_adapter import NemoServiceAdapter
from nhx.testing import (
    TEST_ADMIN_EMAIL,
    ClientContext,
    as_user,
    create_test_client,
    grant_workspace_role,
    short_unique_name,
    unique_email,
)
from typer.testing import CliRunner

pytestmark = pytest.mark.integration

SKILL = "pdf-tools"
ClientT = TypeVar("ClientT", bound=NemoClient)


class _TestAgentsService(NemoServiceAdapter):
    def __init__(self) -> None:
        super().__init__(AgentsService())


class _InProcessClientProvider(DefaultNemoClientProvider):
    """Builds async clients with the platform's client factory over the in-process ASGI transport.

    Plugin code builds its delegated service client with ``get_async_nemo_client(as_service=...)``,
    which the test harness does not route in-process; this keeps the platform's headers (service
    principal, on-behalf-of caller) and only swaps the transport.
    """

    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self._http_client = http_client

    def get_async_nemo_client(
        self,
        *,
        as_service: str | None = None,
        internal: bool = False,
        on_behalf_of: str | RuntimePrincipal | None = None,
        workspace: str | None = None,
    ) -> AsyncNemoClient:
        headers = _build_headers(as_service=as_service, internal=internal, on_behalf_of=on_behalf_of)
        return AsyncNemoClient(
            base_url=str(self._http_client.base_url),
            workspace=workspace,
            default_headers=headers or None,
            http_client=self._http_client,
            owns_http_client=False,
        )


@contextmanager
def _platform(*, auth_enabled: bool = True) -> Generator[ClientContext]:
    with create_test_client(
        _TestAgentsService,
        FilesService,
        auth_enabled=auth_enabled,
        client_type=ClientContext,
        workspaces=[] if auth_enabled else ["default"],
        projects=[],
    ) as ctx:
        set_nemo_client_provider(_InProcessClientProvider(ctx.async_client._http))
        try:
            yield ctx
        finally:
            set_nemo_client_provider(None)


def _skill_md(description: str, tags: str = "[docs, pdf]") -> bytes:
    return f"---\nname: {SKILL}\ndescription: {description}\nmetadata:\n  tags: {tags}\n---\nRead PDFs.\n".encode()


def _upload_source(client: NemoClient, workspace: str, fileset: str, skill_md: bytes) -> str:
    files = FilesClient.from_client(client)
    files.create_fileset(workspace=workspace, body=CreateFilesetRequest(name=fileset), exist_ok=True).data()
    files.upload_file(workspace=workspace, name=fileset, path=f"skills/{SKILL}/SKILL.md", content=skill_md).data()
    files.upload_file(
        workspace=workspace, name=fileset, path=f"skills/{SKILL}/scripts/run.sh", content=b"echo hi\n"
    ).data()
    return f"{workspace}/{fileset}#skills/{SKILL}"


def _input(source: str, tags: list[str] | None = None) -> SkillInput:
    return SkillInput(source=SkillSource(fileset=source), tags=tags or [])


def test_skill_lifecycle_and_permissions() -> None:
    with _platform() as ctx:
        admin = as_user(ctx.client, TEST_ADMIN_EMAIL)
        workspace, private = short_unique_name("skills"), short_unique_name("skills-private")
        for name in (workspace, private):
            WorkspacesClient.from_client(admin).create_workspace(
                query_params=CreateWorkspaceQueryParams(wait_role_propagation=True),
                body=CreateWorkspaceRequest(name=name),
            ).data()

        editor_email, viewer_email = unique_email("skill-editor"), unique_email("skill-viewer")
        grant_workspace_role(admin, workspace=workspace, principal=editor_email, roles=["Editor"])
        grant_workspace_role(admin, workspace=workspace, principal=viewer_email, roles=["Viewer"])
        editor = as_user(ctx.client, editor_email)
        editor_skills = AgentsClient.from_client(editor)
        viewer_skills = AgentsClient.from_client(as_user(ctx.client, viewer_email))
        outsider_skills = AgentsClient.from_client(as_user(ctx.client, unique_email("skill-outsider")))

        source = _upload_source(editor, workspace, "src", _skill_md("Use it to read PDFs."))
        private_source = _upload_source(admin, private, "src", _skill_md("Private."))

        # Create needs agents.skills.create; Viewer has only read and list.
        with pytest.raises(PermissionDeniedError):
            viewer_skills.create_skill(workspace=workspace, name=SKILL, body=_input(source))
        # The caller must be able to read the source, even though the service's own client could.
        with pytest.raises(PermissionDeniedError):
            editor_skills.create_skill(workspace=workspace, name=SKILL, body=_input(private_source))
        with pytest.raises(NotFoundError):
            editor_skills.get_skill(workspace=workspace, name=SKILL)

        first = editor_skills.create_skill(workspace=workspace, name=SKILL, body=_input(source, ["v1"])).data()
        assert (first.revision, first.tags) == (1, {"latest": 1, "v1": 1})
        assert first.frontmatter.tags == ["docs", "pdf"]
        assert first.archive.archive_ref.startswith(f"{workspace}/{SKILLS_FILESET_NAME}#{SKILL}/")
        with pytest.raises(ConflictError):
            editor_skills.create_skill(workspace=workspace, name=SKILL, body=_input(source))

        # Republishing identical content cuts no revision; new content cuts revision 2.
        same = editor_skills.publish_skill(workspace=workspace, name=SKILL, body=_input(source)).data()
        assert same.revision == 1
        _upload_source(editor, workspace, "src", _skill_md("Use it to read and split PDFs.", "[pdf]"))
        second = editor_skills.publish_skill(workspace=workspace, name=SKILL, body=_input(source)).data()
        assert second.revision == 2 and second.content_hash != first.content_hash

        # Reads by tag, by digest and at latest; Viewer may read, an outsider may not.
        latest = viewer_skills.get_skill(workspace=workspace, name=SKILL).data()
        assert (latest.revision, latest.tags) == (2, {"latest": 2, "v1": 1})
        by_tag = viewer_skills.get_skill_revision(workspace=workspace, name=SKILL, revision="v1").data()
        by_digest = viewer_skills.get_skill_revision(
            workspace=workspace, name=SKILL, revision=first.content_hash
        ).data()
        assert by_tag.content_hash == by_digest.content_hash == first.content_hash
        with pytest.raises(NotFoundError):
            viewer_skills.get_skill_revision(workspace=workspace, name=SKILL, revision="nope")
        with pytest.raises(PermissionDeniedError):
            outsider_skills.get_skill(workspace=workspace, name=SKILL)

        revisions = viewer_skills.list_skill_revisions(workspace=workspace, name=SKILL).page().items
        assert [r.revision for r in revisions] == [2, 1]

        # List filters match the frontmatter of any published revision (an entities-service
        # relationship filter over the skill's revisions), not only the latest.
        def listed(query: str | None = None) -> list[str]:
            params: ListSkillsQueryParams = {"filter": query} if query else {}
            return [s.name for s in viewer_skills.list_skills(workspace=workspace, query_params=params).page().items]

        assert listed() == [SKILL]
        assert listed('{"tags": "pdf"}') == [SKILL]
        assert listed('{"tags": "docs"}') == [SKILL]  # only revision 1 carries "docs"
        assert listed('{"tags": "nope"}') == []
        assert listed('{"tags": {"$in": ["nope", "pdf"]}}') == [SKILL]
        assert listed('{"description": {"$like": "split"}}') == [SKILL]
        assert listed('{"description": {"$like": "nothing-like-this"}}') == []

        # Tags: user tags move; 'latest' is reserved; Viewer may not tag.
        tagged = editor_skills.tag_skill_revision(
            workspace=workspace, name=SKILL, tag="stable", query_params={"revision": first.content_hash}
        ).data()
        assert (tagged.revision, tagged.tags) == (1, {"stable": 1, "v1": 1})
        with pytest.raises(UnprocessableEntityError):
            editor_skills.tag_skill_revision(
                workspace=workspace, name=SKILL, tag="latest", query_params={"revision": "v1"}
            )
        with pytest.raises(PermissionDeniedError):
            viewer_skills.tag_skill_revision(
                workspace=workspace, name=SKILL, tag="stable", query_params={"revision": "v1"}
            )

        # Delete needs agents.skills.delete.
        with pytest.raises(PermissionDeniedError):
            viewer_skills.delete_skill(workspace=workspace, name=SKILL)
        editor_skills.delete_skill(workspace=workspace, name=SKILL)
        with pytest.raises(NotFoundError):
            viewer_skills.get_skill(workspace=workspace, name=SKILL)


@dataclass
class _CLIState:
    """The slice of the ``nemo`` CLI state the skills commands use, over the in-process platform."""

    client: NemoClient

    def typed_client(self, client_cls: type[ClientT], timeout: float = 60.0) -> ClientT:
        return client_cls.from_client(self.client)

    def get_workspace(self) -> str | None:
        return "default"

    def get_base_url(self, default: str | None = None) -> str | None:
        return default

    def get_output_format(self, override: str | None = None, *, apply_non_tty_default: bool = True) -> str:
        return override or "json"

    def get_no_truncate(self, override: bool | None = None) -> bool:
        return True

    def get_timestamp_format(self, override: str | None = None) -> str:
        return "iso8601"


def test_cli_publishes_from_a_local_directory_without_auth(tmp_path: Path) -> None:
    directory = tmp_path / SKILL
    (directory / "scripts").mkdir(parents=True)
    (directory / "SKILL.md").write_bytes(_skill_md("Use it to read PDFs."))
    (directory / "scripts" / "run.sh").write_text("echo hi\n")

    with _platform(auth_enabled=False) as ctx:
        app, state = AgentsCLI().get_cli(), _CLIState(ctx.client)

        def run(*args: str) -> dict:
            result = CliRunner().invoke(app, ["skills", *args, "-f", "json"], obj=state)
            assert result.exit_code == 0, result.output
            return json.loads(result.stdout)

        created = run("create", SKILL, "--from-dir", str(directory), "--tag", "v1")
        assert (created["revision"], created["tags"]) == (1, {"latest": 1, "v1": 1})
        # An unchanged directory reuses its upload and publishes no new revision.
        assert run("publish", SKILL, "--from-dir", str(directory))["revision"] == 1

        assert run("get", SKILL, "--revision", created["content_hash"])["revision"] == 1
        assert [s["name"] for s in run("list", "--tag", "pdf")["data"]] == [SKILL]
        assert run("list", "--tag", "nope")["data"] == []
        assert [r["revision"] for r in run("revisions", SKILL)["data"]] == [1]
