# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Skill route handlers: status codes, error mapping, path validation and list filters.

The service is mocked; publishing itself is covered by ``test_skill_service.py`` and the routes end
to end, with real authorization, by ``tests/integration/test_skills_integration.py``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, create_autospec

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from nemo_agents_plugin.api.v2 import skills as skills_routes
from nemo_agents_plugin.api.v2.dependencies import get_skill_service
from nemo_agents_plugin.skill_service import (
    LinkedTasksetNotFoundError,
    SkillExistsError,
    SkillRevisionConflictError,
    SkillRevisionContentMismatchError,
    SkillRevisionNotFoundError,
    SkillService,
    SkillValidationError,
)
from nemo_agents_plugin.skill_storage import SkillSourceError
from nemo_helix_plugin.agents.skills import Skill, SkillArchiveDescriptor, SkillFrontmatter, SkillInput, SkillSource
from nemo_helix_plugin.entity_client import NemoEntityConflictError
from nemo_helix_plugin.filter_ops import ComparisonOperation, FilterOperator, LogicalOperation
from nemo_helix_plugin.schema import Page, PaginationData

BASE = "/apis/agents/v2/workspaces/default/skills"
DIGEST = "c" * 64
BODY = {"source": {"fileset": "default/src#skills/my-skill"}, "tags": ["v1"]}


def _skill(revision: int = 1) -> Skill:
    return Skill(
        name="my-skill",
        workspace="default",
        revision=revision,
        content_hash=DIGEST,
        tags={"latest": revision},
        frontmatter=SkillFrontmatter(name="my-skill", description="Use it."),
        archive=SkillArchiveDescriptor(
            archive_ref=f"default/nemo-agents-skills#my-skill/{'d' * 64}/skill_archive",
            files_hash="d" * 64,
            skill_name="my-skill",
        ),
    )


def _page(*skills: Skill) -> Page[Skill]:
    return Page(
        data=list(skills),
        pagination=PaginationData(
            page=1, page_size=100, current_page_size=len(skills), total_pages=1, total_results=len(skills)
        ),
    )


@pytest.fixture
def service() -> AsyncMock:
    return create_autospec(SkillService, instance=True)


@pytest.fixture
def client(service: AsyncMock) -> Iterator[TestClient]:
    app = FastAPI()
    app.include_router(skills_routes.router, prefix="/apis/agents/v2/workspaces/{workspace}")
    app.dependency_overrides[get_skill_service] = lambda: service
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_create_returns_201_and_passes_the_body(client: TestClient, service: AsyncMock) -> None:
    service.create_skill.return_value = _skill()

    response = client.post(f"{BASE}/my-skill", json=BODY, params={"project": "p"})

    assert response.status_code == 201, response.text
    assert response.json()["content_hash"] == DIGEST
    service.create_skill.assert_awaited_once_with(
        "default",
        "my-skill",
        SkillInput(source=SkillSource(fileset="default/src#skills/my-skill"), tags=["v1"]),
        project="p",
    )


@pytest.mark.parametrize(("published", "expected"), [(True, 201), (False, 200)])
def test_publish_status_says_whether_a_revision_was_cut(
    client: TestClient, service: AsyncMock, published: bool, expected: int
) -> None:
    service.publish_skill.return_value = (_skill(2), published)

    response = client.put(f"{BASE}/my-skill", json=BODY)

    assert response.status_code == expected, response.text
    assert response.json()["revision"] == 2


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (SkillExistsError("exists"), 409),
        (SkillRevisionConflictError("contended"), 409),
        (NemoEntityConflictError("stale"), 409),
        (SkillValidationError("bad tag"), 422),
        (SkillSourceError("no SKILL.md"), 422),
        (LinkedTasksetNotFoundError("no taskset"), 422),
        (SkillRevisionNotFoundError("no revision"), 404),
        (SkillRevisionContentMismatchError("modified"), 500),
        (RuntimeError("boom"), 500),
        (HTTPException(status_code=403, detail="denied"), 403),
    ],
)
def test_service_errors_map_to_statuses(client: TestClient, service: AsyncMock, error: Exception, status: int) -> None:
    service.create_skill.side_effect = error

    response = client.post(f"{BASE}/my-skill", json=BODY)

    assert response.status_code == status, response.text
    if status == 500:
        assert "boom" not in response.text and "modified" not in response.text


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"source": {"fileset": "no-path"}},
        {"source": {"fileset": "default/src#x"}, "taskset": "default/ts#latest"},
        {"source": {"fileset": "default/src#x"}, "extra": 1},
    ],
)
def test_invalid_bodies_are_422(client: TestClient, service: AsyncMock, body: dict[str, Any]) -> None:
    assert client.post(f"{BASE}/my-skill", json=body).status_code == 422
    service.create_skill.assert_not_awaited()


@pytest.mark.parametrize("name", ["My-Skill", "my--skill", "-skill", "a" * 65])
def test_invalid_skill_names_are_422(client: TestClient, service: AsyncMock, name: str) -> None:
    assert client.post(f"{BASE}/{name}", json=BODY).status_code == 422
    assert client.get(f"{BASE}/{name}").status_code == 422
    service.create_skill.assert_not_awaited()
    service.get_skill.assert_not_awaited()


def test_get_latest_by_tag_and_by_digest(client: TestClient, service: AsyncMock) -> None:
    service.get_skill.return_value = _skill()

    assert client.get(f"{BASE}/my-skill").status_code == 200
    assert client.get(f"{BASE}/my-skill/revisions/v1").status_code == 200
    assert client.get(f"{BASE}/my-skill/revisions/{DIGEST}").status_code == 200

    assert [c.args for c in service.get_skill.await_args_list] == [
        ("default", "my-skill"),
        ("default", "my-skill", "v1"),
        ("default", "my-skill", DIGEST),
    ]


def test_missing_skill_is_404_everywhere(client: TestClient, service: AsyncMock) -> None:
    service.get_skill.return_value = None
    service.list_revisions.return_value = None
    service.tag_revision.return_value = None
    service.delete_skill.return_value = False

    assert client.get(f"{BASE}/absent").status_code == 404
    assert client.get(f"{BASE}/absent/revisions/v1").status_code == 404
    assert client.get(f"{BASE}/absent/revisions").status_code == 404
    assert client.put(f"{BASE}/absent/tags/stable", params={"revision": "latest"}).status_code == 404
    assert client.delete(f"{BASE}/absent").status_code == 404


def test_unknown_revision_is_404(client: TestClient, service: AsyncMock) -> None:
    service.get_skill.side_effect = SkillRevisionNotFoundError("no revision tagged 'nope'")

    response = client.get(f"{BASE}/my-skill/revisions/nope")

    assert response.status_code == 404
    assert "nope" in response.json()["detail"]


def test_tag_requires_a_revision_and_maps_reserved_tags(client: TestClient, service: AsyncMock) -> None:
    assert client.put(f"{BASE}/my-skill/tags/stable").status_code == 422
    service.tag_revision.assert_not_awaited()

    service.tag_revision.side_effect = SkillValidationError("'latest' is managed automatically")
    assert client.put(f"{BASE}/my-skill/tags/latest", params={"revision": DIGEST}).status_code == 422

    service.tag_revision.side_effect = None
    service.tag_revision.return_value = _skill()
    assert client.put(f"{BASE}/my-skill/tags/stable", params={"revision": DIGEST}).status_code == 200
    service.tag_revision.assert_awaited_with("default", "my-skill", "stable", DIGEST)


def test_delete_returns_204(client: TestClient, service: AsyncMock) -> None:
    service.delete_skill.return_value = True
    assert client.delete(f"{BASE}/my-skill").status_code == 204


def test_list_revisions_paginates(client: TestClient, service: AsyncMock) -> None:
    service.list_revisions.return_value = _page(_skill(2), _skill(1))

    response = client.get(f"{BASE}/my-skill/revisions", params={"page": 2, "page_size": 5})

    assert response.status_code == 200
    assert [s["revision"] for s in response.json()["data"]] == [2, 1]
    service.list_revisions.assert_awaited_once_with("default", "my-skill", page=2, page_size=5)


def test_list_translates_frontmatter_filters_and_sort(client: TestClient, service: AsyncMock) -> None:
    service.list_skills.return_value = _page(_skill())

    response = client.get(
        BASE,
        params={
            "filter[description][$like]": "pdf",
            "filter[tags][$in]": "docs,pdf",
            "filter[name]": "my-skill",
            "sort": "-name",
        },
    )

    assert response.status_code == 200, response.text
    kwargs = service.list_skills.await_args.kwargs
    assert kwargs["sort"] == "-name"
    operation = kwargs["filter_operation"]
    assert isinstance(operation, LogicalOperation)
    assert set(map(repr, operation.operations)) == set(
        map(
            repr,
            [
                ComparisonOperation(field="data.frontmatter.description", operator=FilterOperator.LIKE, value="pdf"),
                LogicalOperation(
                    operator=FilterOperator.OR,
                    operations=[
                        ComparisonOperation(field="data.frontmatter.tags", operator=FilterOperator.CONTAINS, value=v)
                        for v in ("docs", "pdf")
                    ],
                ),
                ComparisonOperation(field="name", operator=FilterOperator.EQ, value="my-skill"),
            ],
        )
    )


@pytest.mark.parametrize(
    "params",
    [{"filter[tags][$like]": "x"}, {"filter[nope]": "x"}, {"sort": "frontmatter"}],
)
def test_list_rejects_unsupported_filters_and_sorts(
    client: TestClient, service: AsyncMock, params: dict[str, str]
) -> None:
    assert client.get(BASE, params=params).status_code in {400, 422}
    service.list_skills.assert_not_awaited()


def test_list_ignores_a_workspace_override_in_the_filter(client: TestClient, service: AsyncMock) -> None:
    service.list_skills.return_value = _page()

    assert client.get(BASE, params={"filter[workspace]": "other"}).status_code == 200

    assert service.list_skills.await_args.args == ("default",)
    assert service.list_skills.await_args.kwargs["filter_operation"] is None
