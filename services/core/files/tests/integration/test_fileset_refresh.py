# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests: refreshing a fileset against the revision it tracks."""

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import httpx

DEFAULT_WORKSPACE = "default"
FILESETS_URL = f"/apis/files/v2/workspaces/{DEFAULT_WORKSPACE}/filesets"

FIRST_SHA = "1" * 40
SECOND_SHA = "2" * 40


class _FakeResponse:
    def __init__(self, sha: str, status: int = 200):
        self._sha = sha
        self.status = status
        self.headers: dict[str, str] = {}

    async def json(self) -> dict[str, Any]:
        return {"sha": self._sha}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc) -> None:
        return None


class _FakeSession:
    """Answers every GitHub API read with one commit SHA."""

    def __init__(self, sha: str, status: int = 200):
        self.sha = sha
        self.status = status
        self.requests: list[str] = []

    def get(self, url: str, **_kwargs):
        self.requests.append(url)
        return _FakeResponse(self.sha, status=self.status)


@contextmanager
def _github_at(sha: str, status: int = 200) -> Iterator[_FakeSession]:
    session = _FakeSession(sha, status=status)
    with patch("nhx.core.files.app.backends.github.get_http_session", return_value=session):
        yield session


def _create_github_fileset(test_client: httpx.Client, name: str, revision: str = "main") -> dict[str, Any]:
    response = test_client.post(
        FILESETS_URL,
        json={
            "name": name,
            "storage": {"type": "github", "owner": "acme", "repo": "agents", "revision": revision},
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


class TestRefreshFileset:
    def test_moves_a_tracked_fileset_to_the_commit_the_ref_names_now(self, test_client: httpx.Client) -> None:
        name = f"gh-refresh-{uuid.uuid4().hex[:8]}"
        with _github_at(FIRST_SHA):
            created = _create_github_fileset(test_client, name)
        assert created["storage"]["revision"] == FIRST_SHA

        with _github_at(SECOND_SHA) as github:
            response = test_client.post(f"{FILESETS_URL}/{name}/refresh")

        assert response.status_code == 200, response.text
        storage = response.json()["storage"]
        assert storage["revision"] == SECOND_SHA
        assert storage["original_revision"] == "main"
        # The ref is what gets re-resolved; re-resolving the pinned SHA would report
        # the same move without ever asking about the branch.
        assert any(url.endswith("/commits/main") for url in github.requests), github.requests

        # The refreshed revision is what a later read serves, not just what the call returned.
        assert test_client.get(f"{FILESETS_URL}/{name}").json()["storage"]["revision"] == SECOND_SHA

    def test_leaves_the_repository_and_directory_alone(self, test_client: httpx.Client) -> None:
        name = f"gh-scoped-{uuid.uuid4().hex[:8]}"
        with _github_at(FIRST_SHA):
            response = test_client.post(
                FILESETS_URL,
                json={
                    "name": name,
                    "storage": {
                        "type": "github",
                        "owner": "acme",
                        "repo": "agents",
                        "revision": "main",
                        "path": "agents/calc",
                    },
                },
            )
            assert response.status_code == 200, response.text

        with _github_at(SECOND_SHA):
            storage = test_client.post(f"{FILESETS_URL}/{name}/refresh").json()["storage"]

        assert (storage["owner"], storage["repo"], storage["path"]) == ("acme", "agents", "agents/calc")

    def test_refuses_a_fileset_pinned_to_a_commit(self, test_client: httpx.Client) -> None:
        """A revision the user pinned themselves tracks nothing, so there is nowhere to move."""
        name = f"gh-pinned-{uuid.uuid4().hex[:8]}"
        with _github_at(FIRST_SHA):
            _create_github_fileset(test_client, name, revision=FIRST_SHA)

        response = test_client.post(f"{FILESETS_URL}/{name}/refresh")

        assert response.status_code == 409
        assert "does not track a revision" in response.json()["detail"]

    def test_refuses_a_pinned_fileset_without_resolving_its_secrets(self, test_client: httpx.Client) -> None:
        """Eligibility is a property of the stored config, so it is decided first.

        Resolving secrets first would report a missing or forbidden secret as 400
        for a fileset whose real answer is 409.
        """
        name = f"gh-pinned-secret-{uuid.uuid4().hex[:8]}"
        with _github_at(FIRST_SHA):
            _create_github_fileset(test_client, name, revision=FIRST_SHA)

        with patch(
            "nhx.core.files.api.v2.filesets.endpoints.resolve_storage_secrets_for_user",
            side_effect=AssertionError("secrets must not be resolved for an ineligible fileset"),
        ):
            response = test_client.post(f"{FILESETS_URL}/{name}/refresh")

        assert response.status_code == 409

    def test_refuses_a_fileset_holding_uploaded_files(self, test_client: httpx.Client) -> None:
        name = f"local-{uuid.uuid4().hex[:8]}"
        assert test_client.post(FILESETS_URL, json={"name": name}).status_code == 200

        response = test_client.post(f"{FILESETS_URL}/{name}/refresh")

        assert response.status_code == 409

    def test_reports_github_being_unavailable(self, test_client: httpx.Client) -> None:
        name = f"gh-down-{uuid.uuid4().hex[:8]}"
        with _github_at(FIRST_SHA):
            _create_github_fileset(test_client, name)

        with _github_at(SECOND_SHA, status=503):
            response = test_client.post(f"{FILESETS_URL}/{name}/refresh")

        assert response.status_code == 502
        assert test_client.get(f"{FILESETS_URL}/{name}").json()["storage"]["revision"] == FIRST_SHA

    def test_reports_a_ref_github_no_longer_has(self, test_client: httpx.Client) -> None:
        name = f"gh-gone-{uuid.uuid4().hex[:8]}"
        with _github_at(FIRST_SHA):
            _create_github_fileset(test_client, name)

        with _github_at(SECOND_SHA, status=404):
            response = test_client.post(f"{FILESETS_URL}/{name}/refresh")

        assert response.status_code == 400
        assert test_client.get(f"{FILESETS_URL}/{name}").json()["storage"]["revision"] == FIRST_SHA

    def test_reports_a_fileset_that_does_not_exist(self, test_client: httpx.Client) -> None:
        response = test_client.post(f"{FILESETS_URL}/never-created/refresh")

        assert response.status_code == 404
