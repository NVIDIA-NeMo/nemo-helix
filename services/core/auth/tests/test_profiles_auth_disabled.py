# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""/me and preferences on local installs where auth is disabled."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from nhx.core.auth.service import AuthService
from nhx.testing.client import create_test_client

ME = "/apis/auth/v2/me"


@pytest.fixture(scope="module")
def http_client() -> Iterator[TestClient]:
    with create_test_client(AuthService, client_type=TestClient) as client:
        yield client


def test_anonymous_browser_gets_stable_local_account(http_client: TestClient) -> None:
    first = http_client.get(ME)
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["auth_method"] == "disabled"
    assert body["principal_id"] == ""
    assert body["profile"]["display_name"] == "Local user"
    assert body["can_edit_profile"] is True
    assert http_client.get(ME).json()["account_id"] == body["account_id"]

    renamed = http_client.patch(f"{ME}/profile", json={"display_name": "Sam"})
    assert renamed.json()["profile"]["display_name"] == "Sam"

    saved = http_client.patch("/apis/auth/v2/me/preferences/studio.ui", json={"theme": "light"})
    assert saved.status_code == 200, saved.text
    assert http_client.get("/apis/auth/v2/me/preferences/studio.ui").json()["value"] == {"theme": "light"}


def test_principal_headers_resolve_their_own_account(http_client: TestClient) -> None:
    local = http_client.get(ME).json()
    cli = http_client.get(
        ME, headers={"X-NHX-Principal-Id": "dev@example.com", "X-NHX-Principal-Email": "dev@example.com"}
    )

    assert cli.status_code == 200, cli.text
    assert cli.json()["account_id"] != local["account_id"]
    assert cli.json()["profile"]["display_name"] == "dev@example.com"
    assert (
        http_client.get(
            "/apis/auth/v2/me/preferences/studio.ui", headers={"X-NHX-Principal-Id": "dev@example.com"}
        ).json()["value"]
        == {}
    )
