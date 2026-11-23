# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the taskset E2E Authentik helper without contacting an identity provider."""

import json
import stat
import urllib.parse
import urllib.request
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest

from e2e.k8s.scripts import authentik_admin

BOOTSTRAP_TOKEN = "fake-bootstrap-token"
APP_PASSWORD = "fake-app-password"
USER_LOOKUP = "/api/v3/core/users/?username=demo%2Btester"
USER_DETAIL = "/api/v3/core/users/42/"
Response = tuple[str, str, dict[str, Any] | None]


@pytest.fixture
def fake_authentik(monkeypatch: pytest.MonkeyPatch) -> Callable[..., list[urllib.request.Request]]:
    """Stub the network boundary and retain each outgoing request for assertions."""
    monkeypatch.setenv("AUTHENTIK_API", "https://authentik.test/")
    monkeypatch.setenv("AUTHENTIK_BOOTSTRAP_TOKEN", BOOTSTRAP_TOKEN)

    def install(*responses: Response) -> list[urllib.request.Request]:
        pending = iter(responses)
        requests: list[urllib.request.Request] = []

        def urlopen(request: urllib.request.Request, *, timeout: int) -> BytesIO:
            requests.append(request)
            method, path, body = next(pending)
            assert timeout == 30
            assert request.get_method() == method
            assert request.full_url == f"https://authentik.test{path}"
            assert request.get_header("Authorization") == f"Bearer {BOOTSTRAP_TOKEN}"
            return BytesIO(json.dumps(body).encode() if body is not None else b"")

        monkeypatch.setattr(
            authentik_admin,
            "urllib",
            SimpleNamespace(
                parse=urllib.parse, request=SimpleNamespace(Request=urllib.request.Request, urlopen=urlopen)
            ),
        )
        return requests

    return install


@pytest.mark.parametrize("existing_file", [False, True], ids=["new-file", "replace-existing"])
def test_app_password_is_written_privately_without_logging_the_key(
    fake_authentik: Callable[..., list[urllib.request.Request]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    existing_file: bool,
) -> None:
    monkeypatch.setattr(authentik_admin, "uuid", SimpleNamespace(uuid4=lambda: UUID(int=0)))
    identifier = "e2e-demo+tester-00000000"
    requests = fake_authentik(
        ("GET", USER_LOOKUP, {"results": [{"pk": 42}]}),
        ("POST", "/api/v3/core/tokens/", {}),
        ("GET", f"/api/v3/core/tokens/{identifier}/view_key/", {"key": APP_PASSWORD}),
    )
    output = tmp_path / "app-password"
    if existing_file:
        output.write_text("old password with trailing data")
        output.chmod(0o644)

    authentik_admin.app_password("demo+tester", str(output))

    assert len(requests) == 3
    body = requests[1].data
    assert body is not None
    assert json.loads(body) == {"identifier": identifier, "intent": "app_password", "user": 42, "expiring": False}
    assert output.read_text() == APP_PASSWORD
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    captured = capsys.readouterr()
    assert captured.out == ""
    assert identifier in captured.err
    assert APP_PASSWORD not in captured.err and BOOTSTRAP_TOKEN not in captured.err


@pytest.mark.parametrize("current_email", ["old@example.test", "admin@example.test"])
def test_set_email_patches_only_when_the_address_differs(
    fake_authentik: Callable[..., list[urllib.request.Request]], current_email: str
) -> None:
    target_email = "admin@example.test"
    responses: list[Response] = [
        ("GET", USER_LOOKUP, {"results": [{"pk": 42}]}),
        ("GET", USER_DETAIL, {"email": current_email}),
    ]
    if current_email != target_email:
        responses.append(("PATCH", USER_DETAIL, None))
    requests = fake_authentik(*responses)

    authentik_admin.set_email("demo+tester", target_email)

    assert len(requests) == len(responses)
    if current_email != target_email:
        body = requests[-1].data
        assert body is not None and json.loads(body) == {"email": target_email}


def test_app_password_for_a_missing_user_does_not_create_a_file(
    fake_authentik: Callable[..., list[urllib.request.Request]], tmp_path: Path
) -> None:
    requests = fake_authentik(("GET", USER_LOOKUP, {"results": []}))
    output = tmp_path / "app-password"

    with pytest.raises(SystemExit, match=r"no Authentik user 'demo\+tester'"):
        authentik_admin.app_password("demo+tester", str(output))

    assert len(requests) == 1
    assert not output.exists()
