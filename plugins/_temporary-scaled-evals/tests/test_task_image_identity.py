# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json

import httpx
import pytest

pytest.importorskip("scaled_evals")

from scaled_evals.api.build.task_image_identity import (
    TaskImageIdentityError,
    parse_task_image_ref,
    require_allowed_task_image,
    resolve_task_image,
    resolve_upstream_image,
    verify_stored_task_image,
)
from scaled_evals.api.settings import settings

DIGEST = "sha256:" + "a" * 64
OTHER_DIGEST = "sha256:" + "b" * 64


@pytest.fixture(autouse=True)
def _registry_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "task_image_validation_mode", "resolve")
    monkeypatch.setattr(settings, "task_image_allowed_registries", "registry.example.com")
    monkeypatch.setattr(settings, "task_image_allowed_repositories", "")
    monkeypatch.setattr(settings, "task_image_registry_auth_file", "")
    monkeypatch.setattr(settings, "task_image_registry_insecure", False)
    monkeypatch.setattr(
        settings,
        "harbor_dataset_upstream_allowed_registries",
        "docker.io,registry.example.com",
    )


def test_parse_tag_and_digest_references() -> None:
    tagged = parse_task_image_ref("registry.example.com/team/task:release-1")
    pinned = parse_task_image_ref(f"registry.example.com/team/task@{DIGEST}")

    assert tagged.tag == "release-1"
    assert tagged.digest is None
    assert pinned.digest == DIGEST
    assert pinned.digest_ref(DIGEST) == f"registry.example.com/team/task@{DIGEST}"


@pytest.mark.parametrize(
    "image_ref",
    [
        "team/task:latest",
        "registry.example.com/team/task",
        f"registry.example.com/team/task:tag@{DIGEST}",
        "https://registry.example.com/team/task:tag",
        "registry.example.com/Team/task:tag",
    ],
)
def test_parse_rejects_ambiguous_or_malformed_references(image_ref: str) -> None:
    with pytest.raises(TaskImageIdentityError):
        parse_task_image_ref(image_ref)


def test_registry_and_optional_repository_policy_use_path_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref = parse_task_image_ref("registry.example.com/team/tasks/one:tag")

    require_allowed_task_image(ref)
    monkeypatch.setattr(
        settings,
        "task_image_allowed_repositories",
        "registry.example.com/team/tasks/*",
    )
    require_allowed_task_image(ref)
    monkeypatch.setattr(
        settings,
        "task_image_allowed_repositories",
        "registry.example.com/team/task",
    )
    with pytest.raises(TaskImageIdentityError, match="repository .* is not approved"):
        require_allowed_task_image(ref)


def test_resolve_tag_preserves_runtime_ref_and_records_manifest_digest() -> None:
    request_seen: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_seen
        request_seen = request
        return httpx.Response(200, headers={"Docker-Content-Digest": DIGEST})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = resolve_task_image("registry.example.com/team/task:signed", client=client)

    assert result.runtime_ref == "registry.example.com/team/task:signed"
    assert result.digest == DIGEST
    assert result.immutable_ref == f"registry.example.com/team/task@{DIGEST}"
    assert request_seen is not None
    assert request_seen.url.path == "/v2/team/task/manifests/signed"


def test_resolve_rejects_claimed_or_builder_digest_mismatch() -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(200, headers={"Docker-Content-Digest": OTHER_DIGEST})
    )
    with (
        httpx.Client(transport=transport) as client,
        pytest.raises(TaskImageIdentityError, match="digest mismatch"),
    ):
        resolve_task_image(
            "registry.example.com/team/task:signed",
            expected_digest=DIGEST,
            client=client,
        )


def test_resolve_supports_registry_bearer_challenge() -> None:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        if request.url.path == "/token":
            return httpx.Response(200, json={"token": "manifest-token"})
        if request.headers.get("Authorization") == "Bearer manifest-token":
            return httpx.Response(200, headers={"Docker-Content-Digest": DIGEST})
        return httpx.Response(
            401,
            headers={
                "WWW-Authenticate": (
                    'Bearer realm="https://registry.example.com/token",'
                    'service="registry.example.com",scope="repository:team/task:pull"'
                )
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = resolve_task_image("registry.example.com/team/task:signed", client=client)

    assert result.digest == DIGEST
    assert any("service=registry.example.com" in url for url in requests)


def test_external_bearer_realm_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # noqa: ANN001
    encoded = base64.b64encode(b"user:secret").decode()
    auth_file = tmp_path / ".dockerconfigjson"
    auth_file.write_text(json.dumps({"auths": {"registry.example.com": {"auth": encoded}}}))
    monkeypatch.setattr(settings, "task_image_registry_auth_file", str(auth_file))

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == f"Basic {encoded}"
        return httpx.Response(
            401,
            headers={"WWW-Authenticate": 'Bearer realm="https://auth.example.com/token"'},
        )

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(TaskImageIdentityError, match="host must match"),
    ):
        resolve_task_image("registry.example.com/team/task:signed", client=client)


def test_resolve_uses_docker_config_basic_auth(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # noqa: ANN001
    encoded = base64.b64encode(b"user:secret").decode()
    auth_file = tmp_path / ".dockerconfigjson"
    auth_file.write_text(json.dumps({"auths": {"registry.example.com": {"auth": encoded}}}))
    monkeypatch.setattr(settings, "task_image_registry_auth_file", str(auth_file))

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == f"Basic {encoded}"
        return httpx.Response(200, headers={"Docker-Content-Digest": DIGEST})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        resolve_task_image("registry.example.com/team/task:signed", client=client)


def test_resolve_uses_inline_docker_config_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    encoded = base64.b64encode(b"user:secret").decode()
    monkeypatch.setattr(settings, "task_image_registry_auth_file", "")
    monkeypatch.setattr(
        settings,
        "task_image_registry_auth_json",
        json.dumps({"auths": {"registry.example.com": {"auth": encoded}}}),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == f"Basic {encoded}"
        return httpx.Response(200, headers={"Docker-Content-Digest": DIGEST})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        resolve_task_image("registry.example.com/team/task:signed", client=client)


def test_upstream_docker_hub_uses_anonymous_auth_when_config_has_no_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:  # noqa: ANN001
    auth_file = tmp_path / ".dockerconfigjson"
    auth_file.write_text(json.dumps({"auths": {"registry.example.com": {"auth": "unused"}}}))
    monkeypatch.setattr(settings, "task_image_registry_auth_file", str(auth_file))

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "auth.docker.io":
            assert "Authorization" not in request.headers
            return httpx.Response(200, json={"token": "anonymous-token"})
        assert request.url.host == "registry-1.docker.io"
        if request.headers.get("Authorization") == "Bearer anonymous-token":
            return httpx.Response(200, headers={"Docker-Content-Digest": DIGEST})
        assert "Authorization" not in request.headers
        return httpx.Response(
            401,
            headers={
                "WWW-Authenticate": (
                    'Bearer realm="https://auth.docker.io/token",'
                    'service="registry.docker.io",scope="repository:library/ubuntu:pull"'
                )
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = resolve_upstream_image("ubuntu:latest", client=client)

    assert result.immutable_ref == f"docker.io/library/ubuntu@{DIGEST}"
    assert len(requests) == 3


def test_upstream_resolution_ignores_configured_registry_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # noqa: ANN001
    encoded = base64.b64encode(b"user:secret").decode()
    auth_file = tmp_path / ".dockerconfigjson"
    auth_file.write_text(json.dumps({"auths": {"registry.example.com": {"auth": encoded}}}))
    monkeypatch.setattr(settings, "task_image_registry_auth_file", str(auth_file))

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/token":
            assert "Authorization" not in request.headers
            return httpx.Response(200, json={"token": "anonymous-token"})
        if request.headers.get("Authorization") == "Bearer anonymous-token":
            return httpx.Response(200, headers={"Docker-Content-Digest": DIGEST})
        assert "Authorization" not in request.headers
        return httpx.Response(
            401,
            headers={
                "WWW-Authenticate": (
                    'Bearer realm="https://registry.example.com/token",'
                    'service="registry.example.com",scope="repository:team/task:pull"'
                )
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = resolve_upstream_image("registry.example.com/team/task:latest", client=client)

    assert result.immutable_ref == f"registry.example.com/team/task@{DIGEST}"
    assert len(requests) == 3


def test_upstream_resolution_rejects_private_image_without_using_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:  # noqa: ANN001
    encoded = base64.b64encode(b"user:secret").decode()
    auth_file = tmp_path / ".dockerconfigjson"
    auth_file.write_text(json.dumps({"auths": {"registry.example.com": {"auth": encoded}}}))
    monkeypatch.setattr(settings, "task_image_registry_auth_file", str(auth_file))

    def handler(request: httpx.Request) -> httpx.Response:
        assert "Authorization" not in request.headers
        if request.url.path == "/token":
            return httpx.Response(401)
        return httpx.Response(
            401,
            headers={
                "WWW-Authenticate": (
                    'Bearer realm="https://registry.example.com/token",'
                    'service="registry.example.com",scope="repository:team/private:pull"'
                )
            },
        )

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(TaskImageIdentityError, match="not publicly readable"),
    ):
        resolve_upstream_image("registry.example.com/team/private:latest", client=client)


def test_dispatch_revalidation_rejects_moved_tag() -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(200, headers={"Docker-Content-Digest": OTHER_DIGEST})
    )
    with (
        httpx.Client(transport=transport) as client,
        pytest.raises(TaskImageIdentityError, match="digest mismatch"),
    ):
        verify_stored_task_image(
            "registry.example.com/team/task:signed",
            DIGEST,
            client=client,
        )


def test_dispatch_accepts_consistent_digest_without_registry_io() -> None:
    transport = httpx.MockTransport(lambda _request: pytest.fail("unexpected registry request"))
    with httpx.Client(transport=transport) as client:
        verify_stored_task_image(
            f"registry.example.com/team/task@{DIGEST}",
            DIGEST,
            client=client,
        )


def test_hosted_policy_rejects_digest_runtime_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "task_image_hosted_mode", True)
    transport = httpx.MockTransport(lambda _request: pytest.fail("unexpected registry request"))
    with (
        httpx.Client(transport=transport) as client,
        pytest.raises(TaskImageIdentityError, match="signed tag-form reference"),
    ):
        resolve_task_image(f"registry.example.com/team/task@{DIGEST}", client=client)


def test_legacy_ready_revision_fails_with_actionable_error() -> None:
    with pytest.raises(TaskImageIdentityError, match="create, upload, and finalize a new revision"):
        verify_stored_task_image("registry.example.com/team/task:legacy", None)
