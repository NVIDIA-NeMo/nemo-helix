# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The push step behind the broker: a token per image, a signature it cannot make itself, and a
delivery that is the only way its image completes.

The broker and the platform answer through ``httpx.MockTransport``, and crane is replaced by a
recorder, so each test states what they said and asserts what the step did about it -- above all,
that it never delivers a signature for bytes it did not push.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from nemo_builder_plugin.client import BuilderClient
from nemo_builder_plugin.completion import SignatureDelivery
from nemo_builder_plugin.run import push as push_module
from nemo_builder_plugin.run.push import (
    POD_TOKEN_PATH,
    POD_TOKEN_USERNAME,
    Broker,
    BrokerRefused,
    Courier,
    Signed,
    _publish_all,
    _push_one,
    step_token_path,
)
from nemo_builder_plugin.steps import PushImage, PushStepConfig
from nemo_helix_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR
from nemo_helix_plugin.client.errors import NemoHTTPError
from nemo_helix_plugin.jobs.constants import PERSISTENT_JOB_STORAGE_PATH_ENVVAR

REGISTRY = "reg.example.com"
SIGNED = {"payload": base64.b64encode(b"{}").decode(), "signature": base64.b64encode(b"sig").decode()}


def _layout(root: Path) -> str:
    """A well-formed one-manifest OCI layout at ``root``; returns its manifest digest."""
    manifest = json.dumps({"schemaVersion": 2, "layers": []}).encode()
    digest = hashlib.sha256(manifest).hexdigest()
    (root / "blobs" / "sha256").mkdir(parents=True)
    (root / "blobs" / "sha256" / digest).write_bytes(manifest)
    (root / "oci-layout").write_text(json.dumps({"imageLayoutVersion": "1.0.0"}))
    (root / "index.json").write_text(json.dumps({"schemaVersion": 2, "manifests": [{"digest": f"sha256:{digest}"}]}))
    return f"sha256:{digest}"


def _image(name: str = "demo-1-0", *, caller: bool = True) -> PushImage:
    return PushImage(
        image=name,
        system_ref=f"{REGISTRY}/ws/team/app:ws--demo-1-{name[-1]}",
        caller_ref=f"{REGISTRY}/ws/team/app:v1" if caller else None,
    )


class FakeBroker:
    """The broker as the step sees it."""

    def __init__(self, digest: str) -> None:
        self.digest = digest
        self.calls: list[str] = []

    def credentials(self) -> dict[str, object]:
        self.calls.append("credentials")
        return {"auths": {REGISTRY: {"registrytoken": f"token-{len(self.calls)}"}}}

    def sign(self, image: str) -> Signed:
        self.calls.append(f"sign:{image}")
        return Signed(digest=self.digest, delivery=SignatureDelivery.model_validate(SIGNED))


class FakeCourier:
    def __init__(self, statuses: dict[str, str] | None = None) -> None:
        self.statuses = statuses or {}
        self.delivered: list[str] = []

    def deliver(self, image: str, delivery: SignatureDelivery) -> None:
        self.delivered.append(image)

    def status(self, image: str) -> str:
        return self.statuses.get(image, "pending")


@pytest.fixture
def crane(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def run(args: list[str], *, env: object = None) -> str:
        calls.append(args)
        return ""

    monkeypatch.setattr(push_module, "run_tool", run)
    return calls


class TestTheStepsToken:
    def test_the_workload_token_when_jobs_projects_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With platform auth on, the broker reviews it for the workload audience."""
        monkeypatch.setenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, "/var/run/secrets/nhx/token")
        assert step_token_path() == Path("/var/run/secrets/nhx/token")

    def test_otherwise_the_pods_service_account_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, raising=False)
        assert step_token_path() == Path(POD_TOKEN_PATH)


class TestTheBrokerClient:
    def _broker(self, tmp_path: Path, handler: Callable[[httpx.Request], httpx.Response]) -> Broker:
        (tmp_path / "token").write_text("pod-token-1\n")
        return Broker(
            "http://broker:8080/",
            token_path=tmp_path / "token",
            http=httpx.Client(transport=httpx.MockTransport(handler)),
        )

    def test_credentials_are_asked_for_with_the_pods_token_and_no_scope(self, tmp_path: Path) -> None:
        seen: list[httpx.Request] = []

        def broker(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={"auths": {REGISTRY: {"registrytoken": "t"}}, "repositories": ["ws/app"]})

        assert self._broker(tmp_path, broker).credentials() == {"auths": {REGISTRY: {"registrytoken": "t"}}}
        expected = base64.b64encode(f"{POD_TOKEN_USERNAME}:pod-token-1".encode()).decode()
        assert seen[0].url == "http://broker:8080/credentials"
        assert seen[0].headers["Authorization"] == f"Basic {expected}"
        assert seen[0].content in (b"", b"null")

    def test_the_token_is_read_fresh_for_each_request(self, tmp_path: Path) -> None:
        """The kubelet rotates it; a publish of many images can outlive one."""
        tokens: list[str] = []

        def broker(request: httpx.Request) -> httpx.Response:
            tokens.append(base64.b64decode(request.headers["Authorization"].split()[1]).decode().split(":")[1])
            return httpx.Response(200, json={"auths": {REGISTRY: {}}})

        client = self._broker(tmp_path, broker)
        client.credentials()
        (tmp_path / "token").write_text("pod-token-2")
        client.credentials()
        assert tokens == ["pod-token-1", "pod-token-2"]

    def test_a_signature_is_asked_for_by_row_name_only(self, tmp_path: Path) -> None:
        bodies: list[object] = []

        def broker(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json={"digest": "sha256:" + "d" * 64, "signed": SIGNED})

        signed = self._broker(tmp_path, broker).sign("demo-1-0")
        assert bodies == [{"image": "demo-1-0"}]
        assert signed.digest == "sha256:" + "d" * 64 and signed.delivery.signature == SIGNED["signature"]

    @pytest.mark.parametrize(
        "response",
        [
            httpx.Response(403, json={"errors": [{"code": "DENIED"}]}),
            httpx.Response(200, json={"digest": "latest", "signed": SIGNED}),
            httpx.Response(200, json=["not", "an", "object"]),
        ],
    )
    def test_anything_but_a_signature_is_refused(self, tmp_path: Path, response: httpx.Response) -> None:
        with pytest.raises(BrokerRefused):
            self._broker(tmp_path, lambda request: response).sign("demo-1-0")

    def test_no_credential_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(BrokerRefused, match="no credential"):
            self._broker(tmp_path, lambda request: httpx.Response(200, json={"auths": {}})).credentials()


class TestTheCourier:
    def _courier(self, responses: list[httpx.Response]) -> tuple[Courier, list[float]]:
        def platform(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/apis/builder/v2/workspaces/ws/container-images/demo-1-0/signature"
            return responses.pop(0)

        sleeps: list[float] = []
        client = BuilderClient(
            base_url="http://platform", http_client=httpx.Client(transport=httpx.MockTransport(platform))
        )
        return Courier(client, workspace="ws", sleep=sleeps.append), sleeps

    def _ready(self) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "name": "demo-1-0",
                "workspace": "ws",
                "status": "ready",
                "registry": REGISTRY,
                "repository": "ws/app",
                "provenance": {
                    "build_set": "demo",
                    "revision": 1,
                    "job": "ws/demo-1",
                    "system_tag": "ws--demo-1-0",
                    "request_digest": "sha256:" + "a" * 64,
                },
            },
        )

    def test_delivery_that_succeeds(self) -> None:
        courier, sleeps = self._courier([self._ready()])
        courier.deliver("demo-1-0", SignatureDelivery.model_validate(SIGNED))
        assert sleeps == []

    def test_a_platform_that_is_briefly_down_is_retried(self) -> None:
        """The route is idempotent for the same signature, so a retry is safe."""
        courier, sleeps = self._courier([httpx.Response(503, json={"detail": "down"}), self._ready()])
        courier.deliver("demo-1-0", SignatureDelivery.model_validate(SIGNED))
        assert sleeps == [1.0]

    def test_a_refused_signature_is_not_retried(self) -> None:
        courier, sleeps = self._courier([httpx.Response(422, json={"detail": "Signature refused"})])
        with pytest.raises(NemoHTTPError):
            courier.deliver("demo-1-0", SignatureDelivery.model_validate(SIGNED))
        assert sleeps == []

    def test_a_platform_that_stays_down_is_given_up_on(self) -> None:
        courier, sleeps = self._courier([httpx.Response(503, json={"detail": "down"}) for _ in range(5)])
        with pytest.raises(NemoHTTPError):
            courier.deliver("demo-1-0", SignatureDelivery.model_validate(SIGNED))
        assert sleeps == [1.0, 2.0, 4.0, 8.0]


class TestPushOne:
    def test_pushes_the_callers_tag_then_the_system_tag_then_delivers(
        self, tmp_path: Path, crane: list[list[str]]
    ) -> None:
        digest = _layout(tmp_path / "out")
        broker, courier = FakeBroker(digest), FakeCourier()
        _push_one(_image(), tmp_path / "out", broker=broker, courier=courier, docker_config=tmp_path, plain_http=False)
        assert [call[3] for call in crane] == [f"{REGISTRY}/ws/team/app:v1", f"{REGISTRY}/ws/team/app:ws--demo-1-0"]
        assert broker.calls == ["credentials", "sign:demo-1-0"]
        assert courier.delivered == ["demo-1-0"]
        written = json.loads((tmp_path / "config.json").read_text())
        assert written == {"auths": {REGISTRY: {"registrytoken": "token-1"}}}
        assert (tmp_path / "config.json").stat().st_mode & 0o777 == 0o600

    def test_a_plain_http_registry_is_pushed_over_http_only_when_configured(
        self, tmp_path: Path, crane: list[list[str]]
    ) -> None:
        digest = _layout(tmp_path / "out")
        _push_one(
            _image(caller=False),
            tmp_path / "out",
            broker=FakeBroker(digest),
            courier=FakeCourier(),
            docker_config=tmp_path,
            plain_http=True,
        )
        assert crane == [["crane", "push", str(tmp_path / "out"), f"{REGISTRY}/ws/team/app:ws--demo-1-0", "--insecure"]]

    def test_a_signature_over_other_bytes_is_never_delivered(self, tmp_path: Path, crane: list[list[str]]) -> None:
        """The system tag moved between the push and the signature: that signature is not this image's."""
        _layout(tmp_path / "out")
        courier = FakeCourier()
        with pytest.raises(BrokerRefused, match="moved"):
            _push_one(
                _image(),
                tmp_path / "out",
                broker=FakeBroker("sha256:" + "e" * 64),
                courier=courier,
                docker_config=tmp_path,
                plain_http=False,
            )
        assert courier.delivered == []


class TestPublishAll:
    def _config(self, *names: str) -> PushStepConfig:
        return PushStepConfig(images=[_image(name) for name in names], registry=REGISTRY, broker="http://broker:8080")

    def _work(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *built: str) -> str:
        monkeypatch.setenv(PERSISTENT_JOB_STORAGE_PATH_ENVVAR, str(tmp_path))
        digest = ""
        for name in built:
            digest = _layout(tmp_path / "out" / name)
        return digest

    def test_every_built_image_is_published_and_delivered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crane: list[list[str]]
    ) -> None:
        digest = self._work(tmp_path, monkeypatch, "demo-1-0", "demo-1-1")
        courier = FakeCourier()
        code = _publish_all(
            self._config("demo-1-0", "demo-1-1"),
            broker=FakeBroker(digest),
            courier=courier,
            docker_config=tmp_path,
        )
        assert code == 0 and courier.delivered == ["demo-1-0", "demo-1-1"]

    def test_a_missing_layout_fails_that_image_and_not_the_rest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crane: list[list[str]]
    ) -> None:
        digest = self._work(tmp_path, monkeypatch, "demo-1-1")
        courier = FakeCourier()
        code = _publish_all(
            self._config("demo-1-0", "demo-1-1"),
            broker=FakeBroker(digest),
            courier=courier,
            docker_config=tmp_path,
        )
        assert code == 1 and courier.delivered == ["demo-1-1"]

    def test_a_rerun_skips_what_already_settled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crane: list[list[str]]
    ) -> None:
        """An evicted step reruns: a ready row needs nothing, and the broker would grant nothing for it."""
        digest = self._work(tmp_path, monkeypatch, "demo-1-0", "demo-1-1")
        broker, courier = FakeBroker(digest), FakeCourier({"demo-1-0": "ready"})
        code = _publish_all(
            self._config("demo-1-0", "demo-1-1"),
            broker=broker,
            courier=courier,
            docker_config=tmp_path,
        )
        assert code == 0 and courier.delivered == ["demo-1-1"]
        assert "sign:demo-1-0" not in broker.calls

    def test_a_row_that_already_failed_counts_as_a_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crane: list[list[str]]
    ) -> None:
        digest = self._work(tmp_path, monkeypatch, "demo-1-0")
        code = _publish_all(
            self._config("demo-1-0"),
            broker=FakeBroker(digest),
            courier=FakeCourier({"demo-1-0": "failed"}),
            docker_config=tmp_path,
        )
        assert code == 1
