# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The push step: the workspace's credential from its environment, then push, sign and complete each image."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from nemo_builder_plugin.client import BuilderClient
from nemo_builder_plugin.entities import ContainerImage
from nemo_builder_plugin.registry import RegistryError
from nemo_builder_plugin.run import push as push_module
from nemo_builder_plugin.run import tools
from nemo_builder_plugin.run.push import Builder, Credentials, Refused, _publish_all, _push_one, signer
from nemo_builder_plugin.signing import SigningKey, key_fingerprint
from nemo_builder_plugin.steps import (
    REGISTRY_PASSWORD_ENV,
    REGISTRY_USERNAME_ENV,
    SIGNING_KEY_ENV,
    PushImage,
    PushStepConfig,
)
from nemo_helix_plugin.client.errors import NemoHTTPError
from nemo_helix_plugin.jobs.constants import PERSISTENT_JOB_STORAGE_PATH_ENVVAR

REGISTRY = "reg.example.com"
IMAGES = "/apis/builder/v2/workspaces/ws/container-images"
KEY = ec.generate_private_key(ec.SECP256R1())
PEM = KEY.private_bytes(
    serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
).decode()


def _layout(root: Path) -> str:
    blobs = root / "blobs" / "sha256"
    blobs.mkdir(parents=True)
    config = b"{}"
    (blobs / hashlib.sha256(config).hexdigest()).write_bytes(config)
    manifest = json.dumps(
        {"schemaVersion": 2, "config": {"digest": f"sha256:{hashlib.sha256(config).hexdigest()}"}, "layers": []}
    ).encode()
    digest = hashlib.sha256(manifest).hexdigest()
    (blobs / digest).write_bytes(manifest)
    (root / "oci-layout").write_text(json.dumps({"imageLayoutVersion": "1.0.0"}))
    (root / "index.json").write_text(json.dumps({"schemaVersion": 2, "manifests": [{"digest": f"sha256:{digest}"}]}))
    return f"sha256:{digest}"


def _image(name: str = "demo-1-0", *, caller: bool = True) -> PushImage:
    return PushImage(
        image=name,
        system_ref=f"{REGISTRY}/ws/team/app:ws--demo-1-{name[-1]}",
        caller_ref=f"{REGISTRY}/ws/team/app:v1" if caller else None,
    )


def _row(name: str = "demo-1-0", digest: str | None = None, status: str = "pending") -> ContainerImage:
    return ContainerImage.model_validate(
        {
            "name": name,
            "workspace": "ws",
            "status": status,
            "registry": REGISTRY,
            "repository": "ws/team/app",
            "digest": digest,
            "provenance": {
                "build_set": "demo",
                "revision": 1,
                "job": "ws/demo-1",
                "request_digest": "sha256:" + "a" * 64,
            },
        }
    )


class FakeBuilder:
    """Reads each row as ``statuses`` says, and completes each at ``digest``."""

    def __init__(self, digest: str, events: list[str], statuses: dict[str, str] | None = None) -> None:
        self.digest = digest
        self.events = events
        self.statuses = statuses or {}

    def row(self, image: str) -> ContainerImage:
        return _row(image, status=self.statuses.get(image, "pending"))

    def complete(self, image: str, digest: str) -> ContainerImage:
        self.events.append(f"complete {image} {digest}")
        return _row(image, self.digest, "ready")


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """What the step did, in order: each crane push, signature and completion."""
    happened: list[str] = []

    def run(args: list[str], *, env: object = None) -> str:
        happened.append(" ".join(args))
        # As `crane push LAYOUT REF` does: print where it pushed, by the digest it read from the layout.
        layout, ref = Path(args[2]), args[3]
        return f"{ref.rsplit(':', 1)[0]}@{json.loads((layout / 'index.json').read_text())['manifests'][0]['digest']}"

    monkeypatch.setattr(push_module, "run_tool", run)
    return happened


def _sign(events: list[str]):
    def sign(row: ContainerImage, digest: str) -> None:
        events.append(f"sign {row.name} {digest}")

    return sign


class TestRunningATool:
    def test_its_output_is_logged_a_line_at_a_time(self, caplog: pytest.LogCaptureFixture) -> None:
        script = "printf 'one\\ntwo\\n'; printf 'oops\\nfailed\\n' >&2; exit 3"
        with caplog.at_level(logging.INFO, logger=tools.__name__), pytest.raises(RuntimeError, match="exit 3"):
            tools.run_tool(["sh", "-c", script])
        messages = [record.getMessage() for record in caplog.records]
        assert {"one", "two", "oops", "failed"} <= set(messages)


class TestCredentials:
    def _environ(self, **overrides: str) -> dict[str, str]:
        return {
            REGISTRY_USERNAME_ENV: "builder",
            REGISTRY_PASSWORD_ENV: "secret\n",
            SIGNING_KEY_ENV: PEM,
            "HOME": "/root",
        } | overrides

    def test_they_are_read_from_the_environment_and_taken_out_of_it(self) -> None:
        environ = self._environ()
        credentials = Credentials.take_from(environ, log_in=True, sign=True)
        assert credentials.login == ("builder", "secret")
        assert credentials.key is not None
        assert credentials.key.fingerprint == key_fingerprint(KEY.public_key())
        assert environ == {"HOME": "/root"}

    def test_a_pem_whose_final_newline_was_trimmed_still_loads(self) -> None:
        credentials = Credentials.take_from(self._environ(**{SIGNING_KEY_ENV: PEM.strip()}), log_in=True, sign=True)
        assert credentials.key is not None
        assert credentials.key.fingerprint == key_fingerprint(KEY.public_key())

    def test_a_missing_one_is_named(self) -> None:
        environ = self._environ()
        del environ[SIGNING_KEY_ENV]
        with pytest.raises(Refused, match=SIGNING_KEY_ENV):
            Credentials.take_from(environ, log_in=True, sign=True)

    def test_a_key_that_is_not_one_is_refused_without_repeating_it(self) -> None:
        with pytest.raises(Refused, match="not an unencrypted") as caught:
            Credentials.take_from(self._environ(**{SIGNING_KEY_ENV: "hunter2"}), log_in=True, sign=True)
        assert "hunter2" not in str(caught.value)

    def test_crane_gets_the_credential_for_the_registry_only(self) -> None:
        config = Credentials.take_from(self._environ(), log_in=True, sign=True).docker_config(REGISTRY)
        assert config == {"auths": {REGISTRY: {"auth": base64.b64encode(b"builder:secret").decode()}}}

    def test_ones_the_deployment_turned_off_are_not_needed(self) -> None:
        credentials = Credentials.take_from({"HOME": "/root"}, log_in=False, sign=False)
        assert (credentials.login, credentials.key) == (None, None)

    def test_without_a_credential_crane_pushes_anonymously(self) -> None:
        assert Credentials(None, None).docker_config(REGISTRY) == {"auths": {}}


class TestSigning:
    def test_a_deployment_that_does_not_sign_has_no_signer(self) -> None:
        config = PushStepConfig(images=[_image()], registry=REGISTRY, sign=False)
        assert signer(config, Credentials(("builder", "secret"), None), httpx.Client()) is None

    def test_without_a_credential_the_signature_is_stored_anonymously(self, monkeypatch: pytest.MonkeyPatch) -> None:
        authorizations: list[str | None] = []
        monkeypatch.setattr(push_module, "login", lambda *args, **kwargs: pytest.fail("logged in"))
        monkeypatch.setattr(
            push_module, "Registry", lambda host, authorization, **kwargs: authorizations.append(authorization)
        )
        monkeypatch.setattr(push_module, "sign_image", lambda registry, row, digest, key: None)
        config = PushStepConfig(images=[_image()], registry=REGISTRY, log_in=False)
        sign = signer(config, Credentials(None, SigningKey(PEM.encode())), httpx.Client())
        assert sign is not None
        sign(_row(), "sha256:" + "b" * 64)
        assert authorizations == [None]


class TestTheBuilderClient:
    def _builder(self, responses: list[httpx.Response]) -> tuple[Builder, list[httpx.Request], list[float]]:
        seen: list[httpx.Request] = []

        def platform(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return responses.pop(0)

        sleeps: list[float] = []
        client = BuilderClient(
            base_url="http://platform", http_client=httpx.Client(transport=httpx.MockTransport(platform))
        )
        return Builder(client, workspace="ws", sleep=sleeps.append), seen, sleeps

    def test_the_row_is_read_by_name(self) -> None:
        builder, seen, _ = self._builder([httpx.Response(200, json=_row().model_dump(mode="json"))])
        assert builder.row("demo-1-0").status == "pending"
        assert (seen[0].method, seen[0].url.path) == ("GET", f"{IMAGES}/demo-1-0")

    def test_completing_names_the_digest_this_step_pushed(self) -> None:
        digest = "sha256:" + "d" * 64
        builder, seen, _ = self._builder(
            [httpx.Response(200, json=_row(digest=digest, status="ready").model_dump(mode="json"))]
        )
        assert builder.complete("demo-1-0", digest).digest == digest
        assert seen[0].url.path == f"{IMAGES}/demo-1-0/complete"
        assert json.loads(seen[0].content) == {"digest": digest}

    def test_a_platform_that_is_briefly_down_is_retried(self) -> None:
        digest = "sha256:" + "d" * 64
        builder, _, sleeps = self._builder(
            [
                httpx.Response(503, json={"detail": "down"}),
                httpx.Response(200, json=_row(digest=digest, status="ready").model_dump(mode="json")),
            ]
        )
        builder.complete("demo-1-0", digest)
        assert sleeps == [1.0]

    def test_reading_a_row_is_retried_too(self) -> None:
        builder, seen, sleeps = self._builder(
            [httpx.Response(503, json={"detail": "down"}), httpx.Response(200, json=_row().model_dump(mode="json"))]
        )
        assert builder.row("demo-1-0").status == "pending"
        assert (len(seen), sleeps) == (2, [1.0])

    def test_a_refusal_is_not_retried(self) -> None:
        builder, _, sleeps = self._builder([httpx.Response(409, json={"detail": "already ready at another digest"})])
        with pytest.raises(NemoHTTPError):
            builder.complete("demo-1-0", "sha256:" + "d" * 64)
        assert sleeps == []

    def test_a_platform_that_stays_down_is_given_up_on(self) -> None:
        builder, _, sleeps = self._builder([httpx.Response(503, json={"detail": "down"}) for _ in range(5)])
        with pytest.raises(NemoHTTPError):
            builder.complete("demo-1-0", "sha256:" + "d" * 64)
        assert sleeps == [1.0, 2.0, 4.0, 8.0]


class TestPushOne:
    def test_pushes_the_callers_tag_then_the_system_tag_then_signs_then_completes(
        self, tmp_path: Path, events: list[str]
    ) -> None:
        digest = _layout(tmp_path / "out")
        builder = FakeBuilder(digest, events)
        _push_one(_row(), _image(), tmp_path / "out", builder=builder, sign=_sign(events), plain_http=False)
        assert events == [
            f"crane push {tmp_path / 'out'} {REGISTRY}/ws/team/app:v1",
            f"crane push {tmp_path / 'out'} {REGISTRY}/ws/team/app:ws--demo-1-0",
            f"sign demo-1-0 {digest}",
            f"complete demo-1-0 {digest}",
        ]

    def test_an_image_the_deployment_does_not_sign_is_pushed_and_completed(
        self, tmp_path: Path, events: list[str]
    ) -> None:
        digest = _layout(tmp_path / "out")
        _push_one(_row(), _image(), tmp_path / "out", builder=FakeBuilder(digest, events), sign=None, plain_http=False)
        assert events == [
            f"crane push {tmp_path / 'out'} {REGISTRY}/ws/team/app:v1",
            f"crane push {tmp_path / 'out'} {REGISTRY}/ws/team/app:ws--demo-1-0",
            f"complete demo-1-0 {digest}",
        ]

    def test_a_plain_http_registry_is_pushed_over_http_only_when_configured(
        self, tmp_path: Path, events: list[str]
    ) -> None:
        digest = _layout(tmp_path / "out")
        _push_one(
            _row(),
            _image(caller=False),
            tmp_path / "out",
            builder=FakeBuilder(digest, events),
            sign=_sign(events),
            plain_http=True,
        )
        assert events[0] == f"crane push {tmp_path / 'out'} {REGISTRY}/ws/team/app:ws--demo-1-0 --insecure"

    def test_an_image_that_could_not_be_signed_is_not_completed(self, tmp_path: Path, events: list[str]) -> None:
        digest = _layout(tmp_path / "out")

        def refuse(row: ContainerImage, digest: str) -> None:
            raise RegistryError("writing the signature: the registry answered 403")

        with pytest.raises(RegistryError):
            _push_one(
                _row(), _image(), tmp_path / "out", builder=FakeBuilder(digest, events), sign=refuse, plain_http=False
            )
        assert not any(event.startswith("complete") for event in events)

    def test_an_image_crane_read_differently_is_neither_signed_nor_completed(
        self, tmp_path: Path, events: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        digest = _layout(tmp_path / "out")
        other = "sha256:" + "b" * 64

        def run(args: list[str], *, env: object = None) -> str:
            events.append(" ".join(args))
            return f"{args[3].rsplit(':', 1)[0]}@{other}"

        monkeypatch.setattr(push_module, "run_tool", run)
        with pytest.raises(Refused, match=f"crane pushed {other}, not the checked {digest}"):
            _push_one(
                _row(),
                _image(),
                tmp_path / "out",
                builder=FakeBuilder(digest, events),
                sign=_sign(events),
                plain_http=False,
            )
        assert events == [f"crane push {tmp_path / 'out'} {REGISTRY}/ws/team/app:v1"]

    def test_an_image_completed_at_another_digest_is_a_failure(self, tmp_path: Path, events: list[str]) -> None:
        _layout(tmp_path / "out")
        with pytest.raises(Refused, match="not ready at"):
            _push_one(
                _row(),
                _image(),
                tmp_path / "out",
                builder=FakeBuilder("sha256:" + "e" * 64, events),
                sign=_sign(events),
                plain_http=False,
            )


class TestPublishAll:
    def _config(self, *names: str) -> PushStepConfig:
        return PushStepConfig(images=[_image(name) for name in names], registry=REGISTRY)

    def _work(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *built: str) -> str:
        monkeypatch.setenv(PERSISTENT_JOB_STORAGE_PATH_ENVVAR, str(tmp_path))
        digest = ""
        for name in built:
            digest = _layout(tmp_path / "out" / name)
        return digest

    def _completed(self, events: list[str]) -> list[str]:
        return [event.split()[1] for event in events if event.startswith("complete")]

    def test_every_built_image_is_published_signed_and_completed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: list[str]
    ) -> None:
        digest = self._work(tmp_path, monkeypatch, "demo-1-0", "demo-1-1")
        code = _publish_all(
            self._config("demo-1-0", "demo-1-1"), builder=FakeBuilder(digest, events), sign=_sign(events)
        )
        assert code == 0 and self._completed(events) == ["demo-1-0", "demo-1-1"]
        assert [event.split()[1] for event in events if event.startswith("sign")] == ["demo-1-0", "demo-1-1"]

    def test_a_missing_layout_fails_that_image_and_not_the_rest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: list[str]
    ) -> None:
        digest = self._work(tmp_path, monkeypatch, "demo-1-1")
        code = _publish_all(
            self._config("demo-1-0", "demo-1-1"), builder=FakeBuilder(digest, events), sign=_sign(events)
        )
        assert code == 1 and self._completed(events) == ["demo-1-1"]

    def test_a_rerun_skips_what_already_settled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: list[str]
    ) -> None:
        digest = self._work(tmp_path, monkeypatch, "demo-1-0", "demo-1-1")
        builder = FakeBuilder(digest, events, {"demo-1-0": "ready"})
        code = _publish_all(self._config("demo-1-0", "demo-1-1"), builder=builder, sign=_sign(events))
        assert code == 0 and self._completed(events) == ["demo-1-1"]
        assert not any("demo-1-0" in event for event in events)

    def test_a_row_that_already_failed_counts_as_a_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: list[str]
    ) -> None:
        digest = self._work(tmp_path, monkeypatch, "demo-1-0")
        builder = FakeBuilder(digest, events, {"demo-1-0": "failed"})
        assert _publish_all(self._config("demo-1-0"), builder=builder, sign=_sign(events)) == 1
