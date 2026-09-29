# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The broker process: who a credential identifies, what it is handed, and what gets signed.

The API server is represented by what ``TokenReview`` and a pod read return, the platform by the
rows it lists, and the registry by a token client and a signer -- so each test states exactly what
each said. The point of most of them is a request that must NOT be served: a raw job with no rows,
a step asking for another job's image, a pod Jobs did not make.
"""

from __future__ import annotations

import base64
import http.client
import json
import socket
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import cast

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from nemo_builder_plugin.broker import (
    JOB_LABEL,
    MANAGED_BY_JOBS,
    MANAGED_BY_LABEL,
    PROFILE_LABEL,
    STEP_LABEL,
    WORKSPACE_LABEL,
    NotIdentified,
    Pod,
    StepIdentity,
)
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.entities import ContainerImage, Provenance
from nemo_builder_plugin.execution import ExecutionBackend
from nemo_builder_plugin.registry_auth import IssuedToken, RegistryAuthError
from nemo_builder_plugin.run import broker as broker_module
from nemo_builder_plugin.run.broker import (
    POD_NAME_EXTRA,
    POD_UID_EXTRA,
    BoundedServer,
    Broker,
    BrokerConfig,
    BrokerUnavailable,
    BuilderRouteRows,
    CosignSigner,
    Review,
    SignedImage,
    SigningFailed,
    TokenExchange,
    TokenExchangeConfig,
    _handler,
    cosign_sign_args,
)
from nemo_builder_plugin.signing import signature_annotations

REGISTRY = "reg.example.com"
PUSH_SA = "system:serviceaccount:nhx-builds:nhx-build-push"
DIGEST = "sha256:" + "d" * 64
PUBLIC_KEY = (
    ec.generate_private_key(ec.SECP256R1())
    .public_key()
    .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    .decode()
)
BUILDER = BuilderConfig(registry=REGISTRY, credential_broker="http://broker:8080", signing_public_key=PUBLIC_KEY)
CONFIG = BrokerConfig(
    platform_url="http://platform",
    registry_username="nhx-build-broker",
    registry_password_file="/unused",
    signing_key="/etc/nhx-build-broker/cosign/cosign.key",
)


def _basic(token: str) -> str:
    return "Basic " + base64.b64encode(f"nhx-build-push-pod-token:{token}".encode()).decode()


def _pod(**labels: str) -> Pod:
    base = {
        MANAGED_BY_LABEL: MANAGED_BY_JOBS,
        WORKSPACE_LABEL: "ws-a",
        JOB_LABEL: "build-1",
        STEP_LABEL: "push",
        PROFILE_LABEL: "build-push",
    }
    return Pod(
        name="build-1-push-x", uid="uid-1", service_account="nhx-build-push", phase="Running", labels=base | labels
    )


def _row(name: str = "build-1-0", *, job: str = "ws-a/build-1", repository: str = "ws-a/app") -> ContainerImage:
    return ContainerImage(
        name=name,
        workspace=job.split("/")[0],
        registry=REGISTRY,
        repository=repository,
        provenance=Provenance(
            build_set="build",
            revision=1,
            job=job,
            system_tag=f"{job.split('/')[0]}--{name}",
            request_digest="sha256:" + "a" * 64,
        ),
    )


class FakeReviewer:
    def __init__(self, reviews: dict[str, Review] | None = None) -> None:
        self.reviews = (
            reviews
            if reviews is not None
            else {
                "push-token": Review(
                    authenticated=True,
                    username=PUSH_SA,
                    extra={POD_NAME_EXTRA: ["build-1-push-x"], POD_UID_EXTRA: ["uid-1"]},
                )
            }
        )
        self.audiences: list[list[str] | None] = []
        self.fail = False

    def review(self, token: str, audiences: list[str] | None) -> Review:
        self.audiences.append(audiences)
        if self.fail:
            raise BrokerUnavailable("api server unreachable")
        return self.reviews.get(token, Review(authenticated=False))


class FakePods:
    def __init__(self, pods: dict[str, Pod] | None = None) -> None:
        self.pods = pods if pods is not None else {"build-1-push-x": _pod()}

    def read(self, namespace: str, name: str) -> Pod | None:
        assert namespace == "nhx-builds"
        return self.pods.get(name)


class FakeRows:
    """What the builder's list route would return for the job -- which may include strays."""

    def __init__(self, rows: list[ContainerImage] | None = None) -> None:
        self.rows = rows if rows is not None else [_row()]
        self.fail: Exception | None = None
        self.tokens: list[str] = []

    def pending(self, identity: StepIdentity, token: str) -> list[ContainerImage]:
        self.tokens.append(token)
        if self.fail:
            raise self.fail
        return [row for row in self.rows if row.status == "pending"]

    def get(self, identity: StepIdentity, token: str, name: str) -> ContainerImage | None:
        if self.fail:
            raise self.fail
        return next((row for row in self.rows if row.name == name and row.workspace == identity.workspace), None)


class FakeRegistry:
    def __init__(self) -> None:
        self.asked: list[tuple[list[str], list[str]]] = []
        self.fail = False

    def issue(self, repositories: Sequence[str], actions: Sequence[str]) -> IssuedToken:
        self.asked.append((list(repositories), list(actions)))
        if self.fail:
            raise RegistryAuthError("token service down")
        return IssuedToken(token="registry-token", expires_in=300, narrowed=True)


class FakeSigner:
    """Resolves every system tag to ``DIGEST``, then records it, then signs -- in that order."""

    def __init__(self) -> None:
        self.signed: list[tuple[str, dict[str, str]]] = []
        self.fail = False

    def sign(
        self, row: ContainerImage, annotations: Mapping[str, str], *, record: Callable[[str], None]
    ) -> SignedImage:
        if self.fail:
            raise SigningFailed("registry unreachable")
        record(DIGEST)
        self.signed.append((row.name, dict(annotations)))
        return SignedImage(digest=DIGEST, payload=b'{"payload":1}', signature=b"sig")


class World:
    """A broker and everything it talks to."""

    def __init__(self, *, rows: list[ContainerImage] | None = None, audience: str | None = None) -> None:
        self.reviewer = FakeReviewer()
        self.pods = FakePods()
        self.rows = FakeRows(rows)
        self.registry = FakeRegistry()
        self.signer = FakeSigner()
        self.audit: list[Mapping[str, object]] = []
        self.audit_fails = False
        config = CONFIG.model_copy(update={"step_token_audience": audience})
        self.broker = Broker(
            BUILDER,
            config,
            backend=ExecutionBackend(BUILDER),
            reviewer=self.reviewer,
            pods=self.pods,
            rows=self.rows,
            registry=self.registry,
            signer=self.signer,
            audit=self._audit,
        )

    def _audit(self, record: Mapping[str, object]) -> None:
        if self.audit_fails:
            raise OSError("stdout closed")
        self.audit.append(record)


class TestIdentity:
    def test_a_push_steps_token_identifies_its_job(self) -> None:
        assert World().broker.identify("push-token") == StepIdentity(
            workspace="ws-a", job="build-1", pod="build-1-push-x", role="push"
        )

    def test_the_token_is_reviewed_for_the_configured_audience(self) -> None:
        """With platform auth on, the step presents the workload token, bound to another audience."""
        world = World(audience="nhx-workload")
        world.broker.identify("push-token")
        assert world.reviewer.audiences == [["nhx-workload"]]
        world = World()
        world.broker.identify("push-token")
        assert world.reviewer.audiences == [None]

    def test_a_token_the_api_server_rejects_identifies_nothing(self) -> None:
        with pytest.raises(NotIdentified, match="does not accept"):
            World().broker.identify("forged")

    def test_a_service_account_outside_the_build_namespace_identifies_nothing(self) -> None:
        world = World()
        world.reviewer.reviews["other"] = Review(
            authenticated=True,
            username="system:serviceaccount:nhx-platform:nemo-helix",
            extra={POD_NAME_EXTRA: ["p"], POD_UID_EXTRA: ["u"]},
        )
        with pytest.raises(NotIdentified, match="build namespace"):
            world.broker.identify("other")

    def test_a_token_bound_to_no_pod_identifies_nothing(self) -> None:
        world = World()
        world.reviewer.reviews["legacy"] = Review(authenticated=True, username=PUSH_SA)
        with pytest.raises(NotIdentified, match="not bound to a pod"):
            world.broker.identify("legacy")

    def test_a_token_whose_pod_is_gone_identifies_nothing(self) -> None:
        world = World()
        world.pods.pods.clear()
        with pytest.raises(NotIdentified, match="no longer exists"):
            world.broker.identify("push-token")


class TestCredentials:
    def test_the_step_gets_a_token_for_exactly_its_jobs_destinations(self) -> None:
        world = World(rows=[_row("build-1-0", repository="ws-a/app"), _row("build-1-1", repository="ws-a/tools")])
        reply = world.broker.credentials(authorization=_basic("push-token"))
        assert reply.status == 200
        assert reply.body["auths"] == {REGISTRY: {"registrytoken": "registry-token"}}
        assert reply.body["repositories"] == ["ws-a/app", "ws-a/tools"]
        assert world.registry.asked == [(["ws-a/app", "ws-a/tools"], ["pull", "push"])]
        assert world.audit[-1]["decision"] == "credentials" and world.audit[-1]["job"] == "ws-a/build-1"

    def test_rows_are_read_with_the_steps_own_token(self) -> None:
        """So with auth on they are read as the submitter, and the broker has no identity of its own."""
        world = World()
        world.broker.credentials(authorization=_basic("push-token"))
        assert world.rows.tokens == ["push-token"]

    def test_a_raw_job_with_no_rows_is_handed_nothing(self) -> None:
        """The attack the broker exists for: any Editor can run a job as the push step's identity."""
        world = World(rows=[])
        reply = world.broker.credentials(authorization=_basic("push-token"))
        assert reply.status == 403
        assert world.registry.asked == []
        assert world.audit[-1]["decision"] == "credentials-refused"

    def test_stray_rows_from_elsewhere_are_never_granted(self) -> None:
        world = World(rows=[_row(job="ws-a/build-2"), _row(repository="ws-b/app")])
        assert world.broker.credentials(authorization=_basic("push-token")).status == 403
        assert world.registry.asked == []

    @pytest.mark.parametrize("authorization", [None, "Bearer push-token", "Basic !!!", _basic("forged")])
    def test_a_credential_that_identifies_no_step_is_a_401(self, authorization: str | None) -> None:
        world = World()
        reply = world.broker.credentials(authorization=authorization)
        assert reply.status == 401 and "WWW-Authenticate" in reply.headers
        assert world.registry.asked == []

    def test_a_step_of_another_role_is_a_401(self) -> None:
        world = World()
        world.pods.pods["build-1-push-x"] = _pod(**{STEP_LABEL: "fetch", PROFILE_LABEL: "build-fetch"})
        assert world.broker.credentials(authorization=_basic("push-token")).status == 401

    def test_the_platform_refusing_the_exchange_is_a_401(self) -> None:
        """With auth on, the platform exchanges only while the pod's delegation is live."""
        world = World()
        world.rows.fail = NotIdentified("delegation is not live")
        assert world.broker.credentials(authorization=_basic("push-token")).status == 401

    def test_unreadable_rows_are_a_503_and_nothing_is_issued(self) -> None:
        world = World()
        world.rows.fail = BrokerUnavailable("platform down")
        assert world.broker.credentials(authorization=_basic("push-token")).status == 503
        assert world.registry.asked == []

    def test_an_unreachable_api_server_is_a_503(self) -> None:
        world = World()
        world.reviewer.fail = True
        assert world.broker.credentials(authorization=_basic("push-token")).status == 503

    def test_a_token_service_failure_is_a_502(self) -> None:
        world = World()
        world.registry.fail = True
        assert world.broker.credentials(authorization=_basic("push-token")).status == 502

    def test_no_audit_line_means_no_token(self) -> None:
        world = World()
        world.audit_fails = True
        reply = world.broker.credentials(authorization=_basic("push-token"))
        assert reply.status == 503 and "auths" not in reply.body


class TestSign:
    def _sign(self, world: World, image: str = "build-1-0") -> broker_module.Reply:
        return world.broker.sign(authorization=_basic("push-token"), body=json.dumps({"image": image}).encode())

    def test_the_broker_signs_the_steps_own_row_with_the_rows_annotations(self) -> None:
        world = World()
        reply = self._sign(world)
        assert reply.status == 200
        assert reply.body["digest"] == DIGEST
        assert reply.body["reference"] == f"{REGISTRY}/ws-a/app@{DIGEST}"
        signed = cast(dict[str, str], reply.body["signed"])
        assert base64.b64decode(signed["payload"]) == b'{"payload":1}'
        assert base64.b64decode(signed["signature"]) == b"sig"
        assert world.signer.signed == [("build-1-0", signature_annotations(_row()))]

    def test_the_audit_names_the_digest_before_signing_and_the_outcome_after(self) -> None:
        world = World()
        self._sign(world)
        assert [(line["decision"], line["reference"]) for line in world.audit] == [
            ("sign", f"{REGISTRY}/ws-a/app@{DIGEST}"),
            ("signed", f"{REGISTRY}/ws-a/app@{DIGEST}"),
        ]

    def test_another_jobs_image_is_refused(self) -> None:
        world = World(rows=[_row(), _row("build-2-0", job="ws-a/build-2")])
        reply = self._sign(world, "build-2-0")
        assert reply.status == 403 and world.signer.signed == []

    def test_an_image_that_does_not_exist_is_refused(self) -> None:
        assert self._sign(World(), "nope").status == 403

    def test_a_ready_row_is_not_signed_again(self) -> None:
        row = _row()
        row.status = "ready"
        assert self._sign(World(rows=[row])).status == 403

    @pytest.mark.parametrize("body", [b"", b"[]", b'{"image": 3}', b"{"])
    def test_a_body_that_names_no_image_is_a_400(self, body: bytes) -> None:
        assert World().broker.sign(authorization=_basic("push-token"), body=body).status == 400

    def test_a_signing_failure_is_a_502_and_audited_as_one(self) -> None:
        world = World()
        world.signer.fail = True
        assert self._sign(world).status == 502
        assert [line["decision"] for line in world.audit] == ["sign-failed"]

    def test_no_audit_line_means_no_signature(self) -> None:
        world = World()
        world.audit_fails = True
        reply = self._sign(world)
        assert reply.status == 503 and "signed" not in reply.body
        assert world.signer.signed == []


def _serve(world: World, *, max_requests: int = 8, timeout: float | None = None) -> tuple[BoundedServer, str]:
    handler = _handler(world.broker)
    if timeout is not None:
        handler.timeout = timeout
    server = BoundedServer(("127.0.0.1", 0), handler, max_requests=max_requests)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _post_raw(url: str, path: str, headers: Mapping[str, str]) -> http.client.HTTPResponse:
    """A POST with exactly these headers, which no well-behaved client would send."""
    host, port = url.removeprefix("http://").split(":")
    connection = http.client.HTTPConnection(host, int(port), timeout=5)
    connection.putrequest("POST", path, skip_accept_encoding=True)
    for name, value in headers.items():
        connection.putheader(name, value)
    connection.endheaders()
    return connection.getresponse()


def _closed_by_server(sock: socket.socket, *, within: float) -> bool:
    sock.settimeout(within)
    try:
        return sock.recv(1) == b""
    except ConnectionResetError:
        return True
    except TimeoutError:
        return False


class TestOverHttp:
    @pytest.fixture
    def url(self) -> Iterator[str]:
        server, url = _serve(World())
        yield url
        server.shutdown()

    def test_health(self, url: str) -> None:
        assert httpx.get(f"{url}/healthz").json() == {"status": "ok"}

    def test_credentials_and_sign(self, url: str) -> None:
        auth = ("nhx-build-push-pod-token", "push-token")
        assert httpx.post(f"{url}/credentials", auth=auth).status_code == 200
        signed = httpx.post(f"{url}/sign", auth=auth, json={"image": "build-1-0"})
        assert signed.status_code == 200 and signed.headers["Cache-Control"] == "no-store"

    def test_unknown_paths_and_oversized_bodies(self, url: str) -> None:
        assert httpx.post(f"{url}/token").status_code == 404
        assert httpx.post(f"{url}/sign", content=b"x" * 5000).status_code == 413

    @pytest.mark.parametrize("length", ["-1", "abc", "1e3", "\u00b2"])
    def test_a_length_that_is_not_a_non_negative_integer_is_refused_unread(self, url: str, length: str) -> None:
        """-1 once passed the size check and read to the end of the stream, before any authentication."""
        assert _post_raw(url, "/sign", {"Content-Length": length}).status == 400

    def test_anything_the_broker_did_not_expect_is_still_an_answer(self) -> None:
        world = World()
        world.rows.fail = RuntimeError("a bug")
        server, url = _serve(world)
        try:
            reply = httpx.post(f"{url}/credentials", auth=("nhx-build-push-pod-token", "push-token"))
            assert reply.status_code == 500 and world.registry.asked == []
        finally:
            server.shutdown()

    def test_a_client_that_goes_quiet_is_dropped(self) -> None:
        server, url = _serve(World(), timeout=0.2)
        try:
            host, port = url.removeprefix("http://").split(":")
            with socket.create_connection((host, int(port))) as quiet:
                quiet.sendall(b"POST /sign HTTP/1.0\r\nContent-Length: 100\r\n\r\n")
                assert _closed_by_server(quiet, within=5)
        finally:
            server.shutdown()

    def test_connections_beyond_the_bound_are_closed_and_the_slot_comes_back(self) -> None:
        server, url = _serve(World(), max_requests=1, timeout=2.0)
        try:
            host, port = url.removeprefix("http://").split(":")
            with socket.create_connection((host, int(port))) as holder:
                time.sleep(0.1)  # accepted, and holding the one slot while it sends nothing
                with socket.create_connection((host, int(port))) as refused:
                    # Well inside the holder's timeout: served, it would be kept open that long.
                    assert _closed_by_server(refused, within=0.5)
                assert _closed_by_server(holder, within=5)
            assert httpx.get(f"{url}/healthz").status_code == 200
        finally:
            server.shutdown()


class TestBuilderRouteRows:
    IDENTITY = StepIdentity(workspace="ws-a", job="build-1", pod="p", role="push")

    def _rows(self, handler: httpx.MockTransport, exchange: Callable[[str], str] | None = None) -> BuilderRouteRows:
        return BuilderRouteRows("http://platform", exchange=exchange, http=httpx.Client(transport=handler))

    def test_a_jobs_pending_rows_are_read_with_the_job_and_status_filters(self) -> None:
        seen: list[httpx.Request] = []

        def platform(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=[json.loads(_row().model_dump_json())])

        rows = self._rows(httpx.MockTransport(platform)).pending(self.IDENTITY, "push-token")
        assert [row.name for row in rows] == ["build-1-0"]
        assert seen[0].url.path == "/apis/builder/v2/workspaces/ws-a/container-images"
        assert (seen[0].url.params["job"], seen[0].url.params["status"]) == ("build-1", "pending")
        assert "Authorization" not in seen[0].headers

    def test_with_an_exchange_rows_are_read_as_the_submitter(self) -> None:
        seen: list[httpx.Request] = []

        def platform(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=[])

        exchanged: list[str] = []

        def exchange(token: str) -> str:
            exchanged.append(token)
            return "submitter-token"

        BuilderRouteRows(
            "http://platform", exchange=exchange, http=httpx.Client(transport=httpx.MockTransport(platform))
        ).pending(self.IDENTITY, "push-token")
        assert exchanged == ["push-token"]
        assert seen[0].headers["Authorization"] == "Bearer submitter-token"

    def test_a_missing_row_is_none_and_an_unreachable_platform_is_unavailable(self) -> None:
        rows = self._rows(httpx.MockTransport(lambda request: httpx.Response(404, json={"detail": "no"})))
        assert rows.get(self.IDENTITY, "t", "nope") is None
        down = self._rows(httpx.MockTransport(lambda request: httpx.Response(503, json={"detail": "down"})))
        with pytest.raises(BrokerUnavailable):
            down.pending(self.IDENTITY, "t")

    @pytest.mark.parametrize("failure", [NotIdentified("not live"), BrokerUnavailable("down")])
    def test_what_the_exchange_says_is_what_a_read_says(self, failure: Exception) -> None:
        def exchange(token: str) -> str:
            raise failure

        rows = self._rows(httpx.MockTransport(lambda request: httpx.Response(200, json=[])), exchange)
        for read in (lambda: rows.pending(self.IDENTITY, "t"), lambda: rows.get(self.IDENTITY, "t", "x")):
            with pytest.raises(type(failure)):
                read()


class TestTokenExchange:
    EXCHANGE = TokenExchangeConfig(token_endpoint="https://auth/token", client_id="c")

    def test_a_refused_exchange_identifies_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def refuse(**kwargs: object) -> dict[str, object]:
            raise broker_module.WorkloadTokenExchangeError(
                "Workload token exchange failed: invalid_grant - delegation is not live"
            )

        monkeypatch.setattr(broker_module, "token_exchange_grant", refuse)
        with pytest.raises(NotIdentified, match="would not exchange"):
            TokenExchange(self.EXCHANGE)("push-token")

    @pytest.mark.parametrize(
        ("status", "body", "raised"),
        [
            (400, {"error": "invalid_grant", "error_description": "not live"}, NotIdentified),
            (401, {"error": "invalid_client"}, NotIdentified),
            (503, None, BrokerUnavailable),
            (500, {"error": "server_error"}, BrokerUnavailable),
            (200, {"token_type": "Bearer"}, BrokerUnavailable),
        ],
    )
    def test_the_platform_failing_is_unavailable_and_only_a_refusal_identifies_nothing(
        self, monkeypatch: pytest.MonkeyPatch, status: int, body: dict[str, str] | None, raised: type[Exception]
    ) -> None:
        """Through the platform's own ``token_exchange_grant``, which is what the error code is read from."""

        def post(url: str, **kwargs: object) -> httpx.Response:
            request = httpx.Request("POST", url)
            if body is None:
                return httpx.Response(status, text="upstream unavailable", request=request)
            return httpx.Response(status, json=body, request=request)

        monkeypatch.setattr(httpx, "post", post)
        with pytest.raises(raised):
            TokenExchange(self.EXCHANGE)("push-token")

    @pytest.mark.parametrize("endpoint", ["https://auth.example.com/token", "http://localhost:8080/token"])
    def test_an_endpoint_is_https_or_loopback(self, endpoint: str) -> None:
        assert TokenExchangeConfig(token_endpoint=endpoint, client_id="c").token_endpoint == endpoint

    @pytest.mark.parametrize("endpoint", ["http://nemo-helix.nhx-platform.svc:8080/token", "ftp://auth/token"])
    def test_any_other_endpoint_is_refused_at_start(self, endpoint: str) -> None:
        with pytest.raises(ValueError, match="HTTPS"):
            TokenExchangeConfig(token_endpoint=endpoint, client_id="c")

    def test_an_exchange_returns_the_access_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        asked: dict[str, object] = {}

        def grant(**kwargs: object) -> dict[str, object]:
            asked.update(kwargs)
            return {"access_token": "submitter-token", "expires_in": 60}

        monkeypatch.setattr(broker_module, "token_exchange_grant", grant)
        exchange = TokenExchange(TokenExchangeConfig(token_endpoint="https://auth/token", client_id="c", audience="a"))
        assert exchange("push-token") == "submitter-token"
        assert (asked["subject_token"], asked["audience"]) == ("push-token", "a")


class TestKubernetesPods:
    def test_an_api_server_that_cannot_be_reached_is_unavailable_not_a_crash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """urllib3's errors are not ApiExceptions, and once escaped the handler with no answer."""
        from urllib3.exceptions import MaxRetryError

        pods = broker_module.KubernetesPods()

        def unreachable(**kwargs: object) -> object:
            raise MaxRetryError(None, "/api/v1/namespaces/nhx-builds/pods/p", "connection refused")  # ty: ignore[invalid-argument-type]

        monkeypatch.setattr(pods._api, "read_namespaced_pod", unreachable)
        with pytest.raises(BrokerUnavailable, match="MaxRetryError"):
            pods.read("nhx-builds", "p")


class TestCosign:
    def test_it_signs_by_digest_only_with_the_tlog_off_and_the_signature_written_out(self) -> None:
        args = cosign_sign_args(
            f"{REGISTRY}/ws-a/app@{DIGEST}",
            key="/k",
            annotations={"b": "2", "a": "1"},
            output_payload=Path("/tmp/p"),
            output_signature=Path("/tmp/s"),
        )
        assert args[:2] == ["cosign", "sign"] and args[-1] == f"{REGISTRY}/ws-a/app@{DIGEST}"
        assert "--tlog-upload=false" in args
        assert "--output-payload=/tmp/p" in args and "--output-signature=/tmp/s" in args
        assert [a for a in args if a.startswith("--annotations")] == ["--annotations=a=1", "--annotations=b=2"]
        assert not [a for a in args if a.startswith("--allow-")]

    def test_a_plain_http_registry_is_allowed_by_the_flag_for_http(self) -> None:
        """``--allow-insecure-registry`` only skips TLS verification; plain HTTP is its own flag."""
        args = cosign_sign_args(
            f"{REGISTRY}/a@{DIGEST}",
            key="/k",
            annotations={},
            output_payload=Path("p"),
            output_signature=Path("s"),
            plain_http=True,
        )
        assert [a for a in args if a.startswith("--allow-")] == ["--allow-http-registry"]

    def test_a_tag_is_never_signed(self) -> None:
        with pytest.raises(ValueError, match="by digest"):
            cosign_sign_args(
                f"{REGISTRY}/ws-a/app:v1",
                key="/k",
                annotations={},
                output_payload=Path("p"),
                output_signature=Path("s"),
            )

    def test_an_annotation_cosign_would_split_is_refused(self) -> None:
        with pytest.raises(ValueError, match="intact"):
            cosign_sign_args(
                f"{REGISTRY}/a@{DIGEST}",
                key="/k",
                annotations={"a": "1,b=2"},
                output_payload=Path("p"),
                output_signature=Path("s"),
            )

    def _signer(
        self, monkeypatch: pytest.MonkeyPatch, manifest: dict[str, object], config: BrokerConfig = CONFIG
    ) -> tuple[CosignSigner, list[list[str]]]:
        calls: list[list[str]] = []

        def run(args: list[str], *, env: Mapping[str, str] | None = None) -> str:
            calls.append(args)
            assert env is not None
            if args[0] == "cosign":
                calls.append(["env", env["COSIGN_PASSWORD"]])
            config = json.loads((Path(env["DOCKER_CONFIG"]) / "config.json").read_text())
            assert config == {"auths": {REGISTRY: {"registrytoken": "registry-token"}}}
            if args[1] == "digest":
                return DIGEST
            if args[1] == "manifest":
                return json.dumps(manifest)
            payload = next(a.split("=", 1)[1] for a in args if a.startswith("--output-payload="))
            signature = next(a.split("=", 1)[1] for a in args if a.startswith("--output-signature="))
            Path(payload).write_bytes(b'{"critical":{}}')
            Path(signature).write_text(base64.b64encode(b"der-signature").decode())
            return ""

        monkeypatch.setattr(broker_module, "run_tool", run)
        return CosignSigner(BUILDER, config, FakeRegistry()), calls

    MANIFEST: dict[str, object] = {"mediaType": "application/vnd.oci.image.manifest.v1+json"}

    def test_the_signer_resolves_the_system_tag_itself_records_it_then_signs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        signer, calls = self._signer(monkeypatch, self.MANIFEST)
        signed = signer.sign(
            _row(), signature_annotations(_row()), record=lambda digest: calls.append(["record", digest])
        )
        assert calls[0] == ["crane", "digest", f"{REGISTRY}/ws-a/app:ws-a--build-1-0"]
        assert [call[0] for call in calls] == ["crane", "crane", "record", "cosign", "env"]
        assert calls[2] == ["record", DIGEST] and calls[3][-1] == f"{REGISTRY}/ws-a/app@{DIGEST}"
        assert (signed.digest, signed.payload, signed.signature) == (DIGEST, b'{"critical":{}}', b"der-signature")

    def test_nothing_is_signed_if_it_cannot_be_recorded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        signer, calls = self._signer(monkeypatch, self.MANIFEST)

        def record(digest: str) -> None:
            raise broker_module.AuditUnavailable("stdout closed")

        with pytest.raises(broker_module.AuditUnavailable):
            signer.sign(_row(), signature_annotations(_row()), record=record)
        assert "cosign" not in [call[0] for call in calls]

    def test_the_key_password_is_read_without_its_trailing_newline(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        password = tmp_path / "password"
        password.write_text("hunter2\n")
        config = CONFIG.model_copy(update={"signing_key_password_file": str(password)})
        signer, calls = self._signer(monkeypatch, self.MANIFEST, config)
        signer.sign(_row(), signature_annotations(_row()), record=lambda digest: None)
        assert ["env", "hunter2"] in calls

    def test_an_index_is_not_signed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        signer, calls = self._signer(
            monkeypatch, {"mediaType": "application/vnd.oci.image.index.v1+json", "manifests": []}
        )
        recorded: list[str] = []
        with pytest.raises(SigningFailed, match="index"):
            signer.sign(_row(), signature_annotations(_row()), record=recorded.append)
        assert [call[1] for call in calls] == ["digest", "manifest"] and recorded == []
