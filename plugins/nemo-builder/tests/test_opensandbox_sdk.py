# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The OpenSandbox SDK calls behind the provider, against a stand-in for the SDK: only the step image installs it."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from datetime import timedelta
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from nemo_builder_plugin import run as run_package
from nemo_builder_plugin.run.sandbox import Mount
from nemo_builder_plugin.steps import OpenSandboxServer
from nhx_sandbox.egress import EgressPolicy, EgressRule

MODULE = "nemo_builder_plugin.run.opensandbox_sdk"

SERVER = OpenSandboxServer(domain="opensandbox-server.opensandbox-system.svc.cluster.local")


class _Model(SimpleNamespace):
    """Any of the SDK's models: what it was made with."""

    @classmethod
    def model_validate(cls, data: dict[str, Any]) -> _Model:
        return cls(**data)


class Unset:
    """The SDK's marker for a field the response left out."""


class PolicyStatusResponse(SimpleNamespace):
    """What the egress sidecar answers on `GET /policy`."""


class FakeSdkSandbox:
    def __init__(self, server: FakeServer) -> None:
        self.id = "sbx-1"
        self.commands = SimpleNamespace(run=self._run)
        self._egress_service = SimpleNamespace(_client="the sidecar's client")
        self.ran: list[tuple[str, Any, Any]] = []
        self.killed = self.closed = False
        self._server = server

    def _run(self, command: str, *, opts: Any, handlers: Any) -> SimpleNamespace:
        self.ran.append((command, opts, handlers))
        handlers.on_stdout(SimpleNamespace(text="out\n"))
        handlers.on_stderr(SimpleNamespace(text="err\n"))
        return SimpleNamespace(exit_code=3)

    def kill(self) -> None:
        self.killed = True
        if self._server.kill_fails:
            raise ConnectionError("the server is gone")

    def close(self) -> None:
        self.closed = True


class FakeServer:
    """One tenant's sandboxes, as the SDK sees them."""

    def __init__(self) -> None:
        self.pages: list[list[SimpleNamespace]] = [[]]
        self.created: list[dict[str, Any]] = []
        self.killed: list[str] = []
        self.connections: list[Any] = []
        self.managers_closed = 0
        self.create_fails = self.kill_fails = False
        self.sandbox = FakeSdkSandbox(self)
        self.policy_reads: list[Any] = []
        self.policy = SimpleNamespace(
            status_code=200,
            parsed=PolicyStatusResponse(
                enforcement_mode="dns+nft",
                policy=SimpleNamespace(to_dict=lambda: {"defaultAction": "deny", "egress": [{"action": "allow"}]}),
            ),
        )


@pytest.fixture
def server() -> FakeServer:
    return FakeServer()


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch, server: FakeServer) -> Iterator[ModuleType]:
    """``opensandbox_sdk``, imported against the stand-in."""

    class Manager:
        @staticmethod
        def create(*, connection_config: Any) -> Manager:
            server.connections.append(connection_config)
            return Manager()

        def list_sandbox_infos(self, query: Any) -> SimpleNamespace:
            return SimpleNamespace(
                sandbox_infos=server.pages[query.page - 1],
                pagination=SimpleNamespace(has_next_page=query.page < len(server.pages)),
            )

        def kill_sandbox(self, sandbox_id: str) -> None:
            server.killed.append(sandbox_id)

        def close(self) -> None:
            server.managers_closed += 1

    class Sandbox:
        @staticmethod
        def create(image: str, **kwargs: Any) -> FakeSdkSandbox:
            server.created.append({"image": image, **kwargs})
            if server.create_fails:
                # What a create that timed out leaves: a sandbox carrying the attempt's label.
                server.pages[0].append(SimpleNamespace(id="sbx-left", metadata=kwargs["metadata"]))
                raise TimeoutError("the sandbox never became ready")
            return server.sandbox

    def read_policy(*, client: Any) -> SimpleNamespace:
        server.policy_reads.append(client)
        return server.policy

    def module(name: str, **attributes: Any) -> None:
        stand_in = ModuleType(name)
        vars(stand_in).update(attributes)
        monkeypatch.setitem(sys.modules, name, stand_in)

    for package in ("opensandbox", "opensandbox.models", "opensandbox.api", "opensandbox.api.egress"):
        module(package)
    module("opensandbox.api.egress.api.policy", get_policy=SimpleNamespace(sync_detailed=read_policy))
    module("opensandbox.api.egress.models.policy_status_response", PolicyStatusResponse=PolicyStatusResponse)
    module("opensandbox.api.egress.types", Unset=Unset)
    module("opensandbox.config", ConnectionConfigSync=_Model)
    module("opensandbox.models.execd", RunCommandOpts=_Model)
    module("opensandbox.models.execd_sync", ExecutionHandlersSync=_Model)
    module("opensandbox.models.sandboxes", PVC=_Model, NetworkPolicy=_Model, SandboxFilter=_Model, Volume=_Model)
    module("opensandbox.sync", SandboxManagerSync=Manager, SandboxSync=Sandbox)
    monkeypatch.delitem(sys.modules, MODULE, raising=False)
    yield importlib.import_module(MODULE)
    # Not left bound to this test's stand-in.
    sys.modules.pop(MODULE, None)
    vars(run_package).pop("opensandbox_sdk", None)


def _create(sdk: ModuleType, **overrides: Any) -> Any:
    arguments: dict[str, Any] = {
        "image": "nhx-kaniko:test",
        "entrypoint": ["/busybox/sh"],
        "env": {"TMPDIR": "/opt/opensandbox"},
        "claim": "nhx-build-work",
        "mounts": [
            Mount("jobs/abc/context/fs", "/nhx-work/context/fs", True),
            Mount("jobs/abc/out/a", "/out/a", False),
        ],
        "labels": {"nhx.nvidia.com/build-job": "job-1"},
        "cpu": "2",
        "memory": "8Gi",
        "egress": EgressPolicy(rules=(EgressRule(target="*.com"), EgressRule(target="10.0.0.0/8", action="deny"))),
        "ttl_seconds": 3900,
        "ready_timeout_seconds": 300,
        **overrides,
    }
    return sdk.SdkApi(SERVER, "tenant-key").create(**arguments)


class TestTheConnection:
    def test_it_reaches_sandboxes_only_through_the_server_with_the_key(
        self, sdk: ModuleType, server: FakeServer
    ) -> None:
        """Directly, execd would get the key; through the server, the server checks and strips it."""
        sdk.SdkApi(SERVER, "tenant-key").labelled({})
        connection = server.connections[0]
        assert (connection.domain, connection.protocol) == (SERVER.domain, "http")
        assert connection.use_server_proxy is True
        assert connection.api_key == "tenant-key"
        assert connection.headers == {"OPEN-SANDBOX-API-KEY": "tenant-key"}


class TestCreate:
    def test_it_asks_for_the_providers_sandbox(self, sdk: ModuleType, server: FakeServer) -> None:
        _create(sdk)
        created = server.created[0]
        assert (created["image"], created["env"]) == ("nhx-kaniko:test", {"TMPDIR": "/opt/opensandbox"})
        assert created["resource"] == {"cpu": "2", "memory": "8Gi"}
        assert (created["timeout"], created["ready_timeout"]) == (timedelta(seconds=3900), timedelta(seconds=300))
        assert created["metadata"]["nhx.nvidia.com/build-job"] == "job-1"

    def test_each_mount_is_a_subpath_of_the_work_volume(self, sdk: ModuleType, server: FakeServer) -> None:
        _create(sdk)
        volumes = server.created[0]["volumes"]
        assert [(v.pvc.claim_name, v.pvc.create_if_not_exists) for v in volumes] == [("nhx-build-work", False)] * 2
        assert [(v.sub_path, v.mount_path, v.read_only) for v in volumes] == [
            ("jobs/abc/context/fs", "/nhx-work/context/fs", True),
            ("jobs/abc/out/a", "/out/a", False),
        ]

    def test_it_sets_the_egress_policy(self, sdk: ModuleType, server: FakeServer) -> None:
        _create(sdk)
        policy = server.created[0]["network_policy"]
        assert policy.defaultAction == "deny"
        assert policy.egress == [{"action": "allow", "target": "*.com"}, {"action": "deny", "target": "10.0.0.0/8"}]

    def test_a_failed_create_kills_what_it_left_by_its_own_label(self, sdk: ModuleType, server: FakeServer) -> None:
        server.create_fails = True
        server.pages[0].append(SimpleNamespace(id="sbx-other", metadata={"nhx.nvidia.com/build-job": "job-1"}))
        with pytest.raises(TimeoutError):
            _create(sdk)
        assert server.killed == ["sbx-left"]


class TestASandbox:
    def test_a_command_passes_its_output_on_and_keeps_none(self, sdk: ModuleType, server: FakeServer) -> None:
        output: list[str] = []
        assert _create(sdk).run("true", timeout_seconds=30, on_output=output.append) == 3
        command, opts, handlers = server.sandbox.ran[0]
        assert (command, opts.timeout) == ("true", timedelta(seconds=30))
        assert handlers.skip_accumulation is True
        assert output == ["out\n", "err\n"]

    def test_it_reports_its_sidecars_enforcement_mode_with_its_policy(
        self, sdk: ModuleType, server: FakeServer
    ) -> None:
        """SDK 0.1.16's own get_egress_policy drops the mode."""
        applied = _create(sdk).applied_egress()
        assert server.policy_reads == ["the sidecar's client"]
        assert applied.enforcement_mode == "dns+nft"
        assert applied.policy == _Model(defaultAction="deny", egress=[{"action": "allow"}])

    def test_a_sidecar_that_reports_no_mode_has_none(self, sdk: ModuleType, server: FakeServer) -> None:
        server.policy.parsed.enforcement_mode = Unset()
        assert _create(sdk).applied_egress().enforcement_mode is None

    @pytest.mark.parametrize("parsed", ["unauthorized", PolicyStatusResponse(enforcement_mode="dns", policy=Unset())])
    def test_a_policy_that_cannot_be_read_raises(self, sdk: ModuleType, server: FakeServer, parsed: object) -> None:
        server.policy = SimpleNamespace(status_code=401, parsed=parsed)
        with pytest.raises(RuntimeError, match="could not be read: HTTP 401"):
            _create(sdk).applied_egress()

    def test_it_is_closed_even_if_it_cannot_be_killed(self, sdk: ModuleType, server: FakeServer) -> None:
        server.kill_fails = True
        with pytest.raises(ConnectionError):
            _create(sdk).destroy()
        assert server.sandbox.killed and server.sandbox.closed


class TestLabelled:
    def test_it_matches_the_labels_itself_on_every_page(self, sdk: ModuleType, server: FakeServer) -> None:
        """SDK 0.1.16's server-side filter matches nothing on a key with a `/`."""
        job = "nhx.nvidia.com/build-job"
        server.pages = [
            [SimpleNamespace(id="a", metadata={job: "x"}), SimpleNamespace(id="b", metadata={job: "y"})],
            [SimpleNamespace(id="c", metadata={job: "x", "other": "1"}), SimpleNamespace(id="d", metadata=None)],
        ]
        assert sdk.SdkApi(SERVER, "tenant-key").labelled({job: "x"}) == ["a", "c"]
        assert server.managers_closed == 1

    def test_kill_kills_one_sandbox(self, sdk: ModuleType, server: FakeServer) -> None:
        sdk.SdkApi(SERVER, "tenant-key").kill("sbx-old")
        assert (server.killed, server.managers_closed) == (["sbx-old"], 1)
