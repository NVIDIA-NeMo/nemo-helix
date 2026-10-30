# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A mock OpenSandbox SDK and server for the driver and environment tests.

Shapes follow what the real SDK returns where it matters: command output arrives as one message
per line with its line ending stripped, and sandbox listings are paginated.
"""

from __future__ import annotations

import itertools
import re
import shlex
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import httpx


class SandboxApiException(Exception):
    """Same name as the SDK's transient API error, so the driver retries it."""


class Server:
    """What the mock server holds and how it misbehaves; tests set the knobs, then inspect the state."""

    def __init__(self) -> None:
        """A server with no sandboxes that behaves normally until a test turns a knob."""
        self.sandboxes: dict[str, dict[str, str]] = {}
        self.killed: list[str] = []
        self.creates: list[dict[str, Any]] = []
        # Raised by create after the server has made the sandbox, as when a response is lost.
        self.create_failures: list[BaseException] = []
        # Raised by create before any sandbox exists.
        self.rejections: list[BaseException] = []
        # Overrides the default action a sandbox reports back; None reports what was sent.
        self.applied_default: str | None = None
        # False makes sandboxes unable to report their applied egress policy at all.
        self.readback = True
        # States get_info reports, one per call; the last one repeats.
        self.states: list[str] = ["Running"]
        self.healthy = True
        # How many health probes raise before they start answering.
        self.health_errors = 0
        # The uid ``id -u`` reports; None makes the probe fail.
        self.image_uid: int | None = 0
        # Listings return at most this many sandboxes per page.
        self.page_size = 50
        # Sandbox IDs whose kill raises.
        self.unkillable: set[str] = set()
        # What the endpoints route answers instead of ``<pod IP>:<port>``, or raises.
        self.endpoint_answer: httpx.Response | BaseException | None = None
        self.endpoint_requests: list[httpx.Request] = []
        # False gives sandboxes none of the SDK's private services, as a future SDK might.
        self.private_services = True
        self._ids = (f"sbx-{n}" for n in itertools.count(1))

    def live(self) -> set[str]:
        """IDs of sandboxes the server created and nothing has killed."""
        return set(self.sandboxes) - set(self.killed)

    @staticmethod
    def pod_ip(sandbox_id: str) -> str:
        """The pod IP the server reports for ``sbx-<n>``: ``10.244.0.<n>``."""
        return f"10.244.0.{sandbox_id.rsplit('-', 1)[1]}"

    def endpoint_route(self, request: httpx.Request) -> httpx.Response | None:
        """Answer ``GET /v1/sandboxes/{id}/endpoints/{port}`` as a Kubernetes server without a gateway does.

        Returns ``None`` for any other route, so a test's own handler can answer it.
        """
        match = re.fullmatch(r"/v1/sandboxes/([^/]+)/endpoints/(\d+)", request.url.path)
        if match is None:
            return None
        self.endpoint_requests.append(request)
        if isinstance(self.endpoint_answer, BaseException):
            raise self.endpoint_answer
        if self.endpoint_answer is not None:
            return self.endpoint_answer
        sandbox_id, port = match.groups()
        return httpx.Response(200, json={"endpoint": f"{self.pod_ip(sandbox_id)}:{port}"})

    def kill(self, sandbox_id: str) -> None:
        """Record a kill, or raise for a sandbox listed in ``unkillable``."""
        if sandbox_id in self.unkillable:
            raise SandboxApiException(f"kill {sandbox_id} failed")
        self.killed.append(sandbox_id)


class Files:
    """``sandbox.files`` over an in-memory tree of ``path -> bytes``; directories map to ``None``."""

    def __init__(self) -> None:
        """An empty file tree."""
        self.tree: dict[str, bytes | None] = {}
        self.modes: dict[str, int] = {}
        # Paths whose read fails, like a dangling symlink.
        self.unreadable: set[str] = set()
        # Paths the search reports without an entry type, so the caller has to ask.
        self.untyped: set[str] = set()

    async def create_directories(self, entries: list[SimpleNamespace]) -> None:
        """Record each entry as a directory with its mode."""
        for entry in entries:
            self.tree[entry.path] = None
            self.modes[entry.path] = entry.mode

    async def write_files(self, entries: list[SimpleNamespace]) -> None:
        """Record each entry's bytes and mode."""
        for entry in entries:
            self.tree[entry.path] = entry.data
            self.modes[entry.path] = entry.mode

    async def read_bytes(self, path: str) -> bytes:
        """Return a file's bytes; directories, missing and unreadable paths raise."""
        data = self.tree.get(path)
        if path in self.unreadable or data is None:
            raise FileNotFoundError(path)
        return data

    async def search(self, entry: SimpleNamespace) -> list[SimpleNamespace]:
        """Every path under ``entry.path``, typed as the SDK types them unless listed in ``untyped``."""
        root = entry.path.rstrip("/")
        found = []
        for path, data in sorted(self.tree.items()):
            if path.startswith(root + "/"):
                entry_type = (
                    "" if path in self.untyped else ("EntryType.DIRECTORY" if data is None else "EntryType.FILE")
                )
                found.append(SimpleNamespace(path=path, entry_type=entry_type))
        return found


def execution(
    stdout: list[str] | None = None, stderr: list[str] | None = None, exit_code: int | None = 0, error: Any = None
) -> SimpleNamespace:
    """An SDK ``Execution``: each stream is a list of messages, one per line without its line ending."""
    return SimpleNamespace(
        logs=SimpleNamespace(
            stdout=[SimpleNamespace(text=text) for text in stdout or []],
            stderr=[SimpleNamespace(text=text) for text in stderr or []],
        ),
        exit_code=exit_code,
        error=error,
    )


class Commands:
    """``sandbox.commands``: records every run; ``id -u`` reports the server's image uid.

    ``test -d PATH`` and ``test -f PATH``, which Harbor's ``is_dir`` and ``is_file`` send, are
    answered from the sandbox's file tree.
    """

    def __init__(self, server: Server, files: Files) -> None:
        """Commands for one sandbox, answering file checks from that sandbox's ``files``."""
        self._server = server
        self._files = files
        self.runs: list[tuple[str, dict[str, Any]]] = []
        # Results for other commands, by command; default is exit 0 with no output.
        self.results: dict[str, SimpleNamespace] = {}

    async def run(self, command: str, opts: SimpleNamespace) -> SimpleNamespace:
        """Record the command and its options, then return its canned result."""
        self.runs.append((command, dict(vars(opts))))
        if command == "id -u":
            if self._server.image_uid is None:
                raise SandboxApiException("execd unavailable")
            return execution(stdout=[str(self._server.image_uid)])
        if command.startswith(("test -d ", "test -f ")):
            _, flag, path = shlex.split(command)
            if path not in self._files.tree:
                kind = None
            elif self._files.tree[path] is None:
                kind = "-d"
            else:
                kind = "-f"
            return execution(exit_code=0 if kind == flag else 1)
        return self.results.get(command) or execution()

    def uids_sent(self, command: str) -> list[int | None]:
        """The uid sent with each run of ``command``; None where none was sent."""
        return [opts.get("uid") for run, opts in self.runs if run == command]

    def probes(self) -> int:
        """How many times the image uid was probed."""
        return sum(run == "id -u" for run, _ in self.runs)


class Sandbox:
    """A live sandbox handle, as ``Sandbox.create`` returns it."""

    # The SDK's private services. Declared but unset, so a sandbox only has them when mock_sdk sets them, which it
    # skips when the server's ``private_services`` knob is off.
    _sandbox_service: Any
    _connection_config: Any
    _egress_service: Any

    def __init__(self, server: Server, sandbox_id: str, policy: dict[str, Any] | None) -> None:
        """A sandbox the server created with ``policy``; ``None`` means no egress policy was sent."""
        self._server = server
        self.id = sandbox_id
        self._policy = policy or {"defaultAction": "allow", "egress": []}
        self.files = Files()
        self.commands = Commands(server, self.files)
        self.closed = False
        # The SDK only has get_egress_policy on servers that can report it.
        if server.readback:
            self.get_egress_policy = self._get_egress_policy

    async def _get_egress_policy(self) -> SimpleNamespace:
        """The policy as applied: what was sent, unless the server overrides the default action."""
        default = self._server.applied_default or self._policy["defaultAction"]
        rules = [SimpleNamespace(**rule) for rule in self._policy["egress"]]
        return SimpleNamespace(default_action=default, egress=rules)

    async def get_info(self) -> SimpleNamespace:
        """The next state from ``Server.states``, with a failure message for terminal states."""
        states = self._server.states
        state = states.pop(0) if len(states) > 1 else states[0]
        return SimpleNamespace(status=SimpleNamespace(state=state, message="image pull failed"))

    async def is_healthy(self) -> bool:
        """Raise for the first ``Server.health_errors`` probes, then report ``Server.healthy``."""
        if self._server.health_errors:
            self._server.health_errors -= 1
            raise SandboxApiException("probe failed")
        return self._server.healthy

    async def kill(self) -> None:
        """Kill this sandbox on the server."""
        self._server.kill(self.id)

    async def close(self) -> None:
        """Release the client side only."""
        self.closed = True


class Manager:
    """``SandboxManager``: paginated listing by metadata, and kill by ID."""

    def __init__(self, server: Server) -> None:
        """A manager over ``server``'s sandboxes."""
        self._server = server

    async def list_sandbox_infos(self, selector: SimpleNamespace) -> SimpleNamespace:
        """Live sandboxes whose metadata includes ``selector.metadata``, one page at a time."""
        matches = [
            SimpleNamespace(id=sandbox_id)
            for sandbox_id, metadata in self._server.sandboxes.items()
            if sandbox_id not in self._server.killed and selector.metadata.items() <= metadata.items()
        ]
        size = self._server.page_size
        page = matches[(selector.page - 1) * size : selector.page * size]
        return SimpleNamespace(
            sandbox_infos=page, pagination=SimpleNamespace(has_next_page=selector.page * size < len(matches))
        )

    async def kill_sandbox(self, sandbox_id: str) -> None:
        """Kill a sandbox by ID."""
        self._server.kill(sandbox_id)

    async def close(self) -> None:
        """Nothing to release."""


class ConnectionConfig(SimpleNamespace):
    """``opensandbox.config.ConnectionConfig``: its kwargs as attributes, plus what direct server requests read."""

    headers = {"X-Client": "opensandbox-sdk"}

    def get_api_key(self) -> str | None:
        """The API key, as the SDK reports it."""
        return self.api_key

    def get_base_url(self) -> str:
        """The server's API root, as the SDK builds it from the protocol and domain."""
        return f"{self.protocol}://{self.domain}/v1"


def mock_sdk(server: Server) -> dict[str, Any]:
    """SDK classes, as ``harbor_opensandbox_driver.load_sdk`` returns them, backed by ``server``."""

    class MockSandbox:
        """``opensandbox.Sandbox``: only ``create`` is called on the class."""

        @staticmethod
        async def create(image: Any, **kwargs: Any) -> Sandbox:
            """Record the request and create the sandbox, failing as the server's knobs say."""
            if server.rejections:
                raise server.rejections.pop(0)
            server.creates.append({"image": image, **kwargs})
            sandbox_id = next(server._ids)
            server.sandboxes[sandbox_id] = dict(kwargs["metadata"])
            if server.create_failures:
                # The server created the sandbox, but the client never got a handle for it.
                raise server.create_failures.pop(0)
            policy = kwargs["network_policy"].payload if kwargs["network_policy"] is not None else None
            sandbox = Sandbox(server, sandbox_id, policy)
            if server.private_services:
                # Where the SDK keeps them; the driver replaces the egress client for a moved policy API.
                sandbox._sandbox_service = SandboxService()
                sandbox._connection_config = kwargs["connection_config"]
                sandbox._egress_service = EgressService(SimpleNamespace(port=18080))
            return sandbox

    class SandboxService:
        """The SDK's sandbox service: resolves a sandbox's endpoint on a port."""

        async def get_sandbox_endpoint(self, sandbox_id: str, port: int, use_server_proxy: bool) -> SimpleNamespace:
            return SimpleNamespace(sandbox_id=sandbox_id, port=port, use_server_proxy=use_server_proxy)

    @dataclass
    class EgressService:
        """The SDK's egress client, on the endpoint it was built for."""

        endpoint: Any

    class AdapterFactory:
        """``opensandbox.adapters.factory.AdapterFactory``: builds the egress client for an endpoint."""

        def __init__(self, connection_config: Any) -> None:
            self.connection_config = connection_config

        def create_egress_service(self, endpoint: Any) -> EgressService:
            return EgressService(endpoint)

    class MockManager:
        """``opensandbox.manager.SandboxManager``: only ``create`` is called on the class."""

        @staticmethod
        async def create(connection_config: Any) -> Manager:
            """Open a manager on the mock server."""
            return Manager(server)

    # The model classes only need to carry their keyword arguments for the tests to inspect.
    return {
        "Sandbox": MockSandbox,
        "AdapterFactory": AdapterFactory,
        "SandboxManager": MockManager,
        "ConnectionConfig": ConnectionConfig,
        "RunCommandOpts": lambda **kwargs: SimpleNamespace(**kwargs),
        "SearchEntry": lambda **kwargs: SimpleNamespace(**kwargs),
        "WriteEntry": lambda **kwargs: SimpleNamespace(**{"data": None, **kwargs}),
        "NetworkPolicy": SimpleNamespace(model_validate=lambda payload: SimpleNamespace(payload=payload)),
        "SandboxFilter": lambda metadata, page: SimpleNamespace(metadata=metadata, page=page),
    }
