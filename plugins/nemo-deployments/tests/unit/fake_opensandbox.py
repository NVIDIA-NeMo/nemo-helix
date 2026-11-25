# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A fake OpenSandbox installation: lifecycle server, header-mode ingress gateway, and execd."""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, parse_qsl, urlsplit

API_KEY = "osb-key"
SECURE_ACCESS_HEADER = "OpenSandbox-Secure-Access"
INGRESS_HEADER = "OpenSandbox-Ingress-To"
LEGACY_INGRESS_HEADER = "OPEN-SANDBOX-INGRESS"
EXECD_TOKEN_HEADER = "X-EXECD-ACCESS-TOKEN"


@dataclass
class FakeCommand:
    argv: list[str]
    envs: dict[str, str]
    running: bool = True
    exit_code: int | None = None
    output: str = ""


@dataclass
class FakeOpenSandbox:
    base: str = ""
    sandboxes: dict[str, dict[str, Any]] = field(default_factory=dict)
    create_bodies: list[dict[str, Any]] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    files: dict[str, dict[str, tuple[bytes, int]]] = field(default_factory=dict)
    commands: dict[str, FakeCommand] = field(default_factory=dict)
    interrupted: list[str] = field(default_factory=list)
    probes: list[str] = field(default_factory=list)
    gateway_requests: list[tuple[str, str, str]] = field(default_factory=list)
    initial_state: str = "Running"
    create_delay: float = 0.0
    server_status: int | None = None
    patch_status: int | None = None
    execd_status: int | None = None
    logs_status: int | None = None
    app_status: int = 200
    secure_token_generation: int = 0

    def add(self, metadata: dict[str, str], state: str = "Running") -> str:
        sandbox_id = str(uuid.uuid4())
        self.sandboxes[sandbox_id] = {
            "id": sandbox_id,
            "image": {"uri": "registry.example/agent:1"},
            "status": {"state": state, "lastTransitionAt": "2026-01-01T00:00:00Z"},
            "metadata": dict(metadata),
            "entrypoint": ["tail", "-f", "/dev/null"],
            "createdAt": "2026-01-01T00:00:00Z",
        }
        return sandbox_id

    def secure_token(self, sandbox_id: str) -> str:
        return f"secure-{sandbox_id}-{self.secure_token_generation}"

    def execd_token(self, sandbox_id: str) -> str | None:
        body = next((b for b in self.create_bodies if b.get("_id") == sandbox_id), None)
        return (body or {}).get("env", {}).get("EXECD_ACCESS_TOKEN")


def _multipart_parts(content_type: str, body: bytes) -> list[tuple[str, bytes]]:
    """Name and payload of each part, in order."""
    boundary = content_type.split("boundary=", 1)[1].encode()
    parts: list[tuple[str, bytes]] = []
    for part in body.split(b"--" + boundary):
        head, _, payload = part.partition(b"\r\n\r\n")
        name = re.search(rb'name="([^"]+)"', head)
        if name is not None:
            parts.append((name[1].decode(), payload.removesuffix(b"\r\n")))
    return parts


class _Handler(BaseHTTPRequestHandler):
    state: FakeOpenSandbox

    def _send(
        self, status: int, body: Any = None, text: str | None = None, headers: dict[str, str] | None = None
    ) -> None:
        payload = text.encode() if text is not None else json.dumps(body if body is not None else {}).encode()
        headers = {"content-type": "text/plain" if text is not None else "application/json", **(headers or {})}
        self.send_response(status)
        self.send_header("content-length", str(len(payload)))
        for name, value in headers.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(payload)

    def _body(self) -> bytes:
        if self.headers.get("transfer-encoding", "").lower() != "chunked":
            return self.rfile.read(int(self.headers.get("content-length") or 0))
        body = b""
        while size := int(self.rfile.readline().split(b";")[0], 16):
            body += self.rfile.read(size)
            self.rfile.readline()
        self.rfile.readline()
        return body

    def _handle(self) -> None:
        url = urlsplit(self.path)
        parts = url.path.strip("/").split("/")
        body = self._body()
        route = self._gateway_route()
        if route is not None:
            return self._gateway(route, parts, parse_qs(url.query), body)
        if parts[0] == "v1":
            return self._server(parts[1:], parse_qs(url.query), body)
        return self._send(400, text="missing header 'OpenSandbox-Ingress-To' or 'Host'")

    def _gateway_route(self) -> str | None:
        """The route label the ingress gateway reads: the routing header, its legacy name, then ``Host``."""
        label = self.headers.get(INGRESS_HEADER) or self.headers.get(LEGACY_INGRESS_HEADER)
        if label:
            return label
        host = self.headers.get("host", "")
        return host.split(".")[0] if host.endswith(".osb.helix.internal") else None

    do_GET = do_POST = do_PATCH = do_DELETE = _handle

    def _server(self, parts: list[str], query: dict[str, list[str]], body: bytes) -> None:
        state = self.state
        if self.headers.get("OPEN-SANDBOX-API-KEY") != API_KEY:
            return self._send(401, {"code": "UNAUTHORIZED", "message": "bad key"})
        if state.server_status is not None:
            return self._send(state.server_status, {"code": "BROKEN", "message": "server is broken"})
        if parts == ["sandboxes"] and self.command == "POST":
            request = json.loads(body)
            time.sleep(state.create_delay)
            sandbox_id = state.add(request.get("metadata") or {}, state.initial_state)
            state.create_bodies.append({**request, "_id": sandbox_id})
            return self._send(202, state.sandboxes[sandbox_id])
        if parts == ["sandboxes"] and self.command == "GET":
            wanted = dict(parse_qsl(query.get("metadata", [""])[0]))
            items = [
                sandbox
                for sandbox in state.sandboxes.values()
                if all(sandbox["metadata"].get(key) == value for key, value in wanted.items())
            ]
            pagination = {
                "page": 1,
                "pageSize": int(query.get("pageSize", ["20"])[0]),
                "totalItems": len(items),
                "totalPages": 1,
                "hasNextPage": False,
            }
            return self._send(200, {"items": items, "pagination": pagination})
        sandbox = state.sandboxes.get(parts[1]) if len(parts) > 1 else None
        if sandbox is None:
            return self._send(404, {"code": "NOT_FOUND", "message": "no such sandbox"})
        if parts[2:] == ["metadata"] and self.command == "PATCH":
            if state.patch_status is not None:
                return self._send(state.patch_status, {"code": "CONFLICT", "message": "metadata changed"})
            for key, value in json.loads(body).items():
                if value is None:
                    sandbox["metadata"].pop(key, None)
                else:
                    sandbox["metadata"][key] = value
            return self._send(200, sandbox)
        if parts[2:3] == ["endpoints"]:
            headers = {INGRESS_HEADER: f"{parts[1]}-{parts[3]}", SECURE_ACCESS_HEADER: state.secure_token(parts[1])}
            return self._send(200, {"endpoint": state.base, "headers": headers})
        if len(parts) == 2 and self.command == "DELETE":
            state.deleted.append(parts[1])
            del state.sandboxes[parts[1]]
            return self._send(204)
        return self._send(404, {"code": "NOT_FOUND", "message": "no such route"})

    def _gateway(self, route: str, rest: list[str], query: dict[str, list[str]], body: bytes) -> None:
        state = self.state
        sandbox_id, _, port = route.rpartition("-")
        state.gateway_requests.append((sandbox_id, port, urlsplit(self.path).path))
        if sandbox_id not in state.sandboxes:
            return self._send(404)
        if self.headers.get(SECURE_ACCESS_HEADER) != state.secure_token(sandbox_id):
            return self._send(401)
        if port != "44772":
            state.probes.append("/" + "/".join(rest))
            return self._send(state.app_status)
        if state.execd_status is not None:
            return self._send(state.execd_status)
        if self.headers.get(EXECD_TOKEN_HEADER) != state.execd_token(sandbox_id):
            return self._send(401)
        if rest == ["files", "upload"]:
            uploaded = _multipart_parts(self.headers["content-type"], body)
            for (_, metadata_json), (_, content) in zip(uploaded[::2], uploaded[1::2], strict=True):
                metadata = json.loads(metadata_json)
                state.files.setdefault(sandbox_id, {})[metadata["path"]] = (content, metadata["mode"])
            return self._send(200)
        if rest == ["command"] and self.command == "POST":
            request = json.loads(body)
            command_id = uuid.uuid4().hex
            state.commands[command_id] = FakeCommand(argv=request["argv"], envs=request.get("envs") or {})
            now = int(time.time() * 1000)
            events = [
                {"type": "init", "text": command_id, "timestamp": now},
                {"type": "execution_complete", "timestamp": now},
            ]
            stream = "\n\n".join(json.dumps(event) for event in events) + "\n\n"
            return self._send(200, text=stream, headers={"content-type": "text/event-stream"})
        if rest == ["command"] and self.command == "DELETE":
            state.interrupted.append(query["id"][0])
            return self._send(200)
        if rest[:2] == ["command", "status"]:
            command = state.commands.get(rest[2])
            if command is None:
                return self._send(404, {"code": "NOT_FOUND", "message": "no such command"})
            return self._send(200, {"id": rest[2], "running": command.running, "exit_code": command.exit_code})
        if rest[:1] == ["command"] and rest[2:] == ["logs"]:
            if state.logs_status is not None:
                return self._send(state.logs_status)
            output = state.commands[rest[1]].output
            return self._send(200, text=output, headers={"EXECD-COMMANDS-TAIL-CURSOR": str(len(output))})
        return self._send(404)

    def log_message(self, format: str, *args: Any) -> None:
        pass


@contextmanager
def running_fake_opensandbox() -> Generator[FakeOpenSandbox]:
    """Serve a fake installation on a loopback port for the duration of the block."""
    state = FakeOpenSandbox()
    server = ThreadingHTTPServer(("127.0.0.1", 0), type("Handler", (_Handler,), {"state": state}))
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    state.base = f"127.0.0.1:{server.server_address[1]}"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
