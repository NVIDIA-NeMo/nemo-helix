# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A fake OpenSandbox installation: lifecycle server, uri-mode ingress gateway, and execd."""

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
from urllib.parse import parse_qs, urlsplit

API_KEY = "osb-key"
SECURE_ACCESS_HEADER = "OpenSandbox-Secure-Access"
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
    initial_state: str = "Running"
    create_delay: float = 0.0
    server_status: int | None = None
    patch_status: int | None = None
    execd_status: int | None = None
    app_status: int = 200

    def add(self, metadata: dict[str, str], state: str = "Running") -> str:
        sandbox_id = str(uuid.uuid4())
        self.sandboxes[sandbox_id] = {"id": sandbox_id, "status": {"state": state}, "metadata": dict(metadata)}
        return sandbox_id

    def secure_token(self, sandbox_id: str) -> str:
        return f"secure-{sandbox_id}"

    def execd_token(self, sandbox_id: str) -> str | None:
        body = next((b for b in self.create_bodies if b.get("_id") == sandbox_id), None)
        return (body or {}).get("env", {}).get("EXECD_ACCESS_TOKEN")


def _multipart_fields(content_type: str, body: bytes) -> dict[str, bytes]:
    boundary = content_type.split("boundary=", 1)[1].encode()
    fields: dict[str, bytes] = {}
    for part in body.split(b"--" + boundary):
        head, _, payload = part.partition(b"\r\n\r\n")
        name = re.search(rb'name="([^"]+)"', head)
        if name is not None:
            fields[name[1].decode()] = payload.removesuffix(b"\r\n")
    return fields


class _Handler(BaseHTTPRequestHandler):
    state: FakeOpenSandbox

    def _send(self, status: int, body: Any = None, text: str | None = None) -> None:
        payload = text.encode() if text is not None else json.dumps(body if body is not None else {}).encode()
        self.send_response(status)
        self.send_header("content-type", "text/plain" if text is not None else "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _handle(self) -> None:
        url = urlsplit(self.path)
        parts = url.path.strip("/").split("/")
        body = self.rfile.read(int(self.headers.get("content-length") or 0))
        if parts[0] == "v1":
            return self._server(parts[1:], parse_qs(url.query), body)
        return self._gateway(parts, parse_qs(url.query), body)

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
            wanted = dict(pair.split("=", 1) for pair in query.get("metadata", [""])[0].split("&") if pair)
            items = [
                sandbox
                for sandbox in state.sandboxes.values()
                if all(sandbox["metadata"].get(key) == value for key, value in wanted.items())
            ]
            return self._send(200, {"items": items, "pagination": {"hasNextPage": False}})
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
            return self._send(200, sandbox["metadata"])
        if parts[2:3] == ["endpoints"]:
            return self._send(
                200, {"endpoint": "unused", "headers": {SECURE_ACCESS_HEADER: state.secure_token(parts[1])}}
            )
        if len(parts) == 2 and self.command == "DELETE":
            state.deleted.append(parts[1])
            del state.sandboxes[parts[1]]
            return self._send(204)
        return self._send(404, {"code": "NOT_FOUND", "message": "no such route"})

    def _gateway(self, parts: list[str], query: dict[str, list[str]], body: bytes) -> None:
        state = self.state
        sandbox_id, port, rest = parts[0], parts[1], parts[2:]
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
            fields = _multipart_fields(self.headers["content-type"], body)
            metadata = json.loads(fields["metadata"])
            content = fields["file"]
            state.files.setdefault(sandbox_id, {})[metadata["path"]] = (content, metadata["mode"])
            return self._send(200)
        if rest == ["command"] and self.command == "POST":
            request = json.loads(body)
            command_id = uuid.uuid4().hex
            state.commands[command_id] = FakeCommand(argv=request["argv"], envs=request.get("envs") or {})
            events = [{"type": "init", "text": command_id}, {"type": "execution_complete"}]
            return self._send(200, text="\n\n".join(json.dumps(event) for event in events) + "\n\n")
        if rest == ["command"] and self.command == "DELETE":
            state.interrupted.append(query["id"][0])
            return self._send(200)
        if rest[:2] == ["command", "status"]:
            command = state.commands.get(rest[2])
            if command is None:
                return self._send(404, {"code": "NOT_FOUND", "message": "no such command"})
            return self._send(200, {"id": rest[2], "running": command.running, "exit_code": command.exit_code})
        if rest[:1] == ["command"] and rest[2:] == ["logs"]:
            return self._send(200, text=state.commands[rest[1]].output)
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
