# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observe credentials injected into synthetic sandbox requests without exposing them."""

import argparse
import hashlib
import hmac
import json
import ssl
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TypedDict
from urllib.parse import parse_qs, urlsplit


class ProbeReport(TypedDict):
    probe_id: str
    pid: int
    boot_id: str
    attempt: int
    placeholder_sha256: str
    forbidden_status: int | None
    forbidden_error: str | None
    public_status: int | None


class Observation(ProbeReport):
    sequence: int
    received_at: float
    accepted: bool
    version: str | None


class Heartbeat(TypedDict):
    probe_id: str
    pid: int
    boot_id: str
    placeholder_sha256: str
    completed_attempt: int
    probe_status: int | None
    probe_error: str | None


class ReceiverState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.expected: dict[str, str] = {}
        self.observations: deque[Observation] = deque(maxlen=256)
        self.sequence = 0
        self.heartbeats: dict[str, Heartbeat] = {}

    def record(self, authorization: str, report: ProbeReport) -> Observation:
        digest = hashlib.sha256(authorization.encode()).hexdigest()
        with self.lock:
            version = next((name for name, value in self.expected.items() if hmac.compare_digest(value, digest)), None)
            self.sequence += 1
            observation: Observation = {
                **report,
                "sequence": self.sequence,
                "received_at": time.time(),
                "accepted": version is not None,
                "version": version,
            }
            self.observations.append(observation)
            return observation


class ReceiverServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int]) -> None:
        self.state = ReceiverState()
        super().__init__(address, ReceiverHandler)


class ReceiverHandler(BaseHTTPRequestHandler):
    server: ReceiverServer

    def log_message(self, format: str, *args: object) -> None:
        pass

    def reply(self, status: int, body: object) -> None:
        encoded = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        target = urlsplit(self.path)
        if target.path == "/health":
            heartbeat_json = parse_qs(target.query).get("heartbeat", [None])[0]
            if heartbeat_json is not None:
                try:
                    body = json.loads(heartbeat_json)
                    heartbeat = Heartbeat(
                        probe_id=body["probe_id"],
                        pid=body["pid"],
                        boot_id=body["boot_id"],
                        placeholder_sha256=body["placeholder_sha256"],
                        completed_attempt=body["completed_attempt"],
                        probe_status=body["probe_status"],
                        probe_error=body["probe_error"],
                    )
                    with self.server.state.lock:
                        self.server.state.heartbeats[heartbeat["probe_id"]] = heartbeat
                except (ValueError, KeyError, TypeError):
                    self.reply(400, {"error": "Invalid heartbeat."})
                    return
            self.reply(200, {"ready": True})
        elif target.path == "/observations":
            try:
                after = int(parse_qs(target.query).get("after", ["0"])[0])
            except ValueError:
                self.reply(400, {"error": "Invalid sequence."})
                return
            with self.server.state.lock:
                rows = [row for row in self.server.state.observations if row["sequence"] > after]
                latest = self.server.state.sequence
                heartbeats = list(self.server.state.heartbeats.values())
            self.reply(200, {"observations": rows, "latest_sequence": latest, "heartbeats": heartbeats})
        else:
            self.reply(404, {})

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16384:
                raise ValueError
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError
            if self.path == "/expect":
                tokens = body["tokens"]
                if not isinstance(tokens, dict) or len(tokens) > 16:
                    raise ValueError
                expected: dict[str, str] = {}
                for version, token in tokens.items():
                    if not isinstance(version, str) or not 1 <= len(version) <= 64:
                        raise ValueError
                    if not isinstance(token, str) or not 1 <= len(token) <= 4096:
                        raise ValueError
                    expected[version] = hashlib.sha256(f"Bearer {token}".encode()).hexdigest()
                with self.server.state.lock:
                    self.server.state.expected = expected
                self.reply(200, {"versions": list(expected)})
            elif self.path == "/probe":
                probe_id, pid, digest = body["probe_id"], body["pid"], body["placeholder_sha256"]
                boot_id = body["boot_id"]
                attempt = body["attempt"]
                forbidden_status, forbidden_error = body["forbidden_status"], body["forbidden_error"]
                public_status = body["public_status"]
                if not isinstance(probe_id, str) or not 1 <= len(probe_id) <= 128:
                    raise ValueError
                if type(pid) is not int or pid < 1:
                    raise ValueError
                if not isinstance(boot_id, str) or len(boot_id) != 32:
                    raise ValueError
                if type(attempt) is not int or attempt < 1:
                    raise ValueError
                if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ValueError
                if forbidden_status is not None and (
                    type(forbidden_status) is not int or not 100 <= forbidden_status <= 599
                ):
                    raise ValueError
                if forbidden_error is not None and (not isinstance(forbidden_error, str) or len(forbidden_error) > 80):
                    raise ValueError
                report: ProbeReport = {
                    "probe_id": probe_id,
                    "pid": pid,
                    "boot_id": boot_id,
                    "attempt": attempt,
                    "placeholder_sha256": digest,
                    "forbidden_status": forbidden_status,
                    "forbidden_error": forbidden_error,
                    "public_status": public_status,
                }
                observation = self.server.state.record(self.headers.get("Authorization", ""), report)
                self.reply(200 if observation["accepted"] else 401, observation)
            else:
                self.reply(404, {})
        except (ValueError, KeyError, TypeError):
            self.close_connection = True
            self.reply(400, {"error": "Invalid request."})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--cert-file")
    parser.add_argument("--key-file")
    args = parser.parse_args()
    if bool(args.cert_file) != bool(args.key_file):
        parser.error("--cert-file and --key-file must be supplied together")
    with ReceiverServer((args.host, args.port)) as server:
        if args.cert_file:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(args.cert_file, args.key_file)
            server.socket = context.wrap_socket(server.socket, server_side=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
