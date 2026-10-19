# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP shim that satisfies AIPerf's pre-check and proxies chat completions.

AIPerf's ``_check_service`` issues ``GET urljoin(base_url, "/v1/models")`` and
expects a 200, but NHX doesn't serve ``/v1/models`` at the IGW workspace root
the way OpenAI does, and upstream AIPerf doesn't expose a knob to override the
probe path. To unblock the benchmark without patching AIPerf or NHX, we run
this tiny shim on a separate port:

- ``GET /v1/models``  -> ``200 {"object":"list","data":[]}``
- ``POST /v1/chat/completions`` -> reverse-proxy to NHX IGW
- ``GET /__shim/health`` -> ``200 {"status":"ok"}``
- Any other path -> 404

This module is meant to be invoked as ``python -m
nemo_guardrails_plugin.benchmarks.shim`` so it can be supervised by the same
process-group machinery the harness uses for the mock LLMs and ``nemo
services run``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
from nemo_guardrails_plugin.benchmarks.constants import (
    AIPERF_SHIM_HOST,
    AIPERF_SHIM_PORT,
    IGW_CHAT_PATH,
    NHX_BASE_URL,
)

log = logging.getLogger("nemo_guardrails_plugin.benchmarks.shim")


_MODELS_RESPONSE = json.dumps({"object": "list", "data": []}).encode("utf-8")
_HEALTH_RESPONSE = json.dumps({"status": "ok"}).encode("utf-8")


class _ShimHandler(BaseHTTPRequestHandler):
    """Minimal handler that routes the two paths AIPerf actually touches."""

    # Allow override via class attr so tests can swap in a mock httpx client.
    upstream_url: str = f"{NHX_BASE_URL}{IGW_CHAT_PATH}"

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        log.debug("shim: " + format, *args)

    def _send_json(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if self.path == "/v1/models":
            self._send_json(200, _MODELS_RESPONSE)
            return
        if self.path == "/__shim/health":
            self._send_json(200, _HEALTH_RESPONSE)
            return
        self._send_json(404, json.dumps({"detail": "not found"}).encode("utf-8"))

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        if self.path != "/v1/chat/completions":
            self._send_json(404, json.dumps({"detail": "not found"}).encode("utf-8"))
            return

        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length else b""

        forwarded_headers = {
            k: v for k, v in self.headers.items() if k.lower() not in {"host", "content-length", "connection"}
        }

        try:
            response = httpx.post(
                self.upstream_url,
                content=body,
                headers=forwarded_headers,
                timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0),
            )
        except httpx.HTTPError as e:
            log.warning("shim: upstream request failed: %s", e)
            self._send_json(
                502,
                json.dumps({"detail": f"upstream error: {e}"}).encode("utf-8"),
            )
            return

        self.send_response(response.status_code)
        for header, value in response.headers.items():
            if header.lower() in {"transfer-encoding", "connection", "content-length"}:
                continue
            self.send_header(header, value)
        self.send_header("Content-Length", str(len(response.content)))
        self.end_headers()
        self.wfile.write(response.content)


def serve(host: str = AIPERF_SHIM_HOST, port: int = AIPERF_SHIM_PORT) -> None:
    """Run the shim until the process is signalled."""
    httpd = ThreadingHTTPServer((host, port), _ShimHandler)
    log.info("AIPerf shim listening on http://%s:%d", host, port)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nemo-guardrails-benchmark-shim",
        description=__doc__,
    )
    parser.add_argument("--host", default=AIPERF_SHIM_HOST)
    parser.add_argument("--port", type=int, default=AIPERF_SHIM_PORT)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    serve(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
