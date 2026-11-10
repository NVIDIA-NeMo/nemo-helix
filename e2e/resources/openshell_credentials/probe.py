# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep using one startup placeholder while Helix rotates or removes its credential."""

import hashlib
import json
import os
import ssl
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen


class HealthHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        self.send_response(200 if self.path == "/health" else 404)
        self.send_header("Content-Length", "0")
        self.end_headers()


def request_status(request: Request, context: ssl.SSLContext) -> tuple[int | None, str | None]:
    try:
        with urlopen(request, timeout=5, context=context) as response:
            return response.status, None
    except HTTPError as exc:
        exc.close()
        return exc.code, None
    except URLError as exc:
        return None, type(exc.reason).__name__
    except (OSError, ValueError) as exc:
        return None, type(exc).__name__


def main() -> None:
    placeholder = os.environ["API_TOKEN"]
    service_url = os.environ["SERVICE_URL"]
    probe_id = os.environ["PROBE_ID"]
    base_url = os.environ["NHX_BASE_URL"].rstrip("/")
    forbidden_path = os.environ["FORBIDDEN_HELIX_PATH"]
    if not forbidden_path.startswith("/") or forbidden_path.startswith("//"):
        raise ValueError("FORBIDDEN_HELIX_PATH must be a platform-relative path.")
    interval = float(os.environ.get("PROBE_INTERVAL_SECONDS", "1"))
    if not 0 < interval <= 60:
        raise ValueError("PROBE_INTERVAL_SECONDS must be between zero and 60.")
    context = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE"))
    identity = {
        "probe_id": probe_id,
        "pid": os.getpid(),
        "boot_id": uuid.uuid4().hex,
        "placeholder_sha256": hashlib.sha256(placeholder.encode()).hexdigest(),
    }
    health_server = None
    attempt = 0
    last_status: int | None = None
    last_error: str | None = None
    if os.environ.get("PROBE_HEALTH_SERVER") == "1":
        health_server = ThreadingHTTPServer(("0.0.0.0", 8000), HealthHandler)
        threading.Thread(target=health_server.serve_forever, daemon=True).start()
    try:
        while True:
            attempt += 1
            forbidden_status, forbidden_error = request_status(Request(base_url + forbidden_path), context)
            heartbeat = {
                **identity,
                "completed_attempt": attempt - 1,
                "probe_status": last_status,
                "probe_error": last_error,
            }
            health_url = urljoin(service_url, "/health") + "?" + urlencode({"heartbeat": json.dumps(heartbeat)})
            public_status, _ = request_status(Request(health_url), context)
            report = {
                **identity,
                "attempt": attempt,
                "forbidden_status": forbidden_status,
                "forbidden_error": forbidden_error,
                "public_status": public_status,
            }
            status, error = request_status(
                Request(
                    service_url,
                    data=json.dumps(report).encode(),
                    headers={"Authorization": f"Bearer {placeholder}", "Content-Type": "application/json"},
                    method="POST",
                ),
                context,
            )
            last_status, last_error = status, error
            print(json.dumps({**report, "probe_status": status, "probe_error": error}), flush=True)
            time.sleep(interval)
    finally:
        if health_server is not None:
            health_server.shutdown()
            health_server.server_close()


if __name__ == "__main__":
    main()
