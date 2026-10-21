# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A stand-in OpenAI-compatible endpoint, so a sandboxed Gym host can complete a rollout offline.

NeMo-Gym's `inference_provider` model server speaks `/v1/chat/completions`, and its startup probe
only requires that *something* answers at the base URL. That is the whole surface needed to drive a
rollout end to end without a real provider or an API key, which is what makes a local execution
test possible at all.

It answers every question the same way. That is deliberate: the point is to exercise the path, and
a fixed answer makes the resulting reward a property of the environment's grading rather than of
model behaviour that would vary run to run.
"""

from __future__ import annotations

import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

ANSWER = os.environ.get("STUB_ANSWER", "A")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        # The startup probe accepts any answer, including a 404. Models are listed anyway so the
        # endpoint is usable by anything that enumerates first.
        self._json(200, {"object": "list", "data": [{"id": os.environ.get("STUB_MODEL", "dummy-model")}]})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        request = json.loads(self.rfile.read(length) or b"{}")
        self._json(
            200,
            {
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.get("model", "dummy-model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": ANSWER},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )


if __name__ == "__main__":
    HTTPServer(("0.0.0.0", int(os.environ.get("PORT", "8000"))), Handler).serve_forever()
