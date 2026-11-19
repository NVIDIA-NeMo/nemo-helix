# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator-started loopback demo manifest server, not a Helix-managed workload.

Default Tavily responses are deterministic STUBS. --live-tavily changes only the
manifest mapping to Tavily's real hosted MCP API; it never reads a real API key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import yaml


def create_server(port: int = 8099, *, live_tavily: bool = False) -> ThreadingHTTPServer:
    # Optional example UI is available only from a source checkout. The demo
    # never builds or executes third-party code in the platform process.
    root = next((p for p in Path(__file__).resolve().parents if (p / "plugins/example-plugin").is_dir()), None)
    bundle_dir = root / "plugins/example-plugin/src/nemo_example_plugin/web/dist" if root is not None else None

    class Handler(BaseHTTPRequestHandler):
        server: ThreadingHTTPServer

        def reply(self, status: int, data: bytes, content_type: str = "application/json") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def manifest_reply(self, value: dict) -> None:
            if urlsplit(self.path).path.endswith(".yaml"):
                self.reply(200, yaml.safe_dump(value, sort_keys=False).encode(), "application/yaml")
            else:
                self.reply(200, json.dumps(value).encode())

        def do_GET(self) -> None:
            base = f"http://127.0.0.1:{self.server.server_port}"
            path = urlsplit(self.path).path
            if path in {"/tavily.yaml", "/tavily.json"}:
                upstream = "https://mcp.tavily.com/mcp" if live_tavily else f"{base}/mcp"
                manifest = {
                    "schema_version": 1,
                    "plugin": {
                        "name": "tavily",
                        "title": "Tavily" if live_tavily else "Tavily (LOCAL STUB)",
                        "endpoints": [
                            {
                                "method": "POST",
                                "path": "/v2/workspaces/{workspace}/mcp",
                                "access": "write",
                                "upstream_url": upstream,
                            }
                        ],
                        "credential": {"secret_name": "tavily-api-key", "scheme": "bearer"},
                    },
                }
                self.manifest_reply(manifest)
            elif path in {"/example.yaml", "/example.json"} and bundle_dir is not None:
                data = (bundle_dir / "index.js").read_bytes()
                manifest = {
                    "schema_version": 1,
                    "plugin": {
                        "name": "example",
                        "title": "Community Example UI",
                        "endpoints": [
                            {
                                "method": "POST",
                                "path": "/v2/workspaces/{workspace}/mcp",
                                "access": "write",
                                "upstream_url": f"{base}/mcp",
                            }
                        ],
                        "credential": {"secret_name": "tavily-api-key"},
                        "bundle": {
                            "url": f"{base}/index.js",
                            "sha256": hashlib.sha256(data).hexdigest(),
                            "sidecar": json.loads((bundle_dir / "plugin.manifest.json").read_text()),
                        },
                        "navigation": [{"label": "Example", "path": "/workspaces/{workspace}/plugin/example"}],
                    },
                }
                self.manifest_reply(manifest)
            elif path == "/index.js" and bundle_dir is not None:
                self.reply(200, (bundle_dir / "index.js").read_bytes(), "text/javascript")
            else:
                self.reply(404, b'{"detail":"not found"}')

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/mcp" or live_tavily:
                self.reply(404, b'{"detail":"not found"}')
                return
            if self.headers.get("Authorization") != "Bearer demo-tavily-key":
                self.reply(401, b'{"detail":"LOCAL STUB requires demo-tavily-key"}')
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= 65536:
                    raise ValueError("invalid length")
                message = json.loads(self.rfile.read(length))
                if not isinstance(message, dict):
                    raise ValueError("expected object")
            except (ValueError, json.JSONDecodeError):
                self.reply(400, b'{"detail":"invalid JSON-RPC request"}')
                return
            method = message.get("method")
            if "id" not in message:
                self.reply(202, b"")
                return
            if method == "initialize":
                result = {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "LOCAL Tavily STUB", "version": "1"},
                }
            elif method == "tools/list":
                result = {
                    "tools": [
                        {
                            "name": "tavily_search",
                            "description": "LOCAL STUB, no web access",
                            "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}},
                        }
                    ]
                }
            else:
                result = {"content": [{"type": "text", "text": "LOCAL STUB result: no request was sent to Tavily."}]}
            self.reply(200, json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}).encode())

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--live-tavily", action="store_true")
    args = parser.parse_args()
    with create_server(args.port, live_tavily=args.live_tavily) as server:
        print(f"Manifest: http://127.0.0.1:{server.server_port}/tavily.yaml", flush=True)
        print("Mode: LIVE TAVILY mapping" if args.live_tavily else "Mode: LOCAL STUB (no Tavily requests)", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
