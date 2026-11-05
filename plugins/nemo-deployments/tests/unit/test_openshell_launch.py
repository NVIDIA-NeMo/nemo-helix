# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

from nemo_helix_plugin.openshell.launch import run


async def test_agent_uses_proxy_identity_and_preserves_exit_status(httpserver, unused_tcp_port):
    httpserver.expect_request(
        "/apis/inference-gateway/v2/workspaces/team/openai/-/v1/chat/completions",
        method="POST",
        headers={"X-NHX-Principal-Id": "service:agents", "X-NHX-Principal-On-Behalf-Of": "user-1"},
    ).respond_with_json({"ok": True})
    command = [
        sys.executable,
        "-c",
        f"""
import json, urllib.request
request = urllib.request.Request(
    "http://127.0.0.1:{unused_tcp_port}/apis/inference-gateway/v2/workspaces/team/openai/-/v1/chat/completions",
    data=b"{{}}", headers={{"X-NHX-Principal-Id": "spoofed", "X-NHX-Principal-On-Behalf-Of": "spoofed"}},
)
assert json.load(urllib.request.urlopen(request)) == {{"ok": True}}
raise SystemExit(17)
""",
    ]
    assert (
        await run(
            command, upstream=httpserver.url_for("/"), principal="agents", on_behalf_of="user-1", port=unused_tcp_port
        )
        == 17
    )
    httpserver.check_assertions()
