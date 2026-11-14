# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authentik admin calls for the taskset-run E2E on the Authentik demo. Prints nothing secret.

Reads ``AUTHENTIK_API`` (the server's base URL) and ``AUTHENTIK_BOOTSTRAP_TOKEN`` (``run.sh``'s generated
``authentik.env``) from the environment.

Usage:
  authentik_admin.py app-password <username> <out-file>   mint an app password; write its key to out-file (0600)
  authentik_admin.py set-email <username> <email>          set a user's email (idempotent)
"""

import json
import os
import sys
import urllib.parse
import urllib.request
import uuid
from typing import Any


def call(method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    request = urllib.request.Request(
        os.environ["AUTHENTIK_API"].rstrip("/") + path,
        method=method,
        headers={
            "Authorization": f"Bearer {os.environ['AUTHENTIK_BOOTSTRAP_TOKEN']}",
            "Content-Type": "application/json",
        },
        data=json.dumps(body).encode() if body is not None else None,
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read() or b"{}")


def user_pk(username: str) -> int:
    results = call("GET", f"/api/v3/core/users/?username={urllib.parse.quote(username)}")["results"]
    if not results:
        sys.exit(f"no Authentik user {username!r}")
    return results[0]["pk"]


def app_password(username: str, out_file: str) -> None:
    """The password grant rejects a user's own password; an app password is what it accepts."""
    identifier = f"e2e-{username}-{uuid.uuid4().hex[:8]}"
    body = {"identifier": identifier, "intent": "app_password", "user": user_pk(username), "expiring": False}
    call("POST", "/api/v3/core/tokens/", body)
    key = call("GET", f"/api/v3/core/tokens/{identifier}/view_key/")["key"]
    fd = os.open(out_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(key)
    print(f"created app password {identifier} for {username}", file=sys.stderr)


def set_email(username: str, email: str) -> None:
    pk = user_pk(username)
    if call("GET", f"/api/v3/core/users/{pk}/").get("email") != email:
        call("PATCH", f"/api/v3/core/users/{pk}/", {"email": email})
        print(f"set {username}'s email to {email}", file=sys.stderr)


if __name__ == "__main__":
    command, *args = sys.argv[1:]
    {"app-password": app_password, "set-email": set_email}[command](*args)
