# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry checks that ``nemo agents package --publish`` runs before building.

A registry written as a URL, or one Docker holds no login for, otherwise only
fails at push time, after a build that can take minutes.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

DOCKER_HUB_AUTH_KEY = "index.docker.io"
_LOCAL_REGISTRY_HOSTS = ("localhost", "127.0.0.1")


def registry_scheme_error(registry: str) -> str | None:
    """Why *registry* is not a Docker reference prefix, or ``None`` when it is."""
    if "://" not in registry:
        return None
    bare = registry.split("://", 1)[1].rstrip("/")
    return f"--registry must not include a URL scheme; Docker image references have none. Use: --registry {bare}"


def registry_host(registry: str) -> str:
    """The host Docker stores credentials under for *registry* (Docker Hub for bare names)."""
    first = registry.split("/", 1)[0]
    if "." in first or ":" in first or first == "localhost":
        return first
    return DOCKER_HUB_AUTH_KEY


def _normalize_auth_key(key: str) -> str:
    host = urlsplit(key).netloc if "://" in key else key.split("/", 1)[0]
    return DOCKER_HUB_AUTH_KEY if host in ("docker.io", "registry-1.docker.io") else host


def _docker_config() -> dict[str, object]:
    config_dir = os.environ.get("DOCKER_CONFIG") or str(Path.home() / ".docker")
    try:
        loaded = json.loads((Path(config_dir) / "config.json").read_text())
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _credential_store_hosts(store: str) -> set[str] | None:
    try:
        result = subprocess.run(
            [f"docker-credential-{store}", "list"], capture_output=True, text=True, timeout=10, check=True
        )
        listed = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return {_normalize_auth_key(key) for key in listed} if isinstance(listed, dict) else None


def registry_has_credentials(registry: str) -> bool | None:
    """Whether Docker holds credentials for *registry*: ``None`` when the credential store can't be read."""
    host = registry_host(registry)
    if host.split(":", 1)[0] in _LOCAL_REGISTRY_HOSTS:
        return True
    config = _docker_config()
    helpers = config.get("credHelpers")
    if isinstance(helpers, dict) and host in {_normalize_auth_key(key) for key in helpers}:
        return True
    store = config.get("credsStore")
    if isinstance(store, str) and store:
        hosts = _credential_store_hosts(store)
        if hosts is None:
            return None
        return host in hosts
    auths = config.get("auths")
    return isinstance(auths, dict) and host in {_normalize_auth_key(key) for key in auths}


def registry_login_hint(registry: str) -> str:
    host = registry_host(registry)
    if host == "nvcr.io":
        return (
            "Log in with an NGC API key generated in the org you push to: "
            "echo \"$NGC_API_KEY\" | docker login nvcr.io -u '$oauthtoken' --password-stdin"
        )
    return f"Log in first: docker login {host}"
