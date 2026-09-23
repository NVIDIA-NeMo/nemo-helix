# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mint a per-run agent id and manage a platform's Intake workspaces."""

import os
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx
from nemo_helix import NeMoHelix
from nemo_helix_ext.config.config import Config

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})


class _HTTPResponse(Protocol):
    status_code: int
    text: str

    def json(self) -> dict[str, Any]: ...


class _JSONClient(Protocol):
    def post(self, url: str, *, json: dict[str, Any]) -> _HTTPResponse: ...

    def close(self) -> None: ...


class _ExperimentGroupClient(_JSONClient, Protocol):
    def get(self, url: str) -> _HTTPResponse: ...


class _PollingClient(Protocol):
    def get(self, url: str, *, params: dict[str, Any]) -> _HTTPResponse: ...

    def close(self) -> None: ...


class _HelixClient(Protocol):
    def post(
        self,
        path: str,
        *,
        cast_to: type[httpx.Response],
        content: bytes,
        options: dict[str, Any],
    ) -> httpx.Response: ...

    def close(self) -> None: ...


def _make_platform_client(base_url: str) -> NeMoHelix:
    """Build a synchronous SDK client with platform auth for remote URLs."""
    host = (urlparse(base_url).hostname or "").lower()
    config_path = Config.get_default_config_path()
    if host in _LOOPBACK_HOSTS or not config_path.exists():
        return NeMoHelix(base_url=base_url, timeout=30.0)
    return NeMoHelix(base_url=base_url, config_path=config_path, timeout=30.0)


def mint_agent_id(base: str) -> str:
    """A fresh per-run Intake agent id: ``<base>-<YYYYMMDD-HHMMSS>-<4 hex>``."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{base}-{ts}-{os.urandom(2).hex()}"


def ensure_workspace(base_url: str, workspace: str, *, client: httpx.Client | _JSONClient | None = None) -> None:
    """Create the Intake workspace if it does not already exist.

    POSTs to the entities create-workspace route; treats 2xx (created) and 409
    (already exists) as success, raising ``RuntimeError`` on any other status.
    ``client`` is injectable for tests; when None a short-timeout client is
    created and closed here. No auth header (matches the unauthenticated local
    platform, like the OTLP ingest calls).
    """
    url = f"{base_url.rstrip('/')}/apis/entities/v2/workspaces"
    owns_client = client is None
    active_client = client or httpx.Client(timeout=30.0)
    try:
        resp = active_client.post(url, json={"name": workspace})
    finally:
        if owns_client:
            active_client.close()
    if resp.status_code != 409 and not (200 <= resp.status_code < 300):
        raise RuntimeError(f"workspace create failed ({resp.status_code}): {resp.text}")


def ensure_experiment_group(
    base_url: str,
    workspace: str,
    name: str,
    *,
    client: httpx.Client | _ExperimentGroupClient | None = None,
) -> str:
    """Create the Intake ExperimentGroup if absent; return its id.

    POSTs to the experiment-groups route; on 2xx returns the created group's
    ``id``. A 409 means it already exists, so GET it by name and return that id
    (the group is the per-subject container every run's Experiment hangs off).
    Raises ``RuntimeError`` on any other status. No auth header.
    """
    root = f"{base_url.rstrip('/')}/apis/intake/v2/workspaces/{workspace}/experiment-groups"
    owns_client = client is None
    client = client or httpx.Client(timeout=30.0)
    try:
        resp = client.post(root, json={"name": name})
        if 200 <= resp.status_code < 300:
            return resp.json()["id"]
        if resp.status_code == 409:
            existing = client.get(f"{root}/{name}")
            if 200 <= existing.status_code < 300:
                return existing.json()["id"]
            raise RuntimeError(f"experiment-group lookup failed ({existing.status_code}): {existing.text}")
        raise RuntimeError(f"experiment-group create failed ({resp.status_code}): {resp.text}")
    finally:
        if owns_client:
            client.close()


def create_experiment(
    base_url: str,
    workspace: str,
    *,
    name: str,
    experiment_group_id: str,
    dataset_name: str,
    dataset_version: str,
    metadata: Mapping[str, object],
    client: httpx.Client | _JSONClient | None = None,
) -> None:
    """Create the per-run Experiment under ``experiment_group_id``.

    ``name`` is the run id (timestamped + hex), so it is workspace-unique — a
    409 here is a genuine collision and is surfaced as ``RuntimeError`` along
    with every other non-2xx. No auth header.
    """
    url = f"{base_url.rstrip('/')}/apis/intake/v2/workspaces/{workspace}/evaluations"
    body = {
        "name": name,
        "experiment_group_id": experiment_group_id,
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "metadata": {key: str(value) for key, value in metadata.items()},
    }
    owns_client = client is None
    client = client or httpx.Client(timeout=30.0)
    try:
        resp = client.post(url, json=body)
    finally:
        if owns_client:
            client.close()
    if not (200 <= resp.status_code < 300):
        raise RuntimeError(f"experiment create failed ({resp.status_code}): {resp.text}")


def poll_visible(
    base_url: str,
    workspace: str,
    session_ids: set[str],
    *,
    client: httpx.Client | _PollingClient | None = None,
    timeout_s: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
) -> set[str]:
    """Poll Intake's spans list until each ``session_id`` is queryable (or timeout).

    ClickHouse ingest is asynchronous, so freshly-POSTed ATIF is not immediately
    returned by the spans query. Polls until every id is seen or the deadline
    passes; returns the set confirmed visible. Transient GET errors are ignored
    (ingest lag), and ``sleep`` is injected so tests need not wait.
    """
    url = f"{base_url.rstrip('/')}/apis/intake/v2/workspaces/{workspace}/spans"
    owns_client = client is None
    active_client = client or httpx.Client(timeout=30.0)
    seen: set[str] = set()
    deadline = time.monotonic() + timeout_s
    try:
        while True:
            try:
                resp = active_client.get(url, params={"page": 1, "page_size": 1000, "mode": "summary"})
                if resp.status_code == 200:
                    for span in resp.json().get("data", []):
                        sid = span.get("session_id")
                        if sid in session_ids:
                            seen.add(sid)
            except httpx.HTTPError:
                pass
            if seen >= session_ids or time.monotonic() >= deadline:
                return seen
            sleep(2.0)
    finally:
        if owns_client:
            active_client.close()
