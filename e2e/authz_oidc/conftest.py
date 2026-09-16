# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for the authz OIDC E2E harness.

Spawns dedicated ``nemo services run`` instances (fresh tmp data dir, free
port, this checkout's code) configured for **native OIDC only**:
``auth.enabled=true``, ``oidc.enabled=true`` pointing at the in-harness
issuer, and — critically — ``allow_unsigned_jwt=false``, so every identity in
the matrix is established by a real RS256-signed JWT. ``X-NHX-Principal-*``
headers are never sent; provisioning itself authenticates with a signed JWT
whose ``sub`` is ``service:e2e-harness``.

Two platform phases (both lazy, session-scoped):

- ``platform`` — ``on_invalid_plugin=deny_route``. The harness deliberately loads
  broken/unruled fixture plugins, so it pins per-route fencing rather than inheriting
  the strict ``hard_fail`` default, which would abort the bundle and wedge the platform.
- ``platform_knobs`` — ``on_invalid_plugin=quarantine``.

Run: ``pytest e2e/authz_oidc -v --run-e2e`` (see README.md).
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import time
from collections.abc import Generator, Iterator
from contextlib import closing
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import ADMIN_EMAIL, WS_A, WS_B, Platform  # noqa: E402
from idp import DEFAULT_AUDIENCE, MiniOIDCIssuer  # noqa: E402
from report import ReportCollector  # noqa: E402

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_PLATFORM_CONFIG = _REPO_ROOT / "packages/nhx_platform/config/local.yaml"
_FIXTURE_PLUGINS = ["harness-fixture", "harness-unruled", "harness-broken"]

_HEALTH_TIMEOUT = 180
_PROVISION_TIMEOUT = 120
_POLL = 1.0


# --------------------------------------------------------------------------- #
# Issuer                                                                       #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def issuer() -> Iterator[MiniOIDCIssuer]:
    idp = MiniOIDCIssuer()
    url = idp.start()
    logger.info("Mini OIDC issuer serving at %s", url)
    yield idp
    idp.stop()


# --------------------------------------------------------------------------- #
# Platform process                                                             #
# --------------------------------------------------------------------------- #


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _install_missing_fixture_plugins() -> list[str]:
    """Editable-install any not-yet-present fixture plugins into the active venv.

    Returns the names actually installed (empty if all were already present), so the session
    teardown removes exactly what this run added.
    """
    from importlib.metadata import entry_points

    installed = {ep.name for ep in entry_points(group="nemo.services")}
    missing = [p for p in _FIXTURE_PLUGINS if p not in installed]
    if not missing:
        return []
    pip_specs = [f"-e{_HERE / 'fixtures' / p}" for p in missing]
    subprocess.run(
        ["uv", "pip", "install", "--python", sys.executable, *pip_specs],
        check=True,
        capture_output=True,
        timeout=300,
        cwd=_REPO_ROOT,
    )
    logger.info("Installed fixture plugins: %s", ", ".join(missing))
    return missing


def _uninstall_fixture_plugins(names: list[str]) -> None:
    """Best-effort removal of fixture plugins from the active venv.

    Teardown must never fail the session, but a leak must stay visible — so a non-zero uninstall
    is warned, not raised.
    """
    if not names:
        return
    result = subprocess.run(
        ["uv", "pip", "uninstall", "--python", sys.executable, *names],
        capture_output=True,
        timeout=300,
        cwd=_REPO_ROOT,
    )
    if result.returncode != 0:
        logger.warning(
            "Failed to uninstall fixture plugins %s (rc=%s): %s",
            ", ".join(names),
            result.returncode,
            result.stderr.decode(errors="replace")[:500],
        )
    else:
        logger.info("Uninstalled fixture plugins: %s", ", ".join(names))


def _platform_env(issuer_url: str, base_url: str, data_dir: Path, extra: dict[str, str]) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NHX_", "DATABASE_"))}
    env.update(
        {
            "NHX_CONFIG_FILE_PATH": str(_PLATFORM_CONFIG),
            "NHX_CONFIG_WARNINGS_DISABLED": "1",
            "NHX_BASE_URL": base_url,
            "NHX_DATA_DIR": str(data_dir),
            "NHX_SEED_ON_STARTUP": "true",
            # Authz rows need auth seeding, but the default model-provider seed
            # requires a real NGC_API_KEY and is unrelated to this matrix.
            "NHX_PLATFORM_SEED_MODEL_PROVIDER_ENABLED": "false",
            "NHX_AUTH_ENABLED": "true",
            "NHX_AUTH_ALLOW_UNSIGNED_JWT": "false",  # defaults are true; signed JWTs only
            "NHX_AUTH_POLICY_DECISION_POINT_BASE_URL": base_url,
            "NHX_AUTH_OIDC__ENABLED": "true",
            "NHX_AUTH_OIDC__ISSUER": issuer_url,
            "NHX_AUTH_OIDC__AUDIENCE": DEFAULT_AUDIENCE,
            "NHX_AUTH_ADMIN_EMAIL": ADMIN_EMAIL,
            # bundle_cache_seconds must stay NONZERO: at 0 every PDP eval
            # rebuilds policy data, and degraded fixture plugins are never
            # cached — each eval then re-runs full plugin derivation
            # and entity paging, blowing the 5s PDP timeout platform-wide.
            # Fast background refresh + settle-probes handle propagation.
            "NHX_AUTH_BUNDLE_CACHE_SECONDS": "5",
            "NHX_AUTH_POLICY_DATA_REFRESH_INTERVAL": "2",
            # FINDING (harness-discovered): branch rego exceeds the default
            # embedded-PDP fuel budget (100M; config docstring says typical
            # evals are 20-25M) once seeded principal data is loaded — every
            # request 502s. Raised here to unblock; flagged for the branch.
            "NHX_AUTH_EMBEDDED_PDP_CPU_LIMIT": "2000",
        }
    )
    env.update(extra)
    return env


def _spawn_platform(
    issuer: MiniOIDCIssuer,
    tmp_path_factory: pytest.TempPathFactory,
    label: str,
    extra_env: dict[str, str],
) -> Generator[Platform, None, None]:
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    work = tmp_path_factory.mktemp(f"authz-e2e-{label}")
    data_dir = work / "data"
    data_dir.mkdir()
    log_path = work / "services.log"

    nemo_bin = Path(sys.executable).parent / "nemo"
    args = [
        str(nemo_bin),
        "services",
        "run",
        "--service-group",
        "all",
        "--controller-group",
        "all",
        "--port",
        str(port),
    ]
    env = _platform_env(issuer.issuer_url, base_url, data_dir, extra_env)

    logger.info("Spawning platform [%s] on %s (log: %s)", label, base_url, log_path)
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(args, stdout=log_file, stderr=subprocess.STDOUT, env=env)
        try:
            platform = Platform(base_url=base_url, issuer=issuer, log_path=log_path)
            _wait_healthy(platform)
            yield platform
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


def _wait_healthy(platform: Platform) -> None:
    deadline = time.monotonic() + _HEALTH_TIMEOUT
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{platform.base_url}/health/ready", timeout=2.0).status_code == 200:
                break
        except httpx.RequestError:
            pass
        time.sleep(_POLL)
    else:
        pytest.fail(f"Platform on {platform.base_url} not healthy within {_HEALTH_TIMEOUT}s; log: {platform.log_path}")

    # Seeding runs as a startup task and may lag /health/ready: wait until the
    # admin's PlatformAdmin binding is live via the OIDC path itself.
    admin_token = platform.token("admin")
    deadline = time.monotonic() + _PROVISION_TIMEOUT
    while time.monotonic() < deadline:
        resp = platform.request("GET", "/apis/entities/v2/workspaces", token=admin_token)
        if resp.status_code == 200:
            return
        time.sleep(_POLL)
    pytest.fail(f"Admin OIDC token not authorized within {_PROVISION_TIMEOUT}s; log: {platform.log_path}")


# --------------------------------------------------------------------------- #
# Provisioning (signed service JWT only — no principal headers anywhere)       #
# --------------------------------------------------------------------------- #


def _provision(platform: Platform) -> None:
    """Create workspaces + explicit role bindings; revoke the seeded wildcard.

    Everything authenticates as ``service:e2e-harness`` via a signed JWT: the
    IAM role-binding endpoints are service-principal-only at the handler, and
    a Bearer token whose ``sub`` starts with ``service:`` satisfies that
    end-to-end (middleware builds Principal.id straight from the sub claim).
    """
    token = platform.token("provisioner")

    def call(method: str, path: str, body: dict | None = None) -> httpx.Response:
        resp = platform.request(method, path, token=token, body=body)
        if resp.status_code >= 500:
            raise AssertionError(f"provisioning {method} {path} -> {resp.status_code}: {resp.text[:300]}")
        return resp

    # 1. Revoke the seeded wildcard '*' -> Viewer@system binding. With it in
    #    place every authenticated user holds all .read/.list permissions in
    #    the system workspace, which would make the no-workspace permission-deny rows untestable.
    #    Revocation must go through the generic entities API: both dedicated
    #    revocation endpoints are broken for this binding (IAM DELETE looks it
    #    up in the 'default' workspace; members DELETE filters on a
    #    data.workspace key that binding entities don't carry) — see report.
    entity_path = "/apis/entities/v2/workspaces/system/entities/role_binding/wildcard-system-viewer"
    entity = call("GET", entity_path)
    assert entity.status_code == 200, f"seeded wildcard binding not found -> {entity.status_code}: {entity.text[:300]}"
    payload = entity.json()
    payload["data"]["revoked_at"] = "2026-01-01T00:00:00Z"
    resp = call("PUT", entity_path, {"name": payload["name"], "data": payload["data"]})
    assert resp.status_code == 200, f"wildcard revoke -> {resp.status_code}: {resp.text[:300]}"
    logger.info("Revoked seeded wildcard-system-viewer binding via entities API")

    # 2. Workspaces (creator auto-Admin binding is keyed to the service sub; harmless).
    for ws in (WS_A, WS_B):
        resp = call("POST", "/apis/entities/v2/workspaces", {"name": ws})
        assert resp.status_code == 201, f"workspace {ws} -> {resp.status_code}: {resp.text[:300]}"

    # 3. Explicit bindings (wait_role_propagation defaults to true -> synchronous).
    for principal, workspace, role in (
        ("alice@harness.test", WS_A, "Editor"),
        ("victor@harness.test", WS_A, "Viewer"),
        ("sam@harness.test", "system", "Viewer"),
    ):
        resp = call(
            "POST", "/apis/auth/v2/iam/role-bindings", {"principal": principal, "workspace": workspace, "role": role}
        )
        assert resp.status_code in (200, 201), (
            f"binding {principal}/{role}@{workspace} -> {resp.status_code}: {resp.text[:300]}"
        )

    # 4. Settle probe: alice's Editor binding effective AND the wildcard
    #    revocation propagated (alice must now be denied on the no-workspace permission route).
    alice = platform.token("alice")
    deadline = time.monotonic() + _PROVISION_TIMEOUT
    while time.monotonic() < deadline:
        ok = platform.request("GET", f"/apis/auditor/v2/workspaces/{WS_A}/targets", token=alice).status_code == 200
        revoked = platform.request("GET", "/apis/entities/v2/workspaces", token=alice).status_code == 403
        if ok and revoked:
            return
        time.sleep(_POLL)
    raise AssertionError("Provisioned bindings did not settle (alice 200 on wsA targets + 403 on workspaces list)")


# --------------------------------------------------------------------------- #
# Session fixtures                                                             #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def fixture_plugins() -> Iterator[None]:
    """Install the harness fixture plugins for the session, then remove them.

    The fixtures register deliberately broken/unruled ``nemo.services`` entry points. Left
    installed in the shared venv, a later ordinary ``nemo services run`` in this checkout would
    discover them and — under the ``on_invalid_plugin=hard_fail`` default — refuse to build the
    OPA bundle, wedging the platform for unrelated work (``e2e/services_pool.py`` fences its own
    children against exactly this, but a plain CLI run has no such fence). So we uninstall on
    teardown: both what we installed and any stragglers a previously-aborted run leaked, enforcing
    the invariant that the fixtures never outlive this suite.
    """
    from importlib.metadata import entry_points

    installed = _install_missing_fixture_plugins()
    try:
        yield
    finally:
        present = {ep.name for ep in entry_points(group="nemo.services")}
        to_remove = sorted(set(installed) | {p for p in _FIXTURE_PLUGINS if p in present})
        _uninstall_fixture_plugins(to_remove)


@pytest.fixture(scope="session")
def platform(
    issuer: MiniOIDCIssuer, tmp_path_factory: pytest.TempPathFactory, fixture_plugins: None
) -> Iterator[Platform]:
    """deny_route-knob platform, fully provisioned.

    Pins ``deny_route`` explicitly: the harness installs broken/unruled fixture
    plugins on purpose, so it opts out of the strict ``hard_fail`` default (which
    would abort bundle generation and leave the platform degraded) to exercise
    per-route fencing on a running platform.
    """
    gen = _spawn_platform(issuer, tmp_path_factory, "default", {"NHX_AUTH_ON_INVALID_PLUGIN": "deny_route"})
    with closing(gen):
        p = next(gen)
        _provision(p)
        yield p


@pytest.fixture(scope="session")
def platform_knobs(
    issuer: MiniOIDCIssuer, tmp_path_factory: pytest.TempPathFactory, fixture_plugins: None
) -> Iterator[Platform]:
    """Quarantine-knob platform (no extra provisioning)."""
    gen = _spawn_platform(
        issuer,
        tmp_path_factory,
        "knobs",
        {"NHX_AUTH_ON_INVALID_PLUGIN": "quarantine"},
    )
    with closing(gen):
        p = next(gen)
        yield p


# --------------------------------------------------------------------------- #
# Audit report                                                                 #
# --------------------------------------------------------------------------- #


_REPORT_KEY = pytest.StashKey[ReportCollector]()


@pytest.fixture(scope="session")
def report(request: pytest.FixtureRequest) -> ReportCollector:
    collector = ReportCollector()
    request.session.stash[_REPORT_KEY] = collector
    return collector


def pytest_sessionfinish(session: pytest.Session) -> None:
    collector = session.stash.get(_REPORT_KEY, None)
    if collector and collector.rows:
        out = _HERE / "AUTHZ_E2E_REPORT.md"
        out.write_text(collector.render())
        json_out = _HERE / "AUTHZ_E2E_REPORT.json"
        json_out.write_text(json.dumps(collector.as_json(), indent=2))
        print(f"\nAuthz E2E audit report: {out} ({len(collector.rows)} cases)")
