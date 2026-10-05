# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declarative provider seeding for AUT benchmark runs.

Reads a YAML manifest (``providers.yaml``) describing inference providers and
their secrets, then creates them on the platform via the typed NeMo clients.  An optional
``virtual_models`` section in the same manifest declares Switchyard VirtualModels
to create after provider discovery completes.

Usage from nat_runner::

    from seed_providers import seed_all, DEFAULT_MANIFEST
    result = seed_all(DEFAULT_MANIFEST, base_url="http://localhost:8080", workspace="default")
    if not result.ok:
        print(f"Seeding failed: {result.summary()}")

Standalone debugging::

    python seed_providers.py [--manifest providers.yaml] [--base-url http://localhost:8080]
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST = Path(__file__).resolve().parent / "providers.yaml"
DEFAULT_HTTP_TIMEOUT_SEC = 30


@dataclass
class ProviderSpec:
    """A single provider entry parsed from the manifest YAML."""

    name: str
    host_url: str
    secret_name: str
    from_env: str
    wait_for_discovery: bool = False
    discovery_timeout_sec: int = 120


@dataclass
class VirtualModelSpec:
    """A single virtual_models entry parsed from the manifest YAML."""

    name: str
    models: list[dict[str, str]]
    request_middleware: list[dict[str, Any]]
    response_middleware: list[dict[str, Any]]
    depends_on_provider: str | None = None


@dataclass
class ProviderSeedResult:
    """Outcome of seeding a single provider."""

    name: str
    status: str  # "ok", "skipped", "error"
    message: str = ""


@dataclass
class SeedResult:
    """Aggregate result of seeding all providers from a manifest."""

    providers: list[ProviderSeedResult] = field(default_factory=list)
    virtual_models: list[ProviderSeedResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        all_results = self.providers + self.virtual_models
        return all(p.status in ("ok", "skipped") for p in all_results)

    def summary(self) -> str:
        lines = []
        for p in self.providers:
            lines.append(f"  provider/{p.name}: {p.status}" + (f" ({p.message})" if p.message else ""))
        for v in self.virtual_models:
            lines.append(f"  virtual-model/{v.name}: {v.status}" + (f" ({v.message})" if v.message else ""))
        return "\n".join(lines)


def load_manifest(manifest_path: Path) -> list[ProviderSpec]:
    """Load and validate the provider manifest YAML.

    Returns the list of :class:`ProviderSpec` entries.  VirtualModel entries
    are parsed separately by :func:`_load_virtual_model_specs`.
    """
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "providers" not in raw:
        raise ValueError(f"Manifest {manifest_path} must have a top-level 'providers' key")

    specs = []
    for entry in raw["providers"]:
        specs.append(
            ProviderSpec(
                name=entry["name"],
                host_url=entry["host_url"],
                secret_name=entry["secret_name"],
                from_env=entry["from_env"],
                wait_for_discovery=entry.get("wait_for_discovery", False),
                discovery_timeout_sec=entry.get("discovery_timeout_sec", 120),
            )
        )
    return specs


def _load_virtual_model_specs(manifest_path: Path) -> list[VirtualModelSpec]:
    """Parse the optional ``virtual_models`` section from the manifest YAML."""
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    specs = []
    for entry in (raw or {}).get("virtual_models", []):
        specs.append(
            VirtualModelSpec(
                name=entry["name"],
                models=entry["models"],
                request_middleware=entry.get("request_middleware", []),
                response_middleware=entry.get("response_middleware", []),
                depends_on_provider=entry.get("depends_on_provider"),
            )
        )
    return specs


def _is_conflict(exc: Exception) -> bool:
    """Return True if the exception indicates a 409/already-exists conflict."""
    msg = str(exc).lower()
    return "409" in msg or "conflict" in msg or "already exists" in msg


def _create_secret(client: Any, workspace: str, secret_name: str, secret_value: str) -> None:
    """Create a platform secret, ignoring conflicts (already exists).

    On conflict, the existing secret is kept as-is. This is acceptable for
    ephemeral benchmark containers where secrets are always created fresh.
    For long-lived platform instances a stale secret (e.g. rotated key)
    would require manual deletion or an update-on-conflict strategy.
    """
    from nemo_helix_plugin.secrets.client import SecretsClient
    from nemo_helix_plugin.secrets.types import HelixSecretCreateRequest
    from pydantic import SecretStr

    secrets = SecretsClient.from_client(client)
    try:
        secrets.create_secret(
            body=HelixSecretCreateRequest(name=secret_name, value=SecretStr(secret_value)),
            workspace=workspace,
        )
        logger.info("Created secret '%s'", secret_name)
    except Exception as exc:
        if _is_conflict(exc):
            logger.info("Secret '%s' already exists, skipping", secret_name)
        else:
            raise


def _create_provider(client: Any, workspace: str, spec: ProviderSpec) -> None:
    """Create an inference provider, ignoring conflicts.

    Same ephemeral-container tradeoff as :func:`_create_secret` — a
    pre-existing provider with a stale host_url or secret_name is kept.
    """
    from nemo_helix_plugin.models.client import ModelsClient
    from nemo_helix_plugin.models.types import CreateModelProviderRequest

    try:
        ModelsClient.from_client(client).create_provider(
            workspace=workspace,
            body=CreateModelProviderRequest(
                name=spec.name,
                host_url=spec.host_url,
                api_key_secret_name=spec.secret_name,
            ),
        )
        logger.info("Created provider '%s' -> %s", spec.name, spec.host_url)
    except Exception as exc:
        if _is_conflict(exc):
            logger.info("Provider '%s' already exists, skipping", spec.name)
        else:
            raise


def _wait_for_provider_discovery(client: Any, workspace: str, spec: ProviderSpec) -> bool:
    """Poll until the provider has discovered at least one served model."""
    from nemo_helix_plugin.models.client import ModelsClient

    models = ModelsClient.from_client(client)
    deadline = time.monotonic() + spec.discovery_timeout_sec
    logger.info(
        "Waiting up to %ds for model discovery on provider '%s'...",
        spec.discovery_timeout_sec,
        spec.name,
    )
    while time.monotonic() < deadline:
        try:
            provider = models.get_provider(name=spec.name, workspace=workspace).data()
            served = getattr(provider, "served_models", None) or []
            if served:
                model_ids = [getattr(m, "model_entity_id", str(m)) for m in served]
                logger.info("Provider '%s' discovered models: %s", spec.name, model_ids)
                return True
        except Exception as exc:
            logger.debug("Discovery poll for '%s' failed: %s", spec.name, exc)
        time.sleep(3.0)
    logger.warning("Provider '%s' did not discover any models within %ds", spec.name, spec.discovery_timeout_sec)
    return False


def _create_virtual_model(base_url: str, workspace: str, spec: VirtualModelSpec) -> None:
    """Create a VirtualModel via the entities REST API, ignoring conflicts.

    Uses urllib (stdlib) so this helper has no client dependency beyond the
    manifest. Auth is omitted intentionally: local benchmark platforms run with auth
    disabled (NHX_SECRETS_ALLOW_KEY_CREATION=1, no auth service in the
    services list).
    """
    url = f"{base_url.rstrip('/')}/apis/entities/v2/workspaces/{workspace}/entities/virtual_model"
    body = {
        "name": spec.name,
        "data": {
            "default_model_entity": None,
            "models": spec.models,
            "request_middleware": spec.request_middleware,
            "response_middleware": spec.response_middleware,
            "post_response_middleware": [],
            "override_proxy": None,
            "project": None,
        },
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=DEFAULT_HTTP_TIMEOUT_SEC) as resp:
            result = json.loads(resp.read())
            logger.info("Created VirtualModel '%s' (id=%s)", spec.name, result.get("id", "?"))
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode(errors="replace")
        if exc.code == 409 or "already exists" in body_text.lower() or "conflict" in body_text.lower():
            logger.info("VirtualModel '%s' already exists, skipping", spec.name)
        else:
            raise RuntimeError(f"Failed to create VirtualModel '{spec.name}': HTTP {exc.code} {body_text}") from exc


def seed_all(
    manifest_path: Path,
    base_url: str,
    workspace: str = "default",
) -> SeedResult:
    """Seed all providers and VirtualModels described in the manifest.

    For each provider entry:
    1. Read the env var named by ``from_env``; skip with a warning if unset.
    2. Create a platform secret with the env var's value.
    3. Create an inference provider pointing at the secret.
    4. Optionally wait for model discovery.

    For each virtual_models entry (after all providers are done):
    1. Skip if its ``depends_on_provider`` was itself skipped or errored.
    2. Create the VirtualModel via the entities REST API.

    Returns a :class:`SeedResult` with per-provider and per-vm status.
    """
    from nemo_helix_plugin.client.client import NemoClient

    provider_specs = load_manifest(manifest_path)
    vm_specs = _load_virtual_model_specs(manifest_path)
    client = NemoClient(base_url=base_url, workspace=workspace)
    result = SeedResult()

    # Track which providers succeeded so VMs can gate on their dependency.
    provider_ok: set[str] = set()

    for spec in provider_specs:
        secret_value = os.environ.get(spec.from_env)
        if not secret_value:
            msg = f"Env var {spec.from_env} not set; skipping provider '{spec.name}'"
            logger.warning(msg)
            result.providers.append(ProviderSeedResult(name=spec.name, status="skipped", message=msg))
            continue

        try:
            _create_secret(client, workspace, spec.secret_name, secret_value)
            _create_provider(client, workspace, spec)

            if spec.wait_for_discovery:
                discovered = _wait_for_provider_discovery(client, workspace, spec)
                if not discovered:
                    msg = f"Model discovery timed out after {spec.discovery_timeout_sec}s"
                    logger.warning(msg)
                    result.providers.append(ProviderSeedResult(name=spec.name, status="error", message=msg))
                    continue

            result.providers.append(ProviderSeedResult(name=spec.name, status="ok"))
            provider_ok.add(spec.name)
        except Exception as exc:
            msg = f"Failed to seed provider '{spec.name}': {exc}"
            logger.error(msg)
            result.providers.append(ProviderSeedResult(name=spec.name, status="error", message=msg))

    for vm_spec in vm_specs:
        dep = vm_spec.depends_on_provider
        if dep and dep not in provider_ok:
            msg = f"Skipping VirtualModel '{vm_spec.name}': depends_on_provider '{dep}' was not seeded successfully"
            logger.warning(msg)
            result.virtual_models.append(ProviderSeedResult(name=vm_spec.name, status="skipped", message=msg))
            continue

        try:
            _create_virtual_model(base_url, workspace, vm_spec)
            result.virtual_models.append(ProviderSeedResult(name=vm_spec.name, status="ok"))
        except Exception as exc:
            msg = f"Failed to create VirtualModel '{vm_spec.name}': {exc}"
            logger.error(msg)
            result.virtual_models.append(ProviderSeedResult(name=vm_spec.name, status="error", message=msg))

    return result


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Seed inference providers from a YAML manifest")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Path to providers.yaml")
    parser.add_argument("--base-url", default="http://localhost:8080", help="NeMo Helix base URL")
    parser.add_argument("--workspace", default="default", help="NeMo Helix workspace")
    args = parser.parse_args()

    result = seed_all(args.manifest, base_url=args.base_url, workspace=args.workspace)
    print(f"\nSeeding result (ok={result.ok}):")
    print(result.summary())
    if not result.ok:
        raise SystemExit(1)
