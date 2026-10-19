# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin discovery and API router for Studio web UI plugins."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from functools import cache
from importlib.metadata import Distribution
from pathlib import Path
from urllib.parse import unquote, urlparse

from fastapi import APIRouter
from nemo_helix_plugin.discovery import discover_entry_points, discover_manifests, discover_studio
from nemo_helix_plugin.interface import StudioSpec
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Plugin names are constrained to the same character set enforced by the
# frontend security gate: lowercase letter, followed by lowercase letters,
# digits, or hyphens.  This prevents path traversal or URL injection via a
# malicious/buggy plugin's StudioSpec.name value.
_PLUGIN_NAME_RE = re.compile(r"^[a-z][a-z0-9-]+$")


def _editable_source_root(dist: Distribution) -> Path | None:
    """Source directory of an editable install, per PEP 610's direct_url.json.

    For editable installs the dist-info sits in site-packages but the real
    package files live in a source tree elsewhere — ``dist.locate_file('.')``
    returns site-packages, so it's not a usable root for bundle validation.
    """
    try:
        raw = dist.read_text("direct_url.json")
        if not raw or not isinstance(raw, str):
            return None
        data = json.loads(raw)
        if not data.get("dir_info", {}).get("editable"):
            return None
        url = data.get("url", "")
        parsed = urlparse(url)
        if parsed.scheme != "file" or not parsed.path:
            return None
        return Path(unquote(parsed.path)).resolve()
    except Exception:
        return None


def _validate_bundle_path(ep_name: str, bundle_path: Path) -> bool:
    """Return True if *bundle_path* is safe to serve as a plugin bundle.

    Three checks are applied:

    1. The filename must be ``index.js`` — the advertised ``bundleUrl`` is
       always ``/plugin-ui/<name>/index.js``, so any other filename would
       validate here yet 404 in the browser.
    2. The resolved path must be a regular file (prevents accidentally serving
       system directories, e.g. when ``bundle_path`` points to ``/etc/passwd``
       its parent ``/etc`` would otherwise be exposed).
    3. Best-effort: the resolved path must be within the plugin's distribution
       root as reported by ``importlib.metadata``, or — for PEP 660 editable
       installs — within the source directory recorded in ``direct_url.json``.
       Skipped when distribution metadata is unavailable (e.g. in tests).
    """
    if bundle_path.name != "index.js":
        logger.warning(
            "Studio plugin %r bundle_path %r must be named index.js — skipping bundle",
            ep_name,
            bundle_path,
        )
        return False

    resolved = bundle_path.resolve()
    if not resolved.is_file():
        logger.warning(
            "Studio plugin %r bundle_path %r does not point to a regular file — skipping bundle",
            ep_name,
            bundle_path,
        )
        return False

    ep = discover_entry_points("nemo.studio").get(ep_name)
    dist = getattr(ep, "dist", None) if ep is not None else None
    if dist is not None:
        try:
            roots: list[Path] = [Path(dist.locate_file(".")).resolve()]
            editable_root = _editable_source_root(dist)
            if editable_root is not None:
                roots.append(editable_root)
            if not any(resolved.is_relative_to(root) for root in roots):
                logger.warning(
                    "Studio plugin %r bundle_path %r is outside distribution root(s) %s — skipping bundle",
                    ep_name,
                    bundle_path,
                    [str(r) for r in roots],
                )
                return False
        except Exception:
            logger.debug("Could not determine distribution root for plugin %r — skipping path check", ep_name)

    return True


@dataclass
class PluginManifestResponse:
    """Manifest returned by the /apis/plugins endpoint.

    Attributes:
        name: Plugin entry-point key, e.g. ``"example"``.
        bundle_url: URL served by the platform, e.g. ``"/plugin-ui/example/index.js"``.
            ``None`` for plugins that registered without a web bundle.
        bundle_dir: Parent directory of the plugin's built ``index.js``, for static
            file mounting.  ``None`` when there is no bundle or the directory could
            not be resolved.
    """

    name: str
    bundle_url: str | None = field(default=None)
    bundle_dir: Path | None = field(default=None)


class _PluginManifestOut(BaseModel):
    name: str
    bundleUrl: str | None = None


@cache
def discover_plugins() -> list[PluginManifestResponse]:
    """Discover all installed NHX plugins and return their Studio manifests.

    Every plugin found by ``discover_manifests()`` appears in the response.
    Plugins that also register a ``nemo.studio`` entry point get a
    ``bundleUrl`` pointing to their static JS bundle; all others get
    ``bundleUrl: null``.

    ``bundleUrl`` is always derived from the entry-point key — never from
    ``spec.name`` or ``spec.bundle_path`` — to prevent bundle-slot hijacking
    and path-traversal attacks.

    The result is cached so factories are called exactly once per process.
    """
    all_plugin_names = set(discover_manifests())

    # Studio-specific specs, keyed by entry-point name.
    studio_factories = discover_studio()
    studio_specs: dict[str, StudioSpec] = {}
    for name, factory in studio_factories.items():
        try:
            spec = factory()
            if not isinstance(spec, StudioSpec):
                logger.warning(
                    "Studio plugin %r returned %r instead of StudioSpec — skipping bundle",
                    name,
                    type(spec).__name__,
                )
                continue
            if spec.name != name:
                logger.warning(
                    "Studio plugin %r returned spec.name=%r — must match the entry-point key; skipping bundle",
                    name,
                    spec.name,
                )
                continue
            if not _PLUGIN_NAME_RE.match(name):
                logger.warning(
                    "Studio plugin %r has invalid entry-point key (must match [a-z][a-z0-9-]+) — skipping bundle",
                    name,
                )
                continue
            studio_specs[name] = spec
        except Exception:
            logger.warning("Failed to load studio spec for %r — no bundle will be served", name, exc_info=True)

    manifests: list[PluginManifestResponse] = []
    for name in sorted(all_plugin_names):
        spec = studio_specs.get(name)
        if spec is not None and spec.bundle_path is not None and _validate_bundle_path(name, spec.bundle_path):
            bundle_url: str | None = f"/plugin-ui/{name}/index.js"
            bundle_dir: Path | None = spec.bundle_path.resolve().parent
            logger.info("Registered studio plugin %r at %s", name, bundle_url)
        else:
            bundle_url = None
            bundle_dir = None
            logger.info("Registered plugin %r (no web bundle)", name)
        manifests.append(PluginManifestResponse(name=name, bundle_url=bundle_url, bundle_dir=bundle_dir))

    return manifests


def build_plugins_router(manifests: list[PluginManifestResponse]) -> APIRouter:
    """Build the FastAPI router for the /apis/plugins endpoint.

    Args:
        manifests: Pre-computed list of plugin manifests.  Pass the result of
            ``discover_plugins()`` at startup so discovery runs once.
    """
    router = APIRouter()
    _manifests_out = [_PluginManifestOut(name=m.name, bundleUrl=m.bundle_url) for m in manifests]

    @router.get("/apis/plugins", tags=["Studio Plugins"])
    async def list_plugins() -> list[_PluginManifestOut]:
        """List installed Studio web UI plugins."""
        return _manifests_out

    return router
