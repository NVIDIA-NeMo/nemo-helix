# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed, hash-verified assets for explicitly trusted Studio plugins."""

from __future__ import annotations

import hashlib
import hmac
import os
import tempfile
from pathlib import Path

from nemo_helix_plugin.community_plugins.registry import ExternalPlugin, RegistryStore, canonical_json

MAX_BUNDLE_BYTES = 16 * 1024 * 1024


class AssetStore:
    def __init__(self, registry: RegistryStore) -> None:
        self.registry = registry
        self.root = registry.path.parent / "bundles"

    def put(self, plugin: ExternalPlugin, data: bytes) -> None:
        bundle = plugin.bundle
        if (
            bundle is None
            or len(data) > MAX_BUNDLE_BYTES
            or not hmac.compare_digest(hashlib.sha256(data).hexdigest(), bundle.sha256)
        ):
            raise ValueError("Plugin bundle hash or size validation failed")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.root, delete=False) as output:
                temporary = Path(output.name)
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.root / f"{bundle.sha256}.js")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def read(self, name: str, revision: str, filename: str) -> tuple[bytes, str]:
        plugin = next((p for p in self.registry.load().plugins if p.name == name and p.enabled), None)
        if plugin is None or plugin.bundle is None or plugin.bundle.revision != revision:
            raise FileNotFoundError("Plugin asset is not installed or enabled")
        if filename == "plugin.manifest.json":
            return canonical_json(plugin.bundle.sidecar), "application/json"
        if filename != "index.js":
            raise FileNotFoundError("Unknown plugin asset")
        with (self.root / f"{plugin.bundle.sha256}.js").open("rb") as stream:
            data = stream.read(MAX_BUNDLE_BYTES + 1)
        if len(data) > MAX_BUNDLE_BYTES or not hmac.compare_digest(
            hashlib.sha256(data).hexdigest(), plugin.bundle.sha256
        ):
            raise ValueError("Installed bundle failed integrity verification")
        return data, "text/javascript"
