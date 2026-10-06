# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-if-missing provisioning for the Fabric sandbox image.

``FabricAgentRuntime``'s sandbox mode needs a container image with Fabric + its harness adapters. Rather than
make callers hand-write a Dockerfile, the SDK owns the recipe (:mod:`sandbox.Dockerfile`) and
provisions the image opaquely: :func:`ensure_fabric_image` returns a usable image tag, building it
only when it isn't already present locally. This mirrors the ``ensure_task_image`` build-if-missing
pattern (``docker image inspect`` → ``docker build``).

The image installs published wheels, so the build needs no source checkout and no build context
beyond its own pins. It is pinned to the ``nemo-fabric`` version installed alongside this SDK,
so the sandbox cannot run a Fabric that disagrees with the config the host composes for it.
The Pi harness adapter is an npm package (not a wheel), so the image also bakes Node and the
adapter's pinned npm dependencies; its descriptor is rewritten into the well-known
``share/nemo-fabric/adapters`` tree, so Fabric discovers it exactly like the wheel adapters.

The tag is content-addressed on the recipe + version + adapters, so any of those changing produces a
new tag (cache-bust) and an unchanged recipe reuses the cached image.

This is the local-Docker provisioning path. The intended evolution is a remote image registry as a
cache: :func:`ensure_fabric_image` keeps the same "return a usable tag" contract, its body swapping
local build for a registry pull (build-and-push on miss).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# ``localhost/`` prefix so Docker treats it as an explicit local registry and does NOT qualify the tag
# to ``docker.io/…`` — this image is built locally and never pushed to Docker Hub.
DEFAULT_FABRIC_IMAGE_REPO = "localhost/nemo-evaluator/fabric-sandbox"
FABRIC_DISTRIBUTION = "nemo-fabric"
_DOCKERFILE = Path(__file__).with_name("sandbox.Dockerfile")

# Bound the docker subprocess calls so an unresponsive daemon fails fast instead of hanging the runtime.
_INSPECT_TIMEOUT_S = 30
_BUILD_TIMEOUT_S = 900

#: Adapter distributions baked into the (single, harness-agnostic) Fabric image, pinned to the same
#: version as ``nemo-fabric``. Codex and claude are absent: they need their own CLIs, which this image
#: does not provision.
_ADAPTER_DISTRIBUTIONS: tuple[str, ...] = ("nemo-fabric-adapters-hermes",)

#: The Pi harness adapter is an npm package, versioned with ``nemo-fabric`` like the wheel adapters.
_PI_ADAPTER_NPM_PACKAGE = "nemo-fabric-adapters-pi"

#: Exact pins for the Pi SDK peers, keyed by the ``nemo-fabric`` version they pair with. Each adapter
#: release declares the Pi SDK range it supports (``^0.84.2`` for the 0.3.0 adapter, ``^0.86.0`` for
#: 0.4.0); we pin the range floor exactly, because the tag is a digest of these pins. Keep this table
#: in sync with the adapter's peerDependencies when bumping the Fabric version.
_PI_SDK_PINS_BY_FABRIC_VERSION: dict[str, tuple[str, ...]] = {
    "0.3.0": ("@earendil-works/pi-ai==0.84.2", "@earendil-works/pi-coding-agent==0.84.2"),
    "0.4.0": ("@earendil-works/pi-ai==0.86.0", "@earendil-works/pi-coding-agent==0.86.0"),
}

#: Requirements the harness needs that are not versioned with Fabric. ``hermes-agent`` is the harness
#: itself, which no Fabric extra pulls in; ``tomli-w`` is what the adapter writes Relay's plugin config
#: with, and without it the harness fails at start with ``tomli_w is not installed``. Pinned exactly,
#: not compatible-released: the tag is a digest of these lines, so a requirement free to move would
#: let the same tag name different image contents and be reused from cache.
_HARNESS_REQUIREMENTS: tuple[str, ...] = ("hermes-agent==0.19.0", "tomli-w==1.2.0")


class FabricImageError(RuntimeError):
    """Raised when the Fabric sandbox image cannot be provisioned."""


def fabric_version() -> str:
    """The installed ``nemo-fabric`` version, which the image is pinned to."""
    try:
        return importlib.metadata.version(FABRIC_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError as exc:
        raise FabricImageError(
            f"{FABRIC_DISTRIBUTION} is not installed, so the sandbox image has no version to pin to. "
            "Install the SDK's `fabric` extra, or pass an already-built image."
        ) from exc


def _requirements(version: str) -> list[str]:
    return [
        f"nemo-fabric[relay]=={version}",
        *(f"{distribution}=={version}" for distribution in _ADAPTER_DISTRIBUTIONS),
        *_HARNESS_REQUIREMENTS,
    ]


def _pi_sdk_pins(version: str) -> tuple[str, ...]:
    """The Pi SDK peer pins for the given ``nemo-fabric`` version.

    Raises :class:`FabricImageError` for a version with no known pins: a guessed pin could satisfy the
    adapter at build time yet break the harness at runtime, and the tag is a digest of these pins.
    """
    try:
        return _PI_SDK_PINS_BY_FABRIC_VERSION[version]
    except KeyError:
        raise FabricImageError(
            f"no known Pi SDK pins for nemo-fabric {version}; add a mapping to "
            "_PI_SDK_PINS_BY_FABRIC_VERSION (check nemo-fabric-adapters-pi's peerDependencies)"
        ) from None


def _pi_package_json(version: str) -> str:
    """The npm package.json for the image's Pi adapter install, every dependency pinned exactly."""
    dependencies = {_PI_ADAPTER_NPM_PACKAGE: version}
    for pin in _pi_sdk_pins(version):
        name, _, pinned = pin.partition("==")
        dependencies[name] = pinned
    return (
        json.dumps(
            {"name": "nemo-evaluator-fabric-pi-adapter", "private": True, "dependencies": dependencies}, indent=2
        )
        + "\n"
    )


def fabric_image_tag(*, repo: str = DEFAULT_FABRIC_IMAGE_REPO, version: str | None = None) -> str:
    """Content-addressed tag for the harness-agnostic Fabric image: ``<repo>:<digest>``.

    Not keyed by harness: one Fabric install plus the bundled adapters runs any of them, so the image
    is the same regardless of which harness a task's config selects.
    """
    resolved = version or fabric_version()
    recipe = (
        _DOCKERFILE.read_bytes()
        + "\n".join(_requirements(resolved)).encode("utf-8")
        + _pi_package_json(resolved).encode("utf-8")
    )
    return f"{repo}:{hashlib.sha256(recipe).hexdigest()[:12]}"


def image_exists(tag: str, *, docker_bin: str = "docker") -> bool:
    """Whether an image tag is present in the local Docker image store.

    Raises :class:`FabricImageError` when the Docker daemon is unreachable, so a stopped/misconfigured
    daemon surfaces as a clear error instead of masquerading as "image absent" and triggering a build
    that then also fails confusingly.
    """
    try:
        result = subprocess.run(
            [docker_bin, "image", "inspect", tag], capture_output=True, check=False, timeout=_INSPECT_TIMEOUT_S
        )
    except subprocess.TimeoutExpired as exc:
        raise FabricImageError(
            f"`docker image inspect` timed out after {_INSPECT_TIMEOUT_S}s (daemon unresponsive?)"
        ) from exc
    if result.returncode == 0:
        return True
    stderr = result.stderr.decode("utf-8", errors="replace")
    if "cannot connect to the docker daemon" in stderr.lower():
        raise FabricImageError(f"cannot reach the Docker daemon (is it running?): {stderr.strip()}")
    logger.debug("Fabric image %s not present in local store", tag)
    return False


def ensure_fabric_image(*, docker_bin: str = "docker", force_build: bool = False) -> str:
    """Return a usable Fabric image tag, building it only if not already present.

    One harness-agnostic image serves every bundled harness. Idempotent and content-addressed: an
    unchanged recipe reuses the cached image; a changed recipe or Fabric version yields a new tag.
    """
    version = fabric_version()
    tag = fabric_image_tag(version=version)
    if not force_build and image_exists(tag, docker_bin=docker_bin):
        logger.debug("Fabric image %s already present; skipping build.", tag)
        return tag

    logger.info("Building Fabric image...", extra=dict(tag=tag, fabric_version=version))
    with tempfile.TemporaryDirectory(prefix="nemo-fabric-image-") as ctx_dir:
        ctx = Path(ctx_dir)
        shutil.copy2(_DOCKERFILE, ctx / "Dockerfile")
        (ctx / "requirements.txt").write_text("\n".join(_requirements(version)) + "\n", encoding="utf-8")
        (ctx / "pi-package.json").write_text(_pi_package_json(version), encoding="utf-8")
        try:
            subprocess.run(
                [docker_bin, "build", "-t", tag, str(ctx)],
                check=True,
                env={**os.environ, "DOCKER_BUILDKIT": "1"},
                timeout=_BUILD_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as exc:
            raise FabricImageError(f"docker build timed out after {_BUILD_TIMEOUT_S}s for {tag}") from exc
        except subprocess.CalledProcessError as exc:
            raise FabricImageError(f"docker build failed for {tag}: {exc}") from exc
    logger.info("Built Fabric image %s.", tag, extra=dict(tag=tag))
    return tag
