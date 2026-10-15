# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the agent bundle that ``nemo agents package`` bakes into an image, straight from its registry.

The packaged Dockerfile copies the whole agent directory into the image in one
``COPY ./ <dir>`` layer and points ``AGENT_CONFIG_PATH`` at the ``agent.yaml`` inside it.
Only that layer is downloaded, so no Docker daemon or full image pull is needed.

The image reference comes from an authenticated user and the registry answers are
untrusted, so every host this module talks to (registry, token service, redirect
targets) must be on the registry allow-list or resolve to a public address, every
body is read through a byte cap, and the layer is inflated through a byte cap.
"""

from __future__ import annotations

import asyncio
import gzip
import io
import ipaddress
import json
import logging
import os
import posixpath
import re
import socket
import tarfile
import zlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx
from nemo_agents_plugin.entities import AGENT_CONFIG_FILENAME
from nemo_helix_plugin.container_registry import (
    DOCKER_HUB_REGISTRY,
    ImageReference,
    ImageReferenceError,
    parse_challenges,
    parse_image_reference,
)

LOGGER = logging.getLogger(__name__)

IMAGE_REGISTRIES_ENV = "NEMO_AGENT_IMAGE_REGISTRIES"
ANY_REGISTRY = "*"
DEFAULT_IMAGE_REGISTRIES = frozenset(
    {DOCKER_HUB_REGISTRY, "ghcr.io", "nvcr.io", "quay.io", "public.ecr.aws", "gcr.io", "mcr.microsoft.com"}
)

_MANIFEST_ACCEPT = ", ".join(
    [
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ]
)
_PREFERRED_PLATFORM = ("linux", "amd64")

MAX_BUNDLE_LAYER_BYTES = 64 * 1024 * 1024
MAX_INFLATED_LAYER_BYTES = 256 * 1024 * 1024
MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_BUNDLE_FILES = 1_000
MAX_BUNDLE_BYTES = 8 * 1024 * 1024
_MAX_REDIRECTS = 3

# Written into the build context by `nemo agents package`, not part of the agent.
_BUILD_ARTIFACT_NAMES = frozenset({"Dockerfile.generated", "Dockerfile.generated.dockerignore", ".dockerignore"})
_BUILD_ARTIFACT_SUFFIXES = (".whl",)

_GZIP_MAGIC = b"\x1f\x8b"
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"

_BUILDKIT_COPY = re.compile(r"^(?:COPY|ADD)\b(.*?)(?:\s+#\s*buildkit)?$")
_LEGACY_COPY = re.compile(r"#\(nop\)\s+(?:COPY|ADD)\s+\S+\s+in\s+(\S+)\s*$")
_WORKDIR = re.compile(r"(?:^|#\(nop\)\s+)WORKDIR\s+(\S+)")

_INTERNAL_HOST_SUFFIXES = (".localhost", ".local", ".internal", ".svc", ".cluster.local")
# Ranges is_global misses or that are never a public registry.
_BLOCKED_NETWORKS = tuple(
    ipaddress.ip_network(cidr)
    for cidr in ("0.0.0.0/8", "100.64.0.0/10", "169.254.0.0/16", "192.0.0.0/24", "::ffff:0:0/96", "64:ff9b::/96")
)


class ImageBundleError(ValueError):
    """The image cannot supply an agent bundle."""


class ImageNotFoundError(ImageBundleError):
    """The registry has no such image."""


class ImageAccessDeniedError(ImageBundleError):
    """The registry refused anonymous access to the image."""


@dataclass(frozen=True)
class ExtractedBundle:
    """The agent directory pulled out of an image layer, keyed by path relative to that directory."""

    config_path: str
    files: dict[str, bytes]


def allowed_registries() -> frozenset[str]:
    """Registries agents may be read from; ``*`` allows any registry on a public address."""
    raw = os.environ.get(IMAGE_REGISTRIES_ENV)
    if raw is None:
        return DEFAULT_IMAGE_REGISTRIES
    return frozenset(entry.strip().lower() for entry in raw.split(",") if entry.strip())


def _assert_allowed_registry(registry: str) -> None:
    allowed = allowed_registries()
    if ANY_REGISTRY in allowed or registry in allowed:
        return
    raise ImageBundleError(
        f"Registry {registry} is not an allowed image registry. "
        f"Set {IMAGE_REGISTRIES_ENV} to a comma-separated list of registries to allow it."
    )


def _host_of(authority: str) -> str:
    host = urlsplit(f"//{authority}").hostname
    if not host:
        raise ImageBundleError(f"'{authority}' is not a registry host.")
    return host.rstrip(".")


def _resolve_host(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    return [ipaddress.ip_address(info[4][0]) for info in infos]


def _is_public(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    if not address.is_global or address.is_multicast:
        return False
    return not any(address in network for network in _BLOCKED_NETWORKS)


async def assert_public_host(host: str) -> None:
    """Refuse hosts that are, or resolve to, anything other than a public address."""
    bare = host.strip("[]").lower().rstrip(".")
    private = ImageBundleError(f"Host {host} is on a private network and cannot be read.")
    if not bare or bare == "localhost" or bare.endswith(_INTERNAL_HOST_SUFFIXES):
        raise private
    try:
        addresses: Iterable[ipaddress.IPv4Address | ipaddress.IPv6Address] = [ipaddress.ip_address(bare)]
    except ValueError:
        if "." not in bare:
            raise private from None
        try:
            addresses = await asyncio.to_thread(_resolve_host, bare)
        except OSError as exc:
            raise ImageBundleError(f"Host {host} could not be resolved.") from exc
    addresses = list(addresses)
    if not addresses or not all(_is_public(address) for address in addresses):
        raise private


async def _assert_public_https_url(url: str, *, what: str) -> None:
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname:
        raise ImageBundleError(f"The registry's {what} is not an https URL.")
    await assert_public_host(parts.hostname)


def _json(body: bytes, *, what: str) -> dict[str, Any]:
    try:
        parsed = json.loads(body)
    except ValueError as exc:
        raise ImageBundleError(f"The registry's {what} is not JSON.") from exc
    if not isinstance(parsed, dict):
        raise ImageBundleError(f"The registry's {what} is not a JSON object.")
    return parsed


async def _read_capped(response: httpx.Response, limit: int, *, what: str) -> bytes:
    too_large = ImageBundleError(f"The image's {what} is larger than {limit // (1024 * 1024)} MB.")
    declared = response.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise too_large
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > limit:
            raise too_large
        chunks.append(chunk)
    return b"".join(chunks)


class _RegistryClient:
    def __init__(self, http: httpx.AsyncClient, image: ImageReference) -> None:
        self._http = http
        self._image = image
        self._token: str | None = None
        self._challenge = ""

    def _url(self, kind: str, ref: str) -> str:
        return f"https://{self._image.registry}/v2/{self._image.repository}/{kind}/{ref}"

    async def get(self, kind: str, ref: str, *, limit: int, what: str, accept: str | None = None) -> bytes:
        headers = {"Accept": accept} if accept else {}
        status, body = await self._send(kind, ref, headers, limit=limit, what=what)
        if status == 401 and self._token is None:
            self._token = await self._anonymous_token(self._challenge)
            status, body = await self._send(kind, ref, headers, limit=limit, what=what)
        if status in (401, 403):
            raise ImageAccessDeniedError(
                f"{self._display()} is private. Only images that can be pulled without credentials are supported."
            )
        if status == 404:
            raise ImageNotFoundError(f"{self._display()} was not found.")
        if status >= 400:
            LOGGER.warning("Registry %s answered %s for %s %s", self._image.registry, status, kind, ref)
            raise ImageBundleError(f"Registry {self._image.registry} returned an unexpected response.")
        return body

    async def _send(self, kind: str, ref: str, headers: dict[str, str], *, limit: int, what: str) -> tuple[int, bytes]:
        if self._token:
            headers = {**headers, "Authorization": f"Bearer {self._token}"}
        status, body, self._challenge = await self._fetch(self._url(kind, ref), headers, limit=limit, what=what)
        return status, body

    async def _fetch(self, url: str, headers: dict[str, str], *, limit: int, what: str) -> tuple[int, bytes, str]:
        for _ in range(_MAX_REDIRECTS + 1):
            request = self._http.build_request("GET", url, headers=headers)
            try:
                response = await self._http.send(request, stream=True)
            except httpx.HTTPError as exc:
                LOGGER.warning("Could not reach %s: %s", urlsplit(url).hostname, exc)
                raise ImageBundleError(f"Could not reach registry {self._image.registry}.") from exc
            try:
                if response.is_redirect:
                    url = urljoin(url, response.headers.get("location", ""))
                    await _assert_public_https_url(url, what="redirect")
                    headers = {key: value for key, value in headers.items() if key.lower() != "authorization"}
                    continue
                if response.is_error:
                    return response.status_code, b"", response.headers.get("www-authenticate", "")
                return response.status_code, await _read_capped(response, limit, what=what), ""
            finally:
                await response.aclose()
        raise ImageBundleError(f"Registry {self._image.registry} redirected too many times.")

    async def _anonymous_token(self, challenge: str) -> str | None:
        bearer = dict(parse_challenges(challenge)).get("bearer")
        if bearer is None:
            return None
        params = dict(bearer)
        realm = params.pop("realm", "")
        await _assert_public_https_url(realm, what="token service")
        params.setdefault("scope", f"repository:{self._image.repository}:pull")
        url = str(httpx.URL(realm, params=params))
        status, body, _ = await self._fetch(url, {}, limit=MAX_METADATA_BYTES, what="token")
        if status >= 400:
            return None
        grant = _json(body, what="token")
        token = grant.get("token") or grant.get("access_token")
        return token if isinstance(token, str) else None

    def _display(self) -> str:
        return str(self._image)


def _digest_of(entry: Any) -> str:
    digest = entry.get("digest") if isinstance(entry, dict) else None
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise ImageBundleError("The image manifest lists an entry without a digest.")
    return digest


def _select_manifest(index: dict[str, Any]) -> str:
    manifests = [entry for entry in index.get("manifests") or [] if isinstance(entry, dict)]
    platforms = [(entry, entry.get("platform") or {}) for entry in manifests]
    runnable = [(entry, platform) for entry, platform in platforms if platform.get("architecture") != "unknown"]
    for entry, platform in runnable:
        if (platform.get("os"), platform.get("architecture")) == _PREFERRED_PLATFORM:
            return _digest_of(entry)
    if not runnable:
        raise ImageBundleError("The image index lists no runnable platform.")
    return _digest_of(runnable[0][0])


def _env_value(config: dict[str, Any], name: str) -> str | None:
    container = config.get("config")
    env = container.get("Env") if isinstance(container, dict) else None
    for entry in env if isinstance(env, list) else []:
        key, _, value = str(entry).partition("=")
        if key == name:
            return value
    return None


def _copy_destination(created_by: str, workdir: str) -> str | None:
    """Where a ``COPY``/``ADD`` history entry landed, in BuildKit or legacy builder wording."""
    legacy = _LEGACY_COPY.search(created_by)
    if legacy:
        destination = legacy.group(1)
    else:
        buildkit = _BUILDKIT_COPY.match(created_by.strip())
        if not buildkit:
            return None
        operands = [token for token in buildkit.group(1).split() if not token.startswith("--")]
        if len(operands) < 2:
            return None
        destination = operands[-1]
    return posixpath.normpath(posixpath.join(workdir, destination))


def _copies_into(destination: str, bundle_dir: str) -> bool:
    # A copy onto "/" is a base image's rootfs, never the agent directory.
    return bundle_dir == destination or (destination != "/" and bundle_dir.startswith(destination + "/"))


def _bundle_layer_digest(image_config: dict[str, Any], layers: list[dict[str, Any]], bundle_dir: str) -> str:
    """The last layer whose ``COPY`` landed on *bundle_dir* or on a directory above it.

    ``WORKDIR`` is tracked through the history so a relative destination such as
    ``COPY . .`` resolves the way the builder resolved it.
    """
    history = [entry for entry in image_config.get("history") or [] if isinstance(entry, dict)]
    workdir = "/"
    destinations: list[str | None] = []
    for entry in history:
        created_by = str(entry.get("created_by", ""))
        if changed := _WORKDIR.search(created_by):
            workdir = posixpath.normpath(posixpath.join(workdir, changed.group(1)))
        if not entry.get("empty_layer"):
            destinations.append(_copy_destination(created_by, workdir))
    if len(destinations) != len(layers):
        raise ImageBundleError("The image history does not line up with its layers.")
    for destination, layer in zip(reversed(destinations), reversed(layers), strict=True):
        if destination is not None and _copies_into(destination, bundle_dir):
            return _digest_of(layer)
    raise ImageBundleError(f"No layer copies the agent into {bundle_dir}. Package the image with nemo agents package.")


def _is_build_artifact(relative_path: str) -> bool:
    name = posixpath.basename(relative_path)
    return name in _BUILD_ARTIFACT_NAMES or name.endswith(_BUILD_ARTIFACT_SUFFIXES)


class _CappedReader(io.RawIOBase):
    """Stops an inflating stream once it has produced more bytes than *limit*."""

    def __init__(self, inner: io.BytesIO | gzip.GzipFile, limit: int) -> None:
        super().__init__()
        self._inner = inner
        self._limit = limit
        self._seen = 0

    def readable(self) -> bool:
        return True

    def read(self, size: int | None = -1) -> bytes:
        data = self._inner.read(-1 if size is None else size)
        self._seen += len(data)
        if self._seen > self._limit:
            raise ImageBundleError(f"The agent layer inflates to more than {self._limit // (1024 * 1024)} MB.")
        return data


def _open_layer(layer: bytes) -> tarfile.TarFile:
    if layer.startswith(_ZSTD_MAGIC):
        raise ImageBundleError("The agent layer is zstd-compressed, which is not supported.")
    raw: io.BytesIO | gzip.GzipFile = io.BytesIO(layer)
    if layer.startswith(_GZIP_MAGIC):
        raw = gzip.GzipFile(fileobj=raw)
    return tarfile.open(fileobj=_CappedReader(raw, MAX_INFLATED_LAYER_BYTES), mode="r|")


def _extract_bundle(layer: bytes, bundle_dir: str) -> dict[str, bytes]:
    prefix = bundle_dir.strip("/") + "/"
    files: dict[str, bytes] = {}
    total = 0
    try:
        with _open_layer(layer) as archive:
            for member in archive:
                name = posixpath.normpath(member.name.removeprefix("./").lstrip("/"))
                if not member.isfile() or not name.startswith(prefix):
                    continue
                relative = name[len(prefix) :]
                if posixpath.basename(relative).startswith(".wh.") or _is_build_artifact(relative):
                    continue
                if len(files) >= MAX_BUNDLE_FILES:
                    raise ImageBundleError(f"The agent bundle holds more than {MAX_BUNDLE_FILES} files.")
                total += member.size
                if total > MAX_BUNDLE_BYTES:
                    raise ImageBundleError(f"The agent bundle is larger than {MAX_BUNDLE_BYTES // (1024 * 1024)} MB.")
                extracted = archive.extractfile(member)
                if extracted is not None:
                    files[relative] = extracted.read()
    except (tarfile.TarError, OSError, EOFError, zlib.error) as exc:
        raise ImageBundleError("The agent layer is not a readable archive.") from exc
    return files


async def read_agent_bundle(image: str, *, http: httpx.AsyncClient) -> ExtractedBundle:
    """Fetch the agent directory that ``nemo agents package`` copied into *image*."""
    try:
        reference = parse_image_reference(image)
    except ImageReferenceError as exc:
        raise ImageBundleError(str(exc)) from exc
    _assert_allowed_registry(reference.registry)
    await assert_public_host(_host_of(reference.registry))
    registry = _RegistryClient(http, reference)

    async def manifest_json(ref: str) -> dict[str, Any]:
        body = await registry.get("manifests", ref, accept=_MANIFEST_ACCEPT, limit=MAX_METADATA_BYTES, what="manifest")
        return _json(body, what="manifest")

    manifest = await manifest_json(reference.reference)
    if "manifests" in manifest:
        manifest = await manifest_json(_select_manifest(manifest))

    if not isinstance(manifest.get("config"), dict):
        raise ImageBundleError("The image manifest has no config.")
    config_digest = _digest_of(manifest["config"])
    image_config = _json(
        await registry.get("blobs", config_digest, limit=MAX_METADATA_BYTES, what="config"), what="config"
    )

    config_path = _env_value(image_config, "AGENT_CONFIG_PATH")
    if not config_path:
        raise ImageBundleError("The image sets no AGENT_CONFIG_PATH. Package it with nemo agents package.")
    bundle_dir = posixpath.dirname(config_path) or "/"

    layers = [layer for layer in manifest.get("layers") or [] if isinstance(layer, dict)]
    layer_digest = _bundle_layer_digest(image_config, layers, bundle_dir)
    layer = await registry.get("blobs", layer_digest, limit=MAX_BUNDLE_LAYER_BYTES, what="agent layer")
    files = await asyncio.to_thread(_extract_bundle, layer, bundle_dir)

    config_name = posixpath.basename(config_path)
    if config_name not in files:
        raise ImageBundleError(f"The agent layer has no {config_name}.")
    if config_name != AGENT_CONFIG_FILENAME:
        files[AGENT_CONFIG_FILENAME] = files.pop(config_name)
    return ExtractedBundle(config_path=config_path, files=dict(sorted(files.items())))
