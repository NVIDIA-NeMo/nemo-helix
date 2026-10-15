# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the agent directory that ``nemo agents package`` copied into an image, straight from its registry."""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import io
import ipaddress
import json
import logging
import posixpath
import re
import socket
import tarfile
import urllib.request
import zlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx
from nemo_agents_plugin.entities import AGENT_CONFIG_FILENAME, MAX_ETHOS_STAGED_BYTES, MAX_ETHOS_STAGED_FILES
from nemo_helix_plugin.container_registry import (
    ImageReference,
    ImageReferenceError,
    normalize_registry,
    parse_challenges,
    parse_image_reference,
)

LOGGER = logging.getLogger(__name__)

ANY_REGISTRY = "*"

_MANIFEST_ACCEPT = ", ".join(
    [
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ]
)
_PREFERRED_PLATFORM = ("linux", "amd64")

MAX_LAYER_DOWNLOAD_BYTES = 64 * 1024 * 1024
MAX_INFLATED_LAYER_BYTES = 256 * 1024 * 1024
MAX_METADATA_BYTES = 4 * 1024 * 1024
READ_DEADLINE_SECONDS = 120.0
_MAX_REDIRECTS = 3

_SHA256_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")

# Written into the build context by `nemo agents package`, not part of the agent.
_BUILD_ARTIFACT_NAMES = frozenset({"Dockerfile.generated", "Dockerfile.generated.dockerignore", ".dockerignore"})
_BUILD_ARTIFACT_SUFFIXES = (".whl",)
# Mirrors what Studio leaves out of an upload.
_IGNORED_DIRECTORIES = frozenset(
    {
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".idea",
        ".vscode",
    }
)
_IGNORED_FILENAMES = frozenset({".DS_Store", "Thumbs.db"})
_IGNORED_SUFFIXES = (".pyc", ".pyo", ".pyd", ".so", ".dylib", ".dll")

_WHITEOUT_PREFIX = ".wh."
_OPAQUE_WHITEOUT = ".wh..wh..opq"

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


class RegistryUnavailableError(ImageBundleError):
    """The registry failed, throttled the read, or could not be reached."""


class RegistryTimeoutError(RegistryUnavailableError):
    """The read did not finish in time."""


@dataclass(frozen=True)
class ExtractedBundle:
    config_path: str
    files: dict[str, bytes]
    digest: str


@dataclass(frozen=True)
class RegistryPolicy:
    trusted: frozenset[str]
    trusted_hosts: frozenset[str]
    any_public: bool = False

    @classmethod
    def from_entries(cls, entries: Iterable[str]) -> RegistryPolicy:
        cleaned = [entry.strip() for entry in entries if entry.strip()]
        trusted = frozenset(normalize_registry(entry) for entry in cleaned if entry != ANY_REGISTRY)
        return cls(
            trusted=trusted,
            trusted_hosts=frozenset(_host_of(registry) for registry in trusted),
            any_public=ANY_REGISTRY in cleaned,
        )

    def admits(self, registry: str) -> bool:
        return self.any_public or registry in self.trusted

    def trusts_host(self, host: str) -> bool:
        return host.strip("[]").lower().rstrip(".") in self.trusted_hosts


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


def _proxied(host: str) -> bool:
    proxies = urllib.request.getproxies()
    if not (proxies.get("https") or proxies.get("all")):
        return False
    for entry in proxies.get("no", "").split(","):
        bypass = entry.strip().lstrip(".").lower()
        if bypass == "*" or (bypass and (host == bypass or host.endswith(f".{bypass}"))):
            return False
    return True


async def assert_public_host(host: str) -> None:
    """Refuse hosts that are, or resolve to, anything other than a public address."""
    bare = host.strip("[]").lower().rstrip(".")
    private = ImageBundleError(f"Host {host} is on a private network and cannot be read.")
    if not bare or bare == "localhost" or bare.endswith(_INTERNAL_HOST_SUFFIXES):
        raise private
    try:
        addresses = [ipaddress.ip_address(bare)]
    except ValueError:
        if "." not in bare:
            raise private from None
        try:
            addresses = await asyncio.to_thread(_resolve_host, bare)
        except UnicodeError as exc:
            raise ImageBundleError(f"'{host}' is not a valid host name.") from exc
        except OSError as exc:
            # Behind a proxy the local resolver may not see public names; the proxy resolves them.
            if _proxied(bare):
                return
            raise ImageBundleError(f"Host {host} could not be resolved.") from exc
    if not addresses or not all(_is_public(address) for address in addresses):
        raise private


def _json(body: bytes, *, what: str) -> dict[str, Any]:
    try:
        parsed = json.loads(body)
    except ValueError as exc:
        raise ImageBundleError(f"The {what} is not JSON.") from exc
    if not isinstance(parsed, dict):
        raise ImageBundleError(f"The {what} is not a JSON object.")
    return parsed


def _sha256(body: bytes) -> str:
    return f"sha256:{hashlib.sha256(body).hexdigest()}"


async def _read_capped(response: httpx.Response, limit: int, *, what: str) -> bytes:
    too_large = ImageBundleError(f"The {what} exceeds {limit // (1024 * 1024)} MB.")
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
    def __init__(self, http: httpx.AsyncClient, image: ImageReference, policy: RegistryPolicy) -> None:
        self._http = http
        self._image = image
        self._policy = policy
        self._token: str | None = None
        self._challenge = ""

    async def check_host(self, host: str) -> None:
        if not self._policy.trusts_host(host):
            await assert_public_host(host)

    async def get(
        self, kind: str, ref: str, *, limit: int, what: str, accept: str | None = None, digest: str | None = None
    ) -> bytes:
        headers = {"Accept": accept} if accept else {}
        status, body = await self._send(kind, ref, headers, limit=limit, what=what)
        if status == 401 and self._token is None:
            self._token = await self._anonymous_token(self._challenge)
            status, body = await self._send(kind, ref, headers, limit=limit, what=what)
        self._raise_for_status(status, kind, ref)
        if digest is not None and _sha256(body) != digest:
            raise ImageBundleError(f"The {what} does not match its digest {digest}.")
        return body

    def _raise_for_status(self, status: int, kind: str, ref: str) -> None:
        if status < 400:
            return
        if status in (401, 403):
            raise ImageAccessDeniedError(
                f"{self._image} is private. Only images that can be pulled without credentials are supported."
            )
        if status == 404:
            raise ImageNotFoundError(f"{self._image} was not found.")
        LOGGER.warning("Registry %s answered %s for %s %s", self._image.registry, status, kind, ref)
        if status == 429:
            raise RegistryUnavailableError(f"Registry {self._image.registry} is rate limiting reads. Try again later.")
        if status >= 500:
            raise RegistryUnavailableError(f"Registry {self._image.registry} is unavailable. Try again later.")
        raise ImageBundleError(f"Registry {self._image.registry} returned an unexpected response.")

    async def _send(self, kind: str, ref: str, headers: dict[str, str], *, limit: int, what: str) -> tuple[int, bytes]:
        if self._token:
            headers = {**headers, "Authorization": f"Bearer {self._token}"}
        url = f"https://{self._image.registry}/v2/{self._image.repository}/{kind}/{ref}"
        status, body, self._challenge = await self._fetch(url, headers, limit=limit, what=what)
        return status, body

    async def _check_url(self, url: str, *, what: str) -> None:
        try:
            parts = urlsplit(url)
            host = parts.hostname
        except ValueError as exc:
            raise ImageBundleError(f"The registry's {what} URL is invalid.") from exc
        if parts.scheme != "https" or not host:
            raise ImageBundleError(f"The registry's {what} is not an https URL.")
        await self.check_host(host)

    async def _fetch(self, url: str, headers: dict[str, str], *, limit: int, what: str) -> tuple[int, bytes, str]:
        for _ in range(_MAX_REDIRECTS + 1):
            try:
                request = self._http.build_request("GET", url, headers=headers)
                response = await self._http.send(request, stream=True)
            except (httpx.InvalidURL, ValueError) as exc:
                raise ImageBundleError(f"The registry named an invalid URL for the {what}.") from exc
            except httpx.TimeoutException as exc:
                raise RegistryTimeoutError(f"Registry {self._image.registry} did not answer in time.") from exc
            except httpx.RemoteProtocolError as exc:
                raise RegistryUnavailableError(f"Registry {self._image.registry} sent a malformed response.") from exc
            except httpx.HTTPError as exc:
                LOGGER.warning("Could not reach %s: %s", urlsplit(url).hostname, exc)
                raise RegistryUnavailableError(f"Could not reach registry {self._image.registry}.") from exc
            try:
                if response.is_redirect:
                    url = urljoin(url, response.headers.get("location", ""))
                    await self._check_url(url, what="redirect")
                    headers = {key: value for key, value in headers.items() if key.lower() != "authorization"}
                    continue
                if response.is_error:
                    return response.status_code, b"", response.headers.get("www-authenticate", "")
                return response.status_code, await _read_capped(response, limit, what=what), ""
            except httpx.TimeoutException as exc:
                raise RegistryTimeoutError(f"Registry {self._image.registry} did not answer in time.") from exc
            except httpx.HTTPError as exc:
                raise RegistryUnavailableError(f"Registry {self._image.registry} dropped the connection.") from exc
            finally:
                await response.aclose()
        raise RegistryUnavailableError(f"Registry {self._image.registry} redirected too many times.")

    async def _anonymous_token(self, challenge: str) -> str | None:
        bearer = dict(parse_challenges(challenge)).get("bearer")
        if bearer is None:
            return None
        params = dict(bearer)
        realm = params.pop("realm", "")
        await self._check_url(realm, what="token service")
        params.setdefault("scope", f"repository:{self._image.repository}:pull")
        try:
            url = str(httpx.URL(realm, params=params))
        except httpx.InvalidURL as exc:
            raise ImageBundleError("The registry's token service URL is invalid.") from exc
        status, body, _ = await self._fetch(url, {}, limit=MAX_METADATA_BYTES, what="token response")
        if status >= 400:
            return None
        grant = _json(body, what="token response")
        token = grant.get("token") or grant.get("access_token")
        return token if isinstance(token, str) else None


def _digest_of(entry: Any) -> str:
    digest = entry.get("digest") if isinstance(entry, dict) else None
    if not isinstance(digest, str) or not _SHA256_DIGEST.fullmatch(digest):
        raise ImageBundleError("The image manifest lists an entry without a valid sha256 digest.")
    return digest


def _dicts(value: Any) -> list[dict[str, Any]]:
    return [entry for entry in value if isinstance(entry, dict)] if isinstance(value, list) else []


def _select_manifest(index: dict[str, Any]) -> str:
    platforms = [(entry, entry.get("platform")) for entry in _dicts(index.get("manifests"))]
    runnable = [
        (entry, platform)
        for entry, platform in platforms
        if not isinstance(platform, dict) or platform.get("architecture") != "unknown"
    ]
    for entry, platform in runnable:
        if isinstance(platform, dict) and (platform.get("os"), platform.get("architecture")) == _PREFERRED_PLATFORM:
            return _digest_of(entry)
    if not runnable:
        raise ImageBundleError("The image index lists no runnable platform.")
    return _digest_of(runnable[0][0])


def _container_config(image_config: dict[str, Any]) -> dict[str, Any]:
    container = image_config.get("config")
    return container if isinstance(container, dict) else {}


def _env_value(container: dict[str, Any], name: str) -> str | None:
    env = container.get("Env")
    for entry in env if isinstance(env, list) else []:
        key, _, value = str(entry).partition("=")
        if key == name:
            return value
    return None


def _agent_config_path(image_config: dict[str, Any]) -> str:
    container = _container_config(image_config)
    configured = _env_value(container, "AGENT_CONFIG_PATH")
    if not configured:
        raise ImageBundleError("The image sets no AGENT_CONFIG_PATH. Package it with nemo agents package.")
    workdir = container.get("WorkingDir")
    path = posixpath.normpath(posixpath.join("/", workdir if isinstance(workdir, str) else "", configured))
    if posixpath.dirname(path) == "/":
        raise ImageBundleError(
            f"AGENT_CONFIG_PATH {configured} is in the image root. Package the agent into its own directory."
        )
    return path


def _within(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _copy_destination(created_by: str, workdir: str) -> str | None:
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


def _bundle_layer_digests(image_config: dict[str, Any], layers: list[dict[str, Any]], bundle_dir: str) -> list[str]:
    """Every layer, oldest first, whose COPY or ADD landed in or above *bundle_dir*."""
    workdir = "/"
    destinations: list[str | None] = []
    for entry in _dicts(image_config.get("history")):
        created_by = str(entry.get("created_by", ""))
        if changed := _WORKDIR.search(created_by):
            workdir = posixpath.normpath(posixpath.join(workdir, changed.group(1)))
        if not entry.get("empty_layer"):
            destinations.append(_copy_destination(created_by, workdir))
    if len(destinations) != len(layers):
        raise ImageBundleError("The image history does not line up with its layers.")
    digests = [
        _digest_of(layer)
        for destination, layer in zip(destinations, layers, strict=True)
        # A copy onto "/" is a base image's rootfs, never the agent directory.
        if destination not in (None, "/") and (_within(destination, bundle_dir) or _within(bundle_dir, destination))
    ]
    if not digests:
        raise ImageBundleError(
            f"No layer copies the agent into {bundle_dir}. Package the image with nemo agents package."
        )
    return digests


def _is_skipped(relative_path: str) -> bool:
    *directories, name = relative_path.split("/")
    return (
        name in _BUILD_ARTIFACT_NAMES
        or name in _IGNORED_FILENAMES
        or name.endswith(_BUILD_ARTIFACT_SUFFIXES + _IGNORED_SUFFIXES)
        or any(directory in _IGNORED_DIRECTORIES for directory in directories)
    )


class _CappedReader(io.RawIOBase):
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
            raise ImageBundleError(f"An agent layer inflates to more than {self._limit // (1024 * 1024)} MB.")
        return data


def _open_layer(layer: bytes) -> tarfile.TarFile:
    if layer.startswith(_ZSTD_MAGIC):
        raise ImageBundleError("The agent layer is zstd-compressed, which is not supported.")
    raw: io.BytesIO | gzip.GzipFile = io.BytesIO(layer)
    if layer.startswith(_GZIP_MAGIC):
        raw = gzip.GzipFile(fileobj=raw)
    return tarfile.open(fileobj=_CappedReader(raw, MAX_INFLATED_LAYER_BYTES), mode="r|")


def _check_bundle_size(file_count: int, byte_count: int) -> None:
    if file_count > MAX_ETHOS_STAGED_FILES:
        raise ImageBundleError(f"The agent bundle holds more than {MAX_ETHOS_STAGED_FILES} files.")
    if byte_count > MAX_ETHOS_STAGED_BYTES:
        raise ImageBundleError(f"The agent bundle is larger than {MAX_ETHOS_STAGED_BYTES // 1000} KB.")


def _apply_layer(files: dict[str, bytes], layer: bytes, bundle_dir: str) -> None:
    """Overlay one layer onto *files*, keyed by path in the image, honoring OCI whiteouts."""
    root = bundle_dir.strip("/")
    added: dict[str, bytes] = {}
    removed: list[str] = []
    opaque: list[str] = []
    total = 0
    try:
        with _open_layer(layer) as archive:
            for member in archive:
                name = posixpath.normpath(member.name.removeprefix("./").lstrip("/"))
                parent, base = posixpath.split(name)
                if base == _OPAQUE_WHITEOUT:
                    opaque.append(parent)
                    continue
                if base.startswith(_WHITEOUT_PREFIX):
                    removed.append(posixpath.join(parent, base.removeprefix(_WHITEOUT_PREFIX)))
                    continue
                if not member.isfile() or name == root or not _within(name, root):
                    continue
                if _is_skipped(name[len(root) + 1 :]):
                    continue
                total += member.size
                _check_bundle_size(len(added) + 1, total)
                extracted = archive.extractfile(member)
                if extracted is not None:
                    added[name] = extracted.read()
    except (tarfile.TarError, OSError, EOFError, zlib.error) as exc:
        raise ImageBundleError("The agent layer is not a readable archive.") from exc
    for directory in opaque:
        for path in [path for path in files if path != directory and _within(path, directory)]:
            del files[path]
    for gone in removed:
        for path in [path for path in files if _within(path, gone)]:
            del files[path]
    files.update(added)


async def _read_agent_bundle(image: str, http: httpx.AsyncClient, policy: RegistryPolicy) -> ExtractedBundle:
    try:
        reference = parse_image_reference(image)
    except ImageReferenceError as exc:
        raise ImageBundleError(str(exc)) from exc
    if not policy.admits(reference.registry):
        raise ImageBundleError(
            f"Registry {reference.registry} is not an allowed image registry. "
            "Add it to agents.deployments.image_registries in the platform configuration."
        )
    if reference.is_digest and not _SHA256_DIGEST.fullmatch(reference.reference):
        raise ImageBundleError("Only sha256 image digests are supported.")
    registry = _RegistryClient(http, reference, policy)
    await registry.check_host(_host_of(reference.registry))

    async def manifest_json(ref: str, digest: str | None) -> tuple[dict[str, Any], str]:
        body = await registry.get(
            "manifests", ref, accept=_MANIFEST_ACCEPT, limit=MAX_METADATA_BYTES, what="image manifest", digest=digest
        )
        return _json(body, what="image manifest"), _sha256(body)

    pinned = reference.reference if reference.is_digest else None
    manifest, top_digest = await manifest_json(reference.reference, pinned)
    if "manifests" in manifest:
        platform_digest = _select_manifest(manifest)
        manifest, _ = await manifest_json(platform_digest, platform_digest)

    config_digest = _digest_of(manifest.get("config"))
    image_config = _json(
        await registry.get("blobs", config_digest, limit=MAX_METADATA_BYTES, what="image config", digest=config_digest),
        what="image config",
    )
    config_path = _agent_config_path(image_config)
    bundle_dir = posixpath.dirname(config_path)

    files: dict[str, bytes] = {}
    budget = MAX_LAYER_DOWNLOAD_BYTES
    for layer_digest in _bundle_layer_digests(image_config, _dicts(manifest.get("layers")), bundle_dir):
        layer = await registry.get("blobs", layer_digest, limit=budget, what="agent layers", digest=layer_digest)
        budget -= len(layer)
        await asyncio.to_thread(_apply_layer, files, layer, bundle_dir)

    prefix = bundle_dir.strip("/") + "/"
    relative = {path.removeprefix(prefix): content for path, content in files.items()}
    _check_bundle_size(len(relative), sum(len(content) for content in relative.values()))
    config_name = posixpath.basename(config_path)
    if config_name not in relative:
        raise ImageBundleError(f"The agent layers have no {config_name}.")
    if config_name != AGENT_CONFIG_FILENAME:
        relative[AGENT_CONFIG_FILENAME] = relative.pop(config_name)
    return ExtractedBundle(config_path=config_path, files=dict(sorted(relative.items())), digest=top_digest)


async def read_agent_bundle(
    image: str, *, http: httpx.AsyncClient, policy: RegistryPolicy, deadline: float = READ_DEADLINE_SECONDS
) -> ExtractedBundle:
    """Fetch the agent directory that ``nemo agents package`` copied into *image*."""
    try:
        async with asyncio.timeout(deadline):
            return await _read_agent_bundle(image, http, policy)
    except TimeoutError as exc:
        raise RegistryTimeoutError(f"Reading {image.strip()} took longer than {deadline:g} seconds.") from exc
