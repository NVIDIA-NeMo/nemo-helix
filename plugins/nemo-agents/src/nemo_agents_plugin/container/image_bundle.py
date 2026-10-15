# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the agent bundle that ``nemo agents package`` bakes into an image, straight from its registry.

The packaged Dockerfile copies the whole agent directory into the image in one
``COPY ./ <dir>`` layer and points ``AGENT_CONFIG_PATH`` at the ``agent.yaml`` inside it.
Only that layer is downloaded, so no Docker daemon or full image pull is needed.
"""

from __future__ import annotations

import io
import ipaddress
import json
import posixpath
import re
import tarfile
from dataclasses import dataclass
from typing import Any

import httpx
from nemo_agents_plugin.entities import AGENT_CONFIG_FILENAME

DOCKER_HUB_REGISTRY = "registry-1.docker.io"
_DOCKER_HUB_ALIASES = frozenset({"docker.io", "index.docker.io", "registry-1.docker.io"})

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
MAX_BUNDLE_FILES = 1_000
MAX_BUNDLE_BYTES = 8 * 1024 * 1024

# Written into the build context by `nemo agents package`, not part of the agent.
_BUILD_ARTIFACT_NAMES = frozenset({"Dockerfile.generated", "Dockerfile.generated.dockerignore", ".dockerignore"})
_BUILD_ARTIFACT_SUFFIXES = (".whl",)

_AUTH_PARAM = re.compile(r'(\w+)="([^"]*)"')


class ImageBundleError(ValueError):
    """The image cannot supply an agent bundle."""


class ImageNotFoundError(ImageBundleError):
    """The registry has no such image."""


class ImageAccessDeniedError(ImageBundleError):
    """The registry refused anonymous access to the image."""


@dataclass(frozen=True)
class ImageReference:
    registry: str
    repository: str
    reference: str


@dataclass(frozen=True)
class AgentImageBundle:
    config_path: str
    files: dict[str, bytes]


def parse_image_reference(image: str) -> ImageReference:
    """Split ``[registry/]repository[:tag][@digest]`` the way ``docker pull`` does."""
    text = image.strip()
    if not text or "://" in text or any(char.isspace() for char in text):
        raise ImageBundleError(f"'{image}' is not an image reference. Use registry/repository:tag.")

    name, _, digest = text.partition("@")
    last_slash = name.rfind("/")
    colon = name.rfind(":")
    tag = ""
    if colon > last_slash:
        name, tag = name[:colon], name[colon + 1 :]

    first, _, rest = name.partition("/")
    if rest and ("." in first or ":" in first or first == "localhost"):
        registry, repository = first, rest
    else:
        registry, repository = DOCKER_HUB_REGISTRY, name

    if registry in _DOCKER_HUB_ALIASES:
        registry = DOCKER_HUB_REGISTRY
        if "/" not in repository:
            repository = f"library/{repository}"

    if not repository:
        raise ImageBundleError(f"'{image}' is not an image reference. Use registry/repository:tag.")
    _reject_internal_registry(registry)
    return ImageReference(registry=registry, repository=repository, reference=digest or tag or "latest")


def _reject_internal_registry(registry: str) -> None:
    host = registry.rsplit(":", 1)[0].strip("[]")
    if host == "localhost":
        raise ImageBundleError("Images on the platform host's own registry cannot be read.")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return
    if address.is_private or address.is_loopback or address.is_link_local or address.is_reserved:
        raise ImageBundleError(f"Registry {registry} is on a private network and cannot be read.")


class _RegistryClient:
    def __init__(self, http: httpx.AsyncClient, image: ImageReference) -> None:
        self._http = http
        self._image = image
        self._token: str | None = None

    def _url(self, kind: str, ref: str) -> str:
        return f"https://{self._image.registry}/v2/{self._image.repository}/{kind}/{ref}"

    async def get(self, kind: str, ref: str, *, accept: str | None = None) -> httpx.Response:
        headers = {"Accept": accept} if accept else {}
        response = await self._send(kind, ref, headers)
        if response.status_code == 401 and self._token is None:
            self._token = await self._anonymous_token(response.headers.get("www-authenticate", ""))
            response = await self._send(kind, ref, headers)
        if response.status_code in (401, 403):
            raise ImageAccessDeniedError(
                f"{self._display()} is private. Only images that can be pulled without credentials are supported."
            )
        if response.status_code == 404:
            raise ImageNotFoundError(f"{self._display()} was not found.")
        if response.is_error:
            raise ImageBundleError(f"Registry {self._image.registry} answered {response.status_code}.")
        return response

    async def _send(self, kind: str, ref: str, headers: dict[str, str]) -> httpx.Response:
        if self._token:
            headers = {**headers, "Authorization": f"Bearer {self._token}"}
        try:
            return await self._http.get(self._url(kind, ref), headers=headers)
        except httpx.HTTPError as exc:
            raise ImageBundleError(f"Could not reach registry {self._image.registry}: {exc}") from exc

    async def _anonymous_token(self, challenge: str) -> str | None:
        if not challenge.lower().startswith("bearer "):
            return None
        params = dict(_AUTH_PARAM.findall(challenge))
        realm = params.pop("realm", "")
        if not realm.startswith("https://"):
            return None
        params.setdefault("scope", f"repository:{self._image.repository}:pull")
        try:
            response = await self._http.get(realm, params=params)
        except httpx.HTTPError as exc:
            raise ImageBundleError(f"Could not reach registry {self._image.registry}: {exc}") from exc
        if response.is_error:
            return None
        body = response.json()
        return body.get("token") or body.get("access_token")

    def _display(self) -> str:
        return f"{self._image.registry}/{self._image.repository}:{self._image.reference}"


def _select_manifest(index: dict[str, Any]) -> str:
    manifests = [entry for entry in index.get("manifests", []) if isinstance(entry, dict)]
    runnable = [entry for entry in manifests if entry.get("platform", {}).get("architecture") != "unknown"]
    for entry in runnable:
        platform = entry.get("platform", {})
        if (platform.get("os"), platform.get("architecture")) == _PREFERRED_PLATFORM:
            return entry["digest"]
    if not runnable:
        raise ImageBundleError("The image index lists no runnable platform.")
    return runnable[0]["digest"]


def _env_value(config: dict[str, Any], name: str) -> str | None:
    for entry in config.get("config", {}).get("Env") or []:
        key, _, value = str(entry).partition("=")
        if key == name:
            return value
    return None


def _bundle_layer_digest(image_config: dict[str, Any], layers: list[dict[str, Any]], bundle_dir: str) -> str:
    """The layer created by the packaged Dockerfile's ``COPY ./ <bundle_dir>``."""
    layer_history = [entry for entry in image_config.get("history") or [] if not entry.get("empty_layer")]
    if len(layer_history) != len(layers):
        raise ImageBundleError("The image history does not line up with its layers.")
    copy_into_bundle = re.compile(rf"\bCOPY\b.*\s{re.escape(bundle_dir)}/?(\s|$)")
    for entry, layer in zip(reversed(layer_history), reversed(layers), strict=True):
        if copy_into_bundle.search(entry.get("created_by", "")):
            return layer["digest"]
    raise ImageBundleError(f"No layer copies the agent into {bundle_dir}. Package the image with nemo agents package.")


def _is_build_artifact(relative_path: str) -> bool:
    name = posixpath.basename(relative_path)
    return name in _BUILD_ARTIFACT_NAMES or name.endswith(_BUILD_ARTIFACT_SUFFIXES)


def _extract_bundle(layer: bytes, bundle_dir: str) -> dict[str, bytes]:
    prefix = bundle_dir.strip("/") + "/"
    files: dict[str, bytes] = {}
    total = 0
    try:
        archive = tarfile.open(fileobj=io.BytesIO(layer), mode="r:*")
    except tarfile.TarError as exc:
        raise ImageBundleError(f"The agent layer is not a readable archive: {exc}") from exc
    with archive:
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
    return files


async def read_agent_bundle(image: str, *, http: httpx.AsyncClient) -> AgentImageBundle:
    """Fetch the agent directory that ``nemo agents package`` copied into *image*."""
    reference = parse_image_reference(image)
    registry = _RegistryClient(http, reference)

    manifest = (await registry.get("manifests", reference.reference, accept=_MANIFEST_ACCEPT)).json()
    if "manifests" in manifest:
        manifest = (await registry.get("manifests", _select_manifest(manifest), accept=_MANIFEST_ACCEPT)).json()

    config_digest = manifest.get("config", {}).get("digest")
    if not config_digest:
        raise ImageBundleError("The image manifest has no config.")
    image_config = json.loads((await registry.get("blobs", config_digest)).content)

    config_path = _env_value(image_config, "AGENT_CONFIG_PATH")
    if not config_path:
        raise ImageBundleError("The image sets no AGENT_CONFIG_PATH. Package it with nemo agents package.")
    bundle_dir = posixpath.dirname(config_path) or "/"

    layer_digest = _bundle_layer_digest(image_config, manifest.get("layers") or [], bundle_dir)
    layer_size = next((layer.get("size", 0) for layer in manifest["layers"] if layer["digest"] == layer_digest), 0)
    if layer_size > MAX_BUNDLE_LAYER_BYTES:
        raise ImageBundleError("The agent layer is too large to read.")
    files = _extract_bundle((await registry.get("blobs", layer_digest)).content, bundle_dir)

    config_name = posixpath.basename(config_path)
    if config_name not in files:
        raise ImageBundleError(f"The agent layer has no {config_name}.")
    if config_name != AGENT_CONFIG_FILENAME:
        files[AGENT_CONFIG_FILENAME] = files.pop(config_name)
    return AgentImageBundle(config_path=config_path, files=dict(sorted(files.items())))
