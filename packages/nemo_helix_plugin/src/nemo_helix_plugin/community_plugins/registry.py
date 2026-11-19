# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deployment-independent community plugin contract and atomic registration state."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_REGISTRY_BYTES = 1024 * 1024
_SEGMENT = re.compile(r"[A-Za-z0-9_~.-]+\Z")
_PARAMETER = re.compile(r"\{([a-z][a-z0-9_]*)\}\Z")
_VALUE = re.compile(r"[A-Za-z0-9_~.-]{1,128}\Z")


def path_segments(path: str) -> list[str]:
    """Accept only canonical absolute paths with single-segment placeholders."""
    if not path.startswith("/") or path.startswith("//") or len(path) > 2048:
        raise ValueError("Expected a canonical absolute endpoint path")
    segments = path[1:].split("/")
    if any(s in {"", ".", ".."} or not (_SEGMENT.fullmatch(s) or _PARAMETER.fullmatch(s)) for s in segments):
        raise ValueError("Endpoint paths require safe segments or {parameter} placeholders")
    names = [match[1] for s in segments if (match := _PARAMETER.fullmatch(s))]
    if len(names) != len(set(names)):
        raise ValueError("Endpoint path parameters must be unique")
    return segments


def canonical_json(value: BaseModel) -> bytes:
    return json.dumps(value.model_dump(mode="json", by_alias=True), sort_keys=True, separators=(",", ":")).encode()


def validate_remote_url(value: str) -> str:
    if any(ord(character) <= 32 or character == "\\" for character in value):
        raise ValueError("Invalid URL characters")
    url = urlsplit(value)
    if url.scheme not in {"http", "https"} or not url.hostname or url.netloc.endswith(":"):
        raise ValueError("An absolute HTTPS URL is required")
    _ = url.port
    if url.scheme == "http" and url.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Plain HTTP is supported only for loopback development servers")
    if (
        url.username is not None
        or url.password is not None
        or url.fragment
        or url.query
        or "{" in value
        or "}" in value
    ):
        raise ValueError("Download URLs cannot contain credentials, fragments, or templates")
    return value


class StudioSidecar(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True, frozen=True)
    contract_version: str = Field(alias="contractVersion", min_length=1, max_length=80)
    specifiers: tuple[str, ...] = Field(max_length=128)
    built_against: dict[str, str] = Field(alias="builtAgainst", max_length=128)


class BundleSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    url: str = Field(max_length=2048)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sidecar: StudioSidecar

    @model_validator(mode="after")
    def valid_url(self) -> BundleSpec:
        validate_remote_url(self.url)
        return self

    @property
    def revision(self) -> str:
        return hashlib.sha256(self.sha256.encode() + canonical_json(self.sidecar)).hexdigest()


class NavigationItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    label: str = Field(min_length=1, max_length=120)
    path: str = Field(pattern=r"^/[a-zA-Z0-9_~./{}-]*$", max_length=256)
    section: str = Field(default="Community", min_length=1, max_length=120)

    @model_validator(mode="after")
    def safe_path(self) -> NavigationItem:
        if self.path.startswith("//") or any(part in {".", ".."} for part in self.path.split("/")):
            raise ValueError("Navigation paths must be local, without traversal")
        return self


class WorkspaceSecretCredential(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    secret_name: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")
    scheme: Literal["bearer"] = "bearer"


class EndpointMapping(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    method: Literal["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"]
    path: str
    access: Literal["read", "write"]
    upstream_url: str

    @model_validator(mode="after")
    def validate_mapping(self) -> EndpointMapping:
        segments = path_segments(self.path)
        if segments[0] == "-":
            raise ValueError("The /-/ namespace is reserved for gateway metadata")
        if "{workspace}" in segments and (len(segments) < 4 or segments[:3] != ["v2", "workspaces", "{workspace}"]):
            raise ValueError("Workspace parameters must use /v2/workspaces/{workspace}/<endpoint>")
        if segments[:2] == ["v2", "workspaces"] and (len(segments) < 4 or segments[2] == "-"):
            raise ValueError("A concrete workspace and trailing endpoint are required")
        expected = "read" if self.method in {"GET", "HEAD"} else "write"
        if self.access != expected:
            raise ValueError("GET/HEAD must be read; other supported methods must be write")
        if any(ord(character) <= 32 or character == "\\" for character in self.upstream_url):
            raise ValueError("Upstream URLs cannot contain whitespace, controls, or backslashes")
        url = urlsplit(self.upstream_url)
        # URLs are consented operator configuration, not supplied by the caller.
        # HTTPS is required except for explicit loopback development servers.
        if url.scheme not in {"http", "https"} or not url.hostname or url.netloc.endswith(":"):
            raise ValueError("Expected an absolute HTTP(S) upstream URL")
        # Accessing port also rejects malformed/out-of-range values.
        _ = url.port
        if url.scheme == "http" and url.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Plain HTTP is supported only for loopback development upstreams")
        if url.username is not None or url.password is not None or url.query or url.fragment:
            raise ValueError("Upstream URLs cannot contain credentials, query strings, or fragments")
        if "{" in url.netloc or "}" in url.netloc:
            raise ValueError("Upstream authority cannot contain parameters")
        target_segments = path_segments(url.path)
        names = {s for s in segments if _PARAMETER.fullmatch(s)}
        if any(s not in names for s in target_segments if _PARAMETER.fullmatch(s)):
            raise ValueError("Upstream parameters must come from the declared endpoint path")
        return self

    def resolve(self, path: str) -> str | None:
        """Match a caller path and substitute safe path segments, never a host/query."""
        expected = self.path[1:].split("/")
        actual = path[1:].split("/")
        if len(actual) != len(expected):
            return None
        parameters: dict[str, str] = {}
        for pattern, value in zip(expected, actual, strict=True):
            if match := _PARAMETER.fullmatch(pattern):
                if not _VALUE.fullmatch(value) or value in {".", ".."}:
                    return None
                parameters[match[1]] = value
            elif pattern != value:
                return None
        return self.upstream_url.format_map(parameters)


class ExternalPlugin(BaseModel):
    """Consented external endpoint/UI declaration; never a deployment definition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$", min_length=2, max_length=80)
    title: str = Field(min_length=1, max_length=160)
    enabled: bool = True
    endpoints: tuple[EndpointMapping, ...] = Field(min_length=1, max_length=256)
    bundle: BundleSpec | None = None
    navigation: tuple[NavigationItem, ...] = Field(default=(), max_length=32)
    credential: WorkspaceSecretCredential | None = None

    @property
    def bundle_url(self) -> str | None:
        return f"/plugin-ui/{self.name}/{self.bundle.revision}/index.js" if self.bundle is not None else None

    @property
    def manifest_sha256(self) -> str:
        return hashlib.sha256(canonical_json(self)).hexdigest()

    @model_validator(mode="after")
    def reject_ambiguous_mappings(self) -> ExternalPlugin:
        if self.credential is not None:
            for endpoint in self.endpoints:
                parts = path_segments(endpoint.path)
                if len(parts) < 4 or parts[:2] != ["v2", "workspaces"] or parts[2] != "{workspace}":
                    raise ValueError("Workspace-secret endpoints must use /v2/workspaces/{workspace}/...")
        for index, endpoint in enumerate(self.endpoints):
            left = path_segments(endpoint.path)
            for other in self.endpoints[index + 1 :]:
                right = path_segments(other.path)
                if endpoint.method == other.method and len(left) == len(right):
                    if all(a == b or _PARAMETER.fullmatch(a) or _PARAMETER.fullmatch(b) for a, b in zip(left, right)):
                        raise ValueError("Overlapping endpoint mappings for the same method are not allowed")
        return self


class ExternalPluginRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    plugins: tuple[ExternalPlugin, ...] = Field(default=(), max_length=256)

    @model_validator(mode="after")
    def unique_names(self) -> ExternalPluginRegistry:
        names = [plugin.name for plugin in self.plugins]
        if len(names) != len(set(names)):
            raise ValueError("Registered plugin names must be unique")
        return self


class RegistryConflictError(ValueError):
    pass


class RegistryNotFoundError(ValueError):
    pass


class RegistryStore:
    """Atomic local state with POSIX cross-process writer serialization.

    File locking spans the read/modify/replace operation, never network fetches.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> ExternalPluginRegistry:
        try:
            with self.path.open("rb") as stream:
                data = stream.read(MAX_REGISTRY_BYTES + 1)
        except FileNotFoundError:
            return ExternalPluginRegistry()
        if len(data) > MAX_REGISTRY_BYTES:
            raise ValueError("Plugin registry exceeds its size limit")
        return ExternalPluginRegistry.model_validate_json(data)

    def _mutate(self, change: Callable[[ExternalPluginRegistry], ExternalPluginRegistry]) -> None:
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (self.path.parent / ".installed.lock").open("a+b") as lock:
            os.chmod(lock.fileno(), 0o600)
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = change(self.load())
            data = canonical_json(state)
            if len(data) > MAX_REGISTRY_BYTES:
                raise ValueError("Plugin registry exceeds its size limit")
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(dir=self.path.parent, delete=False) as output:
                    temporary = Path(output.name)
                    output.write(data)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, self.path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

    def install(self, plugin: ExternalPlugin) -> None:
        def change(state: ExternalPluginRegistry) -> ExternalPluginRegistry:
            if any(p.name == plugin.name for p in state.plugins):
                raise RegistryConflictError(
                    "Plugin already installed; uninstall before replacing its consented manifest"
                )
            return ExternalPluginRegistry(plugins=(*state.plugins, plugin))

        self._mutate(change)

    def set_enabled(self, name: str, enabled: bool) -> None:
        def change(state: ExternalPluginRegistry) -> ExternalPluginRegistry:
            if not any(p.name == name for p in state.plugins):
                raise RegistryNotFoundError("Plugin is not installed")
            return ExternalPluginRegistry(
                plugins=tuple(p.model_copy(update={"enabled": enabled}) if p.name == name else p for p in state.plugins)
            )

        self._mutate(change)

    def uninstall(self, name: str) -> None:
        def change(state: ExternalPluginRegistry) -> ExternalPluginRegistry:
            if not any(p.name == name for p in state.plugins):
                raise RegistryNotFoundError("Plugin is not installed")
            return ExternalPluginRegistry(plugins=tuple(p for p in state.plugins if p.name != name))

        self._mutate(change)
