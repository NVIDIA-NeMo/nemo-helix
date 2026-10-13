# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workspace seed files for agent-eval tasks.

A task can stage starter files into the agent's workspace before it runs, under
``inputs[SEED_FILES_INPUT_KEY]`` as a ``{relative_path: source}`` map. Each *source* is a
JSON-serializable value whose ``kind`` selects a registered :class:`SeedHandler` (a bare string is
sugar for inline text).

The SDK ships handlers only for the kinds it can resolve with **no external dependency** — ``inline``
and ``path``. Other kinds are contributed by consumers via :func:`register_seed_handler`; the SDK has
no knowledge of them (e.g. a platform ``fileset`` handler lives in the plugin and resolves against the
Files service at run time). An unregistered kind raises :class:`WorkspaceSeedError`.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

#: ``inputs`` key holding the ``{relative_path: seed}`` map of files to stage into the workspace.
SEED_FILES_INPUT_KEY = "files"


class WorkspaceSeedError(ValueError):
    """A workspace seed could not be parsed, resolved, or written (bad value, unknown kind, ...).

    Subclasses ``ValueError`` so a runner's per-task error handling still catches it.
    """


class InlineSeed(BaseModel):
    """File contents carried in the task itself. Portable to any runner."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["inline"] = "inline"
    content: str = Field(description="File contents; UTF-8 text, or base64-encoded bytes when encoding='base64'.")
    encoding: Literal["text", "base64"] = "text"


class PathSeed(BaseModel):
    """A file on the authoring host. Resolvable only where that path exists (local runs)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["path"] = "path"
    path: str = Field(description="Filesystem path on the authoring host (absolute or relative to the cwd).")


@runtime_checkable
class SeedHandler(Protocol):
    """Parses + resolves one seed ``kind`` into the bytes to stage.

    Consumers implement this for kinds the SDK doesn't ship (e.g. a platform ``fileset`` handler) and
    wire them in with :func:`register_seed_handler`. ``resolve`` runs at seeding time, so a handler
    that needs external services (a client, credentials) acquires them there.
    """

    kind: str

    def parse(self, value: Mapping[str, Any]) -> BaseModel:
        """Validate a raw seed mapping into this kind's typed model."""
        ...

    def resolve(self, seed: BaseModel) -> bytes:
        """Resolve a parsed seed to the bytes to write into the workspace."""
        ...


_HANDLERS: dict[str, SeedHandler] = {}


def register_seed_handler(handler: SeedHandler) -> None:
    """Register a :class:`SeedHandler` under its ``kind`` (replacing any handler already registered)."""
    _HANDLERS[handler.kind] = handler


def _handler_for(kind: str) -> SeedHandler:
    handler = _HANDLERS.get(kind)
    if handler is None:
        raise WorkspaceSeedError(f"no handler registered for seed kind {kind!r}")
    return handler


class _InlineSeedHandler:
    kind = "inline"

    def parse(self, value: Mapping[str, Any]) -> BaseModel:
        return InlineSeed.model_validate(value)

    def resolve(self, seed: BaseModel) -> bytes:
        assert isinstance(seed, InlineSeed)
        if seed.encoding == "base64":
            try:
                return base64.b64decode(seed.content, validate=True)
            except (ValueError, TypeError) as exc:
                raise WorkspaceSeedError(f"inline seed is not valid base64: {exc}") from exc
        return seed.content.encode("utf-8")


class _PathSeedHandler:
    kind = "path"

    def parse(self, value: Mapping[str, Any]) -> BaseModel:
        return PathSeed.model_validate(value)

    def resolve(self, seed: BaseModel) -> bytes:
        assert isinstance(seed, PathSeed)
        source = Path(seed.path).expanduser()
        try:
            return source.read_bytes()
        except OSError as exc:
            raise WorkspaceSeedError(f"path seed {seed.path!r} could not be read: {exc}") from exc


register_seed_handler(_InlineSeedHandler())
register_seed_handler(_PathSeedHandler())


def parse_seed(value: str | Mapping[str, Any]) -> BaseModel:
    """Coerce a seed map value into its validated model. A bare string is inline UTF-8 text."""
    if isinstance(value, str):
        return InlineSeed(content=value)
    if not isinstance(value, Mapping):
        raise WorkspaceSeedError(f"seed must be a string or mapping, got {type(value).__name__}")
    kind = value.get("kind")
    if not isinstance(kind, str):
        raise WorkspaceSeedError("seed mapping is missing a string 'kind'")
    handler = _handler_for(kind)
    try:
        return handler.parse(value)
    except WorkspaceSeedError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize a handler's validation error into our type
        raise WorkspaceSeedError(f"invalid {kind!r} seed: {exc}") from exc


def _resolve_seed_bytes(seed: BaseModel) -> bytes:
    """Resolve a parsed seed to bytes via its registered handler."""
    kind = getattr(seed, "kind", None)
    if not isinstance(kind, str):
        raise WorkspaceSeedError("parsed seed has no string 'kind'")
    return _handler_for(kind).resolve(seed)


def seed_workspace(workspace_dir: str | Path, files: Mapping[str, Any] | None) -> list[str]:
    """Write the ``files`` seed map into ``workspace_dir``; return the seeded relative paths.

    Each value is parsed into a seed model and resolved to bytes by its registered handler. Paths that
    escape the workspace (absolute, or ``..`` traversal) are rejected so a task can only stage files
    inside its own sandbox. ``None``/empty seeds nothing.
    """
    if not isinstance(files, Mapping):
        return []
    root = Path(workspace_dir).resolve()
    written: list[str] = []
    for rel_path, value in files.items():
        target = (root / str(rel_path)).resolve()
        if target != root and root not in target.parents:
            raise WorkspaceSeedError(f"seed file path escapes the workspace: {rel_path!r}")
        data = _resolve_seed_bytes(parse_seed(value))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        written.append(str(rel_path))
    return written
