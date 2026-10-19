# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed reference strings for plugin spec fields.

Plugin job specs frequently accept fields that may be either a *reference*
to a platform-managed entity (a bare name, optionally with a workspace
prefix) or a *literal* value of the same wire shape — typically an HTTP
endpoint URL, or a local filesystem path.  Rather than expose two
mutually-exclusive scalar fields and validate "exactly one of" at the
spec layer, plugins model the field once with a union of two ``str``
subclasses and disambiguate by **shape** at run time.

The pattern (and the :class:`StrRef` base hook) follows the
entity-references RFC sketched in the LJ-3 draft of
``nemo_helix_ext.refs``.  This module is the canonical home for the
generic ref types — plugin-specific subclasses (e.g. ``AgentRef``) live
alongside the job that owns them and just inherit from :class:`StrRef`.

Why subclasses of ``str`` rather than a discriminated Pydantic union:

- The wire format stays a single string.  ``--spec`` JSON, REST bodies,
  and YAML configs round-trip without a tagged ``{"kind": "...",
  "value": "..."}`` envelope.
- The auto-generated CLI sees a ``str``-like flag (one ``--output`` flag,
  not ``--output.kind`` + ``--output.value``); the
  :mod:`nemo_helix_plugin._spec_flags` generator collapses unions whose
  arms all subclass the same scalar base into a single flag of that base
  type.
- Disambiguation happens at the resolver, where it can produce an
  actionable error message ("requires NHX_BASE_URL", "URL has no
  scheme", ...).  Pydantic itself never branches on which arm a value
  belongs to; both arms accept any string.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, ClassVar, Union

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema


class StrRef(str):
    """``str`` subclass that teaches Pydantic to coerce values back to *cls*.

    Without this hook Pydantic refuses to derive a core schema for a
    bare ``str`` subclass and requires ``arbitrary_types_allowed=True``
    on every model that uses it.  The hook validates the value as a
    string then wraps it in ``cls(value)`` so ``isinstance`` checks
    downstream (in tests, in resolvers) keep working.

    Subclasses may set :attr:`__cli_metavar__` to a short uppercase
    placeholder (e.g. ``"PATH"``).  The auto-generated CLI joins the
    metavars of every union arm with ``" | "`` to render the
    ``--help`` placeholder — a field typed as ``LocalDir | FilesetRef``
    surfaces as ``--output <PATH | FILESET_REF>``.  ``None`` (the
    default) tells the generator to fall back to upper-casing the
    flag name.
    """

    __cli_metavar__: ClassVar[str | None] = None

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        del source_type
        return core_schema.no_info_after_validator_function(cls, handler(str))


class EndpointURL(StrRef):
    """A literal HTTP(S) endpoint URL.

    Used as-is by callers that already know how to talk to an HTTP
    service (e.g. ``nat eval --endpoint``).  Useful for pointing a
    job at services that do not run on the platform, or for overriding
    the gateway URL during debugging.
    """

    __cli_metavar__: ClassVar[str | None] = "URL"


class LocalDir(StrRef):
    """A local filesystem directory path.

    Path-shaped strings (``./out``, ``../out``, ``/abs/out``, ``~/out``,
    backslash-bearing values for Windows) classify as :class:`LocalDir`.
    Consumers typically treat the value as a literal :class:`~pathlib.Path`
    and create the directory on demand.
    """

    __cli_metavar__: ClassVar[str | None] = "PATH"


#: Regex form of the shape :func:`parse_entity_ref` accepts: ``name`` or ``workspace/name``, each
#: segment using the platform name charset.  Pydantic fields that hold a reference declare
#: ``pattern=ENTITY_REF_PATTERN`` so a malformed ref is rejected at validation rather than surfacing
#: as a confusing failure during parsing; :func:`parse_entity_ref` then only has to split.  Kept
#: beside the parser so the two cannot drift apart.
ENTITY_REF_PATTERN = r"^[\w\-.]+(/[\w\-.]+)?$"


class FilesetRef(StrRef):
    """A NeMo Helix fileset reference (``"name"`` or ``"workspace/name"``).

    Consumers typically use the typed Files-service client to upload to
    or download from the named fileset, creating it on demand if missing
    (``client.files.upload(..., fileset_auto_create=True)``).
    """

    __cli_metavar__: ClassVar[str | None] = "FILESET_REF"


#: A reference to a *file inside* a fileset: ``workspace/fileset#path/inside.ext``.  Unlike
#: :data:`ENTITY_REF_PATTERN` the workspace is mandatory (a stored reference must be unambiguous
#: wherever it is later read from), and the ``#`` fragment is a file path, so it admits ``/`` and
#: ``.``.  Declared as a field pattern so a malformed reference is rejected when it is stored rather
#: than surfacing as a download failure mid-run.
FILESET_REF_PATTERN = r"^[\w\-.]+/[\w\-.]+#[\w\-./]+$"


# Documentary union alias — the wire shape is still ``str``.  The
# ``_spec_flags`` generator collapses this to a single ``--output`` flag
# of type ``str``; the disambiguation between the two arms happens in
# :func:`classify_output_target` at job-run time.
OutputTarget = Union[LocalDir, FilesetRef]
"""A target for job outputs: local directory *or* NeMo Helix fileset."""


_LOCAL_DIR_PREFIXES = ("/", "./", "../", "~/")
# Windows absolute paths with forward slashes ("C:/tmp/out") aren't caught
# by the prefix list or the "\\" check.  Match a single drive letter
# followed by ``:`` and either separator so jobs running on a Windows
# host don't misclassify their --output as a fileset reference.
_WINDOWS_DRIVE_PATH = re.compile(r"^[A-Za-z]:[/\\]")


def classify_output_target(value: str) -> type[StrRef]:
    """Classify *value* as a :class:`LocalDir` or :class:`FilesetRef`.

    Path-shaped values resolve to :class:`LocalDir`; everything else to
    :class:`FilesetRef`.  This mirrors the LJ-3 RFC's rule that callers
    who want a local-path interpretation must mark it explicitly
    (``./out``) — bare names default to the platform interpretation.

    Path-shape markers:

    - ``/abs``, ``./rel``, ``../parent``, ``~/home`` (POSIX);
    - any value containing ``"\\"`` (Windows path separator);
    - ``C:/...`` / ``C:\\...`` (Windows drive-letter absolute paths);
    - the bare ``~`` (home directory shorthand).
    """
    if (
        value == "~"
        or value.startswith(_LOCAL_DIR_PREFIXES)
        or "\\" in value
        or _WINDOWS_DRIVE_PATH.match(value) is not None
    ):
        return LocalDir
    return FilesetRef


@dataclass
class ParsedEntityRef:
    """Parsed entity reference with workspace and name."""

    workspace: str
    name: str


def parse_entity_ref(identifier: str, default_workspace: str | None = None) -> ParsedEntityRef:
    """Parse an entity identifier into workspace and name.

    Accepted formats:

    - ``entity_name``             — uses *default_workspace*
    - ``workspace/entity_name``   — explicit workspace

    Raises :class:`ValueError` when the identifier has more than one ``/``,
    any segment is empty, or *default_workspace* is ``None`` for an
    unqualified name.
    """
    parts = identifier.strip().split("/")
    if len(parts) > 2 or any(p == "" for p in parts):
        raise ValueError(f"invalid entity reference {identifier!r}; expected 'name' or 'workspace/name'")

    if len(parts) == 2:
        return ParsedEntityRef(workspace=parts[0], name=parts[1])

    if default_workspace is None:
        raise ValueError(
            f"Entity identifier '{identifier}' is not qualified with a workspace and default workspace is not provided. "
            "Must be in the format $workspace/$entity_name or $entity_name."
        )

    return ParsedEntityRef(workspace=default_workspace, name=parts[0])


__all__ = [
    "ENTITY_REF_PATTERN",
    "FILESET_REF_PATTERN",
    "EndpointURL",
    "FilesetRef",
    "LocalDir",
    "OutputTarget",
    "ParsedEntityRef",
    "StrRef",
    "classify_output_target",
    "parse_entity_ref",
]
