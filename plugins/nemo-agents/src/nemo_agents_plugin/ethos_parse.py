# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse and lightly validate ETHOS.md.

This module intentionally does not model every markdown section as structured
Python. It validates the machine-readable front matter and the required section
outline, then returns raw markdown sections for humans and agents to consume.

A missing canonical body section raises. Extra ``##`` headings are kept.
Unknown YAML front-matter keys are ignored and do not fail the parse.
``strict`` turns remaining warnings into errors, which today means an
unversioned file that was parsed as version 1.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import yaml
from nemo_agents_plugin.ethos import (
    ETHOS_SCHEMA_VERSION,
    Ethos,
    required_sections,
)


class EthosParseError(ValueError):
    """Raised when ETHOS.md cannot be parsed or fails lightweight validation."""


_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_SECTION_RE = re.compile(r"^## +(.+?)\s*$", re.MULTILINE)


def parse_ethos(markdown: str, *, strict: bool = False) -> Ethos:
    """Parse ETHOS.md into front matter plus raw markdown sections.

    Args:
        markdown: Full ETHOS.md contents, including front matter.
        strict: Turn remaining warnings into errors. Use this when the caller
            cannot act on an unversioned contract. Extra headings and extra
            front-matter keys are still allowed.

    Raises:
        EthosParseError: Front matter is missing or malformed, a required
            section heading is absent, or ``strict`` is set and a warning was
            raised. Unknown headings are not an error.
    """

    front_match = _FRONT_MATTER_RE.match(markdown)
    if front_match is None:
        raise EthosParseError("missing YAML front matter")

    front = yaml.safe_load(front_match.group(1)) or {}
    if not isinstance(front, dict):
        raise EthosParseError("YAML front matter must be a mapping")

    version, version_warnings = _schema_version(front)
    sections = _split_sections(markdown[front_match.end() :])

    for title in required_sections(version):
        if title not in sections:
            raise EthosParseError(f"missing section: ## {title}")

    # Version 1 dropped ``Framework``: nothing read its value, and the container's
    # framework label comes from ``agent.yaml`` instead. A leftover heading must
    # still be resolved if it is present.
    if "Framework" in sections:
        framework = sections["Framework"].strip()
        if not framework or framework == "_(none)_":
            raise EthosParseError("framework section must be resolved")

    warnings = [*version_warnings]

    if strict and warnings:
        raise EthosParseError("; ".join(warnings))

    return Ethos(
        name=_required_str(front, "name"),
        created_timestamp=_required_datetime(front, "created_timestamp"),
        author=_required_str(front, "author"),
        sections=sections,
        schema_version=version,
        updated_timestamp=_optional_datetime(front, "updated_timestamp"),
        owner=_optional_str(front, "owner"),
        warnings=tuple(warnings),
    )


def _schema_version(front: dict[str, Any]) -> tuple[int, list[str]]:
    """Resolve ``schema_version``, defaulting to 1 for pre-versioning files."""
    raw = front.get("schema_version")
    if raw is None:
        return 1, [
            f"front matter has no 'schema_version'; parsed as version 1. Add 'schema_version: {ETHOS_SCHEMA_VERSION}'."
        ]
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise EthosParseError("front matter field 'schema_version' must be an integer")
    if raw < 1:
        raise EthosParseError("front matter field 'schema_version' must be 1 or greater")
    if raw > ETHOS_SCHEMA_VERSION:
        raise EthosParseError(
            f"ETHOS.md declares schema version {raw}, but this tooling supports "
            f"up to {ETHOS_SCHEMA_VERSION}. Upgrade nemo-helix to read it."
        )
    return raw, []


def _required_str(front: dict[str, Any], key: str) -> str:
    value = front.get(key)
    if not isinstance(value, str) or not value.strip():
        raise EthosParseError(f"front matter field {key!r} is required")
    return value.strip()


def _optional_str(front: dict[str, Any], key: str) -> str | None:
    value = front.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise EthosParseError(f"front matter field {key!r} must be a non-empty string when present")
    return value.strip()


def _coerce_datetime(value: Any, key: str) -> datetime:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value.strip():
        raise EthosParseError(f"front matter field {key!r} is required")
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise EthosParseError(f"front matter field {key!r} must be an ISO 8601 datetime") from exc


def _required_datetime(front: dict[str, Any], key: str) -> datetime:
    return _coerce_datetime(front.get(key), key)


def _optional_datetime(front: dict[str, Any], key: str) -> datetime | None:
    value = front.get(key)
    return None if value is None else _coerce_datetime(value, key)


def _split_sections(body: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    matches = list(_SECTION_RE.finditer(body))
    for i, match in enumerate(matches):
        header = match.group(1).strip()
        if header in sections:
            raise EthosParseError(f"duplicate section: ## {header}")
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections[header] = body[start:end].strip("\n")
    return sections
