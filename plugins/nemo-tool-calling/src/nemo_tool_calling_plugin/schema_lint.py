# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic lint of MCP tool input schemas for shapes that local models call badly.

Order-constrained tool-call grammars (llama.cpp) emit required keys first and optional keys only in
declared order, so open objects, long optional tails and prose-only rules turn into corrupted or
missing keys. See NVIDIA-NeMo/nemo-helix#3857 for the schema this was distilled from.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

Severity = Literal["high", "medium", "low"]

MAX_OPTIONAL_KEYS = 5
MIN_DISCRIMINATED_OPTIONAL_KEYS = 3
_QUOTED_VALUE = re.compile(r"\"([^\"\s]{1,40})\"")
_ENUM_HINT = re.compile(r"one of|\||either", re.IGNORECASE)
_DISCRIMINATOR_NAME = re.compile(r"(^|_)(type|kind)$")
_VARIANT_KEYS = ("anyOf", "oneOf", "allOf")


@dataclass(frozen=True)
class LintFinding:
    rule: str
    severity: Severity
    tool: str
    path: str
    title: str
    detail: str
    evidence: tuple[str, ...] = field(default_factory=tuple)

    @property
    def id(self) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", self.path.lower()).strip("-") or "root"
        return f"lint-{self.rule}-{slug}"

    def to_finding(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": "schema-lint",
            "severity": self.severity,
            "tool": self.tool,
            "title": self.title,
            "detail": self.detail,
            "evidence": list(self.evidence),
        }


def _walk(schema: Any, path: str) -> Iterator[tuple[str, Mapping[str, Any]]]:
    if not isinstance(schema, Mapping):
        return
    yield path, schema
    for name, child in (schema.get("properties") or {}).items():
        yield from _walk(child, f"{path}.{name}" if path else name)
    items = schema.get("items")
    if isinstance(items, Mapping):
        yield from _walk(items, f"{path}[]")
    additional = schema.get("additionalProperties")
    if isinstance(additional, Mapping):
        yield from _walk(additional, f"{path}{{*}}")
    for key in _VARIANT_KEYS:
        for index, variant in enumerate(schema.get(key) or []):
            yield from _walk(variant, f"{path}.{key}[{index}]" if path else f"{key}[{index}]")
    for name, definition in (schema.get("$defs") or {}).items():
        yield from _walk(definition, f"$defs.{name}")


def _label(path: str) -> str:
    return path or "<arguments>"


def _has_shape(schema: Mapping[str, Any]) -> bool:
    return any(key in schema for key in ("type", "properties", "$ref", "enum", "const", *_VARIANT_KEYS))


def _open_object(tool: str, path: str, schema: Mapping[str, Any]) -> Iterator[LintFinding]:
    properties = schema.get("properties") or {}
    additional = schema.get("additionalProperties")
    if not properties or additional is False or isinstance(additional, Mapping):
        return
    explicit = additional is True
    yield LintFinding(
        rule="open-object",
        severity="high" if explicit or len(properties) > MAX_OPTIONAL_KEYS else "medium",
        tool=tool,
        path=path,
        title=f"{_label(path)} allows any key",
        detail=f"{'additionalProperties: true' if explicit else 'No additionalProperties: false'} on "
        f"{_label(path)}; a misspelled or invented key passes the schema and fails in the tool.",
    )


def _many_optional(tool: str, path: str, schema: Mapping[str, Any]) -> Iterator[LintFinding]:
    properties = list((schema.get("properties") or {}).keys())
    required = set(schema.get("required") or [])
    optional = [name for name in properties if name not in required]
    if len(optional) <= MAX_OPTIONAL_KEYS:
        return
    closed = schema.get("additionalProperties") is False
    yield LintFinding(
        rule="many-optional",
        severity="medium" if closed else "high",
        tool=tool,
        path=path,
        title=f"{_label(path)} has {len(optional)} optional keys",
        detail=f"{len(optional)} optional keys on one object ({', '.join(optional[:8])}"
        f"{', ...' if len(optional) > 8 else ''}); split it into variants with few optional keys each.",
    )


def _required_not_first(tool: str, path: str, schema: Mapping[str, Any]) -> Iterator[LintFinding]:
    properties = list((schema.get("properties") or {}).keys())
    required = set(schema.get("required") or [])
    seen_optional: str | None = None
    for name in properties:
        if name not in required:
            seen_optional = seen_optional or name
            continue
        if seen_optional is not None:
            yield LintFinding(
                rule="required-not-first",
                severity="medium",
                tool=tool,
                path=path,
                title=f"Required key {name!r} declared after optional {seen_optional!r}",
                detail=f"On {_label(path)}, declare required keys first: grammars that follow declaration order "
                "write optional keys before the model has committed to the required ones.",
            )
            return


def _untyped(tool: str, path: str, schema: Mapping[str, Any]) -> Iterator[LintFinding]:
    if schema.get("type") == "array":
        items = schema.get("items")
        if not isinstance(items, Mapping) or not _has_shape(items) or _bare_object(items):
            yield LintFinding(
                rule="untyped-items",
                severity="medium",
                tool=tool,
                path=path,
                title=f"{_label(path)} items have no shape",
                detail=f"Items of {_label(path)} are untyped or a bare object; the expected keys exist only in prose.",
            )
        return
    if _bare_object(schema) and path:
        yield LintFinding(
            rule="untyped-object",
            severity="low",
            tool=tool,
            path=path,
            title=f"{_label(path)} is an object with no properties",
            detail=f"{_label(path)} is {{type: object}} with no properties; the model has to guess its keys.",
        )


def _bare_object(schema: Mapping[str, Any]) -> bool:
    return (
        schema.get("type") == "object"
        and not schema.get("properties")
        and not isinstance(schema.get("additionalProperties"), Mapping)
        and not any(key in schema for key in _VARIANT_KEYS)
    )


def _enum_in_prose(tool: str, path: str, schema: Mapping[str, Any]) -> Iterator[LintFinding]:
    if schema.get("type") != "string" or any(key in schema for key in ("enum", "const", "pattern", "format")):
        return
    description = str(schema.get("description") or "")
    values = _QUOTED_VALUE.findall(description)
    if len(values) < 2 or not _ENUM_HINT.search(description):
        return
    yield LintFinding(
        rule="enum-in-prose",
        severity="medium",
        tool=tool,
        path=path,
        title=f"{_label(path)} lists its values only in the description",
        detail=f"{_label(path)} is a free string whose description names {', '.join(values[:6])}; use an enum.",
    )


def _const(tool: str, path: str, schema: Mapping[str, Any]) -> Iterator[LintFinding]:
    if "const" not in schema:
        return
    yield LintFinding(
        rule="const",
        severity="low",
        tool=tool,
        path=path,
        title=f"{_label(path)} uses const",
        detail=f"Use enum: [{schema['const']!r}] instead of const on {_label(path)}; some OpenAPI-subset "
        "backends reject const.",
    )


def _discriminator_in_prose(tool: str, path: str, schema: Mapping[str, Any]) -> Iterator[LintFinding]:
    properties: Mapping[str, Any] = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    optional = [name for name in properties if name not in required]
    if len(optional) < MIN_DISCRIMINATED_OPTIONAL_KEYS or any(key in schema for key in _VARIANT_KEYS):
        return
    for name, child in properties.items():
        values = child.get("enum") if isinstance(child, Mapping) else None
        described = isinstance(child, Mapping) and "discriminator" in str(child.get("description", "")).lower()
        if not isinstance(values, list) or len(values) < 2:
            continue
        if not (_DISCRIMINATOR_NAME.search(name) or described):
            continue
        yield LintFinding(
            rule="discriminator-in-prose",
            severity="high",
            tool=tool,
            path=path,
            title=f"{_label(path)} switches on {name!r} without variants",
            detail=f"{name!r} selects one of {len(values)} shapes, but {_label(path)} is one object with "
            f"{len(optional)} optional keys, so which keys each {name} needs is only in prose. Express each value "
            "as its own anyOf variant with its required keys.",
        )
        return


_RULES = (
    _open_object,
    _many_optional,
    _required_not_first,
    _untyped,
    _enum_in_prose,
    _const,
    _discriminator_in_prose,
)


def lint_tool(tool: Mapping[str, Any]) -> list[LintFinding]:
    """Lint one tool definition (``name``, ``description``, ``inputSchema``)."""
    name = str(tool.get("name") or "tool")
    schema = tool.get("inputSchema") or tool.get("input_schema") or {}
    findings: list[LintFinding] = []
    for path, node in _walk(schema, ""):
        for rule in _RULES:
            findings.extend(rule(name, path, node))
    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(findings, key=lambda finding: order[finding.severity])
