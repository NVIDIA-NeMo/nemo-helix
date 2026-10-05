#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema validation for the audit-spec coverage denominator."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from _markdown import extract_schema_block

AUDIT_SCHEMA = "nemo.eval_author.audit.v1"
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "audit.schema.json"
ITEM_KINDS = frozenset({"tool", "capability", "failure_case"})
_ZERO_SOURCE_DIGEST = "sha256:" + ("0" * 64)


class AuditSpecError(ValueError):
    """Raised when an audit spec fails schema validation."""


class AuditEnvironmentError(RuntimeError):
    """Raised when the validator cannot load required local dependencies."""


def load_audit_spec(path: Path) -> dict[str, Any]:
    """Load and validate an ``audit.md`` file."""
    block = extract_schema_block(path)
    try:
        import yaml
    except ImportError as exc:
        raise AuditEnvironmentError("PyYAML is required to parse audit specs") from exc

    try:
        payload = yaml.safe_load(block)
    except yaml.YAMLError as exc:
        raise AuditSpecError(f"audit YAML does not parse: {exc}") from exc
    return validate_audit_spec(payload, audit_path=path)


def validate_audit_spec(payload: Any, *, audit_path: Path | None = None) -> dict[str, Any]:
    """Return *payload* when it satisfies the audit spec."""
    _validate_json_schema(payload)
    if not isinstance(payload, dict):
        raise AuditSpecError("audit spec must be a mapping")
    _validate_sources(payload, audit_path=audit_path)
    return _validate_semantics(payload)


def item_counts(spec: dict[str, Any]) -> dict[str, int]:
    """Return item counts by kind."""
    counts = Counter(item["kind"] for item in spec["items"])
    return {kind: counts.get(kind, 0) for kind in sorted(ITEM_KINDS)}


def _validate_json_schema(payload: Any) -> None:
    try:
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import SchemaError
    except ImportError as exc:
        raise AuditEnvironmentError("jsonschema is required to validate audit specs") from exc

    try:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditEnvironmentError(f"could not load audit JSON Schema from {SCHEMA_PATH}: {exc}") from exc

    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise AuditEnvironmentError(f"bundled audit JSON Schema is invalid: {exc.message}") from exc
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda error: (list(error.absolute_path), error.message))
    if errors:
        raise AuditSpecError("\n".join(_format_schema_error(error) for error in errors))


def _validate_semantics(payload: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    items = payload["items"]
    source_names: set[str] = set()
    item_names: set[str] = set()
    tool_names: set[str] = set()
    capability_names: set[str] = set()

    # Source records must have unique names so provenance references are unambiguous.
    for index, source in enumerate(payload.get("sources", [])):
        name = _check_name(f"audit.sources[{index}].name", source.get("name"), errors)
        if name is not None:
            if name in source_names:
                errors.append(f"audit.sources[{index}].name {name!r} is duplicated")
            source_names.add(name)

    # Item names must be globally unique because name is the stable coverage key.
    for index, item in enumerate(items):
        path = f"audit.items[{index}]"
        kind = item.get("kind")
        name = _check_name(f"{path}.name", item.get("name"), errors)
        if name is not None:
            if name in item_names:
                errors.append(f"{path}.name {name!r} is duplicated")
            item_names.add(name)
            if kind == "tool":
                tool_names.add(name)
            elif kind == "capability":
                capability_names.add(name)

    if errors:
        raise AuditSpecError("\n".join(errors))

    # Cross-item references must resolve after tool and capability namespaces are known.
    for index, item in enumerate(items):
        path = f"audit.items[{index}]"
        # Required and expected tools must reference declared tool items.
        for field in ("required_tools", "expected_tools"):
            _check_known_tools(f"{path}.{field}", item.get(field), tool_names, errors)
        # Evidence tool references must reference declared tool items.
        for evidence_index, evidence in enumerate(item["evidence_required"]):
            if "tool" in evidence:
                _check_known_tools(
                    f"{path}.evidence_required[{evidence_index}].tool", [evidence["tool"]], tool_names, errors
                )
        if item["kind"] == "failure_case":
            # Failure cases must apply only to declared capability items.
            for ref in item["applies_to"]:
                if ref not in capability_names:
                    errors.append(f"{path}.applies_to references unknown capability name {ref!r}")

    if errors:
        raise AuditSpecError("\n".join(errors))
    return payload


def _validate_sources(payload: dict[str, Any], *, audit_path: Path | None) -> None:
    errors: list[str] = []
    for index, source in enumerate(payload.get("sources", [])):
        digest = source.get("sha256")
        if digest is None:
            continue
        path = f"audit.sources[{index}].sha256"
        if digest == _ZERO_SOURCE_DIGEST:
            errors.append(f"{path} must not be the all-zero placeholder digest")
            continue
        if audit_path is None:
            continue

        source_path = Path(source["path"])
        if not source_path.is_absolute():
            source_path = audit_path.parent / source_path
        try:
            with source_path.open("rb") as stream:
                actual = f"sha256:{hashlib.file_digest(stream, 'sha256').hexdigest()}"
        except OSError as exc:
            errors.append(f"audit.sources[{index}].path could not be read at {source_path}: {exc}")
            continue
        if actual != digest:
            errors.append(f"{path} does not match {source_path}: expected {digest}, got {actual}")

    if errors:
        raise AuditSpecError("\n".join(errors))


def _check_name(path: str, value: Any, errors: list[str]) -> str | None:
    if not isinstance(value, str):
        errors.append(f"{path} must be a string")
        return None
    return value


def _check_known_tools(path: str, values: Any, tool_names: set[str], errors: list[str]) -> None:
    if values is None:
        return
    for value in values:
        if value not in tool_names:
            errors.append(f"{path} references unknown tool name {value!r}")


def _format_schema_error(error: Any) -> str:
    if error.validator == "oneOf":
        matching_kind_errors = list(_matching_kind_context_errors(error))
        if matching_kind_errors:
            return "\n".join(_format_schema_error(context) for context in matching_kind_errors)
    return f"{_json_path(error.absolute_path)}: {error.message}"


def _matching_kind_context_errors(error: Any) -> Iterable[Any]:
    instance = error.instance
    if not isinstance(instance, dict):
        return ()
    kind = instance.get("kind")
    if not isinstance(kind, str):
        return ()
    return (
        context
        for context in error.context
        if _schema_kind_matches(context.schema.get("properties", {}).get("kind", {}), kind)
    )


def _schema_kind_matches(kind_schema: dict[str, Any], kind: str) -> bool:
    return kind_schema.get("const") == kind or kind in kind_schema.get("enum", [])


def _json_path(parts: Iterable[Any]) -> str:
    path = "audit"
    for part in parts:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            path += f".{part}"
    return path
