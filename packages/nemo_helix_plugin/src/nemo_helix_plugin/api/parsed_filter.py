# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ParsedFilter — a single-source-of-truth wrapper around a FilterOperation tree.

Services receive a ``ParsedFilter`` from ``make_filter_dep``.  The operation
tree is the canonical query — services that need to inspect or remove specific
fields use ``extract`` / ``remove``.  The full (possibly mutated) tree is then
forwarded to the entity store.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from fastapi import HTTPException, Request, status
from nemo_helix_plugin.api.filter import (
    ComparisonOperation,
    FilterOperation,
    FilterOperator,
    LogicalOperation,
    parse_bracket_filter,
    parse_json_filter,
)
from nemo_helix_plugin.api.text_filter import parse_text_filter
from nemo_helix_plugin.jobs.openapi_utils import parse_deep_object
from pydantic import BaseModel

# Top-level entity fields that exist as columns rather than under data.*
# Filter models implicitly extend this set via _get_valid_fields.
ENTITY_BASE_FIELDS: set[str] = {"id", "name", "workspace", "created_at", "updated_at", "entity_type", "project"}


def _reverse_translate_dict(d: dict[str, Any], reverse_map: dict[str, str]) -> dict[str, Any]:
    """Reverse-translate entity field names back to user-facing names in a dict tree."""
    result: dict[str, Any] = {}
    for key, value in d.items():
        if key in ("$and", "$or"):
            result[key] = [
                _reverse_translate_dict(item, reverse_map) if isinstance(item, dict) else item for item in value
            ]
        elif key == "$not":
            result[key] = _reverse_translate_dict(value, reverse_map) if isinstance(value, dict) else value
        else:
            new_key = reverse_map.get(key, key)
            result[new_key] = value
    return result


def _get_valid_fields(filter_model: type[BaseModel]) -> set[str]:
    return set(filter_model.model_fields.keys()) | ENTITY_BASE_FIELDS


def _operation_references_any(operation: FilterOperation | None, fields: set[str]) -> bool:
    """Return True if any ``ComparisonOperation`` in the tree targets a field in ``fields``."""
    if operation is None:
        return False
    if isinstance(operation, ComparisonOperation):
        return operation.field in fields
    if isinstance(operation, LogicalOperation):
        return any(_operation_references_any(child, fields) for child in operation.operations)
    return False


def _validate_fields(
    operation: FilterOperation, valid_fields: set[str], namespaces: frozenset[str] = frozenset()
) -> None:
    """Walk the operation tree and raise ValueError for any unknown field names.

    Validates that each field is one of:

    * a known filter/entity field, or
    * a ``data.*`` path (which addresses fields stored in the entity's data JSON
      column), or
    * a sub-path of a declared *namespace* field — e.g. ``labels.<key>`` when the
      filter model maps ``labels`` to a nested entity namespace. ``namespaces`` is
      supplied generically by the model; this function has no per-service knowledge.
    """
    if isinstance(operation, ComparisonOperation):
        f = operation.field
        if f.startswith("data."):
            return
        if "." in f:
            # Dotted sub-paths are only valid under a declared namespace field
            # (``labels.<key>`` → ``data.labels.<key>``). Otherwise reject, so
            # paths like ``name.foo`` can't smuggle past validation.
            if f.split(".", 1)[0] in namespaces:
                return
            raise ValueError(f"Unknown filter field '{f}'. Valid fields: {sorted(valid_fields)}")
        if f not in valid_fields:
            raise ValueError(f"Unknown filter field '{f}'. Valid fields: {sorted(valid_fields)}")
    elif isinstance(operation, LogicalOperation):
        for child in operation.operations:
            _validate_fields(child, valid_fields, namespaces)


@dataclass
class ParsedFilter:
    """Parsed, validated, and entity-translated filter.

    ``operation`` is the single source of truth for the entity query.
    Use ``extract`` to read a field value and ``remove`` to strip a field
    before forwarding to the entity store.
    """

    operation: FilterOperation | None
    _field_map: dict[str, str] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def extract(self, field_name: str) -> Any | None:
        """Return the value of a top-level ``$eq`` field, or ``None``.

        Looks at the entity-mapped field name (e.g., ``data.purpose`` not
        ``purpose``).  For convenience, also accepts the *unmapped* name and
        checks the filter model's field map.
        """
        mapped = self._resolve_field(field_name)
        if self.operation is None:
            return None
        for op in self._top_level_operands():
            if isinstance(op, ComparisonOperation) and op.field == mapped and op.operator == FilterOperator.EQ:
                return op.value
        return None

    def remove(self, field_name: str) -> Any | None:
        """Remove a top-level ``$eq`` field from the operation tree and return its value.

        Returns ``None`` if the field was not present.
        """
        mapped = self._resolve_field(field_name)
        if self.operation is None:
            return None

        if isinstance(self.operation, ComparisonOperation):
            if self.operation.field == mapped and self.operation.operator == FilterOperator.EQ:
                value = self.operation.value
                self.operation = None
                return value
            return None

        if isinstance(self.operation, LogicalOperation) and self.operation.operator == FilterOperator.AND:
            remaining: list[FilterOperation] = []
            removed_value = None
            for op in self.operation.operations:
                if (
                    isinstance(op, ComparisonOperation)
                    and op.field == mapped
                    and op.operator == FilterOperator.EQ
                    and removed_value is None
                ):
                    removed_value = op.value
                else:
                    remaining.append(op)
            if removed_value is not None:
                if len(remaining) == 0:
                    self.operation = None
                elif len(remaining) == 1:
                    self.operation = remaining[0]
                else:
                    self.operation = LogicalOperation(operator=FilterOperator.AND, operations=remaining)
            return removed_value

        return None

    def has(self, field_name: str) -> bool:
        """Return True if any ``ComparisonOperation`` in the tree references ``field_name``.

        Operator-agnostic and tree-walking — unlike ``extract`` and ``remove``,
        which only inspect top-level ``$eq`` operands.  Use this to assert a
        field is *absent* from the filter (e.g. cross-entity guards that only
        support a top-level shape).

        Accepts the user-facing name or the entity-mapped name and matches
        against both forms in the tree.  A caller asking about ``lora_enabled``
        therefore catches a tree containing either ``lora_enabled`` (when the
        field map hasn't been applied) or ``data.lora_enabled`` (post-translate).
        """
        candidates = {field_name, self._resolve_field(field_name)}
        return _operation_references_any(self.operation, candidates)

    def and_with(self, extra: FilterOperation) -> None:
        """AND-merge ``extra`` into the operation tree.

        Use this to compose cross-entity conditions (resolved by the service
        layer) with the user-supplied filter, so the canonical operation tree
        remains the single source of truth forwarded to the entity store.
        """
        if self.operation is None:
            self.operation = extra
            return
        if isinstance(self.operation, LogicalOperation) and self.operation.operator == FilterOperator.AND:
            self.operation = LogicalOperation(
                operator=FilterOperator.AND,
                operations=[*self.operation.operations, extra],
            )
            return
        self.operation = LogicalOperation(
            operator=FilterOperator.AND,
            operations=[self.operation, extra],
        )

    def to_response(self) -> dict[str, Any] | None:
        """Serialize for the API response ``filter`` field.

        Reverses the entity field mapping so the response uses user-facing
        field names (e.g., ``dataset`` instead of ``data.dataset``).
        """
        if self.operation is None:
            return None
        if self._field_map:
            reverse_map = {entity: user for user, entity in self._field_map.items()}
            return _reverse_translate_dict(self.operation.to_dict(), reverse_map)
        return self.operation.to_dict()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_field(self, field_name: str) -> str:
        """Map a user-facing field name to its entity-store name."""
        if field_name.startswith("data."):
            return field_name
        if field_name in self._field_map:
            return self._field_map[field_name]
        return field_name

    def _top_level_operands(self) -> list[FilterOperation]:
        if self.operation is None:
            return []
        if isinstance(self.operation, LogicalOperation) and self.operation.operator == FilterOperator.AND:
            return list(self.operation.operations)
        return [self.operation]


def make_filter_dep(
    filter_model: type[BaseModel],
    *,
    param_name: str = "filter",
) -> Callable[..., Awaitable[ParsedFilter]]:
    """Create a FastAPI dependency that parses a unified ``filter`` query parameter.

    Returns a ``ParsedFilter`` whose ``operation`` is the full, validated,
    entity-translated filter tree.  Services use ``extract`` / ``remove`` to
    inspect or strip fields before forwarding to the entity store.
    """
    # Pre-compute the field map once at dependency creation time. The hooks
    # below are duck-typed: any filter model exposing ``_get_entity_field_map``
    # / ``translate_operation`` (e.g. ``nhx.common.entities.values.Filter``)
    # gets entity-store translation; plain ``nemo_helix_plugin.schema.Filter``
    # subclasses don't and skip these branches.
    get_field_map = getattr(filter_model, "_get_entity_field_map", None)
    entity_field_map: dict[str, str] = get_field_map() if callable(get_field_map) else {}
    # Namespace fields (e.g. ``labels`` → ``data.labels``) whose dotted sub-paths
    # are valid. Supplied by the model, so the parser stays service-agnostic.
    get_namespace_map = getattr(filter_model, "_get_entity_namespace_map", None)
    entity_namespace_map: dict[str, str] = get_namespace_map() if callable(get_namespace_map) else {}
    namespace_fields: frozenset[str] = frozenset(entity_namespace_map)

    async def _dependency(request: Request) -> ParsedFilter:
        try:
            operation: FilterOperation | None = None

            # Check for raw string filter param (JSON or text)
            raw = request.query_params.get(param_name)
            if raw is not None:
                if raw.startswith("{"):
                    operation = parse_json_filter(raw)
                else:
                    operation = parse_text_filter(raw)
            else:
                # Check for bracket notation params (filter[...])
                bracket_dict = parse_deep_object(name=param_name, params=request.query_params)
                if bracket_dict:
                    operation = parse_bracket_filter(bracket_dict)

            # Validate field names
            if operation is not None:
                valid_fields = _get_valid_fields(filter_model)
                _validate_fields(operation, valid_fields, namespace_fields)

            # Translate field names for the entity store (e.g., purpose → data.purpose)
            translate = getattr(filter_model, "translate_operation", None)
            if operation is not None and callable(translate):
                operation = translate(operation)

            return ParsedFilter(operation=operation, _field_map=entity_field_map)

        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

    return _dependency
