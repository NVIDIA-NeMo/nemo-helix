# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Filter utilities for entities.

Text filter syntax (auto-detected when value doesn't start with '{'):

?filter=name:"llama"
?filter=status:"active" AND amount>500
?filter=name~"llama"
?filter='name:"a" OR name:"b"'
?filter='status IN ["active", "pending"]'

Object (JSON) syntax:

?filter={"name":"llama","created_at":{"$lte":"2024-01-01"}}
?filter={"$not": {"$and": [{"name":"llama"},{"name":"llama2"}]}}

Bracket notation:

?filter[name][$like]=llama
?filter[name][$in]=llama,llama2

Relationship traversal (requires registered relationships for the entity type):

?filter[adapters][$exists]=true
?filter[adapters][finetuning_type]=LoRA
?filter={"adapters":{"$exists":true}}
"""

import json
import re
from collections.abc import Awaitable
from typing import Annotated, Any, Callable, Dict, Optional

from fastapi import Depends, HTTPException, Query, Request
from nemo_helix_plugin.api.filter import _normalize_value, _parse_field_operation, _wrap_operations
from nhx.common.api.filter import (
    ComparisonOperation,
    FilterOperation,
    FilterOperator,
    FilterRepository,
    LogicalOperation,
)
from nhx.core.entities.utils.relationships import Relationship, get_relationships, resolve_child_field
from pydantic import ConfigDict


class RelationshipFilterOperation(FilterOperation):
    """Filter via a related entity using an EXISTS subquery.

    Supports two modes:
    - Existence check: filter[adapters][$exists]=true/false
    - Conditional: filter[adapters][finetuning_type]=LoRA (implies exists=True)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    operator: FilterOperator = FilterOperator.EXISTS
    relationship_name: str
    relationship: Relationship
    condition: FilterOperation | None = None
    exists: bool = True

    def apply(self, repository: "FilterRepository") -> Any:
        return repository.relationship_exists(
            target_entity_type=self.relationship.target_entity_type,
            join_field=self.relationship.via,
            child_condition=self.condition,
            negate=not self.exists,
        )

    def to_dict(self) -> Dict[str, Any]:
        if self.condition is None:
            return {self.relationship_name: {"$exists": self.exists}}
        return {self.relationship_name: self.condition.to_dict()}


def _parse_dict_to_operation(
    filter_dict: Dict[str, Any],
    entity_type: str | None = None,
) -> FilterOperation:
    """Convert dictionary to FilterOperation."""
    relationships = get_relationships(entity_type)

    # Check for top-level logical operators
    for logical_op in (FilterOperator.AND, FilterOperator.OR):
        op_key = logical_op.value
        if op_key in filter_dict:
            operations = [_parse_dict_to_operation(item, entity_type=entity_type) for item in filter_dict[op_key]]
            return LogicalOperation(operator=logical_op, operations=operations)

    if "$not" in filter_dict:
        operation = _parse_dict_to_operation(filter_dict["$not"], entity_type=entity_type)
        return LogicalOperation(operator=FilterOperator.NOT, operations=[operation])

    operations: list[FilterOperation] = []
    for field, value in filter_dict.items():
        if field in relationships:
            operations.append(_parse_relationship_value(field, value, relationships[field]))
            continue

        if isinstance(value, dict):
            operations.append(_parse_field_operation(field, value))
        else:
            operations.append(ComparisonOperation(operator=FilterOperator.EQ, field=field, value=value))

    return _wrap_operations(operations)


def _parse_relationship_value(
    rel_name: str,
    value: Any,
    relationship: Relationship,
) -> RelationshipFilterOperation:
    """Parse the value for a relationship key into a RelationshipFilterOperation."""
    if not isinstance(value, dict):
        raise ValueError(
            f"Relationship '{rel_name}' requires a dict value, got {type(value).__name__}. "
            f'Example: {{"{rel_name}": {{"$exists": true}}}} or {{"{rel_name}": {{"field": "value"}}}}'
        )

    if "$exists" in value and len(value) == 1:
        exists_val = value["$exists"]
        if isinstance(exists_val, str):
            exists_val = exists_val.lower() in ("true", "1", "yes")
        return RelationshipFilterOperation(
            relationship_name=rel_name,
            relationship=relationship,
            condition=None,
            exists=bool(exists_val),
        )

    child_dict = {resolve_child_field(k): v for k, v in value.items() if k != "$exists"}
    child_op = _parse_dict_to_operation(child_dict)
    return RelationshipFilterOperation(
        relationship_name=rel_name,
        relationship=relationship,
        condition=child_op,
        exists=True,
    )


def _parse_json_filter(filter_json: str, entity_type: str | None = None) -> FilterOperation:
    """Parse JSON filter parameter."""
    try:
        filter_dict = json.loads(filter_json)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in filter parameter: {filter_json}") from None
    return _parse_dict_to_operation(filter_dict, entity_type=entity_type)


def _parse_bracket_params(
    query_params: Dict[str, str],
    entity_type: str | None = None,
    param_name: str = "filter",
) -> FilterOperation:
    """Parse bracket-style filter parameters."""
    relationships = get_relationships(entity_type)
    filter_params: Dict[str, Any] = {}
    rel_params: Dict[str, Dict[str, Any]] = {}

    for key, value in query_params.items():
        if not key.startswith(f"{param_name}["):
            continue

        match = re.match(rf"^{re.escape(param_name)}\[([^\]]+)\](.*)$", key)
        if not match:
            continue

        field = match.group(1)
        rest = match.group(2)

        if field in ["$or", "$and", "$not"]:
            if rest:
                raise ValueError(
                    f"Invalid filter parameter: {key}. Logical operators cannot have additional parameters."
                )
            try:
                parsed_value = json.loads(value)
                if field == "$or" or field == "$and":
                    operations = [_parse_dict_to_operation(item, entity_type=entity_type) for item in parsed_value]
                    return LogicalOperation(operator=FilterOperator(field), operations=operations)
                elif field == "$not":
                    operation = _parse_dict_to_operation(parsed_value, entity_type=entity_type)
                    return LogicalOperation(operator=FilterOperator.NOT, operations=[operation])
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON for {field}: {value}") from None

        if field in relationships:
            _accumulate_relationship_bracket(field, rest, value, rel_params)
            continue

        operators = []
        while rest:
            op_match = re.match(r"^\[(\$[^\]]+)\](.*)$", rest)
            if op_match:
                operators.append(op_match.group(1))
                rest = op_match.group(2)
            else:
                break

        if field not in filter_params:
            filter_params[field] = {}

        if operators:
            current = filter_params[field]
            for op in operators[:-1]:
                if op not in current:
                    current[op] = {}
                current = current[op]

            final_op = operators[-1]
            current[final_op] = _normalize_value(FilterOperator(final_op), value)
        else:
            filter_params[field] = value

    operations: list[FilterOperation] = []

    if filter_params:
        operations.append(_parse_dict_to_operation(filter_params, entity_type=entity_type))

    for rel_name, child_params in rel_params.items():
        rel = relationships[rel_name]
        operations.append(_parse_relationship_value(rel_name, child_params, rel))

    if not operations:
        raise ValueError("No filter parameters found")

    return _wrap_operations(operations)


def _accumulate_relationship_bracket(
    rel_name: str,
    rest: str,
    value: str,
    rel_params: Dict[str, Dict[str, Any]],
) -> None:
    """Parse remaining brackets after a relationship name and accumulate into rel_params.

    Handles:
      [rel][$exists]=true          -> {"$exists": "true"}
      [rel][field]=val             -> {"field": "val"}
      [rel][field][$op]=val        -> {"field": {"$op": val}}
    """
    all_brackets = re.findall(r"\[([^\]]+)\]", rest)
    if not all_brackets:
        raise ValueError(
            f"Relationship '{rel_name}' requires at least one sub-bracket. "
            f"Example: filter[{rel_name}][$exists]=true or filter[{rel_name}][field_name]=value"
        )

    first = all_brackets[0]
    target = rel_params.setdefault(rel_name, {})

    if first == "$exists":
        target["$exists"] = value
        return

    child_field = first
    child_operators = [b for b in all_brackets[1:] if b.startswith("$")]

    if child_operators:
        current = target.setdefault(child_field, {})
        for op in child_operators[:-1]:
            current = current.setdefault(op, {})
        final_op = child_operators[-1]
        current[final_op] = _normalize_value(FilterOperator(final_op), value)
    else:
        target[child_field] = value


def make_filter_dep(
    param_name: str = "filter",
    description: str = (
        "Query filter expression. Supports text and JSON syntaxes:\n"
        '- Text: name:"value" AND status>500 with operators : ~ > >= < <= IN NOT IN AND OR and negation prefix -\n'
        '- Object (JSON): {"name":{"$like":"value"}} with operators $eq, $like, $lt, $lte, $gt, $gte, $in, $nin, $and, $or, $not\n'
        "- Bracket notation: ?filter[name][$like]=value\n"
        "- Relationship traversal: ?filter[relationship][$exists]=true or ?filter[relationship][field]=value"
    ),
) -> Callable[..., Awaitable[Optional[FilterOperation]]]:
    """Create a FastAPI dependency for parsing filter parameters.

    Auto-detects the filter syntax:
    - If the string starts with '{', parses as JSON object syntax
    - Otherwise, parses as text syntax (e.g., name:"value" AND status>500)
    """

    async def _filter_dep(
        request: Request,
        filter: Annotated[Optional[str], Query(alias=param_name, description=description)] = None,
    ) -> Optional[FilterOperation]:
        from nhx.core.entities.utils.text_filter import parse_text_filter

        entity_type = request.path_params.get("entity_type")

        # Accept both 'filter' and 'search' query params for backward compat.
        # 'filter' takes priority; 'search' is a deprecated alias.
        raw_value = filter
        if raw_value is None:
            raw_value = request.query_params.get("search")

        if raw_value:
            raw_value = raw_value.strip()
            try:
                if raw_value.startswith("{"):
                    return _parse_json_filter(raw_value, entity_type=entity_type)
                return parse_text_filter(raw_value)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid filter query: {e}")

        # Check bracket notation for both 'filter[...]' and 'search[...]' params
        bracket_params = {}
        for key, value in request.query_params.items():
            if key.startswith(f"{param_name}["):
                bracket_params[key] = value
            elif key.startswith("search["):
                # Rewrite search[x] -> filter[x] for backward compat
                rewritten_key = f"{param_name}[" + key[len("search[") :]
                bracket_params[rewritten_key] = value

        if bracket_params:
            try:
                return _parse_bracket_params(bracket_params, entity_type=entity_type, param_name=param_name)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

        return None

    return _filter_dep


FilterDep = Annotated[Optional[FilterOperation], Depends(make_filter_dep())]
