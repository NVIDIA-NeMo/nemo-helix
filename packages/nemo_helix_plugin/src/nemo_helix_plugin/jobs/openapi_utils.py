# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAPI deepObject parameter utilities for job route factories."""

import json
import textwrap
from json import JSONDecodeError
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from starlette.datastructures import QueryParams

# Module-level registry of query-parameter schemas (filter/search classes) referenced
# via ``generate_openapi_extra_params`` so FastAPI doesn't see them through pydantic's
# response/body walk. ``register_query_param_schemas`` injects these into
# ``components.schemas`` of the generated OpenAPI spec.
_query_param_schemas: Dict[str, type[BaseModel]] = {}


def generate_openapi_extra_params(
    filter_schema: Optional[type[BaseModel]] = None,
    filter_description: Optional[str] = None,
    search_schema: Optional[type[BaseModel]] = None,
    search_description: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Generate openapi_extra parameter definitions for deepObject filter and search params.

    The filter/search classes are referenced by ``$ref`` in the parameter definition,
    and registered so ``register_query_param_schemas`` can inject them into
    ``components.schemas`` of the final spec.

    Args:
        filter_schema: Pydantic model class for the filter (e.g. ``ModelEntityFilter``)
        filter_description: Description for the filter parameter
        search_schema: Pydantic model class for the search
        search_description: Description for the search parameter

    Returns:
        Dict with "parameters" key suitable for use as ``openapi_extra``
    """
    parameters = []

    def _ref_for(model: type[BaseModel]) -> Dict[str, str]:
        _query_param_schemas[model.__name__] = model
        return {"$ref": f"#/components/schemas/{model.__name__}"}

    if filter_schema:
        parameters.append(
            {
                "in": "query",
                "name": "filter",
                "style": "deepObject",
                "required": False,
                "explode": True,
                "schema": _ref_for(filter_schema),
                "description": textwrap.dedent(
                    filter_description
                    or (
                        "Filter results on various criteria. "
                        "Supports bracket notation (?filter[field][$op]=value), "
                        'JSON ({"field":{"$op":"value"}}), and text syntax (field:"value"). '
                        "Operators: $eq (exact match), $like (substring), "
                        "$lt, $lte, $gt, $gte (comparison), $in, $nin (set membership), "
                        "$and, $or, $not (logical). Default operator is $eq."
                    )
                ),
            }
        )

    if search_schema:
        parameters.append(
            {
                "in": "query",
                "name": "search",
                "style": "deepObject",
                "required": False,
                "explode": True,
                "schema": _ref_for(search_schema),
                "description": textwrap.dedent(search_description or "Search results by text fields."),
            }
        )

    return {"parameters": parameters}


def clear_query_param_schemas() -> None:
    """Reset the module-level query-param schema registry.

    Call between isolated spec generations so a stale registration from a prior
    service doesn't leak into the next one. Today the offline spec-gen script
    forks a subprocess per service so this is defensive; in-process loaders
    (tests, dev multiplexers) need it.
    """
    _query_param_schemas.clear()


def _strip_null_defaults(schema: Any) -> Any:
    """Return *schema* without cosmetic ``default: null`` entries."""
    if isinstance(schema, dict):
        return {
            key: _strip_null_defaults(value)
            for key, value in schema.items()
            if not (key == "default" and value is None)
        }
    if isinstance(schema, list):
        return [_strip_null_defaults(item) for item in schema]
    return schema


def _promote_schema_defs(schema_name: str, schema: Dict[str, Any], components: Dict[str, Any]) -> None:
    """Move a schema's local ``$defs`` into ``components.schemas``.

    Query filter schemas are emitted with refs like
    ``#/components/schemas/DatetimeFilter``. Those refs only resolve if the
    matching nested definitions are promoted to global components.
    """
    local_defs = schema.pop("$defs", None)
    if not local_defs:
        return

    for name, local_def in local_defs.items():
        existing = components.get(name)
        if existing is None:
            components[name] = local_def
            continue
        if existing == local_def or _strip_null_defaults(existing) == _strip_null_defaults(local_def):
            continue
        raise ValueError(
            f"Schema '{name}' from query parameter schema '{schema_name}' conflicts with existing component"
        )


def register_query_param_schemas(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Inject filter/search classes registered via ``generate_openapi_extra_params``
    into ``components.schemas`` of ``spec`` if not already present.

    Nested ``$defs`` are promoted because these schemas use component-level refs
    (``#/components/schemas/{model}``) for nested filter models. Without promotion,
    live ``/openapi.json`` output can contain dangling refs until the offline
    postprocessing pipeline runs.
    """
    if not _query_param_schemas:
        return spec
    components = spec.setdefault("components", {}).setdefault("schemas", {})
    for name, cls in _query_param_schemas.items():
        if name in components:
            existing = components[name]
            if isinstance(existing, dict):
                _promote_schema_defs(name, existing, components)
            continue
        schema = cls.model_json_schema(ref_template="#/components/schemas/{model}")
        _promote_schema_defs(name, schema, components)
        components[name] = schema
    return spec


def install_query_param_schema_openapi_hook(app: FastAPI) -> None:
    """Wrap ``app.openapi`` so live specs include registered query-param schemas."""
    default_openapi = app.openapi

    def custom_openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        openapi_schema = default_openapi()
        try:
            openapi_schema = register_query_param_schemas(openapi_schema)
        except Exception:
            app.openapi_schema = None
            raise
        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi  # type: ignore[method-assign]


def parse_deep_object(name: str, params: QueryParams) -> Dict:
    """ "Helper function to parse 'deepObject'-like query parameters."""
    result = {}

    # Get all unique keys (without duplicates)
    unique_keys = set(params.keys())

    for key in unique_keys:
        # Only process keys with bracket notation (e.g., "search[name]=value").
        # Bare keys like "search={"field":"value"}" are not deepObject-style params
        # and should be handled by other mechanisms (e.g., parse_json_search).
        if "[" not in key:
            continue

        # Get all values for this key (handles multiple values)
        values = params.getlist(key)

        # Split key by '[' and ']', handling nested keys like 'sub_item[name]'
        keys = key.replace("]", "").split("[")
        current = result
        for part in keys[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        final_key = keys[-1]

        # Process all values for this key
        processed_values = []
        for value in values:
            # Depending on how the encoding is done, this could be an encoded JSON
            if value.startswith("{") or value.startswith("["):
                try:
                    value = json.loads(value)
                except JSONDecodeError:
                    raise ValueError(
                        f"Invalid filter value: '{value}'. "
                        f"Array and object values must be valid JSON with proper quoting. "
                        f'Example: ["item1","item2"] not [item1,item2]'
                    )
            elif value == "":  # support `null` value
                value = None
            elif isinstance(value, str) and "," in value:
                # Handle comma-delimited values by splitting them
                # This supports e.g., filter[status]=pending,active
                comma_split_values = [v.strip() for v in value.split(",")]
                processed_values.extend(comma_split_values)
                continue
            processed_values.append(value)

        # Handle multiple values for the same key (OR logic)
        if len(processed_values) == 1:
            current[final_key] = processed_values[0]
        else:
            current[final_key] = processed_values

    return result.get(name)
