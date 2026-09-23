# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Various utility functions."""

import json
import logging
import os
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from fastapi import HTTPException
from nemo_helix_plugin.jobs.openapi_utils import clear_query_param_schemas as clear_query_param_schemas  # noqa: F401
from nemo_helix_plugin.jobs.openapi_utils import (
    generate_openapi_extra_params as generate_openapi_extra_params,  # noqa: F401
)
from nemo_helix_plugin.jobs.openapi_utils import (
    install_query_param_schema_openapi_hook as install_query_param_schema_openapi_hook,  # noqa: F401
)
from nemo_helix_plugin.jobs.openapi_utils import parse_deep_object as parse_deep_object  # noqa: F401
from nemo_helix_plugin.jobs.openapi_utils import (
    register_query_param_schemas as register_query_param_schemas,  # noqa: F401
)
from pydantic import BaseModel, ValidationError
from pydantic._internal._model_construction import ModelMetaclass
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined
from starlette import status
from starlette.convertors import Convertor
from starlette.datastructures import QueryParams as QueryParams  # noqa: F401 — re-exported for type compat
from starlette.requests import Request

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


class IDConvertor(Convertor):
    """Convertor that matches either one or two URL components separated by "/".

    Also, it does not allow for certain reserved names in the second component, to
    avoid ambiguity in the resource URLs.

    This is a small price to pay for a smooth user experience.
    """

    regex = r"[^/]+|[^/]+/(?!checkpoints)(?!files)[^/]+"

    def convert(self, value: str) -> str:
        return value

    def to_string(self, value: str) -> str:
        return value


class HealthCheckFilter(logging.Filter):
    """
    Filters health check and status logging (GET /health*, GET /status) in the Uvicorn access log.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == "uvicorn.access" and len(record.args) >= 3:
            request_method = record.args[1]
            path = record.args[2]
            return not (request_method == "GET" and ("/health" in path or path == "/status"))
        else:
            return True


def filter_health_checks(logger: logging.Logger, log_enabled: bool = False):
    log_health_endpoints = os.getenv("LOG_HEALTH_ENDPOINTS", log_enabled)
    if not log_health_endpoints:
        logger.info("Logging for health and status endpoints is filtered, enable with env LOG_HEALTH_ENDPOINTS")
        logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())


def _anyof_null_visitor(key: str, value: Any, parent: Dict):
    """Collapse ``anyOf: [<schema>, {type: null}]`` into just ``<schema>``.

    Pydantic emits this pattern for ``Optional`` fields.  We flatten it so
    that nullable unions don't clutter the spec.

    Any ``default: null`` on the parent is also dropped: once the ``null``
    member is gone, ``default: null`` no longer matches the collapsed type
    (e.g. ``{"type": "string", "default": null}``), which breaks downstream
    codegen (Orval's zod output rejects ``z.string().default(null)``).
    """
    if key != "anyOf" or not isinstance(value, list):
        return

    _NULL = {"type": "null"}
    if _NULL not in value:
        return

    value.remove(_NULL)
    if "default" in parent and parent["default"] is None:
        del parent["default"]
    if len(value) == 1:
        non_null = value[0]
        del parent["anyOf"]
        if not non_null.keys() & {"type", "$ref", "oneOf", "anyOf"}:
            raise ValueError(f"Unsupported anyOf member format: {non_null}")
        # Hoist every key from the non-null branch (type, format, writeOnly,
        # readOnly, items, pattern, enum, examples, $ref, oneOf, anyOf, ...) onto
        # the parent without overwriting parent-provided metadata like
        # title/description. An Optional[Union[...]] collapses to a bare oneOf.
        for k, v in non_null.items():
            parent.setdefault(k, v)


def normalize_schema_name(name: str) -> str:
    """Apply all naming normalization rules to a raw FastAPI/Pydantic schema name.

    Rules applied in order:
    1. Dash removal: ``Foo-Input`` -> ``FooInput``, ``Foo-Output`` -> ``FooOutput``
    2. Page simplification: ``Page_Job_Filter_`` -> ``JobsPage``
    3. Namespace stripping: ``nemo__api__Foo`` -> ``Foo``
    """
    name = name.replace("-Input", "Input")
    name = name.replace("-Output", "Output")

    if name.startswith("Page"):
        parts = name.replace("__", "_").split("_")
        if len(parts) >= 2:
            name = parts[1] + "sPage"

    if "__" in name:
        name = name.split("__")[-1]

    return name


def _walk_spec(d: Dict, visitor: Callable[[str, Any, Dict], None]):
    """Recursively walk all dicts in the spec tree, calling *visitor(key, value, parent)* for each entry.

    The visitor may mutate *parent[key]* (e.g. replace a ``$ref`` string).
    Because each value is bound to a local variable before recursion, the
    walk will still descend into the **original** value, not the replacement.
    This is safe when replaced values are leaf scalars (strings, numbers) but
    means a visitor that swaps a dict subtree will **not** cause the walker
    to descend into the new subtree.
    """
    for k, v in list(d.items()):
        visitor(k, v, d)
        if isinstance(v, dict):
            _walk_spec(v, visitor)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    _walk_spec(item, visitor)


def _normalize_refs_and_schema_keys(spec: Dict) -> Dict:
    """Normalize all ``$ref`` values and schema dictionary keys.

    Schema keys are renamed first so that ``$ref`` values can be rewritten
    consistently.  When two raw keys normalize to the same target and have
    identical content the duplicate is silently dropped.

    When the content *differs*, two distinct Pydantic models are fighting over
    one schema name, and keeping one silently makes the other's ``$ref``\\ s point
    at the wrong contract.  This raises ``ValueError`` so the build fails loudly
    rather than shipping a wrong contract in the generated SDK.
    """
    schemas = spec["components"]["schemas"]

    # Normalize each raw key once, then group by the name it maps to.  Two
    # distinct models from different modules (e.g. each customization backend's
    # own ``TrainingSpec``) produce module-qualified raw keys that normalize to
    # the same bare name; collapsing them would make one silently steal the
    # other's ``$ref``\\ s, shipping a wrong contract.
    key_to_target: Dict[str, str] = {old_key: normalize_schema_name(old_key) for old_key in schemas}

    by_target: Dict[str, List[str]] = defaultdict(list)
    for old_key, target in key_to_target.items():
        by_target[target].append(old_key)

    for target, old_keys in by_target.items():
        # Identical-content duplicates are harmless — they dedup to one schema.
        # Keep one representative per distinct content shape; more than one means
        # genuinely different models are colliding on a single name.
        reps: List[str] = []
        for key in old_keys:
            if not any(schemas[key] == schemas[rep] for rep in reps):
                reps.append(key)
        if len(reps) > 1:
            raise ValueError(
                f"OpenAPI schema name collision: {sorted(old_keys)} all normalize to "
                f"'{target}' with differing content. Two distinct Pydantic models share a "
                f"class name across modules; give them distinct names — rename or dedupe the "
                f"models, or namespace them (e.g. via a per-backend NamespacedModel base) — so "
                f"they emit distinct schema names."
            )

    # Iterating ``key_to_target`` preserves ``schemas`` order, so the first raw
    # key that maps to a given target still wins the collapse below.
    rename_map: Dict[str, str] = {old_key: target for old_key, target in key_to_target.items() if target != old_key}

    for old_key, new_key in rename_map.items():
        if new_key not in schemas:
            schemas[new_key] = deepcopy(schemas[old_key])
        del schemas[old_key]

    discriminator_mappings: List[Dict] = []

    def _resolve(raw_name: str) -> str:
        if raw_name in rename_map:
            return rename_map[raw_name]
        return normalize_schema_name(raw_name)

    def _ref_visitor(key: str, value: Any, parent: Dict):
        if key == "$ref" and isinstance(value, str):
            schema_name = value.split("/")[-1]
            new_name = _resolve(schema_name)
            if new_name != schema_name:
                parent[key] = value.rsplit("/", 1)[0] + "/" + new_name

        if key == "discriminator" and isinstance(value, dict):
            mapping = value.get("mapping")
            if isinstance(mapping, dict):
                for map_key, map_ref in list(mapping.items()):
                    if isinstance(map_ref, str):
                        schema_name = map_ref.split("/")[-1]
                        new_name = _resolve(schema_name)
                        if new_name != schema_name:
                            mapping[map_key] = map_ref.rsplit("/", 1)[0] + "/" + new_name
                discriminator_mappings.append(mapping)

    _walk_spec(spec, _ref_visitor)

    return _fix_discriminator_dangling_refs(spec, discriminator_mappings)


def _fix_discriminator_dangling_refs(spec: Dict, discriminator_mappings: List[Dict]) -> Dict:
    """Fix discriminator mapping refs that point to non-existent schemas after renaming.

    Handles dangling suffixes like ``FooInput`` when only ``Foo`` exists.
    """
    schema_name_set = set(spec["components"]["schemas"].keys())
    for mapping in discriminator_mappings:
        for key, ref in list(mapping.items()):
            if not isinstance(ref, str) or not ref.startswith("#/components/schemas/"):
                continue
            name = ref.split("/")[-1]
            if name in schema_name_set:
                continue
            resolved = False
            for suffix in ("Input", "Output"):
                if name.endswith(suffix):
                    base = name[: -len(suffix)]
                    if base in schema_name_set:
                        mapping[key] = f"#/components/schemas/{base}"
                        resolved = True
                        break
            if not resolved:
                logger.warning(
                    "Discriminator mapping ref '%s' points to non-existent schema and could not be auto-resolved",
                    ref,
                )
    return spec


def _split_input_output_schemas(spec: Dict) -> Dict:
    """Clone output schemas referenced from Input contexts so they get their own Input variant.

    When an ``*Input`` schema references a schema that has ``updated_at``
    (i.e. an output/entity schema), create a copy with an ``Input`` suffix
    and repoint the ``$ref``.
    """
    schemas = spec["components"]["schemas"]

    def _get_input_ref(ref_id: str) -> str:
        if ref_id.endswith("Output"):
            base = ref_id[: -len("Output")]
            if base in schemas:
                return base + "Input"
        return ref_id + "Input"

    def _maybe_replace_with_input(schema_id: str, ref_id: str) -> Optional[str]:
        ref_schema = schemas.get(ref_id)
        if ref_schema is None:
            logger.warning("Referenced schema '%s' not found (from '%s'), skipping.", ref_id, schema_id)
            return None
        if "Input" in schema_id and "updated_at" in ref_schema.get("properties", {}) and "Input" not in ref_id:
            new_ref_id = _get_input_ref(ref_id)
            if new_ref_id not in schemas:
                logger.debug(f"{new_ref_id} does not exist. Using original schema from {ref_id}.")
                schemas[new_ref_id] = deepcopy(schemas[ref_id])
            logger.debug(f"Replacing {ref_id} with {new_ref_id} in {schema_id}")
            return new_ref_id
        return None

    for schema_id, schema in list(schemas.items()):
        for prop_name, prop_data in schema.get("properties", {}).items():
            prop_variants = [prop_data]
            if "anyOf" in prop_data:
                prop_variants = prop_data["anyOf"]

            for prop_variant in prop_variants:
                if "$ref" in prop_variant:
                    ref_id = prop_variant["$ref"].split("/")[-1]
                    new_ref = _maybe_replace_with_input(schema_id, ref_id)
                    if new_ref is not None:
                        prop_variant["$ref"] = "#/components/schemas/" + new_ref

                elif prop_variant.get("type") == "array" and "$ref" in prop_variant.get("items", {}):
                    ref_id = prop_variant["items"]["$ref"].split("/")[-1]
                    new_ref = _maybe_replace_with_input(schema_id, ref_id)
                    if new_ref is not None:
                        prop_variant["items"]["$ref"] = "#/components/schemas/" + new_ref

    return spec


def _annotate_string_references(spec: Dict) -> Dict:
    """Add human-readable descriptions to ``anyOf: [string, $ref]`` patterns."""
    all_schemas = spec["components"]["schemas"]

    def _visitor(key: str, value: Any, parent: Dict):
        if key != "anyOf" or not isinstance(value, list) or len(value) != 2:
            return
        string_item, ref_item = value[0], value[1]
        if string_item.get("type") != "string" or not ref_item.get("$ref"):
            return
        ref_id = ref_item["$ref"].split("/")[-1]
        ref_entity = all_schemas.get(ref_id)
        if ref_entity is None:
            return
        ref_title = ref_entity.get("title", ref_id)
        if ref_title.endswith("Input"):
            ref_title = ref_title[: -len("Input")]
        string_item["title"] = "Reference"
        string_item["description"] = "A reference to " + ref_title + "."

    _walk_spec(spec, _visitor)
    return spec


def _sync_schema_titles(spec: Dict) -> Dict:
    """Ensure every schema's ``title`` field matches its dictionary key."""
    for key, schema in spec["components"]["schemas"].items():
        schema["title"] = key
    return spec


def _sort_schemas(spec: Dict) -> Dict:
    """Sort ``components.schemas`` keys alphabetically."""
    schemas = spec["components"]["schemas"]
    spec["components"]["schemas"] = dict(sorted(schemas.items()))
    return spec


def _apply_schema_default_extensions(spec: Dict) -> Dict:
    """Promote opt-in schema default metadata into the emitted OpenAPI schema."""

    def _visitor(key: str, value: Any, parent: Dict):
        if key == "x-schema-default":
            parent.setdefault("default", value)
            del parent[key]

    _walk_spec(spec, _visitor)
    return spec


def tweak_spec(spec: Dict) -> Dict:
    spec = _apply_schema_default_extensions(spec)
    _walk_spec(spec, _anyof_null_visitor)
    spec = _normalize_refs_and_schema_keys(spec)
    spec = _split_input_output_schemas(spec)
    spec = _annotate_string_references(spec)
    spec = _sync_schema_titles(spec)
    spec = _sort_schemas(spec)
    return spec


class FlattenBaseModel(ModelMetaclass):
    """Helper class for flattening the fields definitions with extra features.

    Goes up the dependency chain and collects all field definitions.
    Allows the hiding and reordering of certain fields.
    """

    @staticmethod
    def collect_fields(cls):
        annotations = {}
        fields = cls.model_fields

        # Traverse up the inheritance chain
        while cls is not BaseModel and cls is not object:
            if hasattr(cls, "__annotations__"):
                # Merge annotations, later classes take precedence, so only update, if we don't have a value yet.
                for key, value in cls.__annotations__.items():
                    annotations.setdefault(key, value)
            cls = cls.__bases__[0]  # Go to the next class in the chain

        return annotations, fields.copy()

    def __new__(cls, name, bases, dct):
        # Check if the class already has annotations
        annotations = dct.get("__annotations__", {})

        existing_annotations, existing_fields = FlattenBaseModel.collect_fields(dct["_base_model"])

        ## TODO: remove the Optional

        # Only add existing annotations that don't exist in the schema's own annotations
        # This preserves field overrides defined in the schema class
        annotations |= {k: v for k, v in existing_annotations.items() if k not in annotations}

        for field in dct.get("_exclude", []):
            if field in annotations:
                del annotations[field]
                del existing_fields[field]

        final_annotations = {}
        for field in dct.get("_order", []):
            if field in annotations:
                final_annotations[field] = annotations[field]

        for field in annotations:
            if field not in final_annotations:
                final_annotations[field] = annotations[field]

        is_response_model = dct.get("_is_response_model", False)

        # Make sure the class has the updated annotations
        dct["__annotations__"] = final_annotations
        dct.update(existing_fields)
        dct["__doc__"] = dct["_base_model"].__dict__.get("__doc__")
        # Process optionals
        optionals = dct.get("_optionals", [])
        make_all_optional = "*" in optionals
        for field, anno in final_annotations.items():
            # Ignore non-pydantic fields and internal fields
            field_info = dct.get(field)
            if not isinstance(field_info, FieldInfo) or field.startswith("_"):
                continue

            # TODO: This handles an edge case with nested recursive fields. In `entities.Model`, `base_model` can be of type
            # `entities.Model`. For some reason, that type actually points to the `Model` defined in `nemohelix/api/models`
            # once this class resolves. Here, we need to ensure that it correctly points to `entities.Model`, to ensure the
            # correct validation rules are applied.
            if field == "base_model" and "_base_model" in dct:
                if dct["_base_model"].__name__ == "Model":
                    try:
                        from nhx_common.datamodel.datastore.models import entities
                        from nhx_common.datamodel.types import URN

                        final_annotations[field] = Optional[Union[URN, "entities.Model"]]
                    except ImportError:
                        # If imports aren't available, skip this special handling
                        pass

            # For output models, ensure that fields with a default value are required in the OpenAPI spec
            if is_response_model:
                # custom_fields should not be marked as required
                if field == "custom_fields":
                    continue

                if field_info.default is not None or field_info.default_factory is not None:
                    field_info = deepcopy(field_info)

                    # If the field has a default value, store in `json_schema_extra` so it appears in OpenAPI spec
                    if field_info.default is not None and field_info.default is not PydanticUndefined:
                        if not field_info.json_schema_extra:
                            field_info.json_schema_extra = {}
                        field_info.json_schema_extra["default"] = field_info.default

                    # Explicitly remove any default value so these fields are required in the OpenAPI spec
                    field_info.default = PydanticUndefined
                    field_info.default_factory = None

                    dct[field] = field_info
            if make_all_optional or field in optionals:
                # Make annotation optional, no-op if annotation is already optional
                final_annotations[field] = Optional[anno]

                # Set default value to None if the field is required
                field_info = deepcopy(field_info)
                field_info.annotation = final_annotations[field]
                if field_info.is_required():
                    field_info.default = None
                dct[field] = field_info

        return super().__new__(cls, name, bases, dct)


def _apply_comparison_operators(item_value, filter_operators: Dict) -> bool:
    """Apply comparison operators to item values."""

    # Convert string dates to datetime objects for comparison
    def to_comparable(value):
        if isinstance(value, str):
            try:
                # Try to parse as ISO datetime
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                return value
        return value

    item_comparable = to_comparable(item_value)

    for operator, filter_value in filter_operators.items():
        filter_comparable = to_comparable(filter_value)

        if operator == "gt":
            if not (item_comparable > filter_comparable):
                return False
        elif operator == "gte":
            if not (item_comparable >= filter_comparable):
                return False
        elif operator == "lt":
            if not (item_comparable < filter_comparable):
                return False
        elif operator == "lte":
            if not (item_comparable <= filter_comparable):
                return False
        elif operator == "eq":
            if not (item_comparable == filter_comparable):
                return False
        elif operator == "neq":
            if not (item_comparable != filter_comparable):
                return False
        elif operator == "in":
            if item_value not in filter_value:
                return False

    return True


def filter_match(item: Dict, filter_dict: Dict, strict: bool = True) -> bool:
    """Helper function with generic filtering on dicts."""
    for key, filter_value in filter_dict.items():
        # Skip the "raw" maker
        if key == "*":
            continue

        if filter_value is None:
            continue

        if key not in item:
            if filter_value is True:
                # Key must exist but doesn't
                return False
            elif filter_value is False:
                # Key must not exist and doesn't
                continue
            else:
                # Key doesn't exist, but filter expects a specific value
                return False
        else:
            item_value = item[key]
            if filter_value is True:
                # Key exists, which is expected
                continue
            elif filter_value is False:
                # Key exists but shouldn't
                return False
            elif isinstance(filter_value, dict):
                # Handle comparison operators
                if any(op in filter_value for op in ["gt", "gte", "lt", "lte", "eq", "neq", "in"]):
                    if not _apply_comparison_operators(item_value, filter_value):
                        return False
                elif "start" in filter_value or "end" in filter_value:
                    # Handle date range filters
                    start = filter_value.get("start")
                    end = filter_value.get("end")

                    if start:
                        if item_value < start:
                            return False
                    if end:
                        if item_value > end:
                            return False
                elif isinstance(item_value, dict):
                    # Recursively match nested dictionaries
                    if not filter_match(item_value, filter_value, strict):
                        return False
                elif isinstance(item_value, str):
                    # Try to parse string as JSON and match recursively if possible
                    try:
                        parsed = json.loads(item_value)
                    except Exception:
                        return False
                    if isinstance(parsed, dict):
                        if not filter_match(parsed, filter_value, strict):
                            return False
                    else:
                        return False
                else:
                    # filter_value is dict but item_value is not - no match
                    return False
            elif isinstance(filter_value, list):
                # Handle OR logic for multiple values
                match_found = False
                for single_filter_value in filter_value:
                    if _match_single_value(key, item_value, single_filter_value, strict):
                        match_found = True
                        break
                if not match_found:
                    return False
            else:
                # Check for equality with the filter value
                if not _match_single_value(key, item_value, filter_value, strict):
                    return False
    return True


def _match_single_value(key: str, item_value: Any, filter_value: Any, strict: bool) -> bool:
    """Helper function to match a single filter value against an item value."""
    if isinstance(filter_value, str) and isinstance(item_value, dict):
        if not strict:
            if key in item_value and isinstance(item_value[key], str):
                return filter_value.lower() in item_value[key].lower()
            if len(item_value) > 1 and (
                any(
                    key in field_name
                    for field_name in item_value.keys()
                    if isinstance(field_name, str) and len(field_name) > 3
                )
            ):
                for dict_key, dict_value in item_value.items():
                    if isinstance(dict_value, str) and filter_value.lower() in dict_value.lower():
                        return True
            return False
        else:
            if item_value.get(key) and isinstance(item_value.get(key), str):
                return filter_value == item_value.get(key)
            return False
    elif not strict:
        if isinstance(item_value, str) and isinstance(filter_value, str):
            if filter_value.lower() not in item_value.lower():
                return False
        elif (
            isinstance(item_value, str)
            and item_value is not None
            and isinstance(filter_value, str)
            and filter_value not in item_value
        ):
            return False
        elif item_value is not None and filter_value != item_value:
            return False
        elif item_value is None:
            # item_value is None, so it can't contain filter_value
            return False
    elif item_value != filter_value:
        return False
    return True


def parse_deep_object_query(field_name: str, value_type: Type[T]) -> Callable[[Request], Optional[T]]:
    """Create a FastAPI dependency that parses deepObject-style query params into a typed Pydantic model.

    Args:
        field_name: The query parameter prefix (e.g. "filter")
        value_type: Pydantic model class to parse into

    Returns:
        FastAPI dependency returning the parsed model, or None if no params present
    """

    def _dep(request: Request) -> Optional[T]:
        try:
            parsed = parse_deep_object(name=field_name, params=request.query_params) or {}
            if not parsed:
                return None
            return value_type(**parsed)
        except ValidationError as e:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e))

    return _dep


def filter_by_created_at(filter_value: Dict, item_date: datetime) -> bool:
    """Helper function to filter entities based on created_at property."""
    conditions = {
        "gt": lambda item_date, value: item_date > value,
        "gte": lambda item_date, value: item_date >= value,
        "lt": lambda item_date, value: item_date < value,
        "lte": lambda item_date, value: item_date <= value,
        "eq": lambda item_date, value: item_date == value,
        "neq": lambda item_date, value: item_date != value,
    }
    for compare_key, value in filter_value.items():
        if compare_key not in conditions:
            return False
        if value and not conditions[compare_key](item_date, value):
            return False
    return True


def split_named_entity_urn(urn: str) -> List[str]:
    """Split a URN into namespace and name components.

    Args:
        urn: The URN string in format "namespace/name"

    Returns:
        List containing [namespace, name]

    Raises:
        ValueError: If the URN is not in the correct format
    """
    result = urn.split("/", 1)

    if len(result) < 2:
        raise ValueError(f"Incorrect entity full name: {urn}. Required format is <namespace>/<name>")

    return result
