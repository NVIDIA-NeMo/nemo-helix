# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import Any, ClassVar, Optional, get_args

from nemo_helix_plugin.api.parsed_filter import ENTITY_BASE_FIELDS as ENTITY_BASE_FIELDS  # noqa: F401
from nemo_helix_plugin.filter_ops import ComparisonOperation, FilterOperation, FilterOperator, LogicalOperation
from nemo_helix_plugin.schema import Filter as _PluginFilter
from pydantic import ConfigDict, Field, model_validator


class EntityFieldMapping:
    """Annotation marker for custom entity field path mapping.

    Use with Annotated to override the default data.{field_name} mapping::

        class MyFilter(Filter):
            storage_type: Annotated[str | None, map_entity_field("data.storage.type")] = None

    Set ``namespace=True`` to map a field that addresses a *nested namespace*
    rather than a single scalar. The user-facing prefix is rewritten and
    arbitrary dotted sub-keys under it become valid filter fields::

        class MyFilter(Filter):
            labels: Annotated[dict[str, str] | None, map_entity_field("data.labels", namespace=True)] = None

    With the above, both ``labels`` and ``labels.<key>`` are accepted and
    translate to ``data.labels`` / ``data.labels.<key>``.
    """

    def __init__(self, entity_path: str, *, namespace: bool = False):
        self.entity_path = entity_path
        self.namespace = namespace


def map_entity_field(entity_path: str, *, namespace: bool = False) -> EntityFieldMapping:
    """Annotate a filter field with its entity store path.

    Use when the entity path differs from the default ``data.{field_name}`` convention.
    Pass ``namespace=True`` when the field maps a nested namespace rather than a single
    scalar, so arbitrary dotted sub-keys (``field.<key>``) are accepted and rewritten
    under ``entity_path`` (e.g. ``labels.x`` → ``data.labels.x``).
    """
    return EntityFieldMapping(entity_path, namespace=namespace)


_BOOL_TRUE_VALUES = (True, "true")
_BOOL_FALSE_VALUES = (False, "false")


def _map_field(field: str, field_map: dict[str, str], namespace_map: dict[str, str] | None) -> str:
    """Resolve a user-facing field name to its entity-store path.

    Exact scalar mappings (``field_map``) win first. Otherwise a namespace
    mapping rewrites the prefix: a field equal to ``labels`` or starting with
    ``labels.`` maps to ``data.labels`` / ``data.labels.<sub>`` when ``labels``
    is a declared namespace. Unmapped fields pass through unchanged.
    """
    if field in field_map:
        return field_map[field]
    if namespace_map:
        for prefix, entity_prefix in namespace_map.items():
            if field == prefix:
                return entity_prefix
            if field.startswith(prefix + "."):
                return entity_prefix + field[len(prefix) :]
    return field


def _apply_field_map(
    operation: FilterOperation,
    field_map: dict[str, str],
    bool_coercible: set[str] | None = None,
    namespace_map: dict[str, str] | None = None,
) -> FilterOperation:
    """Walk a FilterOperation tree, translate field names, and coerce booleans.

    For fields in *bool_coercible*, an ``$eq`` with a boolean-like value is
    rewritten to a null / not-null existence check:

    * ``"false"`` / ``False`` → ``$eq null``  (field is absent)
    * ``"true"``  / ``True``  → ``$not { $eq null }`` (field has a value)

    *namespace_map* maps user-facing namespace prefixes to entity-store prefixes
    so dotted sub-paths (``labels.x`` → ``data.labels.x``) translate generically.
    """
    if isinstance(operation, ComparisonOperation):
        field = operation.field
        mapped_field = _map_field(field, field_map, namespace_map)

        # Bool coercion for Union[..., bool, str] fields
        if bool_coercible and field in bool_coercible and operation.operator == FilterOperator.EQ:
            if operation.value in _BOOL_FALSE_VALUES:
                return ComparisonOperation(operator=FilterOperator.EQ, field=mapped_field, value=None)
            if operation.value in _BOOL_TRUE_VALUES:
                return LogicalOperation(
                    operator=FilterOperator.NOT,
                    operations=[ComparisonOperation(operator=FilterOperator.EQ, field=mapped_field, value=None)],
                )

        if field != mapped_field:
            return ComparisonOperation(operator=operation.operator, field=mapped_field, value=operation.value)
        return operation
    elif isinstance(operation, LogicalOperation):
        return LogicalOperation(
            operator=operation.operator,
            operations=[_apply_field_map(op, field_map, bool_coercible, namespace_map) for op in operation.operations],
        )
    return operation


from nemo_helix_plugin.schema import Value as Value  # noqa: E402


class Filter(_PluginFilter):
    """Filter with entity field mapping and translation support.

    Extends nemo_helix_plugin's Filter base with translate_operation() for the entity store.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    _entity_field_map_cache: ClassVar[dict[str, str] | None] = None
    _entity_namespace_map_cache: ClassVar[dict[str, str] | None] = None
    _bool_coercible_fields_cache: ClassVar[set[str] | None] = None

    @classmethod
    def _get_entity_field_map(cls) -> dict[str, str]:
        cached = cls.__dict__.get("_entity_field_map_cache")
        if cached is not None:
            return cached
        field_map: dict[str, str] = {}
        for field_name, field_info in cls.model_fields.items():
            # Check for explicit map_entity_field annotation
            mapping = next(
                (m for m in (field_info.metadata or []) if isinstance(m, EntityFieldMapping)),
                None,
            )
            if mapping is not None:
                # Namespace mappings are handled by _get_entity_namespace_map
                # (prefix rewriting), not by exact-match translation here.
                if mapping.namespace:
                    continue
                field_map[field_name] = mapping.entity_path
            elif field_name not in ENTITY_BASE_FIELDS:
                field_map[field_name] = f"data.{field_name}"
            # Base fields map to themselves — no entry needed
        cls._entity_field_map_cache = field_map
        return field_map

    @classmethod
    def _get_entity_namespace_map(cls) -> dict[str, str]:
        """User-facing namespace prefixes → entity-store prefixes.

        Populated from ``map_entity_field(..., namespace=True)`` annotations.
        Sub-paths under a namespace (e.g. ``labels.<key>``) are valid filter
        fields and translate to ``<entity_prefix>.<key>``. This is what lets a
        plugin expose a nested data blob (labels, tags, metadata, …) without the
        global parser knowing the namespace's name.
        """
        cached = cls.__dict__.get("_entity_namespace_map_cache")
        if cached is not None:
            return cached
        namespace_map: dict[str, str] = {}
        for field_name, field_info in cls.model_fields.items():
            mapping = next(
                (m for m in (field_info.metadata or []) if isinstance(m, EntityFieldMapping) and m.namespace),
                None,
            )
            if mapping is not None:
                namespace_map[field_name] = mapping.entity_path
        cls._entity_namespace_map_cache = namespace_map
        return namespace_map

    @classmethod
    def _get_bool_coercible_fields(cls) -> set[str]:
        """Fields where ``bool`` means exists / not-exists rather than a literal match.

        Detected by: ``bool`` appears in a Union alongside at least one non-bool,
        non-None type.  For example ``Optional[Union[SomeFilter, bool, str]]`` is
        coercible, but ``Optional[bool]`` is not.
        """
        cached = cls.__dict__.get("_bool_coercible_fields_cache")
        if cached is not None:
            return cached
        result: set[str] = set()
        for field_name, field_info in cls.model_fields.items():
            args = get_args(field_info.annotation)
            if bool in args and any(a not in (bool, type(None)) for a in args):
                result.add(field_name)
        cls._bool_coercible_fields_cache = result
        return result

    @classmethod
    def translate_operation(cls, operation: FilterOperation) -> FilterOperation:
        """Rewrite field names and coerce boolean existence checks.

        1. Translates filter field names for the entity store
           (e.g., ``model`` → ``data.model``).
        2. For fields whose type includes ``bool`` alongside other types
           (e.g., ``Union[BaseModelFilter, bool, str]``), rewrites
           ``$eq "true"/"false"`` to null / not-null checks.
        """
        field_map = cls._get_entity_field_map()
        bool_coercible = cls._get_bool_coercible_fields()
        namespace_map = cls._get_entity_namespace_map()
        if not field_map and not bool_coercible and not namespace_map:
            return operation
        return _apply_field_map(operation, field_map, bool_coercible or None, namespace_map or None)

    @model_validator(mode="before")
    @classmethod
    def skip_validation_on_raw_objects(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("*"):
            data = {}
        return data


class DatetimeFilter(Filter):
    gte: Optional[datetime] = Field(
        None,
        alias="$gte",
        serialization_alias="$gte",
        description="Filter for results greater than or equal to this datetime.",
    )
    lte: Optional[datetime] = Field(
        None,
        alias="$lte",
        serialization_alias="$lte",
        description="Filter for results less than or equal to this datetime.",
    )

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
        populate_by_name=True,  # Accept both "gte" and "$gte" as input
    )


class StringFilter(Filter):
    eq: Optional[str] = Field(
        None,
        alias="$eq",
        serialization_alias="$eq",
        description="Filter for results equal to this value.",
    )
    like: Optional[str] = Field(
        None,
        alias="$like",
        serialization_alias="$like",
        description="Filter for results matching this pattern.",
    )
    in_: Optional[list[str]] = Field(
        None,
        alias="$in",
        serialization_alias="$in",
        description="Filter for results in this list of values.",
    )
    nin: Optional[list[str]] = Field(
        None,
        alias="$nin",
        serialization_alias="$nin",
        description="Filter for results not in this list of values.",
    )

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
        populate_by_name=True,  # Accept both "eq" and "$eq" as input
    )


class NumberFilter(Filter):
    gte: Optional[float] = Field(
        None,
        alias="$gte",
        serialization_alias="$gte",
        description="Filter for results greater than or equal to this value.",
    )
    lte: Optional[float] = Field(
        None,
        alias="$lte",
        serialization_alias="$lte",
        description="Filter for results less than or equal to this value.",
    )
    gt: Optional[float] = Field(
        None,
        alias="$gt",
        serialization_alias="$gt",
        description="Filter for results greater than this value.",
    )
    lt: Optional[float] = Field(
        None,
        alias="$lt",
        serialization_alias="$lt",
        description="Filter for results less than this value.",
    )
    eq: Optional[float] = Field(
        None,
        alias="$eq",
        serialization_alias="$eq",
        description="Filter for results equal to this value.",
    )

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
        populate_by_name=True,  # Accept both "gte" and "$gte" as input
        # Reject an empty predicate ({}) at the schema/contract level — it carries no comparison.
        json_schema_extra={"minProperties": 1},
    )
