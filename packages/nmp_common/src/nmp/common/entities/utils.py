# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for entity handling."""

import types
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union, get_args, get_origin

import base58
from nemo_platform_plugin.refs import ParsedEntityRef as ParsedEntityRef
from nemo_platform_plugin.refs import parse_entity_ref as parse_entity_ref
from nmp.common.entities.values import DatetimeFilter
from pydantic import BaseModel, ConfigDict, create_model

__all__ = [
    "get_random_id",
    "get_random_bytes",
    "normalize_filter_list",
    "make_filter_class",
    # Backward-compat aliases
    "normalize_search_list",
    "make_search_class",
    "parse_entity_ref",
    "parse_model_entity_ref",
    "ModelEntityId",
    "ADAPTERS_INFIX",
    "parse_adapters_suffix",
]

#: The infix that separates a base model from its LoRA adapter inside a composite
#: model-entity name: ``{base}&adapters/{adapter_workspace}/{adapter_name}``. Defined
#: once here so the grammar cannot drift between the parse and format directions.
ADAPTERS_INFIX = "&adapters/"


def parse_adapters_suffix(name: str) -> tuple[str, str, str] | None:
    """Decompose a (workspace-less) model-entity *name* on the ``&adapters/`` infix.

    The single home for the LoRA-composite grammar's structural split. Given a bare
    entity name (the part after the workspace, i.e. what
    :attr:`ParsedEntityRef.name` holds), returns ``(base, adapter_workspace,
    adapter_name)`` when *name* is a well-formed
    ``base&adapters/adapter_workspace/adapter_name`` composite, else ``None``.

    Returning ``None`` (rather than raising) for a non-composite lets callers that
    only need "is this a LoRA composite, and if so its parts" branch cheaply; callers
    that require a valid composite treat ``None`` as their own error. A name that
    *contains* ``&adapters/`` but is malformed (missing/empty segments) also returns
    ``None`` — it is not a valid composite.
    """
    if ADAPTERS_INFIX not in name:
        return None
    base, _, adapter_part = name.partition(ADAPTERS_INFIX)
    adapter_workspace, separator, adapter_name = adapter_part.partition("/")
    # The grammar has exactly one adapter-name segment, so a surplus ``/`` in the
    # adapter tail (e.g. ``base&adapters/ws/name/extra``) is not a valid composite.
    if not base or not separator or not adapter_workspace or not adapter_name or "/" in adapter_name:
        return None
    return base, adapter_workspace, adapter_name


def get_random_id(prefix: str) -> str:
    """Generate a random ID with the given prefix.

    Format: {prefix}-{base58_encoded_uuid}
    """
    return f"{prefix}-{get_random_bytes()}"


def get_random_bytes() -> str:
    """Generate random bytes as a base58-encoded string."""
    u = uuid.uuid4()
    return base58.b58encode(u.bytes).decode("ascii")


def normalize_filter_list(value: str | list[str] | None) -> list[str] | None:
    """Normalize filter field values to lists, handling comma-separated values.

    This is needed to handle SDK serialization artifacts where list parameters
    may be serialized as comma-separated strings.

    Args:
        value: A single string, list of strings, or None

    Returns:
        A list of strings, or None if input is None
    """
    if value is None:
        return None
    if isinstance(value, str):
        # Split comma-separated values (SDK serialization artifact)
        return [v.strip() for v in value.split(",") if v.strip()]
    # If it's a list, check if any items contain commas and split them
    result = []
    for item in value:
        if isinstance(item, str) and "," in item:
            result.extend([v.strip() for v in item.split(",") if v.strip()])
        else:
            result.append(item)
    return result if result else None


# Backward-compat alias
normalize_search_list = normalize_filter_list


def _make_filter_field_type(field_type: Any) -> Any:
    """Create an optional filter field type that supports single or list values.

    If the type is already a list, unwrap it to avoid nested arrays (e.g. list[list[X]]).
    """
    origin = get_origin(field_type)
    if origin is list:
        args = get_args(field_type)
        item_type = args[0] if args else Any
        return Optional[Union[item_type, List[item_type]]]
    return Optional[Union[field_type, List[field_type]]]


def _remove_optional_from_type(field_type: Any) -> Any:
    """Remove Optional wrapper from a type annotation."""
    origin = get_origin(field_type)
    if origin is Union or origin is types.UnionType:
        args = get_args(field_type)
        # Check if there is a None type in the Union
        if len(args) >= 2 and type(None) in args:
            # Return the non-None type
            return next(arg for arg in args if arg is not type(None))
    return field_type


def make_filter_class(
    name: str,
    model_cls: Type[BaseModel],
    *,
    base_class: Optional[Type[BaseModel]] = None,
    include_model_fields: Optional[list[str]] = None,
    base_classes: Optional[list[Type[BaseModel]]] = None,
    date_filters: Optional[list[str | tuple[str, Type]]] = None,
    explicit_base_classes: bool = True,
    json_filter: bool = False,
    # Backward-compat aliases
    date_searches: Optional[list[str | tuple[str, Type]]] = None,
    json_search: bool = False,
) -> Type[BaseModel]:
    """Create a Pydantic 2 filter class with optional fields from specified sources.

    Args:
        name: Name of the new class
        model_cls: Primary class to extract fields from
        base_class: Class to inherit from (keyword-only, optional)
        include_model_fields: List of field names to include from model_cls (keyword-only, optional)
        base_classes: List of base classes to extract all fields from (keyword-only, optional)
        date_filters: List of (field_name, date_range_type) tuples for date range filters (keyword-only, optional)
        explicit_base_classes: If True, apply include_model_fields filter to base_classes as well (default: True)
        json_filter: If True, wrap the schema in oneOf with string type to support advanced JSON filter strings

    Returns:
        New Pydantic class with optional fields
    """
    # Support backward-compat aliases
    effective_date_filters = date_filters or date_searches
    effective_json_filter = json_filter or json_search

    field_definitions: Dict[str, Any] = {}

    # Build alias-to-field mapping for the model class
    # This allows filtering by either field name or alias
    model_fields = model_cls.model_fields
    alias_map: Dict[str, str] = {}  # alias -> actual field name
    for actual_name, field_info in model_fields.items():
        if field_info.alias and field_info.alias != actual_name:
            alias_map[field_info.alias] = actual_name

    # Get fields from model_cls that are in include_model_fields
    if include_model_fields:
        for field_name in include_model_fields:
            # Check if the requested name is an alias
            actual_field_name = alias_map.get(field_name, field_name)
            if actual_field_name in model_fields:
                field_info = model_fields[actual_field_name]
                # Remove Optional wrapper from field_type before creating field_condition_type
                field_type = _remove_optional_from_type(field_info.annotation)
                field_condition_type = _make_filter_field_type(field_type)
                # Use the requested name (may be alias) for the filter field
                field_definitions[field_name] = (field_condition_type, None)

    # Get all fields from include_base_classes
    if base_classes:
        for base_cls in base_classes:
            base_fields = base_cls.model_fields
            # Build alias map for this base class
            base_alias_map: Dict[str, str] = {}
            reverse_alias_map: Dict[str, str] = {}  # field name -> alias
            for actual_name, fi in base_fields.items():
                if fi.alias and fi.alias != actual_name:
                    base_alias_map[fi.alias] = actual_name
                    reverse_alias_map[actual_name] = fi.alias

            for actual_field_name, field_info in base_fields.items():
                # Determine the name to use for this field in the filter class
                # If it has an alias, prefer that name
                output_name = reverse_alias_map.get(actual_field_name, actual_field_name)

                # Apply include_model_fields filter if explicit_base_classes is True
                if explicit_base_classes and include_model_fields:
                    # Check if either the actual name or alias is in include_model_fields
                    if actual_field_name not in include_model_fields and output_name not in include_model_fields:
                        continue

                if output_name not in field_definitions:  # Don't override existing fields
                    field_type = _remove_optional_from_type(field_info.annotation)
                    field_condition_type = _make_filter_field_type(field_type)
                    field_definitions[output_name] = (field_condition_type, None)

    if effective_date_filters:
        for date_filter in effective_date_filters:
            if isinstance(date_filter, tuple):
                date_field, date_range_type = date_filter
            else:
                date_field = date_filter
                date_range_type = DatetimeFilter
            field_definitions[date_field] = (Optional[date_range_type], None)
    # Create the new class using create_model
    # Only pass __base__ if base_class is provided
    create_kwargs = {}
    if base_class is not None:
        create_kwargs["__base__"] = base_class

    create_kwargs["__config__"] = ConfigDict(extra="forbid")
    filter_cls = create_model(name, **create_kwargs, **field_definitions)

    if effective_json_filter:

        def _json_filter_schema_extra(schema: dict) -> None:
            obj_schema = {k: v for k, v in schema.items() if k != "title"}
            schema.clear()
            schema["oneOf"] = [obj_schema, {"type": "string"}]
            schema["title"] = name

        filter_cls.model_config["json_schema_extra"] = _json_filter_schema_extra

    return filter_cls


# Backward-compat alias
make_search_class = make_filter_class


def parse_model_entity_ref(identifier: str, default_workspace: str | None = None) -> ParsedEntityRef:
    """Parse a model-entity identifier into a workspace and name, preserving composite names.

    This is the model-entity-aware counterpart to :func:`parse_entity_ref`. It splits on
    the **first** ``/`` only, so the returned ``name`` may itself contain ``/``.

    This matches the cache-key convention used by
    :meth:`nmp.core.inference_gateway.api.model_cache.ModelCache.rebuild_model_entity_map`,
    which keys on ``(workspace, model_entity_name)`` after a single split. Use this when
    parsing identifiers that may legitimately encode a LoRA composite of the form
    ``{base}&adapters/{adapter_workspace}/{adapter_name}`` — the entire composite is the
    entity name, not a workspace path.

    Accepted formats:

    - ``$entity_name`` — uses *default_workspace* as workspace if not provided in the
      identifier. The entity name itself may contain ``/`` (LoRA composite case).
    - ``$workspace/$entity_name`` — explicit workspace; everything after the first ``/``
      is the entity name (which may contain further ``/`` segments).

    Args:
        identifier: The model-entity identifier to parse.
        default_workspace: The workspace to use if the identifier is not qualified with one.

    Returns:
        A :class:`ParsedEntityRef` with the workspace and (possibly composite) name.

    Raises:
        ValueError: If the identifier is empty, contains an empty workspace or name segment,
            or is unqualified and *default_workspace* is ``None``.
    """
    stripped = identifier.strip()
    if not stripped:
        raise ValueError(f"invalid model entity reference {identifier!r}; must not be empty")

    workspace, separator, name = stripped.partition("/")

    if separator:
        # Qualified form: workspace/name (where name itself may contain '/').
        if not workspace or not name:
            raise ValueError(
                f"invalid model entity reference {identifier!r}; "
                "expected 'name' or 'workspace/name' with non-empty segments"
            )
        return ParsedEntityRef(workspace=workspace, name=name)

    # Unqualified form: just a bare name.
    if default_workspace is None:
        raise ValueError(
            f"Model entity identifier '{identifier}' is not qualified with a workspace and "
            "default workspace is not provided. Must be in the format $workspace/$entity_name "
            "or $entity_name."
        )
    return ParsedEntityRef(workspace=default_workspace, name=workspace)


@dataclass(frozen=True)
class ModelEntityId:
    """A parsed model-entity id, LoRA-composite aware.

    Consolidates the ad-hoc ``&adapters/`` partitioning that was hand-rolled across
    the inference-gateway and models services (validation, reconciler served-model
    checks + composite construction, VM routing, error formatting). Both the parse
    and format directions of the composite grammar live here so they cannot drift.

    A model-entity id has one of two shapes:

    - **Plain**: ``{workspace}/{base_name}`` — ``is_lora`` is ``False`` and the three
      adapter fields are ``None``.
    - **LoRA composite**: ``{workspace}/{base_name}&adapters/{adapter_workspace}/{adapter_name}``
      — a fine-tuned adapter parented on a base model. It has no standalone ModelEntity;
      routing rides the base model's VirtualModel (see the IGW ``resolve_vm_for_model``).

    This is a *composition* of :func:`parse_model_entity_ref` (the first-``/``-only split
    that yields ``(workspace, name)`` while preserving a composite ``name``) plus a
    decomposition of that ``name`` on :data:`ADAPTERS_INFIX`. It deliberately does **not**
    extend :class:`ParsedEntityRef`: that is the platform-wide entity ref (filesets,
    guardrails, agents, secrets, …) whose own parser rejects composites, so LoRA/adapter
    semantics have no business on it.

    Fields:
        workspace: The base model's workspace (the segment before the first ``/``).
        base_name: The base model entity name (the segment before ``&adapters/``).
        adapter_workspace: The adapter's workspace, or ``None`` for a plain (non-LoRA) id.
        adapter_name: The adapter's name, or ``None`` for a plain id.
        is_lora: ``True`` iff this id encodes a LoRA adapter composite.
    """

    workspace: str
    base_name: str
    adapter_workspace: str | None = None
    adapter_name: str | None = None

    def __post_init__(self) -> None:
        # The adapter fields are all-or-nothing: a partial pair would set is_lora False
        # and silently drop the supplied field on to_composite(). Empty strings would
        # serialize an invalid composite. Reject both so a constructed instance is always
        # a well-formed plain-or-LoRA id.
        if (self.adapter_workspace is None) != (self.adapter_name is None):
            raise ValueError("adapter_workspace and adapter_name must be provided together")
        if self.adapter_workspace == "" or self.adapter_name == "":
            raise ValueError("adapter fields must not be empty")

    @property
    def is_lora(self) -> bool:
        """Whether this id encodes a LoRA adapter composite."""
        return self.adapter_workspace is not None and self.adapter_name is not None

    @classmethod
    def parse(cls, identifier: str, default_workspace: str | None = None) -> "ModelEntityId":
        """Parse a (possibly composite) model-entity id into its components.

        Layers on top of :func:`parse_model_entity_ref`: first split off the workspace
        (first ``/`` only, composite-preserving), then decompose the resulting name on
        :data:`ADAPTERS_INFIX`.

        .. important::
           *identifier* must be **workspace-qualified** (``workspace/...``) for the LoRA
           composite to be recognized. A **bare** composite name (no leading workspace,
           e.g. ``base&adapters/ws/adapter`` — as seen in an IGW request body where the
           workspace comes from the URL path) would have its first ``/`` consumed as the
           workspace split and would NOT be detected as LoRA. For a bare name, use
           :func:`parse_adapters_suffix` directly on the name instead.

        Args:
            identifier: A model-entity id, plain or LoRA composite.
            default_workspace: Workspace to use when *identifier* is unqualified.

        Returns:
            The parsed :class:`ModelEntityId`.

        Raises:
            ValueError: If the id is empty / has empty segments / is unqualified without a
                default workspace (propagated from :func:`parse_model_entity_ref`), or if it
                carries an ``&adapters/`` infix that is not a well-formed
                ``base&adapters/adapter_workspace/adapter_name`` composite.
        """
        ref = parse_model_entity_ref(identifier, default_workspace=default_workspace)

        if ADAPTERS_INFIX not in ref.name:
            return cls(workspace=ref.workspace, base_name=ref.name)

        parts = parse_adapters_suffix(ref.name)
        if parts is None:
            raise ValueError(
                f"invalid LoRA composite model entity id {identifier!r}; expected "
                "'workspace/base&adapters/adapter_workspace/adapter_name' with non-empty segments"
            )
        base_name, adapter_workspace, adapter_name = parts
        return cls(
            workspace=ref.workspace,
            base_name=base_name,
            adapter_workspace=adapter_workspace,
            adapter_name=adapter_name,
        )

    @property
    def base_id(self) -> str:
        """The workspace-qualified base model id (``{workspace}/{base_name}``).

        For a plain id this is the whole id; for a LoRA composite it is the base model
        the adapter is parented on (the VM-routing anchor).
        """
        return f"{self.workspace}/{self.base_name}"

    def to_composite(self) -> str:
        """Render back to the wire id string.

        Round-trips :meth:`parse`: a plain id renders as ``{workspace}/{base_name}``; a
        LoRA id renders as ``{workspace}/{base_name}&adapters/{adapter_workspace}/{adapter_name}``.
        This is the format side of the grammar — the single home for constructing composite
        ids (previously hand-rolled f-strings in the reconciler and the IGW proxy splice).
        """
        if self.is_lora:
            return f"{self.base_id}{ADAPTERS_INFIX}{self.adapter_workspace}/{self.adapter_name}"
        return self.base_id

    def __str__(self) -> str:
        return self.to_composite()
