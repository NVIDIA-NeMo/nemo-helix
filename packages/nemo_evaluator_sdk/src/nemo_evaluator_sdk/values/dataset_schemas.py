# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public Pydantic models for canonical evaluator schema and dataset column mapping."""

from __future__ import annotations

import re
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Wildcard array segment. Declared here rather than imported from ``dataset_schemas.common``
#: so this values module keeps no dependency on that package.
_ARRAY_WILDCARD = "[]"

_KNOWN_BINDING_FIELDS = (
    "input",
    "output",
    "context",
    "reference",
    "trajectory",
    "messages",
    "tool_calls",
    "tools",
)

#: A dotted path, optionally with array segments: ``answer``, ``messages[1].content``,
#: ``turns[0][2].text``, ``messages[role=assistant].content``. A bracket group holds digits, or a
#: ``key=value`` predicate whose sides exclude ``.``, quotes, brackets, ``=`` and whitespace.
#: The wildcard ``[]`` parses here and is rejected by
#: :meth:`FieldMapping.validate_supported_dataset_paths`, which can say why; so is a predicate
#: followed by a further bracket group. Anything else bracketed -- ``[-1]``, ``[x]``, ``a[1]b`` --
#: fails here.
_FIELD_MAPPING_PATH_PATTERN = r"""^[^\[\]]*(?:\[(?:[0-9]*|[^\[\]"'.=\s]+=[^\[\]"'.=\s]+)\](?:\.[^\[\]]*)?)*$"""

#: A predicate group immediately followed by another bracket group. The predicate already binds one
#: element, so whatever follows can only fail to resolve. Rejecting it keeps the slot free: a bare
#: predicate means the last match, and this position can later name one explicitly --
#: ``[role=assistant][0]`` for the first -- without changing what the bare form means.
_PREDICATE_THEN_BRACKET = re.compile(r"\[[^\[\]]*=[^\[\]]*\]\[")

_FieldMappingPath = Annotated[str, Field(pattern=_FIELD_MAPPING_PATH_PATTERN, min_length=1)]


class InputSchema(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True)

    schema_: dict = Field(
        alias="schema",
        description=(
            "Canonical evaluator input schema expressed as JSON Schema. "
            "This describes the normalized template context required by the metric, "
            "not the raw dataset row shape."
        ),
    )

    @model_validator(mode="after")
    def validate_schema(self) -> Self:
        from nemo_evaluator_sdk.dataset_schemas.common import validate_json_schema

        validate_json_schema(self.schema_)
        return self


class _FieldMappingBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: _FieldMappingPath | None = Field(
        default=None, description="Binding for the canonical 'input' evaluator field."
    )
    output: _FieldMappingPath | None = Field(
        default=None, description="Binding for the canonical 'output' evaluator field."
    )
    context: _FieldMappingPath | None = Field(
        default=None, description="Binding for the canonical 'context' evaluator field."
    )
    reference: _FieldMappingPath | None = Field(
        default=None,
        description="Binding for the canonical 'reference' evaluator field.",
    )
    trajectory: _FieldMappingPath | None = Field(
        default=None,
        description="Binding for the canonical 'trajectory' evaluator field.",
    )
    messages: _FieldMappingPath | None = Field(
        default=None,
        description="Binding for the canonical 'messages' evaluator field.",
    )
    tool_calls: _FieldMappingPath | None = Field(
        default=None,
        description="Binding for the canonical 'tool_calls' evaluator field.",
    )
    tools: _FieldMappingPath | None = Field(
        default=None, description="Binding for the canonical 'tools' evaluator field."
    )
    custom: dict[str, _FieldMappingPath] = Field(
        default_factory=dict,
        description="Additional evaluator field bindings keyed by canonical field name.",
    )

    @model_validator(mode="after")
    def validate_custom_keys(self) -> Self:
        duplicates = sorted(set(self.custom).intersection(_KNOWN_BINDING_FIELDS))
        if duplicates:
            raise ValueError(f"custom binding keys overlap with reserved evaluator fields: {duplicates}")
        return self

    def mapping(self) -> dict[str, str]:
        result = {name: value for name in _KNOWN_BINDING_FIELDS if (value := getattr(self, name)) is not None}
        result.update(self.custom)
        return result


class FieldMapping(_FieldMappingBase):
    """Maps canonical evaluator fields to raw dataset column paths.
    Example: {'input': 'question', 'output': 'answer', 'reference': 'gold', 'trajectory': 'steps'}
    """

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_supported_dataset_paths(self) -> Self:
        """Reject paths that parse but can never bind a single element.

        The pattern is deliberately permissive so these two cases can be explained rather than
        rejected as a regex mismatch.
        """
        wildcard = sorted(
            canonical_name for canonical_name, dataset_path in self.mapping().items() if _ARRAY_WILDCARD in dataset_path
        )
        if wildcard:
            raise ValueError(
                f"wildcard array segments ({_ARRAY_WILDCARD}) are not supported for column mappings: "
                f"{wildcard}. Select a single element, with a positional index such as "
                "messages[1].content or a predicate such as messages[role=assistant].content. "
                "This is not JSONPath; filters like [?(@.role=='assistant')] are not accepted."
            )
        chained = sorted(
            canonical_name
            for canonical_name, dataset_path in self.mapping().items()
            if _PREDICATE_THEN_BRACKET.search(dataset_path)
        )
        if chained:
            raise ValueError(
                f"a predicate already selects a single element, so it cannot be followed by another "
                f"array segment: {chained}. Use one or the other."
            )
        return self
