# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Response models for the Safe Synthesizer JSON job results."""

from typing import Any

from nemo_safe_synthesizer.config import evaluation_report_data, external_results
from pydantic import Field, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema

_NULL_SCHEMA = {"type": "null"}


def _mark_nullable(node: Any, handler: GetJsonSchemaHandler, visited_refs: set[str]) -> None:
    if isinstance(node, list):
        for item in node:
            _mark_nullable(item, handler, visited_refs)
        return
    if not isinstance(node, dict):
        return
    if _NULL_SCHEMA in node.get("anyOf", []):
        node["nullable"] = True
    ref = node.get("$ref")
    if isinstance(ref, str) and ref not in visited_refs:
        visited_refs.add(ref)
        _mark_nullable(handler.resolve_ref_schema(node), handler, visited_refs)
    for value in node.values():
        _mark_nullable(value, handler, visited_refs)


class NullableFieldsSchema:
    """Marks every ``X | None`` in the model tree ``nullable`` in its JSON schema."""

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        # The spec pipeline collapses `X | None` to `X`; `nullable` is the convention that keeps the null.
        json_schema = handler(core_schema)
        _mark_nullable(json_schema, handler, set())
        return json_schema


class EvaluationReportData(NullableFieldsSchema, evaluation_report_data.EvaluationReportData):
    __doc__ = evaluation_report_data.EvaluationReportData.__doc__


class SafeSynthesizerTiming(NullableFieldsSchema, external_results.SafeSynthesizerTiming):
    __doc__ = external_results.SafeSynthesizerTiming.__doc__


class SafeSynthesizerSummary(NullableFieldsSchema, external_results.SafeSynthesizerSummary):
    __doc__ = external_results.SafeSynthesizerSummary.__doc__

    # NSS pins validation schema mode; marking timing only via the parent splits it into Input/Output.
    timing: SafeSynthesizerTiming = Field(
        description=external_results.SafeSynthesizerSummary.model_fields["timing"].description
    )
