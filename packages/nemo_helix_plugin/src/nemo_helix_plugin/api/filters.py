# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI filter utilities for entity endpoints."""

from copy import deepcopy
from typing import Callable, Type, TypeVar

from fastapi import HTTPException
from nemo_helix_plugin.jobs.openapi_utils import parse_deep_object
from pydantic import BaseModel
from starlette import status
from starlette.requests import Request

FilterType = TypeVar("FilterType")


def _extract_validated_fields(validated: BaseModel, raw: dict) -> dict:
    """Extract fields from validated model that correspond to raw input fields.

    This ensures we get type conversions from Pydantic validation (e.g., datetime strings
    to datetime objects) while only including fields that were actually provided in the
    query parameters, avoiding issues with default values.
    """
    result = {}
    for key in raw.keys():
        if hasattr(validated, key):
            value = getattr(validated, key)
            if isinstance(raw[key], dict) and isinstance(value, BaseModel):
                result[key] = value.model_dump(exclude_none=True, by_alias=True, mode="json")
            elif isinstance(value, BaseModel):
                result[key] = value.model_dump(exclude_none=True, by_alias=True, mode="json")
            else:
                result[key] = value
    return result


def make_filter_obj_dep(filter_model: Type[FilterType], param_name: str = "filter") -> Callable[[Request], FilterType]:
    """Create a FastAPI dependency for parsing deepObject-style filter query parameters.

    This function creates a dependency that parses ``filter[field][subfield]=value``
    style query parameters into a validated Pydantic model.

    Args:
        filter_model: The Pydantic model class to validate the filter against.
        param_name: The name of the query parameter prefix (default: ``"filter"``).

    Returns:
        A FastAPI dependency function that returns the parsed and validated filter.
    """

    async def _dep(request: Request) -> FilterType:
        try:
            raw = parse_deep_object(name=param_name, params=request.query_params) or {}
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

        if raw.get("*"):
            return raw
        else:
            validated = filter_model.model_validate(deepcopy(raw))
            return _extract_validated_fields(validated, raw)

    return _dep
