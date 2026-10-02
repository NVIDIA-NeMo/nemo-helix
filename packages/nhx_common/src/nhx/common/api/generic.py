# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic endpoint implementations."""

import math
from operator import attrgetter
from typing import List, Type

from fastapi import HTTPException, Request, status
from nhx.common.api.common import Page, PaginationData
from nhx.common.api.utils import filter_match, parse_deep_object
from pydantic import BaseModel


async def generic_get(
    all_items: List[BaseModel],
    request: Request,
    page: int,
    page_size: int,
    sort: str,
    filter_cls: Type[BaseModel],
):
    """Generic logic for applying filtering and pagination.

    # TODO: add support for sorting.
    """
    raw_filter = parse_deep_object(name="filter", params=request.query_params)
    filter_obj: BaseModel | None = filter_cls.model_validate(raw_filter) if raw_filter else None

    if filter_obj:
        filtered_items = [item for item in all_items if filter_match(item.model_dump(), filter_obj.model_dump())]
    else:
        filtered_items = all_items

    if sort:
        reverse = sort.startswith("-")
        sort_field = sort.lstrip("-")
        try:
            filtered_items.sort(key=attrgetter(sort_field), reverse=reverse)
        except AttributeError:
            raise ValueError(f"Invalid sort field: {sort_field}")

    if page < 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Page index cannot be less than 0",
        )

    if page_size <= 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Page size cannot be less than 1",
        )

    page_items = filtered_items[(page - 1) * page_size : page * page_size]

    return Page(
        data=page_items,
        pagination=PaginationData(
            page=page,
            page_size=page_size,
            current_page_size=len(page_items),
            total_pages=int(math.ceil(len(filtered_items) / page_size)),
            total_results=len(filtered_items),
        ),
        sort=sort,
        filter=filter_obj.model_dump(mode="json", exclude_none=True) if filter_obj else None,
    )
