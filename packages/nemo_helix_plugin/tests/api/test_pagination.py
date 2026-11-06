# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQL pagination arithmetic independent of HTTP and repository dependencies."""

import pytest
from nemo_helix_plugin.entities.pagination import pagination_offset


@pytest.mark.parametrize("page,offset", [(1, 0), (2, 2**63 - 1)])
def test_maximum_sql_limit_is_representable(page: int, offset: int):
    assert pagination_offset(page, 2**63 - 1) == offset


@pytest.mark.parametrize(
    "page,page_size,detail",
    [(0, 1, "positive"), (1, 0, "positive"), (1, 2**63, "page_size.*LIMIT")],
)
def test_invalid_sql_pagination(page: int, page_size: int, detail: str):
    with pytest.raises(ValueError, match=detail):
        pagination_offset(page, page_size)
