# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import Optional, TypeVar

from nemo_helix_plugin.schema import Filter as Filter
from nemo_helix_plugin.schema import Value as Value
from pydantic import Field

V = TypeVar("V")


class DatetimeFilter(Filter):
    gte: Optional[datetime] = Field(None, description="Filter for results greater than or equal to this datetime.")
    lte: Optional[datetime] = Field(None, description="Filter for results less than or equal to this datetime.")
