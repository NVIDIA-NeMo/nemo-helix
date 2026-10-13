# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime

from nhx.common.api.common import DateRange


class TestDateRange:
    def test_mixed_naive_and_defined(self):
        data = {"start": "2025-08-14T23:24:14.921757", "end": "2025-08-15T10:30:00+05:00"}

        date_range = DateRange(**data)

        assert date_range.start.tzinfo is None
        assert date_range.end.tzinfo is None
        assert date_range.start == datetime(2025, 8, 14, 23, 24, 14, 921757)
        assert date_range.end == datetime(2025, 8, 15, 5, 30, 0)

    def test_mixed_timezones(self):
        data = {"start": "2025-08-14T23:24:14.921757Z", "end": "2025-08-15T13:30:00+08:00"}

        date_range = DateRange(**data)

        assert date_range.start.tzinfo is None
        assert date_range.end.tzinfo is None
        assert date_range.start == datetime(2025, 8, 14, 23, 24, 14, 921757)
        assert date_range.end == datetime(2025, 8, 15, 5, 30, 0)

    def test_end_before_start(self):
        data = {"start": "2025-08-15T23:24:14.921757Z", "end": "2025-08-14T10:30:00+08:00"}
        import pytest

        with pytest.raises(ValueError, match="Start date must be before end date"):
            DateRange(**data)

    def test_default_values(self):
        date_range = DateRange()

        assert date_range.start is None
        assert date_range.end is None
