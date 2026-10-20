// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { DatetimeFilterValue } from '@nemo/common/src/components/DataView/dateTimeFilter';
import type { DataViewColumn } from '@nemo/common/src/components/DataView/FilterPanel/types';
import { DateRangeFilterField } from '@nemo/common/src/components/FilterFields/DateRangeFilterField';

export function DateTimeFilterControl({ column }: { column: DataViewColumn }) {
  return (
    <DateRangeFilterField
      dataTestId={`column-filter-${column.id}`}
      value={column.getFilterValue() as DatetimeFilterValue | undefined}
      onValueChange={(value) => column.setFilterValue(value)}
    />
  );
}
