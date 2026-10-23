// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  isDateTimeFilter,
  type DatetimeFilterValue,
} from '@nemo/common/src/components/DataView/dateTimeFilter';
import {
  formatNumberRange,
  isNumberRangeFilter,
  type NumberRangeFilterValue,
} from '@nemo/common/src/components/DataView/FilterPanel/NumberRangeFilter/util';
import {
  useInnerDataViewContext,
  type TanstackTable,
} from '@nemo/common/src/components/DataView/internal';
import { formatDateRange } from '@nemo/common/src/utils/formatDateRange';
import { Close } from '@nemo/icons/icons';
import { Button, Flex, Tag } from '@nvidia/foundations-react-core';

type DataViewColumn = TanstackTable.Column<unknown>;

function getFilterLabel(column: DataViewColumn): string {
  const filterDef = column.columnDef.meta?.filter;
  const header = column.columnDef.header;
  return filterDef?.label ?? (typeof header === 'string' ? header : column.id);
}

function formatFilterValue(column: DataViewColumn): string {
  const filter = column.columnDef.meta?.filter;
  const value = column.getFilterValue();

  if (isDateTimeFilter(filter)) {
    const dt = value as DatetimeFilterValue | undefined;
    return formatDateRange(dt?.$gte, dt?.$lte);
  }

  if (isNumberRangeFilter(filter)) {
    const range = value as NumberRangeFilterValue | undefined;
    return formatNumberRange(range?.$gte, range?.$lte);
  }

  if (typeof value === 'string') {
    const options =
      (filter?.type === 'single-select' || filter?.type === 'multi-select') && filter.options;
    if (options) {
      const match = options.find((opt) => opt.value === value);
      return match?.label ?? value;
    }
    return value;
  }

  if (Array.isArray(value)) {
    const options =
      (filter?.type === 'single-select' || filter?.type === 'multi-select') && filter.options;
    return value
      .map((v) => {
        if (options) {
          const match = options.find((opt) => opt.value === v);
          return match?.label ?? v;
        }
        return String(v);
      })
      .join(', ');
  }

  if (typeof value === 'object' && value != null) {
    const keys = Object.keys(value);
    const options =
      (filter?.type === 'single-select' || filter?.type === 'multi-select') && filter.options;
    if (options) {
      return keys.map((k) => options.find((opt) => opt.value === k)?.label ?? k).join(', ');
    }
    return keys.join(', ');
  }

  return value != null ? String(value) : '';
}

function FilterTag({ column }: { column: DataViewColumn }) {
  const label = getFilterLabel(column);
  const displayValue = formatFilterValue(column);

  return (
    <Tag
      color="gray"
      density="compact"
      kind="outline"
      className="whitespace-nowrap"
      onClick={() => column.setFilterValue(undefined)}
    >
      <b>{label}: </b>
      {displayValue}
      <Close width={12} height={12} />
    </Tag>
  );
}

export function StudioAppliedFilters() {
  const { table } = useInnerDataViewContext();
  const filteredValues = table.getState().columnFilters;

  if (filteredValues.length === 0) return null;

  return (
    <Flex align="center" gap="density-xs" wrap="wrap">
      {filteredValues.map(({ id: columnId }) => {
        const column = table.getColumn(columnId);
        if (!column || !column.columnDef.meta?.filter) return null;

        return <FilterTag key={columnId} column={column} />;
      })}
      <Button
        className="text-nowrap"
        data-testid="clear-filters"
        onClick={() => table.setColumnFilters([])}
        size="small"
        kind="tertiary"
      >
        <span className="hide-mobile">Clear Filters</span>
        <Close />
      </Button>
    </Flex>
  );
}
