// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ColumnFilter } from '@nemo/common/src/components/DataView/internal/ColumnFilter';
import { useInnerDataViewContext } from '@nemo/common/src/components/DataView/internal/context';
import type {
  FilterValue,
  IntentionalAny,
} from '@nemo/common/src/components/DataView/internal/types';
import {
  Button,
  DropdownContent,
  DropdownRoot,
  DropdownTrigger,
  Flex,
  type FlexProps,
  Group,
  Tag,
} from '@nvidia/foundations-react-core';
import { Close } from '@nvidia/react-gui-icons/icons';
import type { Column } from '@tanstack/react-table';
import type { JSX } from 'react';

function truncateString(str: string, maxLength = 20): string {
  return str.length > maxLength ? `${str.slice(0, maxLength)}...` : str;
}

/**
 * Displays the list of applied filters in the DataView. Should be rendered below the toolbar.
 */
export function AppliedFilters(props: Partial<FlexProps>): JSX.Element | null {
  const { table } = useInnerDataViewContext();
  const filteredValues = table.getState().columnFilters;
  if (filteredValues.length === 0) {
    return null;
  }
  return (
    <Flex align="center" gap="density-xs" wrap="wrap" {...props}>
      {filteredValues.map(({ id: columnId, value }) => (
        <ColumnFilterTag
          key={columnId}
          column={table.getColumn(columnId)}
          value={value as FilterValue}
        />
      ))}
      <Button
        className="text-nowrap"
        data-testid="clear-filters"
        onClick={() => table.setColumnFilters([])}
        size="small"
        kind="tertiary"
      >
        <Close variant="fill" />
        <span className="hide-mobile">Clear Filters</span>
      </Button>
    </Flex>
  );
}

function formatValue(
  value: FilterValue | string[],
  optionsValuesOverrides?: Record<string, string>
): string {
  if (typeof value === 'string') {
    return optionsValuesOverrides?.[value] ?? value;
  }
  if (Array.isArray(value)) {
    return value.map((val) => optionsValuesOverrides?.[val] ?? val).join(', ');
  }
  if (typeof value === 'object') {
    return Object.keys(value)
      .map((val) => optionsValuesOverrides?.[val] ?? val)
      .join(', ');
  }
  return JSON.stringify(value);
}

/** A tag that represents a filter applied to a table. */
export function ColumnFilterTag({
  column,
  value,
}: {
  column: Column<IntentionalAny> | undefined;
  value: FilterValue | string[];
}): JSX.Element | null {
  if (!column || !column.columnDef.meta?.filter) {
    return null;
  }
  const filterDef = column.columnDef.meta.filter;
  const columnLabel = filterDef.label ?? column.columnDef.header;
  const optionsValuesOverrides =
    (filterDef.type === 'single-select' || filterDef.type === 'multi-select') &&
    filterDef.options?.reduce<Record<string, string>>((acc, curr) => {
      acc[curr.value] = curr.label ?? curr.value;
      return acc;
    }, {});
  const truncatedValue = value
    ? truncateString(formatValue(value, optionsValuesOverrides || undefined))
    : '';
  return (
    <DropdownRoot>
      <Group>
        <DropdownTrigger asChild>
          <Tag className="whitespace-nowrap" color="gray" density="compact" kind="outline">
            <b>{columnLabel as React.ReactNode}: </b>
            {truncatedValue}
          </Tag>
        </DropdownTrigger>
        <DropdownContent>
          <ColumnFilter column={column} />
        </DropdownContent>
        <Tag
          aria-label="Clear filter"
          color="gray"
          density="compact"
          kind="outline"
          onClick={() => {
            column.setFilterValue(undefined);
          }}
        >
          <Close variant="fill" />
        </Tag>
      </Group>
    </DropdownRoot>
  );
}
