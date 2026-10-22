// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ColumnFilter } from '@nemo/common/src/components/DataView/internal/ColumnFilter';
import { useInnerDataViewContext } from '@nemo/common/src/components/DataView/internal/context';
import {
  DropdownContent,
  DropdownRoot,
  DropdownSub,
  DropdownSubContent,
  DropdownSubTrigger,
  DropdownTrigger,
  type DropdownTriggerProps,
} from '@nvidia/foundations-react-core';
import { Filter } from '@nvidia/react-gui-icons/icons';
import { Fragment, useEffect, useState, type JSX, type ReactNode } from 'react';

/**
 * FilterMenu component used to display the filter menu in the data view. Should be rendered
 * inside a `DataView.Toolbar` component.
 */
export function FilterMenu({
  children,
  closeOnFilterChange = false,
  disabled = false,
  size,
  ...props
}: {
  children?: ReactNode;
  /** Whether the menu should be closed when a filter is applied. @defaultValue false */
  closeOnFilterChange?: boolean;
  size?: 'small' | 'medium' | 'large';
} & Omit<DropdownTriggerProps, 'children'>): JSX.Element {
  const [menuKey, rerender] = useState(0);
  const { table, state } = useInnerDataViewContext();
  const filterableColumns = table.getAllLeafColumns().filter((col) => col.getCanFilter());
  useEffect(() => {
    if (closeOnFilterChange) rerender((prev) => prev + 1);
  }, [state.columnFiltering.state, closeOnFilterChange]);
  return (
    <DropdownRoot key={menuKey} open={disabled ? false : undefined} size={size}>
      <DropdownTrigger disabled={disabled} {...props}>
        {children ?? (
          <Fragment>
            <Filter />
            <span className="hide-mobile">Filter</span>
          </Fragment>
        )}
      </DropdownTrigger>
      <DropdownContent>
        {filterableColumns.map((column) => (
          <DropdownSub key={`option-${column.id}`}>
            <DropdownSubTrigger>
              {(column.columnDef.meta?.filter?.label as ReactNode) ??
                (column.columnDef.header as ReactNode)}
            </DropdownSubTrigger>
            <DropdownSubContent>
              <ColumnFilter column={column} onClose={() => rerender((prev) => prev + 1)} />
            </DropdownSubContent>
          </DropdownSub>
        ))}
      </DropdownContent>
    </DropdownRoot>
  );
}
