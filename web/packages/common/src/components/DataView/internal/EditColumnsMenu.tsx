// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useInnerDataViewContext } from '@nemo/common/src/components/DataView/internal/context';
import {
  DropdownCheckboxItem,
  DropdownContent,
  DropdownHeading,
  DropdownRoot,
  type DropdownRootProps,
  DropdownSection,
  DropdownTrigger,
  type DropdownTriggerProps,
} from '@nvidia/foundations-react-core';
import { childrenToText } from '@nvidia/foundations-react-core/lib';
import { Sliders } from '@nvidia/react-gui-icons/icons';
import { flexRender } from '@tanstack/react-table';
import { Fragment, type JSX, type ReactNode } from 'react';

export interface EditColumnsMenuProps
  extends Pick<DropdownRootProps, 'size'>, Omit<DropdownTriggerProps, 'children' | 'size'> {
  /** Content to render inside the dropdown trigger, before the columns. */
  children?: ReactNode;
  /** Additional content rendered inside the menu, after the columns. */
  slotContent?: ReactNode;
}

/**
 * When clicked this button renders a menu to control column settings.
 */
export function EditColumnsMenu({
  children,
  size,
  slotContent,
  ...props
}: EditColumnsMenuProps): JSX.Element {
  const { table } = useInnerDataViewContext();
  const columns = table.getAllColumns();
  return (
    <DropdownRoot size={size}>
      <DropdownTrigger aria-label="Toggle column visibility" {...props}>
        {children ?? (
          <Fragment>
            <Sliders variant="fill" />
            <span className="hide-mobile">View</span>
          </Fragment>
        )}
      </DropdownTrigger>
      <DropdownContent>
        <DropdownSection>
          <DropdownHeading>Toggle Columns</DropdownHeading>
          {columns
            .filter((col) => col.getCanHide())
            .map((column) => {
              const headerContext =
                typeof column.columnDef.header !== 'string'
                  ? table
                      .getFlatHeaders()
                      .find((header) => header.id === column.id)
                      ?.getContext()
                  : undefined;
              return (
                <DropdownCheckboxItem
                  key={column.id}
                  checked={column.getIsVisible()}
                  filterValue={childrenToText(column.columnDef.header as ReactNode)}
                  onCheckedChange={(value) => {
                    column.toggleVisibility(!!value);
                  }}
                >
                  {flexRender(column.columnDef.header, headerContext!)}
                </DropdownCheckboxItem>
              );
            })}
        </DropdownSection>
        {slotContent}
      </DropdownContent>
    </DropdownRoot>
  );
}
