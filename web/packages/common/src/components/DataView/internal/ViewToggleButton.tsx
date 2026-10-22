// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useInnerDataViewContext } from '@nemo/common/src/components/DataView/internal/context';
import { Button, type ButtonProps } from '@nvidia/foundations-react-core';
import { LayoutGrid, LayoutList } from '@nvidia/react-gui-icons/icons';
import type { JSX, ReactElement } from 'react';

interface ViewToggleItem {
  /** The label to display for the view. Rendered when there is enough space. */
  children: string;
  /** The icon to display for the view. Always rendered. */
  slotLeft: ReactElement;
  /** The value to set when the view is toggled to. */
  value: string;
}

export const DEFAULT_VIEW_ITEMS: ViewToggleItem[] = [
  { children: 'Table View', slotLeft: <LayoutList variant="fill" />, value: 'table' },
  { children: 'Card View', slotLeft: <LayoutGrid variant="fill" />, value: 'card' },
];

/**
 * Toggles between display modes for the DataView. Cycles through items in the order they're
 * defined. Defaults to "table" and "card" views.
 */
export function ViewToggleButton({
  items = DEFAULT_VIEW_ITEMS,
  ...props
}: Omit<ButtonProps, 'children'> & {
  items?: ViewToggleItem[];
}): JSX.Element {
  const {
    state: { displayMode },
  } = useInnerDataViewContext();
  if (items.length === 0) {
    return <Button aria-label="Toggle view" kind="tertiary" disabled {...props} />;
  }
  const currentViewIndex = Math.max(
    0,
    items.findIndex((item) => item.value === displayMode.state)
  );
  const nextViewIndex = (currentViewIndex + 1) % items.length;
  const content = items[nextViewIndex];
  return (
    <Button
      aria-label="Toggle view"
      onClick={() => {
        if (items[nextViewIndex]) {
          displayMode.set(items[nextViewIndex].value);
        }
      }}
      kind="tertiary"
      {...props}
    >
      {content?.slotLeft}
      <span className="hide-mobile">{content?.children}</span>
    </Button>
  );
}
