// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Sparkle } from '@nemo/icons/icons';
import {
  Button,
  Divider,
  DropdownContent,
  DropdownDividerItemEntry,
  DropdownItem,
  DropdownRoot,
  DropdownTrigger,
  Flex,
} from '@nvidia/foundations-react-core';
import React, { type FC, type ReactElement, type ReactNode } from 'react';

export interface ActionMenuItem {
  label: string;
  onSelect: () => void;
  icon?: ReactElement<{ size?: number; fill?: string; className?: string }>;
  disabled?: boolean;
  danger?: boolean;
  divider?: Omit<DropdownDividerItemEntry, 'kind'>;
}

interface ActionsMenuProps {
  actions: ActionMenuItem[];
  /** Trigger label. Omit for an icon-only trigger. */
  label?: ReactNode;
  /** Trigger icon. Defaults to a sparkles icon when a label is shown. */
  icon?: ReactElement;
  kind?: 'primary' | 'secondary' | 'tertiary';
  disabled?: boolean;
  ariaLabel?: string;
  'data-testid'?: string;
}

/*
 * ActionsMenu renders a dropdown of actions behind a single trigger button. Use it to consolidate
 * page-level actions instead of lining up individual buttons. Pass a `label` for a labeled
 * "Actions" trigger, or omit it for an icon-only trigger.
 */
export const ActionsMenu: FC<ActionsMenuProps> = ({
  actions,
  label,
  icon,
  kind = 'tertiary',
  disabled,
  ariaLabel = 'Open actions menu',
  'data-testid': testId = 'actions-menu',
}) => {
  const triggerIcon = icon ?? (label ? <Sparkle /> : undefined);

  return (
    <DropdownRoot>
      <DropdownTrigger
        asChild
        data-testid={`${testId}-trigger`}
        onClick={(e) => e.stopPropagation()}
        showChevron={false}
      >
        <Button kind={kind} disabled={disabled} aria-label={ariaLabel}>
          {triggerIcon}
          {label}
        </Button>
      </DropdownTrigger>
      <DropdownContent
        align="end"
        side="bottom"
        data-testid={`${testId}-content`}
        className="w-[180px] min-w-[180px]"
      >
        {actions.map((action, key) => (
          <React.Fragment key={`action-${key}`}>
            <DropdownItem
              data-testid={`${testId}-item`}
              disabled={action.disabled}
              onClick={(e) => {
                e.stopPropagation();
                action.onSelect();
              }}
              danger={action.danger}
            >
              <Flex align="center" gap="density-sm" className="pr-6">
                {action.icon && React.cloneElement(action.icon, { size: 20, fill: 'solid' })}
                {action.label}
              </Flex>
            </DropdownItem>
            {action.divider && <Divider width={action.divider.width} />}
          </React.Fragment>
        ))}
      </DropdownContent>
    </DropdownRoot>
  );
};
