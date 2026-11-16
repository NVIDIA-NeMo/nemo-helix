// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Sparkle } from '@nemo/icons/icons';
import {
  Button,
  Divider,
  DropdownContent,
  type DropdownDividerItemEntry,
  DropdownItem,
  DropdownRoot,
  DropdownTrigger,
  Flex,
  Stack,
  Text,
  Tooltip,
} from '@nvidia/foundations-react-core';
import React, { type FC, type ReactElement, type ReactNode } from 'react';

export interface ActionMenuItem {
  label: string;
  /** Secondary line under the label, for when the label alone doesn't say what the action does. */
  description?: string;
  onSelect: () => void;
  icon?: ReactElement<{ size?: number; fill?: string; className?: string }>;
  disabled?: boolean;
  disabledReason?: string;
  danger?: boolean;
  divider?: Omit<DropdownDividerItemEntry, 'kind'>;
}

export type DeleteActionLabel = 'Delete' | 'Remove' | `Delete ${string}` | `Remove ${string}`;

export interface DeleteAction {
  onSelect: () => void;
  label?: DeleteActionLabel;
  disabled?: boolean;
  disabledReason?: string;
}

interface ActionsMenuProps {
  actions: ActionMenuItem[];
  deleteAction?: DeleteAction;
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
  deleteAction,
  label,
  icon,
  kind = 'tertiary',
  disabled,
  ariaLabel = 'Open actions menu',
  'data-testid': testId = 'actions-menu',
}) => {
  const triggerIcon = icon ?? (label ? <Sparkle /> : undefined);
  const hasDescriptions = actions.some((action) => action.description);
  const deleteItem: ActionMenuItem | undefined = deleteAction && {
    label: deleteAction.label ?? 'Delete',
    onSelect: deleteAction.onSelect,
    disabled: deleteAction.disabled,
    disabledReason: deleteAction.disabledReason,
    danger: true,
  };
  const items = deleteItem ? [...actions, deleteItem] : actions;
  const showDeleteDivider = actions.length > 0 && !actions.at(-1)?.divider;

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
        className={hasDescriptions ? 'w-[260px] min-w-[260px]' : 'w-[180px] min-w-[180px]'}
      >
        {items.map((action, key) => {
          const disabledReason = action.disabled ? action.disabledReason : undefined;
          const item = (
            <DropdownItem
              data-testid={`${testId}-item`}
              className={disabledReason ? 'pointer-events-none w-full' : undefined}
              disabled={action.disabled}
              onClick={(e) => {
                e.stopPropagation();
                action.onSelect();
              }}
              danger={action.danger}
            >
              <Flex align="center" gap="density-sm" className="pr-6">
                {action.icon && React.cloneElement(action.icon, { size: 20, fill: 'solid' })}
                {action.description ? (
                  <Stack gap="density-xxs" className="items-start text-left">
                    {action.label}
                    <Text kind="body/regular/sm" className="text-secondary">
                      {action.description}
                    </Text>
                  </Stack>
                ) : (
                  action.label
                )}
              </Flex>
            </DropdownItem>
          );

          return (
            <React.Fragment key={`action-${key}`}>
              {action === deleteItem && showDeleteDivider && <Divider />}
              {disabledReason ? (
                <Tooltip
                  slotContent={disabledReason}
                  attributes={{
                    TooltipTrigger: {
                      className: 'flex w-full cursor-not-allowed',
                      onClick: (e) => e.stopPropagation(),
                    },
                  }}
                >
                  {item}
                </Tooltip>
              ) : (
                item
              )}
              {action.divider && <Divider width={action.divider.width} />}
            </React.Fragment>
          );
        })}
      </DropdownContent>
    </DropdownRoot>
  );
};
