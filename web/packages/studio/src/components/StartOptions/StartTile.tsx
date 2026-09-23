// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isDefined } from '@nemo/common/src/utils/isDefined';
import { Card, Flex, Spinner, Text } from '@nvidia/foundations-react-core';
import type { StartTileProps } from '@studio/components/StartOptions/types';
import cn from 'classnames';
import type { FC } from 'react';

/**
 * Picking a tile runs the flow, so this is a button rather than a radio: a radiogroup
 * moves selection on arrow keys, which here would fire an action per keypress.
 *
 * The `.nv-card-content` overrides mirror RadioCard's compact tile so the two stay
 * pixel-identical; Card owns that element, so the grid has to be set through it.
 */
export const StartTile: FC<StartTileProps> = ({
  label,
  description,
  icon,
  slotEnd,
  onSelect,
  disabled = false,
  busy = false,
  busyLabel,
  labelKind,
  descriptionKind,
  className,
}) => {
  const hasDescription = isDefined(description);

  const gapClass = hasDescription
    ? '[&_.nv-card-content]:gap-x-2! [&_.nv-card-content]:gap-y-1!'
    : '[&_.nv-card-content]:gap-0! [&_.nv-card-content]:gap-x-2!';

  const contentClass = `[&_.nv-card-content]:grid [&_.nv-card-content]:grid-cols-[auto_1fr] [&_.nv-card-content]:grid-rows-[auto_auto] [&_.nv-card-content]:items-center! [&_.nv-card-content]:w-full [&_.nv-card-content]:p-3! ${gapClass}`;

  return (
    <button
      type="button"
      onClick={onSelect}
      disabled={disabled}
      className={cn('group block w-full text-left', disabled && 'cursor-not-allowed opacity-50')}
    >
      <Card
        className={cn(
          'w-full',
          !disabled && 'cursor-pointer hover:bg-interaction-hover',
          busy && 'border-interaction-selected',
          contentClass,
          className
        )}
      >
        <Flex
          align="center"
          aria-hidden
          className="col-start-1 row-span-2 row-start-1 shrink-0 self-start pt-0.5 text-base-foreground"
        >
          {icon}
        </Flex>

        <Flex gap="density-md" align="center" className="col-start-2 row-start-1 w-full min-h-0">
          <Text kind={labelKind}>{label}</Text>
          {busy ? (
            <Flex gap="density-md" align="center" className="ml-auto shrink-0">
              {isDefined(busyLabel) && (
                <Text kind="label/regular/sm" color="secondary">
                  {busyLabel}
                </Text>
              )}
              <Spinner size="small" aria-label={busyLabel ?? 'Working'} />
            </Flex>
          ) : (
            slotEnd != null && <div className="ml-auto shrink-0">{slotEnd}</div>
          )}
        </Flex>

        {hasDescription && (
          <Text
            kind={descriptionKind}
            color="secondary"
            className="col-start-2 row-start-2 text-left"
          >
            {description}
          </Text>
        )}
      </Card>
    </button>
  );
};
