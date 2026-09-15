// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Badge, Flex, Text } from '@nvidia/foundations-react-core';
import { SelectableCard } from '@studio/components/SelectableCard';
import type { StartOptionCardProps } from '@studio/components/StartOptions/types';
import type { FC } from 'react';

/**
 * A single "How do you want to start?" tile: a leading icon badge above a title,
 * description, and a metadata badge, in a bordered card rendered as a `<button>`.
 *
 * Disabled options are still shown so the full set of entry points is visible, but
 * they are inert — no hover affordance, no selection, and `aria-disabled`.
 */
export const StartOptionCard: FC<StartOptionCardProps> = ({
  option,
  selected,
  onSelect,
  compact = false,
}) => {
  const Icon = option.icon;
  const interactive = option.enabled;

  return (
    <SelectableCard
      selected={selected}
      onSelect={onSelect}
      disabled={!interactive}
      // `h-full` makes every tile match the tallest in the row; the min-height keeps the
      // row from collapsing when they are all one-liners.
      className={compact ? 'h-full min-h-[104px] gap-2 p-4' : 'h-[240px] gap-3 p-5'}
    >
      {compact ? (
        <Flex align="center" gap="density-sm" className="w-full">
          <Flex
            align="center"
            justify="center"
            className="size-7 shrink-0 rounded-md bg-surface-sunken"
          >
            <Icon size={16} className="text-primary" aria-hidden />
          </Flex>
          <Text kind="body/bold/md" className="text-primary">
            {option.title}
          </Text>
          <div className="flex-1" />
          {option.tag ? (
            <Badge color={option.tag.color} kind={option.tag.kind} size="small">
              {option.tag.label}
            </Badge>
          ) : null}
        </Flex>
      ) : (
        <>
          <Flex
            align="center"
            justify="center"
            className="size-10 shrink-0 rounded-md bg-surface-sunken"
          >
            <Icon size={20} className="text-primary" aria-hidden />
          </Flex>

          <Text kind="body/bold/md" className="text-primary">
            {option.title}
          </Text>
        </>
      )}

      <Text kind="body/regular/sm" className="text-secondary">
        {option.description}
      </Text>

      {compact ? null : (
        <>
          <div className="flex-1" />
          {option.tag ? (
            <Badge color={option.tag.color} kind={option.tag.kind}>
              {option.tag.label}
            </Badge>
          ) : null}
        </>
      )}
    </SelectableCard>
  );
};
