// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Badge, Flex, Text } from '@nvidia/foundations-react-core';
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

  const stateClasses = !interactive
    ? 'cursor-not-allowed border-base opacity-50'
    : selected
      ? 'cursor-pointer border-[#76b900]'
      : 'cursor-pointer border-base hover:-translate-y-0.5 hover:border-[#76b900] hover:bg-surface-hover hover:shadow-md';

  return (
    <button
      type="button"
      onClick={interactive ? onSelect : undefined}
      aria-pressed={interactive ? selected : undefined}
      aria-disabled={!interactive}
      className={`flex w-full flex-col items-start rounded-md border bg-surface-raised text-left transition focus-visible:border-strong focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#76b900] ${
        // `h-full` makes every tile match the tallest in the row (a two-line description
        // otherwise leaves its neighbours short); the min-height keeps the row from
        // collapsing when they all happen to be one-liners.
        compact ? 'h-full min-h-[104px] gap-2 p-4' : 'h-[240px] gap-3 p-5'
      } ${stateClasses}`}
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
    </button>
  );
};
