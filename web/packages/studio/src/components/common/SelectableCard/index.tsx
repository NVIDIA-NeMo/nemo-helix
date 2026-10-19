// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Badge, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { type FC, type ReactNode } from 'react';

export interface CardIconBadgeProps {
  /** The icon (or other glyph) to center inside the badge. */
  children: ReactNode;
  className?: string;
}

/**
 * The small rounded, sunken square that holds a card's leading icon — the icon box
 * in the Data Designer card design.
 */
export const CardIconBadge: FC<CardIconBadgeProps> = ({ children, className }) => (
  <Flex
    align="center"
    justify="center"
    className={`size-[26px] shrink-0 rounded-sm bg-surface-sunken ${className ?? ''}`}
  >
    {children}
  </Flex>
);

export interface SelectableCardProps {
  /** Leading visual: an icon badge ({@link CardIconBadge}), a status dot, etc. */
  leading?: ReactNode;
  /** Primary label. */
  title: string;
  /** Accessible name when visible card text does not convey the full state. */
  ariaLabel?: string;
  /** Optional secondary line beneath the title, in the header row beside `leading`. */
  subtitle?: string;
  /** Extra classes for the subtitle — e.g. an accent color or uppercase type label. */
  subtitleClassName?: string;
  /** Optional muted description line below the header row. */
  description?: string;
  /** Optional tokens rendered as badges below the description. */
  tags?: string[];
  /** Whether the card reads as selected (draws the strong border). */
  selected?: boolean;
  /** When true, the card is non-interactive and muted; `onActivate` never fires. */
  disabled?: boolean;
  /** Tooltip explaining why the card is disabled (shown on hover when `disabled`). */
  disabledReason?: string;
  /** Activation handler; fired on click and on keyboard Enter/Space. */
  onActivate?: () => void;
  /** Prevent the activation click from reaching an interactive parent. */
  stopPropagation?: boolean;
  className?: string;
}

/**
 * A compact, keyboard-activatable card: a leading visual beside a title and an
 * optional subtitle, optionally followed by a description and a row of badges,
 * in a bordered box that highlights on hover, focus, and when selected. Shared by
 * the Data Designer column palette and the DAG canvas nodes so the two stay
 * visually identical.
 *
 * Rendered as a native `<button>` so it is focusable and activatable with no extra
 * wiring.
 */
export const SelectableCard: FC<SelectableCardProps> = ({
  leading,
  title,
  ariaLabel,
  subtitle,
  subtitleClassName,
  description,
  tags,
  selected = false,
  disabled = false,
  disabledReason,
  onActivate,
  stopPropagation = false,
  className,
}) => (
  <button
    type="button"
    onClick={
      disabled
        ? undefined
        : (event) => {
            if (stopPropagation) event.stopPropagation();
            onActivate?.();
          }
    }
    disabled={disabled}
    aria-pressed={selected}
    aria-label={ariaLabel}
    title={disabled ? disabledReason : title}
    className={`flex w-[240px] justify-between flex-col items-start gap-1.5 overflow-hidden rounded-md border bg-surface-raised px-2 py-1.5 text-left transition-colors ${
      disabled
        ? 'cursor-not-allowed opacity-50 border-base'
        : `cursor-pointer hover:border-strong hover:bg-surface-hover focus-visible:border-strong focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-(--color-brand,#76b900) ${selected ? 'border-strong' : 'border-base'}`
    } ${className ?? ''}`}
  >
    <Stack gap="1.5" className="w-full min-w-0">
      <Flex className="w-full min-w-0 items-center gap-2">
        {leading}
        <Stack gap="density-xxs" className="min-w-0 flex-1">
          <Text kind="body/semibold/sm" className="truncate text-primary">
            {title}
          </Text>
          {subtitle ? (
            <Text
              kind="body/regular/xs"
              className={`truncate ${subtitleClassName ?? 'text-secondary'}`}
            >
              {subtitle}
            </Text>
          ) : null}
        </Stack>
      </Flex>

      {description ? (
        <Text kind="body/regular/xs" className="w-full text-secondary">
          {description}
        </Text>
      ) : null}
    </Stack>

    {tags && tags.length > 0 ? (
      <Flex wrap="wrap" gap="density-xs" className="w-full">
        {tags.map((tag) => (
          <Badge key={tag} color="blue" kind="solid" className="text-[10px]">
            {tag}
          </Badge>
        ))}
      </Flex>
    ) : null}
  </button>
);
