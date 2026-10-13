// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Card, Flex, Panel, Stack, Text } from '@nvidia/foundations-react-core';
import cn from 'classnames';
import { ChevronRight } from 'lucide-react';
import type { ComponentProps, FC, ReactNode } from 'react';
import { Link } from 'react-router';

export type StatTileStatus = 'success' | 'warning' | 'error' | 'neutral';

/**
 * `default` is the diagnostics tile: compact label, hint line, capped width.
 * `metric` is the overview tile: larger label, the trailing label reads as a
 * unit sitting on the value's baseline, and the tile fills its grid column.
 * `actionable` is a clickable card: icon + label + chevron header that links
 * elsewhere, with the value rendered below.
 */
export type StatTileVariant = 'default' | 'metric' | 'actionable';

interface StatTileBaseProps {
  label: string;
  value: string;
  className?: string;
  /**
   * The tile's own health, independent of `trailingLabelStatus`/`hintStatus`: only this tints the
   * border, so a value's delta direction (e.g. loss falling) never reads as a threshold breach.
   * `success`/`neutral` keep the default border — only `warning`/`error` stand out.
   */
  status?: StatTileStatus;
}

export interface StatTileDiagnosticProps extends StatTileBaseProps {
  variant?: 'default' | 'metric';
  trailingLabel?: string;
  trailingLabelStatus?: StatTileStatus;
  hint?: string;
  hintStatus?: StatTileStatus;
  bordered?: boolean;
}

type StatTileActionableTarget =
  | { to: string; onClick?: () => void }
  | { to?: never; onClick: () => void };

export type StatTileActionableProps = StatTileBaseProps &
  StatTileActionableTarget & {
    variant: 'actionable';
    /** Header icon shown before the label. */
    icon?: ReactNode;
  };

export type StatTileProps = StatTileDiagnosticProps | StatTileActionableProps;

const MUTED_CLASS_NAME = 'text-placeholder';
const STAT_TILE_SURFACE_TEST_ID = 'stat-tile-surface';
const ACTIONABLE_TARGET_CLASS_NAME = 'group flex w-full flex-col gap-density-md text-left';

const STATUS_CLASS_NAME: Record<StatTileStatus, string> = {
  success: 'text-[color:var(--text-color-feedback-success)]',
  warning: 'text-[color:var(--text-color-feedback-warning)]',
  error: 'text-[color:var(--text-color-feedback-danger)]',
  neutral: MUTED_CLASS_NAME,
};

const BORDER_STATUS_CLASS_NAME: Partial<Record<StatTileStatus, string>> = {
  warning: 'border-(--border-color-feedback-warning)',
  error: 'border-(--border-color-feedback-danger)',
};

const GAP_BY_VARIANT: Record<StatTileVariant, ComponentProps<typeof Stack>['gap']> = {
  default: 'density-sm',
  metric: 'density-xxs',
  actionable: 'density-md',
};

export const StatTile: FC<StatTileProps> = (props) => {
  const { label, value, className, status } = props;

  if (props.variant === 'actionable') {
    const { icon, to, onClick } = props;

    const content = (
      <>
        <Flex align="center" gap="density-sm" className="w-full">
          {icon}
          <Text kind="label/semibold/sm" className="flex-1 text-left">
            {label}
          </Text>
          <ChevronRight
            width={16}
            height={16}
            className="shrink-0 text-placeholder transition-colors group-hover:text-primary group-focus-visible:text-primary"
          />
        </Flex>
        <Text kind="body/regular/2xl" className="tabular-nums">
          {value}
        </Text>
      </>
    );

    return (
      <Card
        asChild
        interactive
        className={cn('h-fit', status && BORDER_STATUS_CLASS_NAME[status], className)}
        data-testid={STAT_TILE_SURFACE_TEST_ID}
      >
        {to ? (
          <Link to={to} onClick={onClick} className={ACTIONABLE_TARGET_CLASS_NAME}>
            {content}
          </Link>
        ) : (
          <button type="button" onClick={onClick} className={ACTIONABLE_TARGET_CLASS_NAME}>
            {content}
          </button>
        )}
      </Card>
    );
  }

  const {
    variant = 'default',
    trailingLabel,
    trailingLabelStatus,
    hint,
    hintStatus,
    bordered = true,
  } = props;
  const isMetric = variant === 'metric';
  const neutralClassName = isMetric ? 'text-secondary' : MUTED_CLASS_NAME;
  const statusClassName = (status: StatTileStatus | undefined) =>
    status && status !== 'neutral' ? STATUS_CLASS_NAME[status] : neutralClassName;

  const content = (
    <Stack gap={GAP_BY_VARIANT[variant]}>
      <Text kind={isMetric ? 'body/regular/md' : 'body/regular/sm'} className={neutralClassName}>
        {label}
      </Text>
      <Flex
        align={isMetric ? 'end' : 'baseline'}
        gap={isMetric ? 'density-md' : 'density-sm'}
        wrap="wrap"
      >
        <Text kind="label/bold/2xl" className="tabular-nums">
          {value}
        </Text>
        {trailingLabel ? (
          <Text
            kind="body/regular/sm"
            className={cn(statusClassName(trailingLabelStatus), isMetric && 'pb-density-xs')}
          >
            {trailingLabel}
          </Text>
        ) : null}
      </Flex>
      {hint ? (
        <Text kind="body/regular/sm" className={statusClassName(hintStatus)}>
          {hint}
        </Text>
      ) : null}
    </Stack>
  );

  if (!bordered) {
    return content;
  }

  return (
    <Panel
      className={cn(
        isMetric ? 'w-full' : 'max-w-sm',
        status && BORDER_STATUS_CLASS_NAME[status],
        className
      )}
      elevation="high"
      data-testid={STAT_TILE_SURFACE_TEST_ID}
    >
      {content}
    </Panel>
  );
};
