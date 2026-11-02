// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  resolveStatusConfig,
  type StatusBadgeProps,
  type StatusConfigEntry,
} from '@nemo/common/src/components/StatusBadge';
import { StatusIndicator, type StatusIndicatorProps } from '@nvidia/foundations-react-core';

type DotColor = Pick<StatusIndicatorProps, 'color' | 'className'>;

// StatusIndicator only ships feedback colors; the rest override its `--color` with accent tokens.
const DOT_COLOR: Record<NonNullable<StatusConfigEntry['color']>, DotColor> = {
  green: { color: 'green' },
  red: { color: 'red' },
  yellow: { color: 'yellow' },
  blue: { color: 'blue' },
  teal: { className: '[--color:var(--text-color-accent-teal)]' },
  purple: { className: '[--color:var(--text-color-accent-purple)]' },
  gray: { className: '[--color:var(--text-color-accent-gray)]' },
};

interface StatusDotProps<T = string> extends StatusBadgeProps<T> {
  size?: StatusIndicatorProps['size'];
  className?: string;
}

/** Compact StatusBadge: a KUI StatusIndicator colored by the same status mapping. */
export const StatusDot = <T extends string = string>({
  status,
  statusConfig,
  fallback,
  label: labelOverride,
  size = 'small',
  className,
}: StatusDotProps<T>) => {
  const config = resolveStatusConfig(status, statusConfig, fallback);
  const label = labelOverride ?? config.label;
  const dot = DOT_COLOR[config.color ?? 'gray'];

  return (
    <StatusIndicator
      role="img"
      aria-label={label}
      title={label}
      size={size}
      color={dot.color}
      className={`shrink-0 ${dot.className ?? ''} ${className ?? ''}`.trim()}
    />
  );
};
