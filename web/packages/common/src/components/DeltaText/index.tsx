// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Text } from '@nvidia/foundations-react-core';
import cn from 'classnames';
import { Equal, Triangle } from 'lucide-react';
import type { ComponentProps, FC } from 'react';

export type DeltaTone = 'improved' | 'regressed' | 'unchanged';

export type DeltaTextSize = 'xs' | 'sm' | 'md' | 'lg' | 'xl';

type TextKind = ComponentProps<typeof Text>['kind'];

export interface DeltaTextProps {
  value: number;
  /** When false, a lower value is the improvement (latency, cost, error rate). */
  higherIsBetter?: boolean;
  format?: (value: number) => string;
  /**
   * One step of the type scale, carrying the glyph with it. `xs` is the compact treatment the
   * design uses inside cards and table cells; step up when the delta sits beside a larger value.
   */
  size?: DeltaTextSize;
  /**
   * Escape hatch for a caller that needs a different weight or family than the semibold `size`
   * picks — e.g. `label/regular/sm`. Wins over `size`, and the glyph still tracks it.
   */
  kind?: TextKind;
  'aria-label'?: string;
  className?: string;
}

const TONE_CLASS_NAME: Record<DeltaTone, string> = {
  improved: 'text-[color:var(--text-color-brand)]',
  regressed: 'text-[color:var(--text-color-accent-red)]',
  unchanged: 'text-secondary',
};

/**
 * Spread across the full type scale — 10, 14, 18, 24, 32px — rather than the adjacent steps, which
 * bunch between 10 and 18 and leave nothing that reads as large next to a headline figure. Each
 * step is roughly a third up on the one below it, so the difference is visible at a glance.
 *
 * The glyph is sized in `em`, so picking the step here is the whole job — the triangle follows the
 * text rather than needing its own scale.
 */
const SIZE_KIND: Record<DeltaTextSize, TextKind> = {
  xs: 'body/semibold/xs',
  sm: 'body/semibold/md',
  md: 'body/semibold/xl',
  lg: 'body/semibold/2xl',
  xl: 'body/semibold/3xl',
};

/**
 * Signs the value and keeps the minus a true minus (U+2212) rather than a hyphen, so a column of
 * deltas lines up against `tabular-nums`.
 */
export const formatSignedDelta = (value: number, fractionDigits = 2): string =>
  `${value > 0 ? '+' : value < 0 ? '−' : ''}${Math.abs(value).toFixed(fractionDigits)}`;

/** Which direction of travel is good news, and so which way the delta is tinted. */
export const deltaTone = (value: number, higherIsBetter = true): DeltaTone => {
  if (value === 0) {
    return 'unchanged';
  }
  return (higherIsBetter ? value > 0 : value < 0) ? 'improved' : 'regressed';
};

/**
 * A signed metric change as bare tinted text with a triangle: green for an improvement, red for a
 * regression.
 */
export const DeltaText: FC<DeltaTextProps> = ({
  value,
  higherIsBetter = true,
  format = formatSignedDelta,
  size = 'xs',
  kind,
  'aria-label': ariaLabel,
  className,
}) => {
  const tone = deltaTone(value, higherIsBetter);
  const Icon = value === 0 ? Equal : Triangle;
  const magnitude = format(Math.abs(value)).replace(/^[+−-]/, '');
  const label =
    ariaLabel ??
    (tone === 'unchanged'
      ? 'No change'
      : `${tone === 'improved' ? 'Improved' : 'Regressed'} by ${magnitude}`);

  return (
    <Text
      kind={kind ?? SIZE_KIND[size]}
      aria-label={label}
      data-testid="delta-text"
      data-delta={tone}
      className={cn(
        'inline-flex items-center gap-[0.25em] whitespace-nowrap tabular-nums',
        TONE_CLASS_NAME[tone],
        className
      )}
    >
      <Icon
        aria-hidden
        data-testid="delta-text-icon"
        size="0.72em"
        strokeWidth={1}
        className={cn(
          'shrink-0 stroke-current',
          tone === 'unchanged' ? 'stroke-2' : 'fill-current',
          value < 0 && 'rotate-180'
        )}
      />
      {format(value)}
    </Text>
  );
};
