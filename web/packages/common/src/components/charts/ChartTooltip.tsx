// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ChartSwatch } from '@nemo/common/src/components/charts/ChartSwatch';
import { Flex, Stack, Text } from '@nvidia/foundations-react-core';
import type { CSSProperties, FC, ReactNode } from 'react';

const TEXT_STYLE: CSSProperties = { color: 'var(--text-color-base)' };

/**
 * Spread onto a recharts `<Tooltip>` that uses the default content so it matches
 * `ChartTooltipSurface`. Theme-aware tokens keep contrast in light and dark mode.
 */
export const chartTooltipProps = {
  contentStyle: {
    fontSize: 12,
    backgroundColor: 'var(--background-color-surface-overlay)',
    border: '1px solid var(--border-color-base)',
    borderRadius: 8,
    boxShadow: 'var(--shadow-sm, 0 1px 2px rgb(0 0 0 / 0.1))',
    ...TEXT_STYLE,
  },
  labelStyle: TEXT_STYLE,
  itemStyle: TEXT_STYLE,
} as const;

/**
 * The shared hover-card surface. Each chart still builds its own rows — going from a recharts
 * payload to a list of values depends on how that chart keys its data.
 */
export const ChartTooltipSurface: FC<{ label?: ReactNode; children?: ReactNode }> = ({
  label,
  children,
}) => (
  <Stack gap="1" className="bg-surface-overlay border border-base shadow-sm rounded-lg p-3">
    {label !== undefined && <Text kind="label/semibold/md">{label}</Text>}
    {children}
  </Stack>
);

/** One series' swatch, name, and value. Omit `value` to render the name alone. */
export const ChartTooltipRow: FC<{
  color?: string;
  dashed?: boolean;
  label: ReactNode;
  value?: ReactNode;
}> = ({ color, dashed, label, value }) => (
  <Flex align="center" gap="2">
    <ChartSwatch color={color} dashed={dashed} />
    <Text kind="body/regular/sm" className="text-placeholder">
      {label}
    </Text>
    {value !== undefined && <Text kind="body/semibold/sm">{value}</Text>}
  </Flex>
);
