// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Flex, Select, Stack, Text } from '@nvidia/foundations-react-core';
import { InfoTip } from '@studio/routes/SafeSynthesizerJobReportRoute/components/InfoTip';
import type { FC, ReactNode } from 'react';

interface ChartFigureProps {
  title: string;
  tooltip?: string;
  slotEnd?: ReactNode;
  children: ReactNode;
}

export const ChartFigure: FC<ChartFigureProps> = ({ title, tooltip, slotEnd, children }) => (
  <Stack gap="density-sm" data-testid="chart-figure">
    <Flex justify="between" align="center" gap="density-md" wrap="wrap">
      <Flex align="center" gap="density-sm">
        <Text kind="label/bold/md">{title}</Text>
        {tooltip ? <InfoTip label={title} text={tooltip} /> : null}
      </Flex>
      {slotEnd}
    </Flex>
    {children}
  </Stack>
);

interface ColumnSelectProps {
  columns: readonly string[];
  value: string;
  onChange: (column: string) => void;
}

/** Picks the numeric column a time-series chart shows; hidden when there is only one. */
export const ColumnSelect: FC<ColumnSelectProps> = ({ columns, value, onChange }) =>
  columns.length > 1 ? (
    <Select
      aria-label="Numeric column"
      className="w-[200px] grow-0"
      items={columns.map((column) => ({ value: column, children: column }))}
      value={value}
      onValueChange={(next) => onChange(String(next))}
    />
  ) : null;
