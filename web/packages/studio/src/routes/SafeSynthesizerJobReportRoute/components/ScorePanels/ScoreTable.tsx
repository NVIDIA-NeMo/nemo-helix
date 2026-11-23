// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ScoreGauge } from '@nemo/common/src/components/ScoreGauge';
import { Stack, Table, Text } from '@nvidia/foundations-react-core';
import type { FC } from 'react';

interface ScoreTableProps {
  scores: {
    name: string;
    value?: number | null;
    description: string;
  }[];
}

export const ScoreTable: FC<ScoreTableProps> = ({ scores }) => {
  return (
    <Table
      data-testid="score-table"
      className="bg-transparent w-full"
      layout="fixed"
      align="left"
      columns={[
        { children: 'Metric' },
        {
          children: 'Score',
          attributes: {
            TableHeaderCell: {
              style: { textAlign: 'right', paddingRight: '20px', width: '90px' },
            },
          },
        },
      ]}
      rows={scores.map((score) => ({
        cells: [
          {
            children: (
              <Stack gap="density-xs">
                <Text kind="body/semibold/md">{score.name}</Text>
                <Text kind="body/regular/sm" className="text-secondary">
                  {score.description}
                </Text>
              </Stack>
            ),
          },
          {
            children: <ScoreGauge score={score.value ?? undefined} size="sm" />,
            attributes: {
              TableDataCell: {
                style: { width: '0px' },
              },
            },
          },
        ],
      }))}
    />
  );
};
