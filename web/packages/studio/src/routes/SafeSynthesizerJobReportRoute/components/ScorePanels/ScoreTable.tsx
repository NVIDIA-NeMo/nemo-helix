// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ScoreGauge } from '@nemo/common/src/components/ScoreGauge';
import { InfoCircle } from '@nemo/icons/icons';
import { Flex, Table, Tooltip } from '@nvidia/foundations-react-core';
import { tooltipClassName } from '@studio/styles/common';
import { FC } from 'react';

interface ScoreTableProps {
  scores: {
    name: string;
    value: number;
    displayValue?: string;
    description?: string;
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
              <Flex gap="density-sm" align="center" className="overflow-visible">
                <span className="truncate">{score.name}</span>
                {score.description && (
                  <Tooltip
                    slotContent={score.description}
                    side="bottom"
                    className={tooltipClassName}
                  >
                    <InfoCircle className="inline shrink-0" />
                  </Tooltip>
                )}
              </Flex>
            ),
            attributes: {
              TableDataCell: {
                className: 'overflow-visible',
              },
            },
          },
          {
            children: <ScoreGauge score={score.value} size="sm" />,
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
