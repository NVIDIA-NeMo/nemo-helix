// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { Badge, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { ReportTable } from '@studio/routes/SafeSynthesizerJobReportRoute/components/ReportTable';
import { GradeLabel } from '@studio/routes/SafeSynthesizerJobReportRoute/components/ScorePanels/GradeBadge';
import { TRAINING_COLUMNS_DESCRIPTION } from '@studio/routes/SafeSynthesizerJobReportRoute/reportCopy';
import type { ColumnResult } from '@studio/routes/SafeSynthesizerJobReportRoute/reportDataTypes';
import { type ComponentProps, type FC, useCallback } from 'react';

const PAGE_SIZE = 10;

const capitalize = (value: string) => value.charAt(0).toUpperCase() + value.slice(1);

const getRowId = (column: ColumnResult) => column.name;

interface TrainingColumnsTabProps {
  columns: ColumnResult[];
  withDistribution: boolean;
  withTransform: boolean;
}

export const TrainingColumnsTab: FC<TrainingColumnsTabProps> = ({
  columns,
  withDistribution,
  withTransform,
}) => {
  const makeColumns = useCallback<
    ComponentProps<typeof StudioDataView<ColumnResult>>['makeColumns']
  >(
    (col) => [
      col.accessor('name', {
        header: 'Column Name',
        cell: ({ getValue }) => <Text kind="body/semibold/sm">{getValue()}</Text>,
      }),
      col.accessor((column) => column.training.unique_count, {
        id: 'unique',
        header: 'Unique',
        cell: ({ getValue }) => getValue().toLocaleString(),
      }),
      col.accessor((column) => column.training.missing_count, {
        id: 'missing',
        header: 'Missing',
        cell: ({ getValue }) => getValue().toLocaleString(),
      }),
      col.accessor((column) => column.training.avg_str_length, {
        id: 'avgLength',
        header: 'Avg. Length',
        cell: ({ getValue }) => Number(getValue().toFixed(2)).toString(),
      }),
      col.accessor((column) => column.training.type, {
        id: 'type',
        header: 'Type',
        cell: ({ getValue }) => (
          <Badge kind="outline" color="gray">
            {capitalize(getValue())}
          </Badge>
        ),
      }),
      ...(withDistribution
        ? [
            col.accessor((column) => column.distribution_stability?.grade ?? null, {
              id: 'distribution',
              header: 'Distribution',
              cell: ({ getValue }) => {
                const grade = getValue();
                if (grade === null) {
                  return 'N/A';
                }
                return <GradeLabel grade={grade} />;
              },
            }),
          ]
        : []),
      ...(withTransform
        ? [
            col.display({
              id: 'entities',
              header: 'Entities (Count)',
              cell: ({ row }) => (
                <Flex gap="density-xs" wrap="wrap">
                  {Object.entries(row.original.pii?.detected_entity_counts ?? {}).map(
                    ([entity, count]) => (
                      <Badge key={entity} kind="outline" color="gray">
                        {`${entity} (${count.toLocaleString()})`}
                      </Badge>
                    )
                  )}
                </Flex>
              ),
            }),
            col.accessor((column) => column.pii?.is_transformed ?? false, {
              id: 'transformed',
              header: 'Transformed',
              cell: ({ getValue }) => (getValue() ? 'True' : 'False'),
            }),
            col.accessor((column) => column.pii?.transform_functions.join(', ') ?? '', {
              id: 'transformFunctions',
              header: 'Transform function',
            }),
          ]
        : []),
    ],
    [withDistribution, withTransform]
  );

  return (
    <Stack gap="density-lg" data-testid="training-columns-tab">
      <Stack gap="density-xs">
        <Flex align="center" gap="density-md">
          <Text kind="body/bold/lg">Training Data</Text>
          <Badge kind="outline" color="gray">
            {`${columns.length} columns`}
          </Badge>
        </Flex>
        <Text kind="body/regular/sm" className="text-secondary">
          {TRAINING_COLUMNS_DESCRIPTION}
        </Text>
      </Stack>
      <ReportTable
        rows={columns}
        makeColumns={makeColumns}
        getRowId={getRowId}
        pageSize={PAGE_SIZE}
      />
    </Stack>
  );
};
