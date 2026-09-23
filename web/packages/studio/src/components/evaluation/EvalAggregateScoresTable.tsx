// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { Badge, Block, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { formatScore, scoreColor } from '@studio/components/evaluation/utils';
import { type ComponentProps, type FC, useCallback, useMemo } from 'react';

export interface EvalAggregateScoreRow {
  name: string;
  mean?: number | null;
  count?: number;
  nan_count?: number;
  min?: number | null;
  max?: number | null;
  value?: number | null;
  score_type?: string;
  mode_category?: string | null;
  rubric_distribution?: { label: string; value?: number; count?: number }[];
}

interface EvalAggregateScoresTableProps {
  scores: EvalAggregateScoreRow[];
  emptyMessage?: string;
  /** When true, score badges are rendered gray instead of the traffic-light color scale.
   *  Use for runner-contributed metrics whose scale is runner-defined and not comparable to [0, 1]. */
  disableScoreColoring?: boolean;
}

const VIEW_PREFIX = 'view.';

const displayMetricName = (name: string): string => {
  if (name.startsWith(VIEW_PREFIX)) return name.slice(VIEW_PREFIX.length);
  const separator = name.indexOf('.');
  if (separator === -1) return name;
  const type = name.slice(0, separator);
  const output = name.slice(separator + 1);
  return !output || output === type ? type : output;
};

const trialsText = (score: EvalAggregateScoreRow): string => {
  const scored = score.count ?? 0;
  return `${scored}/${scored + (score.nan_count ?? 0)}`;
};

const displayScoreValue = (score: EvalAggregateScoreRow): number | null =>
  score.score_type === 'scalar' ? (score.value ?? null) : (score.mean ?? null);

export const EvalAggregateScoresTable: FC<EvalAggregateScoresTableProps> = ({
  scores,
  emptyMessage = 'No scores recorded for this evaluation.',
  disableScoreColoring = false,
}) => {
  const dataViewState = useStudioDataViewState();
  const hasRubric = scores.some((score) => !!score.rubric_distribution?.length);

  const { pageIndex, pageSize } = dataViewState.pagination.state;
  const pageScores = useMemo(
    () => scores.slice(pageIndex * pageSize, (pageIndex + 1) * pageSize),
    [scores, pageIndex, pageSize]
  );

  const makeColumns = useCallback<
    ComponentProps<typeof StudioDataView<EvalAggregateScoreRow>>['makeColumns']
  >(
    (col) => [
      col.accessor((original) => displayMetricName(original.name), {
        id: 'metric',
        header: 'Metric',
        size: 200,
        enableSorting: false,
        cell: ({ row }) => (
          <Text kind="body/semibold/sm" title={row.original.name}>
            {displayMetricName(row.original.name)}
          </Text>
        ),
      }),
      col.accessor(displayScoreValue, {
        id: 'score',
        header: 'Score',
        enableSorting: false,
        size: 100,
        cell: ({ row }) => (
          <Badge
            kind="solid"
            color={disableScoreColoring ? 'gray' : scoreColor(displayScoreValue(row.original))}
          >
            {formatScore(displayScoreValue(row.original))}
          </Badge>
        ),
      }),
      col.accessor((original) => trialsText(original), {
        id: 'trials',
        header: 'Trials',
        size: 90,
        enableSorting: false,
        cell: ({ row }) => (
          <Text className="text-secondary" kind="body/regular/sm">
            {trialsText(row.original)}
          </Text>
        ),
      }),
      ...(hasRubric
        ? [
            col.display({
              id: 'distribution',
              header: 'Distribution',
              size: 240,
              cell: ({ row }) => {
                const { mode_category: mode, rubric_distribution: distribution } = row.original;
                if (!distribution?.length) {
                  return (
                    <Text className="text-secondary" kind="body/regular/sm">
                      —
                    </Text>
                  );
                }
                return (
                  <Stack gap="density-xs">
                    {mode ? (
                      <Text className="text-secondary" kind="body/regular/sm">
                        Most frequent: {mode}
                      </Text>
                    ) : null}
                    <Flex gap="density-xs" wrap="wrap">
                      {distribution.map((bucket) => (
                        <Badge key={bucket.label} kind="outline" color="gray">
                          {bucket.label}: {bucket.count ?? 0}
                        </Badge>
                      ))}
                    </Flex>
                  </Stack>
                );
              },
            }),
          ]
        : []),
    ],
    [hasRubric, disableScoreColoring]
  );

  if (scores.length === 0) {
    return <Block className="text-subtle">{emptyMessage}</Block>;
  }

  return (
    <Stack className="max-h-[420px]">
      <StudioDataView
        dataViewState={dataViewState}
        makeColumns={makeColumns}
        attributes={{
          DataViewRoot: {
            data: pageScores,
            totalCount: scores.length,
            reactTableOptions: { getRowId: (row) => row.name },
          },
          DataViewPagination: { showWhileLessThanPageSize: false },
        }}
      />
    </Stack>
  );
};
