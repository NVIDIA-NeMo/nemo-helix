// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as DataView from '@nemo/common/src/components/DataView/internal';
import { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { DEFAULT_PAGE_SIZE_OPTIONS } from '@nemo/common/src/constants/pagination';
import { useRowNavigation } from '@nemo/common/src/hooks/useRowNavigation';
import { formatDurationMs } from '@nemo/common/src/utils/date';
import { formatEvaluatorScore } from '@nemo/common/src/utils/formatters';
import type { ExperimentResponse } from '@nemo/sdk/generated/platform/schema';
import { Badge, Flex, Text } from '@nvidia/foundations-react-core';
import {
  deriveEvaluatorMetrics,
  LATENCY_METRIC,
  type ParetoMetric,
  TOKENS_METRIC,
} from '@studio/components/charts/ExperimentParetoChart/utils';
import { MetricSelect } from '@studio/components/charts/MetricSelect';
import { useGroupEvaluations } from '@studio/components/charts/useGroupEvaluations';
import {
  type ColumnExtremes,
  columnExtremes,
  type DeltaTone,
  defaultBaselineName,
  describeDelta,
  DURATION_METRIC,
  scoreDelta,
  scoreHeader,
} from '@studio/components/dataViews/ExperimentComparisonDataView/utils';
import type { EvaluationRow } from '@studio/components/dataViews/ExperimentDataView/useExperimentEvaluations';
import { EM_DASH } from '@studio/components/dataViews/OptimizationJobsDataView/utils';
import { getEvaluationDetailRoute } from '@studio/routes/utils';
import { type ComponentProps, type FC, useCallback, useMemo, useState } from 'react';

interface ExperimentComparisonDataViewProps {
  workspace: string;
  group: ExperimentResponse;
  preloadedEvaluations?: EvaluationRow[];
  preloadPending?: boolean;
}

interface ResourceColumn {
  readonly metric: ParetoMetric;
  readonly header: string;
  readonly format: (value: number) => string;
}

const formatDuration = (ms: number): string => formatDurationMs(ms, { hideMsAboveMinute: true });

const RESOURCE_COLUMNS: readonly ResourceColumn[] = [
  { metric: LATENCY_METRIC, header: 'Avg Latency', format: formatDuration },
  {
    metric: TOKENS_METRIC,
    header: 'Avg Tokens',
    format: (value) => Math.round(value).toLocaleString(),
  },
  { metric: DURATION_METRIC, header: 'Duration', format: formatDuration },
];

const DELTA_TONE_CLASS: Readonly<Record<DeltaTone, string>> = {
  better: 'text-feedback-success',
  worse: 'text-feedback-danger',
  neutral: 'text-secondary',
};

const Placeholder: FC = () => (
  <Text kind="body/regular/sm" className="text-placeholder">
    {EM_DASH}
  </Text>
);

interface MetricValueCellProps {
  value: number | null | undefined;
  extremes: ColumnExtremes | undefined;
  format: (value: number) => string;
}

const MetricValueCell: FC<MetricValueCellProps> = ({ value, extremes, format }) => {
  if (value == null || !Number.isFinite(value)) return <Placeholder />;
  const extreme = value === extremes?.best ? 'best' : value === extremes?.worst ? 'worst' : null;
  if (!extreme) return <Text>{format(value)}</Text>;
  return (
    <Text
      kind="body/bold/md"
      className={extreme === 'best' ? 'text-brand' : DELTA_TONE_CLASS.worse}
    >
      {format(value)}
      <span className="sr-only"> ({extreme})</span>
    </Text>
  );
};

export const ExperimentComparisonDataView: FC<ExperimentComparisonDataViewProps> = ({
  workspace,
  group,
  preloadedEvaluations,
  preloadPending = false,
}) => {
  const openRow = useRowNavigation();
  const hasPreloaded = preloadedEvaluations !== undefined;
  const {
    rows: fetchedRows,
    isLoading: isFetching,
    isError,
  } = useGroupEvaluations(workspace, group.id, { enabled: !hasPreloaded && !preloadPending });
  const rows = hasPreloaded ? preloadedEvaluations : fetchedRows;
  const isLoading = hasPreloaded ? false : preloadPending || isFetching;

  const [pickedBaselineName, setPickedBaselineName] = useState<string>();
  const baseline = useMemo(() => {
    const byName = (name: string | undefined) => rows.find((row) => row.name === name);
    return byName(pickedBaselineName) ?? byName(defaultBaselineName(rows));
  }, [rows, pickedBaselineName]);
  const baselineOptions = useMemo(
    () => rows.map((row) => ({ id: row.name, label: row.name })),
    [rows]
  );

  const scoreMetrics = useMemo(() => deriveEvaluatorMetrics(rows), [rows]);
  const extremesByMetric = useMemo(
    () =>
      new Map(
        [...scoreMetrics, ...RESOURCE_COLUMNS.map((column) => column.metric)].map((metric) => [
          metric.id,
          columnExtremes(rows, metric),
        ])
      ),
    [rows, scoreMetrics]
  );

  const dataViewState = DataView.useDataViewState({
    pagination: { paginationOptions: DEFAULT_PAGE_SIZE_OPTIONS },
  });
  const { pageIndex, pageSize } = dataViewState.pagination.state;
  const pageRows = useMemo(
    () => rows.slice(pageIndex * pageSize, (pageIndex + 1) * pageSize),
    [rows, pageIndex, pageSize]
  );

  const makeColumns = useCallback<
    ComponentProps<typeof StudioDataView<EvaluationRow>>['makeColumns']
  >(
    ({ display }) => [
      display({
        id: 'name',
        header: 'Evaluation',
        size: 220,
        cell: ({ row }) => (
          <Flex align="center" gap="density-sm">
            <Text>{row.original.name}</Text>
            {row.original.name === baseline?.name ? (
              <Badge kind="outline" color="gray">
                Baseline
              </Badge>
            ) : null}
          </Flex>
        ),
      }),
      ...scoreMetrics.flatMap((metric) => [
        display({
          id: `score-${metric.id}`,
          header: scoreHeader(metric.id),
          size: 150,
          cell: ({ row }) => (
            <MetricValueCell
              value={metric.accessor(row.original)}
              extremes={extremesByMetric.get(metric.id)}
              format={formatEvaluatorScore}
            />
          ),
        }),
        display({
          id: `delta-${metric.id}`,
          header: 'Δ vs. Baseline',
          size: 130,
          cell: ({ row }) => {
            const delta = scoreDelta(metric, row.original, baseline);
            if (delta === undefined) return <Placeholder />;
            const { text, tone } = describeDelta(delta, metric.direction);
            return <Text className={DELTA_TONE_CLASS[tone]}>{text}</Text>;
          },
        }),
      ]),
      ...RESOURCE_COLUMNS.map(({ metric, header, format }) =>
        display({
          id: metric.id,
          header,
          size: 120,
          cell: ({ row }) => (
            <MetricValueCell
              value={metric.accessor(row.original)}
              extremes={extremesByMetric.get(metric.id)}
              format={format}
            />
          ),
        })
      ),
    ],
    [scoreMetrics, extremesByMetric, baseline]
  );

  const renderBody = () => {
    if (isError) {
      return <Text kind="body/regular/sm">Could not load the evaluations to compare.</Text>;
    }
    if (!isLoading && rows.length === 0) {
      return (
        <Text kind="body/regular/sm" color="subtle">
          No evaluations to compare.
        </Text>
      );
    }
    return (
      <StudioDataView
        dataViewState={dataViewState}
        makeColumns={makeColumns}
        onRowClick={(row, _index, event) =>
          openRow(event, getEvaluationDetailRoute(workspace, group.name, row.name))
        }
        toolbarSlotStart={
          <MetricSelect
            label="Baseline"
            value={baseline?.name ?? ''}
            metrics={baselineOptions}
            onChange={setPickedBaselineName}
            triggerClassName="w-60"
          />
        }
        attributes={{
          DataViewRoot: {
            data: pageRows,
            totalCount: rows.length,
            requestStatus: isLoading ? 'loading' : undefined,
            reactTableOptions: { getRowId: (row) => row.id },
          },
        }}
      />
    );
  };

  return (
    <div className="flex flex-col gap-3 rounded border border-base bg-surface p-4">
      <Text kind="title/xs">Compare Evaluations</Text>
      {renderBody()}
    </div>
  );
};
