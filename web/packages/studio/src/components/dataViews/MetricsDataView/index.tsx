// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { type FilterTree, jsonFilter } from '@nemo/common/src/api/filterOperators';
import { dateTimeFilter } from '@nemo/common/src/components/DataView/dateTimeFilter';
import {
  ROW_ACTIONS_COLUMN_SIZE,
  StudioDataView,
} from '@nemo/common/src/components/DataView/StudioDataView';
import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
import { DeltaText, formatSignedDelta } from '@nemo/common/src/components/DeltaText';
import { EntityEmptyState } from '@nemo/common/src/components/EntityEmptyState';
import { ErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { QuickActionsMenuRoot } from '@nemo/common/src/components/QuickActionsMenu/QuickActionsMenuRoot';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { useRowNavigation } from '@nemo/common/src/hooks/useRowNavigation';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { formatEvaluatorScore } from '@nemo/common/src/utils/formatters';
import { getSortParamWithWhitelist } from '@nemo/common/src/utils/query';
import {
  getEvalsListMetricsQueryKey,
  useEvalsDeleteMetric,
  useEvalsListMetrics,
} from '@nemo/sdk/generated/evals/evals-plugin-metrics-routes';
import { type Metric, type MetricFilter, MetricSort } from '@nemo/sdk/generated/evals/schema';
import { Flex, Stack, Tag, Text } from '@nvidia/foundations-react-core';
import {
  metricOutputNames,
  toMetricRuns,
  toMetricTrendSeries,
} from '@studio/api/metrics/metricScores';
import { useMetricScoreHistory } from '@studio/api/metrics/useMetricScoreHistory';
import { MetricSparkline } from '@studio/components/charts/MetricSparkline';
import { SELECTABLE_METRICS } from '@studio/components/evaluation/EvalWizard/types';
import { getMetricDetailRoute } from '@studio/routes/utils';
import { keepPreviousData, useQueryClient } from '@tanstack/react-query';
import { type ComponentProps, type FC, useMemo, useState } from 'react';

const SORTABLE_FIELDS = Object.values(MetricSort).filter((v) => !v.startsWith('-'));

const TYPE_FILTER = 'metric_type';
const TYPE_OPTIONS = SELECTABLE_METRICS.map(({ type, label }) => ({ value: type, label }));

export interface MetricsDataViewProps {
  workspace: string;
  onCreate?: () => void;
}

export const MetricsDataView: FC<MetricsDataViewProps> = ({ workspace, onCreate }) => {
  const openRow = useRowNavigation();
  const queryClient = useQueryClient();
  const [pendingDelete, setPendingDelete] = useState<{ workspace: string; metric: Metric }>();
  const { mutateAsync: deleteMetric } = useEvalsDeleteMetric();

  const dataViewState = useStudioDataViewState<MetricFilter>({
    defaultSort: [{ id: 'updated_at', desc: true }],
    columnVisibility: { created_at: false },
  });
  const { searchText } = dataViewState.apiFilter;
  const selectedTypes = (dataViewState.apiFilter.filter?.metric_type ?? []) as unknown as string[];
  const { created_at, updated_at } = dataViewState.apiFilter.filter ?? {};
  const clauses: FilterTree<MetricFilter>[] = [
    ...(selectedTypes.length ? [{ metric_type: { $in: selectedTypes } }] : []),
    ...(created_at ? [{ created_at }] : []),
    ...(updated_at ? [{ updated_at }] : []),
    // Search reads both, so a metric is findable by what it measures, not only its name.
    ...(searchText
      ? [{ $or: [{ name: { $like: searchText } }, { description: { $like: searchText } }] }]
      : []),
  ];

  /** Undebounced, so a clicked tag highlights at once. */
  const appliedTypes =
    (dataViewState.columnFiltering.state.find((f) => f.id === TYPE_FILTER)?.value as
      | string[]
      | undefined) ?? [];

  /** Adds the type to the filter, or removes it when already applied. */
  const toggleType = (type: string) =>
    dataViewState.columnFiltering.set((prev) => {
      const current = (prev.find((f) => f.id === TYPE_FILTER)?.value as string[] | undefined) ?? [];
      const next = current.includes(type) ? current.filter((t) => t !== type) : [...current, type];
      const rest = prev.filter((f) => f.id !== TYPE_FILTER);
      return next.length ? [...rest, { id: TYPE_FILTER, value: next }] : rest;
    });

  const { data, isFetching, error } = useEvalsListMetrics(
    workspace,
    {
      page: dataViewState.pagination.state.pageIndex + 1,
      page_size: dataViewState.pagination.state.pageSize,
      sort: getSortParamWithWhitelist(
        dataViewState.sorting.state,
        SORTABLE_FIELDS,
        MetricSort['-updated_at']
      ),
      filter: clauses.length ? jsonFilter<MetricFilter>({ $and: clauses }) : undefined,
    },
    { query: { placeholderData: keepPreviousData } }
  );
  const metrics = useMemo(() => data?.data ?? [], [data?.data]);

  const history = useMetricScoreHistory(workspace);
  // A failed request would leave partial history that reads as complete, so show no scores instead.
  const scoredResults = useMemo(
    () => (history.error ? [] : history.results),
    [history.error, history.results]
  );
  const primaryTrend = useMemo(
    () =>
      new Map(
        metrics.map((metric) => [
          metric.id,
          toMetricTrendSeries(metric, toMetricRuns(metric, scoredResults))[0],
        ])
      ),
    [metrics, scoredResults]
  );

  const fetchingStatus = isFetching ? 'loading' : undefined;
  const requestStatus = error ? 'error' : fetchingStatus;

  const handleDelete = async () => {
    if (!pendingDelete) return false;
    try {
      await deleteMetric({ workspace: pendingDelete.workspace, name: pendingDelete.metric.name });
    } catch {
      return false;
    }
    await queryClient.invalidateQueries({
      queryKey: getEvalsListMetricsQueryKey(pendingDelete.workspace),
    });
    return true;
  };

  const makeColumns: ComponentProps<typeof StudioDataView<Metric>>['makeColumns'] = (
    { accessor },
    { rowActionsColumn }
  ) => [
    accessor('name', { header: 'Name', enableSorting: true, size: 200 }),
    accessor('metric_type', {
      header: 'Type',
      enableSorting: false,
      size: 160,
      meta: { filter: { type: 'multi-select', label: 'Type', options: TYPE_OPTIONS } },
      cell: ({ row }) => {
        const type = row.original.metric_type;
        const selected = appliedTypes.includes(type);
        return (
          <Tag
            density="compact"
            color="gray"
            selected={selected}
            title={selected ? `Remove ${type} from the filter` : `Show only ${type} metrics`}
            onClick={(event) => {
              event.stopPropagation();
              toggleType(type);
            }}
          >
            {type}
          </Tag>
        );
      },
    }),
    accessor((metric) => metricOutputNames(metric).join(', '), {
      id: 'outputs',
      header: 'Outputs',
      enableSorting: false,
      cell: ({ getValue }) => <Text className="truncate">{getValue() || '-'}</Text>,
    }),
    accessor((metric) => primaryTrend.get(metric.id)?.value, {
      id: 'latest',
      header: 'Latest score',
      enableSorting: false,
      size: 160,
      cell: ({ row }) => {
        const trend = primaryTrend.get(row.original.id);
        if (!trend) return <Text className="text-secondary">-</Text>;
        return (
          <Flex align="baseline" gap="density-sm">
            <Text kind="label/semibold/md">{formatEvaluatorScore(trend.value)}</Text>
            {trend.delta !== undefined && (
              <DeltaText value={trend.delta} size="xs" format={(d) => formatSignedDelta(d, 3)} />
            )}
          </Flex>
        );
      },
    }),
    accessor(() => '', {
      id: 'trend',
      header: 'Trend',
      enableSorting: false,
      size: 150,
      cell: ({ row }) => (
        <MetricSparkline points={primaryTrend.get(row.original.id)?.points ?? []} />
      ),
    }),
    accessor('created_at', {
      header: 'Created',
      enableSorting: true,
      size: 150,
      meta: { filter: dateTimeFilter('Created') },
      cell: ({ row }) => <RelativeTime datetime={row.original.created_at} />,
    }),
    accessor('updated_at', {
      header: 'Updated',
      enableSorting: true,
      size: 150,
      meta: { filter: dateTimeFilter('Updated') },
      cell: ({ row }) => <RelativeTime datetime={row.original.updated_at} />,
    }),
    rowActionsColumn({
      size: ROW_ACTIONS_COLUMN_SIZE,
      enableResizing: false,
      cell: ({ row }) => (
        <QuickActionsMenuRoot
          actions={[]}
          deleteAction={{ onSelect: () => setPendingDelete({ workspace, metric: row.original }) }}
        />
      ),
    }),
  ];

  return (
    <Stack gap="density-2xl" className="min-h-0 flex-1">
      <StudioDataView<Metric>
        dataViewState={dataViewState}
        searchField="name"
        makeColumns={makeColumns}
        onRowClick={(metric, _index, event) =>
          openRow(event, getMetricDetailRoute(workspace, metric.name))
        }
        attributes={{
          DataViewSearchBar: { placeholder: 'Search names and descriptions...' },
          DataViewRoot: {
            data: metrics,
            totalCount: data?.pagination?.total_results,
            requestStatus,
          },
          DataViewTableContent: {
            renderEmptyState: ({ hasFiltersApplied, hasSearchApplied }) =>
              hasFiltersApplied || hasSearchApplied ? (
                <EntityEmptyState
                  entity="metrics"
                  variant="no-results"
                  onClearFilters={dataViewState.resetFilters}
                />
              ) : (
                <EntityEmptyState entity="metrics" variant="first-use" onCreate={onCreate} />
              ),
            renderErrorState: () => (
              <ErrorPanel
                errorMessage={getErrorMessage(error ?? new Error('Failed to fetch metrics'))}
              />
            ),
          },
        }}
      />
      {pendingDelete && (
        <DeleteConfirmationModal
          open
          simpleConfirm
          onDelete={handleDelete}
          title={`Delete: ${pendingDelete.metric.name}`}
          description="Evaluations that already ran keep their scores. New runs can no longer reference this metric."
          onClose={() => setPendingDelete(undefined)}
        />
      )}
    </Stack>
  );
};
