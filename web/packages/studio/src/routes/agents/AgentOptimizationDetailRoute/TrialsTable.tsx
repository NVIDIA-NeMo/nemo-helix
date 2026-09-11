// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { StatusBadge, type StatusConfigEntry } from '@nemo/common/src/components/StatusBadge';
import { TableEmptyState } from '@nemo/common/src/components/TableEmptyState';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { formatDurationMs } from '@nemo/common/src/utils/date';
import { Badge, Button, Text } from '@nvidia/foundations-react-core';
import type {
  StudyResults,
  Trial,
} from '@studio/routes/agents/AgentOptimizationDetailRoute/studyResults';
import { Ban, CircleCheck, CircleX, RefreshCw } from 'lucide-react';
import { type ComponentProps, type FC, useMemo } from 'react';

const EM_DASH = '—';

/** Sort id prefix for the one-column-per-objective metric columns. */
const METRIC_SORT_PREFIX = 'metric:';

/** Sort id prefix for the one-column-per-parameter columns. */
const PARAM_SORT_PREFIX = 'param:';

const formatMetric = (value: number | null): string =>
  value === null ? EM_DASH : value.toLocaleString(undefined, { maximumFractionDigits: 4 });

/** `COMPLETE` → `Complete`, matching the sentence-case status text in the design. */
const formatState = (state: string): string =>
  state ? state.charAt(0) + state.slice(1).toLowerCase() : EM_DASH;

/**
 * Optuna's `TrialState` names, mapped onto the shared badge vocabulary.
 */
const TRIAL_STATUS_CONFIG: Record<string, StatusConfigEntry> = {
  COMPLETE: { label: 'Complete', color: 'green', icon: CircleCheck },
  FAIL: { label: 'Failed', color: 'red', icon: CircleX },
  PRUNED: { label: 'Pruned', color: 'yellow', icon: Ban },
  RUNNING: { label: 'Running', color: 'blue', icon: RefreshCw },
  WAITING: { label: 'Waiting', color: 'gray', icon: RefreshCw },
};

const formatParamValue = (value: string): string => {
  const parsed = Number(value);
  if (value.trim() === '' || Number.isNaN(parsed) || Number.isInteger(parsed)) return value;
  return String(Number(parsed.toFixed(3)));
};

const paramsText = (trial: Trial): string =>
  trial.params.map((param) => `${param.name} ${formatParamValue(param.value)}`).join('  ·  ');

const paramValue = (trial: Trial, name: string): string | undefined =>
  trial.params.find((param) => param.name === name)?.value;

/**
 * Every parameter the study tuned, in the order the trials list them (which follows the
 * `params_*` column order of the source CSV). A trial omits parameters it did not record,
 * so the union across trials is what defines the column set.
 */
const collectParamNames = (trials: Trial[]): string[] => {
  const names = new Set<string>();
  for (const trial of trials) {
    for (const param of trial.params) names.add(param.name);
  }
  return [...names];
};

/**
 * Parameters whose every recorded value parses as a number, so the column can sort numerically
 * instead of lexicographically (`10` after `9`, not before it).
 */
const collectNumericParams = (trials: Trial[], names: string[]): Set<string> =>
  new Set(
    names.filter((name) =>
      trials.every((trial) => {
        const value = paramValue(trial, name);
        return value === undefined || value === '' || Number.isFinite(Number(value));
      })
    )
  );

/** What the search bar matches against: the trial id plus every parameter name and value. */
const searchableText = (trial: Trial): string =>
  `trial ${trial.number} ${paramsText(trial)}`.toLowerCase();

/** Missing values sort last in ascending order (and first when the sort is inverted). */
const compareNullable = (a: number | string | null, b: number | string | null): number => {
  if (a === null && b === null) return 0;
  if (a === null) return 1;
  if (b === null) return -1;
  if (typeof a === 'string' || typeof b === 'string') return String(a).localeCompare(String(b));
  return a - b;
};

const sortValue = (
  trial: Trial,
  sortId: string,
  numericParams: ReadonlySet<string>
): number | string | null => {
  if (sortId === 'number') return trial.number;
  if (sortId === 'duration') return trial.durationSeconds;
  if (sortId === 'frontier') return trial.paretoOptimal ? 1 : 0;
  if (sortId === 'state') return trial.state || null;
  if (sortId.startsWith(METRIC_SORT_PREFIX)) {
    const name = sortId.slice(METRIC_SORT_PREFIX.length);
    return trial.metrics.find((metric) => metric.name === name)?.value ?? null;
  }
  if (sortId.startsWith(PARAM_SORT_PREFIX)) {
    const name = sortId.slice(PARAM_SORT_PREFIX.length);
    const value = paramValue(trial, name);
    if (value === undefined || value === '') return null;
    return numericParams.has(name) ? Number(value) : value;
  }
  return null;
};

export interface TrialsTableProps {
  results: StudyResults;
}

export const TrialsTable: FC<TrialsTableProps> = ({ results }) => {
  const { trials, metricNames } = results;
  const primaryMetric = metricNames[0];

  const paramNames = useMemo(() => collectParamNames(trials), [trials]);
  const numericParams = useMemo(
    () => collectNumericParams(trials, paramNames),
    [trials, paramNames]
  );

  const dataViewState = useStudioDataViewState({
    defaultPageSize: 25,
    defaultSort: [
      primaryMetric
        ? { id: `${METRIC_SORT_PREFIX}${primaryMetric}`, desc: true }
        : { id: 'number', desc: false },
    ],
  });

  const { debouncedSearchBar } = dataViewState;
  const sorting = dataViewState.sorting.state;
  const { pageIndex, pageSize } = dataViewState.pagination.state;

  const processedTrials = useMemo(() => {
    const search = debouncedSearchBar.trim().toLowerCase();
    const filtered = search
      ? trials.filter((trial) => searchableText(trial).includes(search))
      : trials;

    const [sort] = sorting;
    if (!sort) return filtered;
    return [...filtered].sort((a, b) => {
      const order = compareNullable(
        sortValue(a, sort.id, numericParams),
        sortValue(b, sort.id, numericParams)
      );
      return sort.desc ? -order : order;
    });
  }, [trials, debouncedSearchBar, sorting, numericParams]);

  const lastPageIndex = Math.max(0, Math.ceil(processedTrials.length / pageSize) - 1);
  const safePageIndex = Math.min(pageIndex, lastPageIndex);
  const pageTrials = useMemo(
    () => processedTrials.slice(safePageIndex * pageSize, safePageIndex * pageSize + pageSize),
    [processedTrials, safePageIndex, pageSize]
  );

  const makeColumns: ComponentProps<typeof StudioDataView<Trial>>['makeColumns'] = ({
    accessor,
    display,
  }) => [
    accessor('number', {
      id: 'number',
      header: 'Trial',
      size: 110,
      enableSorting: true,
      cell: ({ row }) => <Text kind="body/semibold/md">Trial {row.original.number}</Text>,
    }),
    ...metricNames.map((name) =>
      accessor((row: Trial) => row.metrics.find((metric) => metric.name === name)?.value ?? null, {
        id: `${METRIC_SORT_PREFIX}${name}`,
        header: name,
        size: 140,
        enableSorting: true,
        cell: ({ row }) => (
          <Text className="tabular-nums">
            {formatMetric(
              row.original.metrics.find((metric) => metric.name === name)?.value ?? null
            )}
          </Text>
        ),
      })
    ),
    ...paramNames.map((name) =>
      accessor((row: Trial) => paramValue(row, name) ?? '', {
        id: `${PARAM_SORT_PREFIX}${name}`,
        header: name,
        size: 150,
        enableSorting: true,
        cell: ({ row }) => {
          const value = paramValue(row.original, name);
          return (
            <Text className={`${numericParams.has(name) ? 'tabular-nums' : ''}`}>
              {value === undefined || value === '' ? EM_DASH : formatParamValue(value)}
            </Text>
          );
        },
      })
    ),
    accessor('durationSeconds', {
      id: 'duration',
      header: 'Duration',
      size: 120,
      enableSorting: true,
      cell: ({ row }) => (
        <Text className="tabular-nums">
          {row.original.durationSeconds === null
            ? EM_DASH
            : formatDurationMs(row.original.durationSeconds * 1_000)}
        </Text>
      ),
    }),
    accessor('state', {
      id: 'state',
      header: 'Status',
      size: 140,
      enableSorting: true,
      cell: ({ row }) =>
        row.original.state ? (
          <StatusBadge
            status={row.original.state}
            statusConfig={TRIAL_STATUS_CONFIG}
            fallback={{ label: formatState(row.original.state), color: 'gray' }}
          />
        ) : (
          <Text kind="body/regular/sm" className="text-placeholder">
            {EM_DASH}
          </Text>
        ),
    }),
    accessor('paretoOptimal', {
      id: 'frontier',
      header: 'Frontier',
      size: 130,
      enableSorting: true,
      cell: ({ row }) =>
        row.original.paretoOptimal ? (
          <Badge kind="outline" color="green">
            On frontier
          </Badge>
        ) : (
          <Text kind="body/regular/sm" className="text-placeholder">
            {EM_DASH}
          </Text>
        ),
    }),
    display({
      id: 'promote',
      header: 'Promote',
      size: 130,
      cell: () => (
        <Button kind="secondary" size="small" disabled>
          Promote
        </Button>
      ),
    }),
  ];

  return (
    <StudioDataView<Trial>
      dataViewState={dataViewState}
      searchField="params"
      makeColumns={makeColumns}
      attributes={{
        DataViewSearchBar: { placeholder: 'Search parameters or trial ID...' },
        DataViewRoot: {
          data: pageTrials,
          totalCount: processedTrials.length,
        },
        DataViewTableContent: {
          renderEmptyState: () => (
            <TableEmptyState
              header={debouncedSearchBar.trim() ? 'No matching trials' : 'No trials'}
              emptyMessage={
                debouncedSearchBar.trim()
                  ? 'No trial matches this search.'
                  : 'This study did not record any trials.'
              }
            />
          ),
        },
      }}
    />
  );
};
