// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { TableEmptyState } from '@nemo/common/src/components/TableEmptyState';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { useTrialColumns } from '@studio/components/dataViews/OptimizationJobsDataView/useTrialColumns';
import {
  formatParamValue,
  METRIC_SORT_PREFIX,
  PARAM_SORT_PREFIX,
  paramValue,
} from '@studio/components/dataViews/OptimizationJobsDataView/utils';
import type {
  StudyResults,
  Trial,
} from '@studio/routes/agents/AgentOptimizationDetailRoute/studyResults';
import { type FC, memo, useMemo } from 'react';

const paramsText = (trial: Trial): string =>
  trial.params.map((param) => `${param.name} ${formatParamValue(param.value)}`).join('  ·  ');

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

/** Missing values sort last in both directions, so an empty metric never tops the table. */
const compareNullable = (
  a: number | string | null,
  b: number | string | null,
  desc: boolean
): number => {
  if (a === null && b === null) return 0;
  if (a === null) return 1;
  if (b === null) return -1;
  const order =
    typeof a === 'string' || typeof b === 'string' ? String(a).localeCompare(String(b)) : a - b;
  return desc ? -order : order;
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

export interface TrialsDataViewProps {
  results: StudyResults;
  onDeploy?: (trial: Trial) => void;
}

export const TrialsDataView: FC<TrialsDataViewProps> = memo(({ results, onDeploy }) => {
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
    return [...filtered].sort((a, b) =>
      compareNullable(
        sortValue(a, sort.id, numericParams),
        sortValue(b, sort.id, numericParams),
        !!sort.desc
      )
    );
  }, [trials, debouncedSearchBar, sorting, numericParams]);

  const lastPageIndex = Math.max(0, Math.ceil(processedTrials.length / pageSize) - 1);
  const safePageIndex = Math.min(pageIndex, lastPageIndex);
  const pageTrials = useMemo(
    () => processedTrials.slice(safePageIndex * pageSize, safePageIndex * pageSize + pageSize),
    [processedTrials, safePageIndex, pageSize]
  );

  const makeColumns = useTrialColumns({ metricNames, paramNames, numericParams, onDeploy });

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
});

TrialsDataView.displayName = 'TrialsDataView';
