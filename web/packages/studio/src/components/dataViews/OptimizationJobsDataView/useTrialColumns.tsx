// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { StatusBadge, type StatusConfigEntry } from '@nemo/common/src/components/StatusBadge';
import { formatDurationMs } from '@nemo/common/src/utils/date';
import { Badge, Button, Text } from '@nvidia/foundations-react-core';
import {
  EM_DASH,
  formatParamValue,
  METRIC_SORT_PREFIX,
  PARAM_SORT_PREFIX,
  paramValue,
} from '@studio/components/dataViews/OptimizationJobsDataView/utils';
import type { Trial } from '@studio/routes/agents/AgentOptimizationDetailRoute/studyResults';
import { Ban, CircleCheck, CircleX, RefreshCw } from 'lucide-react';
import { type ComponentProps, useCallback } from 'react';

const formatMetric = (value: number | null): string => {
  if (value === null) return EM_DASH;
  const fixed = value.toLocaleString(undefined, { maximumFractionDigits: 4 });
  if (value !== 0 && Number(fixed.replace(/,/g, '')) === 0) {
    return value.toLocaleString(undefined, { maximumSignificantDigits: 4 });
  }
  return fixed;
};

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

type MakeTrialColumns = NonNullable<ComponentProps<typeof StudioDataView<Trial>>['makeColumns']>;

interface UseTrialColumnsOptions {
  metricNames: string[];
  paramNames: string[];
  numericParams: ReadonlySet<string>;
  onDeploy?: (trial: Trial) => void;
}

export const useTrialColumns = ({
  metricNames,
  paramNames,
  numericParams,
  onDeploy,
}: UseTrialColumnsOptions): MakeTrialColumns =>
  useCallback<MakeTrialColumns>(
    ({ accessor, display }) => [
      accessor('number', {
        id: 'number',
        header: 'Trial',
        size: 110,
        enableSorting: true,
        cell: ({ row }) => <Text kind="body/semibold/md">Trial {row.original.number}</Text>,
      }),
      ...metricNames.map((name) =>
        accessor(
          (row: Trial) => row.metrics.find((metric) => metric.name === name)?.value ?? null,
          {
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
          }
        )
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
        id: 'deploy',
        header: 'Deploy',
        size: 130,
        cell: ({ row }) => (
          <Button
            kind="secondary"
            size="small"
            disabled={!onDeploy || row.original.state !== 'COMPLETE'}
            onClick={() => onDeploy?.(row.original)}
            aria-label={`Deploy trial ${row.original.number}`}
          >
            Deploy
          </Button>
        ),
      }),
    ],
    [metricNames, paramNames, numericParams, onDeploy]
  );
