// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { EntityEmptyState } from '@nemo/common/src/components/EntityEmptyState';
import { ErrorMessage } from '@nemo/common/src/components/ErrorMessage';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { getRequestStatus } from '@nemo/common/src/utils/query';
import { Button, Stack, Text } from '@nvidia/foundations-react-core';
import type { RunSummary } from '@studio/routes/agents/AgentMonitorRoute/telemetry';
import { ComponentProps, FC, useEffect, useMemo } from 'react';

interface Props {
  runs: RunSummary[];
  isLoading?: boolean;
  error?: unknown;
  onRetry?: () => void;
}

type RunRow = RunSummary & { id: string };

const formatDuration = (ms: number): string => {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)} s`;
  return `${(ms / 60_000).toFixed(1)} min`;
};

const cmpString = (a: string | undefined, b: string | undefined): number =>
  (a ?? '').localeCompare(b ?? '');

const compareRunsBy = (a: RunRow, b: RunRow, columnId: string): number => {
  switch (columnId) {
    case 'startedAt':
      return a.startedAt.getTime() - b.startedAt.getTime();
    case 'agent':
      return cmpString(a.agent, b.agent);
    case 'model':
      return cmpString(a.model, b.model);
    case 'duration':
      return a.durationMs - b.durationMs;
    case 'prompt_tokens':
      return a.promptTokens - b.promptTokens;
    case 'completion_tokens':
      return a.completionTokens - b.completionTokens;
    case 'tool_calls':
      return a.toolCalls - b.toolCalls;
    default:
      return 0;
  }
};

export const InferenceLogsTable: FC<Props> = ({ runs, isLoading, error, onRetry }) => {
  // Keep the page short enough to fit alongside the cards + chart so the data
  // view's internal overflow scroll doesn't capture the page's wheel events.
  const dataViewState = useStudioDataViewState({
    defaultSort: [{ id: 'startedAt', desc: true }],
    defaultPageSize: 10,
  });

  const rows = useMemo<RunRow[]>(() => runs.map((run) => ({ ...run, id: run.runId })), [runs]);

  // Reset URL pagination when the row count shrinks past the current page (URL
  // bookmark, manual edit, telemetry set shrinking) — otherwise the slice below
  // returns an empty array and StudioDataView shows the empty state for valid
  // data.
  const { pageIndex, pageSize } = dataViewState.pagination.state;
  const lastPageIndex = Math.max(0, Math.ceil(rows.length / pageSize) - 1);
  useEffect(() => {
    if (rows.length > 0 && pageIndex > lastPageIndex) {
      dataViewState.pagination.set({ pageIndex: lastPageIndex, pageSize });
    }
  }, [rows.length, pageIndex, pageSize, lastPageIndex, dataViewState.pagination]);

  // StudioDataView is hardcoded dataMode="manual" — it does not sort or paginate
  // for us. Apply both client-side here against the fully-loaded telemetry rows.
  const pagedRows = useMemo<RunRow[]>(() => {
    const sort = dataViewState.sorting.state[0];
    const sorted = sort
      ? [...rows].sort((a, b) => {
          const cmp = compareRunsBy(a, b, sort.id);
          return sort.desc ? -cmp : cmp;
        })
      : rows;
    const safePageIndex = Math.min(pageIndex, lastPageIndex);
    const start = safePageIndex * pageSize;
    return sorted.slice(start, start + pageSize);
  }, [rows, dataViewState.sorting.state, pageIndex, pageSize, lastPageIndex]);

  const makeColumns: ComponentProps<typeof StudioDataView<RunRow>>['makeColumns'] = ({
    accessor,
  }) => [
    accessor('startedAt', {
      id: 'startedAt',
      header: 'Time',
      enableSorting: true,
      size: 140,
      cell: ({ row }) => <RelativeTime datetime={row.original.startedAt.toISOString()} />,
    }),
    accessor((row) => row.agent ?? '', {
      id: 'agent',
      header: 'Agent',
      enableSorting: true,
      size: 160,
      cell: ({ row }) => (
        <Text className="truncate" title={row.original.agent}>
          {row.original.agent ?? '—'}
        </Text>
      ),
    }),
    accessor((row) => row.model ?? '', {
      id: 'model',
      header: 'Model',
      enableSorting: true,
      size: 220,
      cell: ({ row }) => (
        <Text className="truncate" title={row.original.model}>
          {row.original.model ?? '—'}
        </Text>
      ),
    }),
    {
      id: 'input',
      header: 'Input',
      enableSorting: false,
      cell: ({ row }) => (
        <Text className="truncate" title={row.original.inputPreview}>
          {row.original.inputPreview || '—'}
        </Text>
      ),
    },
    accessor('promptTokens', {
      id: 'prompt_tokens',
      header: 'Tokens in',
      enableSorting: true,
      size: 110,
      cell: ({ row }) => <Text>{row.original.promptTokens}</Text>,
    }),
    accessor('completionTokens', {
      id: 'completion_tokens',
      header: 'Tokens out',
      enableSorting: true,
      size: 110,
      cell: ({ row }) => <Text>{row.original.completionTokens}</Text>,
    }),
    accessor('toolCalls', {
      id: 'tool_calls',
      header: 'Tool calls',
      enableSorting: true,
      size: 110,
      cell: ({ row }) => <Text>{row.original.toolCalls}</Text>,
    }),
    accessor('durationMs', {
      id: 'duration',
      header: 'Duration',
      enableSorting: true,
      size: 110,
      cell: ({ row }) => <Text>{formatDuration(row.original.durationMs)}</Text>,
    }),
  ];

  return (
    <Stack gap="density-md">
      <h3 className="text-lg font-semibold">Inference logs</h3>
      <StudioDataView<RunRow>
        dataViewState={dataViewState}
        makeColumns={makeColumns}
        attributes={{
          DataViewRoot: {
            data: pagedRows,
            totalCount: rows.length,
            requestStatus: getRequestStatus({ error, isLoading }),
          },
          DataViewTableContent: {
            renderErrorState: () => (
              <ErrorMessage
                header="Failed to load telemetry"
                message={
                  onRetry ? (
                    <Button kind="secondary" size="small" onClick={onRetry}>
                      Retry
                    </Button>
                  ) : undefined
                }
              />
            ),
            renderEmptyState: () => (
              <EntityEmptyState entity="agentMonitorRuns" variant="first-use" />
            ),
          },
        }}
      />
    </Stack>
  );
};
