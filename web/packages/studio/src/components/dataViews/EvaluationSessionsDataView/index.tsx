// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  Root as DataViewRoot,
  EditColumnsMenu,
} from '@nemo/common/src/components/DataView/internal';
import { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { EntityEmptyState } from '@nemo/common/src/components/EntityEmptyState';
import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import { useRowNavigation } from '@nemo/common/src/hooks/useRowNavigation';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { formatDurationMs } from '@nemo/common/src/utils/date';
import { formatEvaluatorScore, snakeCaseToTitleCase } from '@nemo/common/src/utils/formatters';
import { useGetEvaluation } from '@nemo/sdk/generated/platform/evaluations';
import {
  listEvaluationSessions,
  useListEvaluationSessions,
} from '@nemo/sdk/generated/platform/experiments';
import type {
  EvaluationSessionResponsesPage,
  EvaluationSessionFilter,
  EvaluationSessionResponse,
  ListEvaluationSessionsParams,
} from '@nemo/sdk/generated/platform/schema';
import { Text, Tooltip } from '@nvidia/foundations-react-core';
import { IntakePayloadPreviewCell } from '@studio/components/IntakeLists/IntakePayloadPreviewCell';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { getEvaluationSessionTraceDetailRoute } from '@studio/routes/utils';
import { tooltipClassName } from '@studio/styles/common';
import { formatInteger } from '@studio/util/intakeTelemetry';
import { keepPreviousData } from '@tanstack/react-query';
import { isAxiosError } from 'axios';
import { Columns3 } from 'lucide-react';
import { type ComponentProps, type FC, useMemo } from 'react';

type SessionRow = EvaluationSessionResponse & { _rowId: string };

interface EvaluationSessionsDataViewProps {
  evaluationName: string;
  experimentName: string;
}

const mapStatusForBadge = (status: EvaluationSessionResponse['status']) =>
  status === 'success' ? 'completed' : status;

const isUnsupportedModeError = (error: unknown): boolean => {
  if (!isAxiosError(error)) return false;
  if (error.response?.status !== 400 && error.response?.status !== 422) return false;
  const detail = (error.response.data as { detail?: unknown } | undefined)?.detail;
  return typeof detail === 'string' && detail === 'Unsupported query parameter(s): mode';
};

const listEvaluationSessionsWithModeFallback = async (
  workspace: string,
  evaluationName: string,
  params: ListEvaluationSessionsParams,
  signal: AbortSignal
): Promise<EvaluationSessionResponsesPage> => {
  try {
    return await listEvaluationSessions(workspace, evaluationName, params, signal);
  } catch (error) {
    if (params.mode !== 'preview' || !isUnsupportedModeError(error)) {
      throw error;
    }
    const fallbackParams = { ...params };
    delete fallbackParams.mode;
    return listEvaluationSessions(workspace, evaluationName, fallbackParams, signal);
  }
};

// Column id → API sort field. All session sort fields are direct 1:1 matches so no
// translation is needed beyond listing the sortable ids.
const SESSION_SORT_FIELD_MAP: Readonly<Record<string, string>> = {
  test_case_name: 'test_case_name',
  latency_ms: 'latency_ms',
  status: 'status',
  tokens: 'tokens',
};

// Converts the table's multi-column sorting state to the comma-separated `sort` API param.
// Returns undefined when nothing is sorted so the endpoint uses its default order.
const getSessionSortParam = (sortingState: { id: string; desc: boolean }[]): string | undefined => {
  if (sortingState.length === 0) return undefined;
  const fields = sortingState
    .map(({ id, desc }) => {
      const field = SESSION_SORT_FIELD_MAP[id];
      return field ? `${desc ? '-' : ''}${field}` : undefined;
    })
    .filter((f): f is string => f !== undefined);
  return fields.length > 0 ? fields.join(',') : undefined;
};

export const EvaluationSessionsDataView: FC<EvaluationSessionsDataViewProps> = ({
  evaluationName,
  experimentName,
}) => {
  const workspace = useWorkspaceFromPath();
  const openRow = useRowNavigation();
  const dataViewState = useStudioDataViewState<EvaluationSessionFilter>({
    columnVisibility: {},
    multiSort: true,
  });
  const { data: experiment } = useGetEvaluation(workspace, evaluationName);

  const page = dataViewState.pagination.state.pageIndex + 1;
  const pageSize = dataViewState.pagination.state.pageSize;
  const sessionParams = useMemo<ListEvaluationSessionsParams>(() => {
    const sort = getSessionSortParam(dataViewState.sorting.state);
    return {
      page,
      page_size: pageSize,
      mode: 'preview',
      ...(sort && { sort }),
      filter: {
        ...dataViewState.apiFilter.filter,
        ...(dataViewState.debouncedSearchBar && {
          test_case_name: dataViewState.debouncedSearchBar,
        }),
      },
    };
  }, [
    dataViewState.apiFilter.filter,
    dataViewState.debouncedSearchBar,
    dataViewState.sorting.state,
    page,
    pageSize,
  ]);

  const { data: sessionsResponse, isLoading } = useListEvaluationSessions(
    workspace,
    evaluationName,
    sessionParams,
    {
      query: {
        placeholderData: keepPreviousData,
        queryFn: ({ signal }) =>
          listEvaluationSessionsWithModeFallback(workspace, evaluationName, sessionParams, signal),
      },
    }
  );

  const sessionsData = sessionsResponse?.data;
  const totalCount = sessionsResponse?.pagination?.total_results ?? sessionsData?.length ?? 0;

  const tableData = useMemo<SessionRow[]>(
    () =>
      (sessionsData ?? []).map((session, i) => ({
        ...session,
        _rowId: session.session_id ?? String(i),
      })),
    [sessionsData]
  );

  // Client-side status filter for instant feedback while the debounced API request catches up.
  const immediateStatusFilter = dataViewState.columnFiltering.state.find((f) => f.id === 'status')
    ?.value as string | undefined;
  const visibleTableData = useMemo(
    () =>
      immediateStatusFilter
        ? tableData.filter((row) => row.status === immediateStatusFilter)
        : tableData,
    [tableData, immediateStatusFilter]
  );

  // One column per evaluator. The experiment's `evaluator_names` is the
  // authoritative, stable set across all sessions; we union in any score keys
  // present in the current page's data so a column never goes missing while the
  // experiment is still loading. Sorted for deterministic column ordering.
  const evaluatorNames = useMemo<string[]>(() => {
    const names = new Set<string>(experiment?.evaluator_names ?? []);
    for (const session of sessionsData ?? []) {
      for (const name of Object.keys(session.evaluator_scores ?? {})) {
        names.add(name);
      }
    }
    return Array.from(names).sort();
  }, [experiment?.evaluator_names, sessionsData]);

  const makeColumns: ComponentProps<typeof DataViewRoot<SessionRow>>['makeColumns'] = ({
    accessor,
  }) => [
    accessor('test_case_name', {
      header: 'Test case',
      enableSorting: true,
      size: 200,
      cell: ({ row }) => {
        const value = row.original.test_case_name;
        if (!value) return <Text>-</Text>;
        return (
          <Tooltip slotContent={value} className={tooltipClassName} side="bottom">
            <Text className="cursor-default truncate max-w-[180px] block">{value}</Text>
          </Tooltip>
        );
      },
    }),
    accessor('status', {
      header: 'Status',
      enableSorting: true,
      meta: {
        filter: {
          type: 'single-select',
          label: 'Status',
          options: [
            { value: 'success', label: 'Completed' },
            { value: 'error', label: 'Error' },
            { value: 'cancelled', label: 'Cancelled' },
            { value: 'unknown', label: 'Unknown' },
          ],
        },
      },
      cell: ({ row }) => <StatusBadge status={mapStatusForBadge(row.original.status)} />,
    }),
    accessor('latency_ms', {
      header: 'Duration',
      enableSorting: true,
      meta: { alignment: 'right' },
      cell: ({ row }) => {
        const ms = row.original.latency_ms;
        return <Text>{ms != null ? formatDurationMs(ms) : '-'}</Text>;
      },
    }),
    accessor('input', {
      header: 'Input',
      enableSorting: false,
      size: 400,
      cell: ({ row }) => <IntakePayloadPreviewCell value={row.original.input} emptyValue="-" />,
    }),
    accessor('output', {
      header: 'Output',
      enableSorting: false,
      size: 400,
      cell: ({ row }) => <IntakePayloadPreviewCell value={row.original.output} emptyValue="-" />,
    }),
    accessor(
      (original) =>
        original.input_tokens != null || original.output_tokens != null
          ? (original.input_tokens ?? 0) + (original.output_tokens ?? 0)
          : undefined,
      {
        id: 'tokens',
        header: 'Tokens',
        enableSorting: true,
        meta: { alignment: 'right' },
        cell: ({ row }) => {
          const { input_tokens, output_tokens } = row.original;
          if (input_tokens == null && output_tokens == null) return <Text>-</Text>;
          return <Text>{formatInteger((input_tokens ?? 0) + (output_tokens ?? 0))}</Text>;
        },
      }
    ),
    ...evaluatorNames.map((name, index) =>
      accessor((original) => original.evaluator_scores?.[name], {
        id: `score-${index}`,
        header: snakeCaseToTitleCase(name),
        enableSorting: false,
        size: 130,
        meta: { alignment: 'right' },
        cell: ({ row }) => {
          const value = row.original.evaluator_scores?.[name];
          return <Text>{formatEvaluatorScore(value)}</Text>;
        },
      })
    ),
  ];

  return (
    <StudioDataView
      dataViewState={dataViewState}
      makeColumns={makeColumns}
      searchField="test_case_name"
      onRowClick={(row, _index, event) => {
        if (row.trace_id) {
          openRow(
            event,
            getEvaluationSessionTraceDetailRoute(
              workspace,
              experimentName,
              evaluationName,
              row.session_id,
              row.trace_id
            )
          );
        }
      }}
      toolbarSlotEnd={
        <EditColumnsMenu
          kind="secondary"
          showChevron={false}
          slotContent={<div aria-hidden className="h-0 w-[230px]" />}
        >
          <>
            <Columns3 />
            <span className="hide-mobile">Columns</span>
          </>
        </EditColumnsMenu>
      }
      attributes={{
        DataViewRoot: {
          data: visibleTableData,
          totalCount,
          requestStatus: isLoading && !sessionsData ? 'loading' : undefined,
        },
        DataViewSearchBar: { placeholder: 'Search case...' },
        DataViewTableContent: {
          renderEmptyState: ({ hasFiltersApplied, hasSearchApplied }) =>
            hasFiltersApplied || hasSearchApplied ? (
              <EntityEmptyState
                entity="evaluationSessions"
                variant="no-results"
                onClearFilters={dataViewState.resetFilters}
              />
            ) : (
              <EntityEmptyState entity="evaluationSessions" variant="first-use" />
            ),
        },
      }}
    />
  );
};
