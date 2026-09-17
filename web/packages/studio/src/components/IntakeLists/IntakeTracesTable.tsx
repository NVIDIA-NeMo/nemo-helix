// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { EditColumnsMenu } from '@nemo/common/src/components/DataView/internal';
import { EntityEmptyState } from '@nemo/common/src/components/EntityEmptyState';
import { ErrorMessage } from '@nemo/common/src/components/ErrorMessage';
import { useRowNavigation } from '@nemo/common/src/hooks/useRowNavigation';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { getSortParamWithWhitelist } from '@nemo/common/src/utils/query';
import type { Trace, TraceFilter, TraceSortField } from '@nemo/sdk/generated/platform/schema';
import { useListTraces } from '@nemo/sdk/generated/platform/traces';
import {
  isDefaultStartedAtFilter,
  makeDefaultStartedAtFilter,
  type StartedAtFilterEntry,
  useSeededStartedAtFilter,
} from '@studio/components/IntakeLists/defaultStartedAtFilter';
import { IntakeTelemetryDataView } from '@studio/components/IntakeLists/IntakeTelemetryDataView';
import { makeIntakeTraceColumns } from '@studio/components/IntakeLists/intakeTraceColumns';
import { useWorkspaceFromPathIfExists } from '@studio/hooks/useWorkspaceFromPath';
import { getIntakeSessionTraceRoute } from '@studio/routes/utils';
import { keepPreviousData } from '@tanstack/react-query';
import { Columns3 } from 'lucide-react';
import { type FC, useState } from 'react';

export interface IntakeTracesTableProps {
  workspace?: string;
  slotEndPortalTargetId?: string;
}

export const IntakeTracesTable: FC<IntakeTracesTableProps> = (props) => {
  // Seed the default started_at filter into the URL before the table mounts,
  // so the dataview state initializes from it directly — one render, one
  // request, no unfiltered first fetch.
  const [defaultStartedAtFilter] = useState(makeDefaultStartedAtFilter);
  const filtersSeeded = useSeededStartedAtFilter(defaultStartedAtFilter);

  if (!filtersSeeded) return null;
  return <SeededIntakeTracesTable {...props} defaultStartedAtFilter={defaultStartedAtFilter} />;
};

const SeededIntakeTracesTable: FC<
  IntakeTracesTableProps & { defaultStartedAtFilter: StartedAtFilterEntry }
> = ({ workspace: workspaceProp, slotEndPortalTargetId, defaultStartedAtFilter }) => {
  const openRow = useRowNavigation();
  const routeWorkspace = useWorkspaceFromPathIfExists();
  const workspace = workspaceProp ?? routeWorkspace;
  const hasWorkspace = Boolean(workspace);
  const requestWorkspace = workspace ?? '';

  const dataViewState = useStudioDataViewState({
    defaultSort: [{ id: 'started_at', desc: true }],
  });

  // The seeded started_at default doesn't count as user filtering: an empty
  // workspace should still get the first-run empty state.
  const hasActiveFilters = dataViewState.debouncedColumnFilters.some(
    (filter) => !isDefaultStartedAtFilter(filter, defaultStartedAtFilter)
  );

  const {
    data: tracesResponse,
    isFetching,
    error,
  } = useListTraces(
    requestWorkspace,
    {
      filter: (dataViewState.apiFilter.filter ?? {}) as TraceFilter,
      mode: 'preview',
      page: dataViewState.pagination.state.pageIndex + 1,
      page_size: dataViewState.pagination.state.pageSize,
      sort: getSortParamWithWhitelist(
        dataViewState.sorting.state,
        ['started_at'],
        '-started_at'
      ) as TraceSortField,
    },
    {
      query: {
        enabled: hasWorkspace,
        placeholderData: keepPreviousData,
      },
    }
  );

  if (error) {
    return <ErrorMessage message={getErrorMessage(error)} />;
  }

  if (!workspace) {
    return <ErrorMessage message="Workspace is required to load traces." />;
  }

  return (
    <IntakeTelemetryDataView<Trace>
      dataViewState={dataViewState}
      makeColumns={makeIntakeTraceColumns({
        traceIdFilter: true,
        sessionIdFilter: true,
        statusFilter: true,
        startedAtSort: true,
        startedAtFilter: true,
      })}
      slotEndPortalTargetId={slotEndPortalTargetId}
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
      onRowClick={(trace, _index, event) =>
        openRow(event, getIntakeSessionTraceRoute(requestWorkspace, trace.session_id, trace.id))
      }
      attributes={{
        DataViewRoot: {
          data: tracesResponse?.data ?? [],
          totalCount: tracesResponse?.pagination?.total_results,
          requestStatus: isFetching ? 'loading' : undefined,
        },
        DataViewTableContent: {
          renderEmptyState: () =>
            hasActiveFilters ? (
              <EntityEmptyState
                entity="telemetryTraces"
                variant="no-results"
                onClearFilters={dataViewState.resetFilters}
              />
            ) : (
              <EntityEmptyState
                entity="telemetryTraces"
                variant="first-use"
                workspace={workspace}
              />
            ),
        },
      }}
    />
  );
};
