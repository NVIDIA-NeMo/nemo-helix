// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { withOperators } from '@nemo/common/src/api/filterOperators';
import { EditColumnsMenu } from '@nemo/common/src/components/DataView/internal';
import { EntityEmptyState } from '@nemo/common/src/components/EntityEmptyState';
import { ErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { useRowNavigation } from '@nemo/common/src/hooks/useRowNavigation';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import type { TraceEvidence } from '@nemo/sdk/generated/insights/schema';
import type { Trace, TraceFilter } from '@nemo/sdk/generated/platform/schema';
import { useListTraces } from '@nemo/sdk/generated/platform/traces';
import { Anchor, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { IntakeTelemetryDataView } from '@studio/components/IntakeLists/IntakeTelemetryDataView';
import { makeIntakeTraceColumns } from '@studio/components/IntakeLists/intakeTraceColumns';
import { getIntakeSessionTraceRoute } from '@studio/routes/utils';
import { Columns3, TriangleAlert } from 'lucide-react';
import { type FC } from 'react';

// Module-level so its identity is stable: DataView rebuilds columns, and remounts every cell,
// whenever makeColumns changes.
const makeTraceColumns = makeIntakeTraceColumns();

function safeEvidenceUrl(value: string | null | undefined): string | undefined {
  if (!value) return undefined;
  try {
    const url = new URL(value);
    return ['https:', 'http:'].includes(url.protocol) ? value : undefined;
  } catch {
    return undefined;
  }
}

export interface InsightTracesTableProps {
  workspace: string;
  evidence: TraceEvidence[];
}

/**
 * Renders an insight's evidence traces using the same columns and DataView shell as
 * `IntakeTracesTable`. Unlike the workspace browse table, this fetches only the referenced
 * traces by id and preserves `traceIds` order (no server sort/filter).
 */
export const InsightTracesTable: FC<InsightTracesTableProps> = ({ workspace, evidence }) => {
  const traceIds = evidence.map((item) => item.trace_id);
  const openRow = useRowNavigation();
  const dataViewState = useStudioDataViewState();
  const { pageIndex, pageSize } = dataViewState.pagination.state;
  const firstVisibleIndex = pageIndex * pageSize;
  const visibleTraceIds = traceIds.slice(firstVisibleIndex, firstVisibleIndex + pageSize);

  const { data, error, isFetching } = useListTraces(
    workspace,
    {
      filter: withOperators<TraceFilter>({ id: { $in: visibleTraceIds } }),
      mode: 'preview',
      page: 1,
      page_size: pageSize,
    },
    {
      query: {
        enabled: Boolean(workspace) && visibleTraceIds.length > 0,
      },
    }
  );

  const tracesById = new Map((data?.data ?? []).map((trace) => [trace.id, trace]));
  const traces = visibleTraceIds
    .map((id) => tracesById.get(id))
    .filter((trace): trace is Trace => trace !== undefined);
  const failedCount = visibleTraceIds.length - traces.length;

  return (
    <Stack className="gap-density-sm">
      {failedCount > 0 && !error && !isFetching ? (
        <Flex className="items-center gap-density-sm">
          <TriangleAlert aria-hidden className="size-4 shrink-0 text-danger" />
          <Text kind="body/regular/sm" className="text-danger">
            {failedCount} of {visibleTraceIds.length} traces couldn&apos;t be loaded.
          </Text>
        </Flex>
      ) : null}
      {evidence.slice(firstVisibleIndex, firstVisibleIndex + pageSize).map((item) => {
        const trace = tracesById.get(item.trace_id);
        const traceUrl =
          safeEvidenceUrl(item.url) ??
          (trace ? getIntakeSessionTraceRoute(workspace, trace.session_id, trace.id) : undefined);
        if (!item.url && !item.spans?.length) return null;
        return (
          <Stack key={item.trace_id} className="gap-density-xs">
            {traceUrl ? (
              <Anchor href={traceUrl}>{item.trace_id}</Anchor>
            ) : (
              <Text>{item.trace_id}</Text>
            )}
            {item.spans?.length ? (
              <ul aria-label={`Supporting spans for ${item.trace_id}`} className="pl-4">
                {item.spans.map((span) => {
                  const url = safeEvidenceUrl(span.url) ?? traceUrl;
                  return (
                    <li key={span.span_id}>
                      {url ? (
                        <Anchor href={url}>{span.span_id}</Anchor>
                      ) : (
                        <Text>{span.span_id}</Text>
                      )}
                    </li>
                  );
                })}
              </ul>
            ) : null}
          </Stack>
        );
      })}
      <IntakeTelemetryDataView<Trace>
        dataViewState={dataViewState}
        makeColumns={makeTraceColumns}
        onRowClick={(trace, _index, event) =>
          openRow(event, getIntakeSessionTraceRoute(workspace, trace.session_id, trace.id))
        }
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
            data: traces,
            totalCount: traceIds.length,
            requestStatus: error ? 'error' : isFetching ? 'loading' : undefined,
          },
          DataViewTableContent: {
            renderEmptyState: () => <EntityEmptyState entity="insightTraces" variant="first-use" />,
            renderErrorState: () => (
              <ErrorPanel
                errorMessage={getErrorMessage(error ?? new Error('Failed to load traces'))}
              />
            ),
          },
        }}
      />
    </Stack>
  );
};
