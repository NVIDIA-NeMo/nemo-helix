// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { dateTimeFilter } from '@nemo/common/src/components/DataView/dateTimeFilter';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import type { Trace } from '@nemo/sdk/generated/platform/schema';
import { Badge } from '@nvidia/foundations-react-core';
import { IntakeTelemetryStatusBadge } from '@studio/components/IntakeDetail/IntakeComponents/IntakeTelemetryStatusBadge';
import { IntakePayloadPreviewCell } from '@studio/components/IntakeLists/IntakePayloadPreviewCell';
import type { IntakeTelemetryDataView } from '@studio/components/IntakeLists/IntakeTelemetryDataView';
import { AGENT_NAME_FILTER_ID } from '@studio/components/IntakeLists/traceFilterIds';
import {
  formatCost,
  formatDurationMs,
  formatInteger,
  getTraceDisplayName,
} from '@studio/util/intakeTelemetry';
import type { ComponentProps } from 'react';

type MakeIntakeTraceColumns = ComponentProps<typeof IntakeTelemetryDataView<Trace>>['makeColumns'];

export interface IntakeTraceColumnOptions {
  /** Expose a Trace ID text filter on the first column (workspace browse table). */
  traceIdFilter?: boolean;
  /** Allow sorting by started_at (server-backed list only). */
  startedAtSort?: boolean;
  /** Expose a Started At datetime filter (workspace browse table). */
  startedAtFilter?: boolean;
  /** Expose a Session ID text filter in the workspace browse table. */
  sessionIdFilter?: boolean;
  /** Expose a Status filter in the workspace browse table. */
  statusFilter?: boolean;
  /** Show an Agent column with an agent-name filter (workspace browse table). */
  agentNameFilter?: boolean;
}

const TRACE_STATUS_FILTER_OPTIONS = [
  { value: 'success', label: 'Success' },
  { value: 'error', label: 'Error' },
  { value: 'cancelled', label: 'Cancelled' },
  { value: 'unknown', label: 'Unknown' },
];

/**
 * Shared trace table columns for Intake browse (`IntakeTracesTable`) and insight evidence
 * (`InsightTracesTable`). Keeps headers, sizes, and formatters in one place.
 */
export const makeIntakeTraceColumns =
  ({
    traceIdFilter = false,
    startedAtSort = false,
    startedAtFilter = false,
    sessionIdFilter = false,
    statusFilter = false,
    agentNameFilter = false,
  }: IntakeTraceColumnOptions = {}): MakeIntakeTraceColumns =>
  ({ accessor }) => [
    accessor('id', {
      id: 'id',
      header: 'Trace',
      size: 280,
      enableSorting: false,
      meta: traceIdFilter
        ? {
            filter: {
              type: 'text' as const,
              label: 'Trace ID',
              placeholder: 'Filter by trace ID',
            },
          }
        : undefined,
      cell: ({ row }) => getTraceDisplayName(row.original),
    }),
    ...(agentNameFilter
      ? [
          accessor('agent_name', {
            id: AGENT_NAME_FILTER_ID,
            header: 'Agent',
            size: 200,
            enableSorting: false,
            meta: {
              filter: {
                type: 'text' as const,
                label: 'Agent',
                placeholder: 'Filter by agent name',
              },
            },
            cell: ({ row }) => row.original.agent_name ?? '—',
          }),
        ]
      : []),
    accessor('session_id', {
      id: 'session_id',
      header: 'Session',
      size: 280,
      enableSorting: false,
      meta: sessionIdFilter
        ? {
            filter: {
              type: 'text' as const,
              label: 'Session ID',
              placeholder: 'Filter by session ID',
            },
          }
        : undefined,
    }),
    accessor('status', {
      id: 'status',
      header: 'Status',
      size: 120,
      enableSorting: false,
      meta: statusFilter
        ? {
            filter: {
              type: 'single-select' as const,
              label: 'Status',
              options: TRACE_STATUS_FILTER_OPTIONS,
            },
          }
        : undefined,
      cell: ({ row }) => <IntakeTelemetryStatusBadge status={row.original.status} />,
    }),
    accessor('input', {
      id: 'input',
      header: 'Input',
      size: 360,
      enableSorting: false,
      cell: ({ row }) => <IntakePayloadPreviewCell value={row.original.input} />,
    }),
    accessor('output', {
      id: 'output',
      header: 'Output',
      size: 360,
      enableSorting: false,
      cell: ({ row }) => <IntakePayloadPreviewCell value={row.original.output} />,
    }),
    {
      id: 'duration_ms',
      header: 'Duration',
      size: 120,
      enableSorting: false,
      cell: ({ row }) => formatDurationMs(row.original.duration_ms),
    },
    {
      id: 'span_count',
      header: 'Spans',
      size: 90,
      enableSorting: false,
      cell: ({ row }) => formatInteger(row.original.span_count),
    },
    {
      id: 'error_count',
      header: 'Errors',
      size: 90,
      enableSorting: false,
      cell: ({ row }) => {
        const errorCount = row.original.error_count ?? 0;
        return errorCount > 0 ? (
          <Badge kind="solid" color="red">
            {formatInteger(errorCount)}
          </Badge>
        ) : (
          formatInteger(errorCount)
        );
      },
    },
    {
      id: 'total_tokens',
      header: 'Tokens',
      size: 120,
      enableSorting: false,
      cell: ({ row }) => formatInteger(row.original.total_tokens),
    },
    {
      id: 'cost_usd',
      header: 'Cost',
      size: 110,
      enableSorting: false,
      cell: ({ row }) => formatCost(row.original.cost_usd),
    },
    accessor('started_at', {
      id: 'started_at',
      header: 'Started',
      size: 150,
      enableSorting: startedAtSort,
      meta: startedAtFilter ? { filter: dateTimeFilter('Started At') } : undefined,
      cell: ({ row }) => <RelativeTime datetime={row.original.started_at} />,
    }),
  ];
