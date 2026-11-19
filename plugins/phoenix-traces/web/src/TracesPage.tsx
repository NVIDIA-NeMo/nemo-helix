// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  RelativeTime,
  StatusBadge,
  StudioDataView,
  TableEmptyState,
  useStudioDataViewState,
} from '@nemo/common';
import { Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { useCallback, useEffect, useMemo, type ComponentProps } from 'react';
import { useSearchParams } from 'react-router';
import { pluginPath } from './paths';
import { useProjects, useSpans } from './queries';
import { formatDuration, groupTraces, type TraceSummary } from './traces';
import type { PluginHost } from './types';

export const STATUS_CONFIG = {
  OK: { label: 'OK', color: 'green' },
  ERROR: { label: 'Error', color: 'red' },
  UNSET: { label: 'Unset', color: 'gray' },
} as const;

export function TracesPage({ host }: { host: PluginHost }) {
  const [params, setParams] = useSearchParams();
  const projects = useProjects(host);
  const project = params.get('project') ?? projects.data?.[0];
  const spans = useSpans(host, project);
  const traces = useMemo(() => groupTraces(spans.data ?? []), [spans.data]);

  const { set: setBreadcrumbs } = host.breadcrumbs;
  useEffect(() => {
    setBreadcrumbs([{ label: 'Phoenix' }, { label: 'Traces' }]);
    return () => setBreadcrumbs([]);
  }, [setBreadcrumbs]);

  const dataViewState = useStudioDataViewState();
  const makeColumns = useCallback<ComponentProps<typeof StudioDataView<TraceSummary>>['makeColumns']>(
    (col) => [
      col.display({
        id: 'trace',
        header: 'Trace',
        size: 320,
        cell: ({ row }) => (
          <Stack gap="0">
            <Text kind="label/bold/sm">{row.original.name}</Text>
            <Text kind="body/regular/xs" color="secondary">
              {row.original.traceId.slice(0, 12)}
            </Text>
          </Stack>
        ),
      }),
      col.display({
        id: 'status',
        header: 'Status',
        size: 100,
        cell: ({ row }) => <StatusBadge status={row.original.status} statusConfig={STATUS_CONFIG} />,
      }),
      col.display({
        id: 'started',
        header: 'Started',
        size: 140,
        cell: ({ row }) => <RelativeTime datetime={new Date(row.original.start).toISOString()} />,
      }),
      col.display({
        id: 'duration',
        header: 'Duration',
        size: 110,
        cell: ({ row }) => <Text kind="body/regular/sm">{formatDuration(row.original.durationMs)}</Text>,
      }),
      col.accessor('spanCount', { header: 'Spans', size: 80 }),
      col.display({
        id: 'tokens',
        header: 'Tokens',
        size: 100,
        cell: ({ row }) => (
          <Text kind="body/regular/sm">{row.original.tokens?.toLocaleString() ?? '—'}</Text>
        ),
      }),
    ],
    []
  );

  return (
    <Stack gap="3" className="h-full p-4">
      <Flex gap="3" align="center" justify="between">
        <Text kind="label/bold/lg">Traces</Text>
        <label className="flex items-center gap-2 text-sm text-subtle">
          Project
          <select
            className="rounded border border-subtle bg-surface-base px-2 py-1 text-primary"
            value={project ?? ''}
            onChange={(e) => setParams({ project: e.target.value })}
          >
            {(projects.data ?? []).map((name) => (
              <option key={name} value={name}>
                {name || '(unassigned)'}
              </option>
            ))}
          </select>
        </label>
      </Flex>

      {projects.isError || spans.isError ? (
        <TableEmptyState
          header="Couldn't load traces"
          emptyMessage="The Phoenix plugin API request failed. Check that the plugin is registered and the Intake adapter is reachable."
        />
      ) : (projects.data?.length === 0 || !spans.isPending) && traces.length === 0 ? (
        <TableEmptyState header="No traces" emptyMessage="No traces are stored in this Intake workspace/project yet." />
      ) : (
        <Stack className="min-h-[320px]">
          <StudioDataView
            dataViewState={dataViewState}
            makeColumns={makeColumns}
            onRowClick={(trace) =>
              host.navigation.navigate(
                `${pluginPath(host.workspaceId, `traces/${encodeURIComponent(trace.traceId)}`)}?project=${encodeURIComponent(project ?? '')}`
              )
            }
            attributes={{
              DataViewRoot: {
                data: traces,
                totalCount: traces.length,
                reactTableOptions: { getRowId: (row) => row.traceId },
              },
            }}
          />
        </Stack>
      )}
    </Stack>
  );
}
