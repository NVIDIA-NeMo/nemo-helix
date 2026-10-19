// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { agentsListAgents } from '@nemo/sdk/generated/agents/agents';
import { useFilesListFilesetFiles } from '@nemo/sdk/generated/platform/files';
import { Combobox, PageHeader, Stack } from '@nvidia/foundations-react-core';
import { FeatureFlagBadge } from '@studio/components/FeatureFlagBadge';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { InferenceLogsTable } from '@studio/routes/agents/AgentMonitorRoute/components/InferenceLogsTable';
import { SummaryCards } from '@studio/routes/agents/AgentMonitorRoute/components/SummaryCards';
import { TokenUsageChart } from '@studio/routes/agents/AgentMonitorRoute/components/TokenUsageChart';
import { RunSummary } from '@studio/routes/agents/AgentMonitorRoute/telemetry';
import {
  FILESET_NAME,
  fetchTelemetryRuns,
  isNotFoundError,
  summarizeRuns,
} from '@studio/routes/agents/AgentMonitorRoute/utils';
import { getAgentsListRoute } from '@studio/routes/utils';
import { useInfiniteQuery, useQuery } from '@tanstack/react-query';
import { useCallback, useEffect, useMemo, useState, type FC } from 'react';

const AGENTS_FILTER_PAGE_SIZE = 25;

export const AgentMonitorRoute: FC = () => {
  const workspace = useWorkspaceFromPath();

  useBreadcrumbs({
    items: [
      {
        slotLabel: 'Agents',
        href: getAgentsListRoute(workspace),
      },
      {
        slotLabel: 'Monitor',
      },
    ],
  });

  const filesListQuery = useFilesListFilesetFiles(workspace, FILESET_NAME, undefined, {
    query: { retry: false, enabled: !!workspace },
  });

  const agentsListQuery = useInfiniteQuery({
    queryKey: [
      'agent-monitor',
      'agents-filter',
      workspace,
      { page_size: AGENTS_FILTER_PAGE_SIZE, sort: 'name' },
    ],
    queryFn: ({ pageParam, signal }) =>
      agentsListAgents(
        workspace,
        { page: pageParam, page_size: AGENTS_FILTER_PAGE_SIZE, sort: 'name' },
        signal
      ),
    initialPageParam: 1,
    getNextPageParam: (last, all) => {
      const totalPages = last.pagination?.total_pages ?? 0;
      return all.length < totalPages ? all.length + 1 : undefined;
    },
    enabled: !!workspace,
    retry: false,
  });

  const loadMoreAgents = useCallback(() => {
    if (agentsListQuery.hasNextPage && !agentsListQuery.isFetchingNextPage) {
      void agentsListQuery.fetchNextPage();
    }
  }, [agentsListQuery]);

  // 404 = brand-new workspace; render empty state, not an error.
  const filesError = isNotFoundError(filesListQuery.error) ? null : filesListQuery.error;
  const files = useMemo(() => filesListQuery.data?.data ?? [], [filesListQuery.data?.data]);
  const hasFiles = files.length > 0;
  const filePaths = useMemo(() => files.map((f) => f.path), [files]);

  const runsQuery = useQuery({
    queryKey: ['agent-monitor', 'telemetry-runs', workspace, filePaths],
    queryFn: ({ signal }) => fetchTelemetryRuns(workspace, files, signal),
    enabled: !!workspace && hasFiles,
    retry: false,
  });

  const runs: RunSummary[] = useMemo(() => runsQuery.data?.runs ?? [], [runsQuery.data?.runs]);
  const isFetching = filesListQuery.isFetching || runsQuery.isFetching;
  const displayError = runsQuery.error ?? filesError;

  const [agentFilter, setAgentFilter] = useState<string[]>([]);

  const agentOptions = useMemo(() => {
    const names = new Set<string>();
    for (const page of agentsListQuery.data?.pages ?? []) {
      for (const agent of page.data ?? []) {
        if (agent.name) names.add(agent.name);
      }
    }
    for (const run of runs) {
      if (run.agent) names.add(run.agent);
    }
    return Array.from(names)
      .sort()
      .map((a) => ({ value: a, children: a }));
  }, [agentsListQuery.data?.pages, runs]);

  useEffect(() => {
    const valid = new Set(agentOptions.map((o) => o.value));
    setAgentFilter((current) => {
      const next = current.filter((a) => valid.has(a));
      return next.length === current.length ? current : next;
    });
  }, [agentOptions]);

  const filteredRuns = useMemo(() => {
    if (agentFilter.length === 0) return runs;
    const allowed = new Set(agentFilter);
    return runs.filter((r) => r.agent && allowed.has(r.agent));
  }, [runs, agentFilter]);

  const summary = useMemo(() => summarizeRuns(filteredRuns), [filteredRuns]);

  const description = `Aggregated from ${runsQuery.data?.fileCount ?? 0} telemetry file${
    runsQuery.data?.fileCount === 1 ? '' : 's'
  } in ${FILESET_NAME}.`;

  return (
    <AccessibleTitle title={`Agent Monitor for ${workspace}`}>
      <Stack gap="density-2xl" padding="density-2xl">
        <PageHeader
          className="p-0"
          slotHeading={
            <>
              Monitor
              <FeatureFlagBadge flag="monitorEnabled" />
            </>
          }
          slotDescription={description}
          slotActions={
            <Combobox
              multiple
              className="w-80 shrink-0"
              items={agentOptions}
              value={agentFilter}
              onValueChange={setAgentFilter}
              onScrollToBottom={loadMoreAgents}
              placeholder="All agents"
              aria-label="Filter by agent"
            />
          }
        />

        <SummaryCards summary={summary} />

        <TokenUsageChart runs={filteredRuns} isPending={isFetching && filteredRuns.length === 0} />

        <InferenceLogsTable
          runs={filteredRuns}
          isFetching={isFetching}
          error={displayError}
          onRetry={() => {
            void filesListQuery.refetch();
            void runsQuery.refetch();
          }}
        />
      </Stack>
    </AccessibleTitle>
  );
};
