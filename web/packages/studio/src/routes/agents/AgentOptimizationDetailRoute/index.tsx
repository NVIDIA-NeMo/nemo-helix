// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { ErrorMessage } from '@nemo/common/src/components/ErrorMessage';
import { ErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { LogViewer } from '@nemo/common/src/components/LogViewer';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import { JOB_POLLING_INTERVAL_MS } from '@nemo/common/src/constants';
import { useJobLogs } from '@nemo/common/src/hooks/useJobLogs';
import { useAgentsGetOptimizeJob } from '@nemo/sdk/generated/agents/agents';
import { PlatformJobStatus } from '@nemo/sdk/generated/platform/schema';
import { Flex, PageHeader, Panel, Spinner, Stack, Text } from '@nvidia/foundations-react-core';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { fetchStudyResults } from '@studio/routes/agents/AgentOptimizationDetailRoute/studyResults';
import { StudyStatTiles } from '@studio/routes/agents/AgentOptimizationDetailRoute/StudyStatTiles';
import { TrialsTable } from '@studio/routes/agents/AgentOptimizationDetailRoute/TrialsTable';
import { getAgentOptimizationsTabRoute, getAgentsListRoute } from '@studio/routes/utils';
import { useRequiredPathParams } from '@studio/util/hooks/useRequiredPathParams';
import { useQuery } from '@tanstack/react-query';
import { ScrollText } from 'lucide-react';
import { type FC, useEffect } from 'react';

/** Statuses that will not change again, so polling can stop. */
const TERMINAL_STATUSES = new Set<PlatformJobStatus>(['completed', 'error', 'cancelled']);
const FAILED_STATUSES = new Set<PlatformJobStatus>(['error', 'cancelled']);

export const AgentOptimizationDetailRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const { optimizeJobName: jobName } = useRequiredPathParams([ROUTE_PARAMS.optimizeJobName]);

  const { data: job, isLoading: isLoadingJob } = useAgentsGetOptimizeJob(workspace, jobName, {
    query: {
      enabled: !!workspace && !!jobName,
      refetchInterval: (query) =>
        query.state.data?.status && TERMINAL_STATUSES.has(query.state.data.status)
          ? false
          : JOB_POLLING_INTERVAL_MS,
    },
  });

  const status = job?.status ?? undefined;
  const isTerminal = status ? TERMINAL_STATUSES.has(status) : false;
  const hasFailed = status ? FAILED_STATUSES.has(status) : false;
  const agentName = job?.spec?.agent?.split('/').pop() ?? undefined;

  const { setBreadcrumbs } = useBreadcrumbs();
  useEffect(() => {
    setBreadcrumbs([
      { slotLabel: 'Agents', href: getAgentsListRoute(workspace) },
      agentName
        ? {
            slotLabel: 'Optimizations',
            href: getAgentOptimizationsTabRoute(workspace, agentName),
            preserveQuery: true,
          }
        : { slotLabel: 'Optimizations' },
      { slotLabel: jobName },
    ]);
    return () => setBreadcrumbs([]);
  }, [setBreadcrumbs, workspace, agentName, jobName]);

  const { data: results, isLoading: isLoadingResults } = useQuery({
    queryKey: ['optimize-study-results', workspace, jobName] as const,
    queryFn: ({ signal }) => fetchStudyResults(workspace, jobName, signal),
    enabled: !!workspace && !!jobName && isTerminal && !hasFailed,
    refetchInterval: (query) => (query.state.data == null ? JOB_POLLING_INTERVAL_MS : false),
  });

  const {
    data: logs,
    isLoading: isLoadingLogs,
    loadProgress,
  } = useJobLogs({
    workspace,
    name: jobName,
    jobStatus: status,
    enabled: hasFailed,
  });

  if (isLoadingJob && !job) {
    return (
      <Flex align="center" justify="center" className="h-full w-full">
        <Spinner size="medium" aria-label="Loading optimization..." />
      </Flex>
    );
  }

  if (!job) {
    return (
      <Stack padding="density-2xl">
        <ErrorMessage
          header="Optimization not found"
          message={`No optimization study named "${jobName}" in workspace "${workspace}".`}
        />
      </Stack>
    );
  }

  const errorMessage =
    typeof job.error_details?.message === 'string' ? job.error_details.message : undefined;

  return (
    <AccessibleTitle title={`Optimization - ${jobName}`}>
      <Stack className="w-full p-density-2xl h-full min-h-0" gap="density-2xl">
        <PageHeader
          className="p-0 shrink-0"
          slotHeading={
            <Flex align="center" gap="3" wrap="wrap">
              <Text kind="title/md">{jobName}</Text>
              <StatusBadge status={job.status} />
              <Flex align="center" gap="2" wrap="wrap">
                {job.updated_at && isTerminal && (
                  <Text kind="body/regular/sm" className="text-secondary">
                    <RelativeTime datetime={job.updated_at} />
                  </Text>
                )}
              </Flex>
            </Flex>
          }
        />

        {hasFailed ? (
          <>
            <ErrorPanel errorMessage={errorMessage} />
            <Panel slotHeading="Logs" slotIcon={<ScrollText />} elevation="high" density="compact">
              <LogViewer
                logs={logs ?? []}
                isLoading={isLoadingLogs}
                loadProgress={loadProgress}
                downloadFilename={`optimize-${jobName}-logs.txt`}
              />
            </Panel>
          </>
        ) : !isTerminal ? (
          <Text kind="body/regular/md" className="text-secondary">
            Trials appear once the study finishes.
          </Text>
        ) : isLoadingResults || !results ? (
          <Flex align="center" justify="center" className="min-h-[200px] w-full">
            <Spinner size="medium" aria-label="Loading trials..." />
          </Flex>
        ) : (
          <>
            <StudyStatTiles results={results} />
            <TrialsTable results={results} />
          </>
        )}
      </Stack>
    </AccessibleTitle>
  );
};
