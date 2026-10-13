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
import {
  useAgentOptimizationGetRunStrategyJob,
  useAgentOptimizationListRunStrategyJobResults,
} from '@nemo/sdk/generated/agent-optimization/agent-optimization';
import type { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import {
  Banner,
  Button,
  Flex,
  PageHeader,
  Panel,
  Spinner,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import { TrialsDataView } from '@studio/components/dataViews/OptimizationJobsDataView';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { DeployTrialModal } from '@studio/routes/agents/AgentOptimizationDetailRoute/DeployTrialModal';
import { StudyInProgress } from '@studio/routes/agents/AgentOptimizationDetailRoute/StudyInProgress';
import {
  fetchStudyResults,
  type Trial,
} from '@studio/routes/agents/AgentOptimizationDetailRoute/studyResults';
import { StudyStatTiles } from '@studio/routes/agents/AgentOptimizationDetailRoute/StudyStatTiles';
import { ArtifactFilesPanel } from '@studio/routes/JobDetailRoute/components/ArtifactFilesPanel';
import { getAgentOptimizationsTabRoute, getAgentsListRoute } from '@studio/routes/utils';
import { useRequiredPathParams } from '@studio/util/hooks/useRequiredPathParams';
import { useQuery } from '@tanstack/react-query';
import { FolderOpen, ScrollText } from 'lucide-react';
import { type FC, useCallback, useEffect, useState } from 'react';

/** Statuses that will not change again, so polling can stop. */
const TERMINAL_STATUSES = new Set<HelixJobStatus>(['completed', 'error', 'cancelled']);
const FAILED_STATUSES = new Set<HelixJobStatus>(['error', 'cancelled']);
const QUEUED_STATUSES = new Set<HelixJobStatus>(['created', 'pending']);

export const AgentOptimizationDetailRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const { optimizeJobName: jobName } = useRequiredPathParams([ROUTE_PARAMS.optimizeJobName]);
  const jobKey = `${workspace}/${jobName}`;
  const [deployTarget, setDeployTarget] = useState<{ jobKey: string; trial: Trial } | null>(null);
  const deployTrial = deployTarget?.jobKey === jobKey ? deployTarget.trial : null;
  const onDeploy = useCallback((trial: Trial) => setDeployTarget({ jobKey, trial }), [jobKey]);

  const {
    data: job,
    isLoading: isLoadingJob,
    error: jobError,
  } = useAgentOptimizationGetRunStrategyJob(workspace, jobName, {
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
  const isQueued = status ? QUEUED_STATUSES.has(status) : false;
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

  const {
    data: results,
    isLoading: isLoadingResults,
    isError: isResultsError,
    error: resultsError,
  } = useQuery({
    queryKey: ['optimize-study-results', workspace, jobName] as const,
    queryFn: ({ signal }) => fetchStudyResults(workspace, jobName, signal),
    enabled: !!workspace && !!jobName && isTerminal && !hasFailed,
    refetchInterval: (query) => (query.state.data === undefined ? JOB_POLLING_INTERVAL_MS : false),
  });

  const {
    data: logs,
    isLoading: isLoadingLogs,
    error: logsError,
    loadProgress,
    refetch: refetchLogs,
  } = useJobLogs({
    workspace,
    name: jobName,
    jobStatus: status,
    enabled: !!job,
  });

  const {
    data: artifacts,
    isLoading: isLoadingArtifacts,
    error: artifactsError,
  } = useAgentOptimizationListRunStrategyJobResults(workspace, jobName, {
    query: {
      queryKey: ['optimization-artifacts', workspace, jobName, status],
      enabled: !!job,
      refetchInterval: isTerminal ? false : JOB_POLLING_INTERVAL_MS,
    },
  });

  if (isLoadingJob && !job) {
    return (
      <Flex align="center" justify="center" className="h-full w-full">
        <Spinner size="medium" aria-label="Loading optimization..." />
      </Flex>
    );
  }

  if (!job && jobError && jobError.response?.status !== 404) {
    return (
      <Stack padding="density-2xl">
        <ErrorMessage header="Could not load optimization" message={jobError.message} />
      </Stack>
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
      <Stack className="w-full p-density-2xl min-h-full" gap="density-2xl">
        <PageHeader
          className="p-0 shrink-0"
          slotHeading={
            <Flex align="center" gap="3" wrap="wrap">
              <Text kind="title/md">{jobName}</Text>
              <StatusBadge status={job.status} />
              {!isTerminal && (
                <Spinner size="small" aria-label={isQueued ? 'Study queued' : 'Study running'} />
              )}
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
          <ErrorPanel errorMessage={errorMessage} />
        ) : !isTerminal ? (
          <StudyInProgress workspace={workspace} job={job} isQueued={isQueued} />
        ) : isResultsError ? (
          <ErrorMessage
            header="Could not load trials"
            message={resultsError.message}
            height="auto"
          />
        ) : isLoadingResults ? (
          <Flex align="center" justify="center" className="min-h-[200px] w-full">
            <Spinner size="medium" aria-label="Loading trials..." />
          </Flex>
        ) : !results ? (
          <Text kind="body/regular/md" className="text-secondary">
            This study did not register any trial results.
          </Text>
        ) : (
          <>
            <StudyStatTiles results={results} />
            <TrialsDataView results={results} onDeploy={onDeploy} />
            <DeployTrialModal
              workspace={workspace}
              job={job}
              trial={deployTrial}
              onClose={() => setDeployTarget(null)}
            />
          </>
        )}
        <Panel slotHeading="Logs" slotIcon={<ScrollText />} elevation="high" density="compact">
          {logsError && logs.length === 0 ? (
            <Banner
              kind="inline"
              status="error"
              slotActions={
                <Button kind="secondary" size="small" onClick={() => void refetchLogs()}>
                  Retry
                </Button>
              }
            >
              Could not load logs for this study.
            </Banner>
          ) : (
            <LogViewer
              logs={logs}
              isLoading={isLoadingLogs && logs.length === 0}
              loadProgress={loadProgress}
              downloadFilename={`optimize-${jobName}-logs.txt`}
              emptyMessage={
                isQueued
                  ? 'Logs appear once the study starts.'
                  : !isTerminal
                    ? 'Waiting for the study to emit its first log lines...'
                    : undefined
              }
            />
          )}
        </Panel>
        <Panel slotHeading="Artifacts" slotIcon={<FolderOpen />} elevation="high" density="compact">
          {artifactsError ? (
            <ErrorMessage
              header="Could not load artifacts"
              message={artifactsError.message}
              height="auto"
            />
          ) : (
            <ArtifactFilesPanel
              workspace={workspace}
              results={artifacts?.data ?? []}
              isLoading={isLoadingArtifacts}
              jobStatus={status}
            />
          )}
        </Panel>
      </Stack>
    </AccessibleTitle>
  );
};
