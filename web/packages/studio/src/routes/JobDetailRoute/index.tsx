// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { CancelJobButton } from '@nemo/common/src/components/CancelJobButton';
import { KVPair } from '@nemo/common/src/components/KVPair';
import { LogViewer } from '@nemo/common/src/components/LogViewer';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import { useJobLogs } from '@nemo/common/src/hooks/useJobLogs';
import { getJobRefetchInterval } from '@nemo/common/src/utils/query';
import { IconClipboard, FolderOpen, IconDocument } from '@nemo/icons/icons';
import { useJobsGetJobSuspense, useJobsListJobResults } from '@nemo/sdk/generated/platform/jobs';
import { Flex, Grid, PageHeader, Panel, Stack } from '@nvidia/foundations-react-core';
import { EvalRunOutcomeBanner } from '@studio/components/evaluation/EvalRunOutcomeBanner';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { ArtifactFilesPanel } from '@studio/routes/JobDetailRoute/components/ArtifactFilesPanel';
import { getWorkspaceJobsRoute } from '@studio/routes/utils';
import { useRequiredPathParams } from '@studio/util/hooks/useRequiredPathParams';
import { FC } from 'react';

export const JobDetailRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const { jobName } = useRequiredPathParams([ROUTE_PARAMS.jobName]);

  const { data: job } = useJobsGetJobSuspense(workspace, jobName, {
    query: {
      refetchInterval: (query) => getJobRefetchInterval(query.state.data?.status),
    },
  });

  const {
    data: logs,
    isLoading: isLoadingLogs,
    loadProgress,
    logStatus,
  } = useJobLogs({
    workspace,
    name: jobName,
    jobStatus: job.status,
  });

  const { data: resultsData, isLoading: isLoadingResults } = useJobsListJobResults(
    workspace,
    jobName
  );

  useBreadcrumbs({
    items: [{ href: getWorkspaceJobsRoute(workspace), slotLabel: 'Jobs' }, { slotLabel: jobName }],
  });

  const results = resultsData?.data ?? [];

  return (
    <AccessibleTitle title={`Job - ${jobName}`}>
      <Stack className="w-full p-density-2xl min-h-full" gap="density-2xl">
        <PageHeader
          slotHeading={jobName}
          slotActions={
            <Flex gap="density-md" align="center">
              <CancelJobButton workspace={workspace} jobName={jobName} jobStatus={job.status} />
            </Flex>
          }
        />

        <EvalRunOutcomeBanner statusDetails={job.status_details} />

        <Grid cols={{ base: 1, xl: 2 }} gap="density-2xl">
          <Panel
            slotHeading="Job Details"
            slotIcon={<IconClipboard />}
            elevation="high"
            density="compact"
          >
            <Stack gap="density-xl">
              <KVPair label="Name" value={job.name} />
              <KVPair label="ID" value={job.id} />
              <KVPair label="Source" value={job.source ?? ''} />
              <KVPair
                label="Status"
                value={job.status ? <StatusBadge status={job.status} /> : '-'}
              />
              <KVPair label="Description" value={job.description || '-'} />
              <KVPair label="Workspace" value={job.workspace ?? ''} />
              <KVPair
                label="Created"
                value={job.created_at ? <RelativeTime datetime={job.created_at} /> : '-'}
              />
              <KVPair
                label="Updated"
                value={job.updated_at ? <RelativeTime datetime={job.updated_at} /> : '-'}
              />
            </Stack>
          </Panel>

          <Panel
            slotHeading="Artifacts"
            slotIcon={<FolderOpen />}
            elevation="high"
            density="compact"
          >
            <ArtifactFilesPanel
              workspace={workspace}
              results={results}
              isLoading={isLoadingResults}
              jobStatus={job.status}
            />
          </Panel>
        </Grid>

        <Panel
          slotHeading="Progress"
          slotIcon={<IconDocument />}
          elevation="high"
          density="compact"
        >
          <LogViewer
            logs={logs ?? []}
            isLoading={isLoadingLogs}
            loadProgress={loadProgress}
            logStatus={logStatus}
            downloadFilename={`job-${jobName}-logs.txt`}
          />
        </Panel>
      </Stack>
    </AccessibleTitle>
  );
};
