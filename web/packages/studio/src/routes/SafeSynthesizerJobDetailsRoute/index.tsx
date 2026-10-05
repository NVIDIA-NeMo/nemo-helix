// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { useJobLogs } from '@nemo/common/src/hooks/useJobLogs';
import { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import {
  useSafeSynthesizerDownloadJobResultSummary as useDownloadJobResultSummaryV1beta1GenerateJobsJobIdResultsSummaryDownloadGet,
  useSafeSynthesizerGetJobSuspense as useGetJobV1beta1GenerateJobsJobIdGetSuspense,
} from '@nemo/sdk/generated/safe-synthesizer/safe-synthesizer';
import { Grid, Stack } from '@nvidia/foundations-react-core';
import { SafeSynthesizerNavigation } from '@studio/components/SafeSynthesizerNavigation';
import { SAFE_SYNTHESIZER_ENABLED } from '@studio/constants/environment';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { JobDetailsPanel } from '@studio/routes/SafeSynthesizerJobDetailsRoute/components/JobDetailsPanel';
import { ProgressSection } from '@studio/routes/SafeSynthesizerJobDetailsRoute/components/ProgressSection';
import { ReportSummaryPanel } from '@studio/routes/SafeSynthesizerJobDetailsRoute/components/ReportSummaryPanel';
import { SAFE_SYNTHESIZER_POLLING_INTERVAL_MS } from '@studio/routes/SafeSynthesizerJobDetailsRoute/util';
import { getSafeSynthesizerRoute } from '@studio/routes/utils';
import { useRequiredPathParams } from '@studio/util/hooks/useRequiredPathParams';
import { isJobSuccessful, isJobTerminated } from '@studio/util/safeSynthesizer';
import { FC, useMemo } from 'react';

export const GenerateJobDetailsRoute: FC | null = SAFE_SYNTHESIZER_ENABLED
  ? () => {
      const workspace = useWorkspaceFromPath();

      const { safeSynthesizerJobName } = useRequiredPathParams([
        ROUTE_PARAMS.safeSynthesizerJobName,
      ]);
      const { data: job } = useGetJobV1beta1GenerateJobsJobIdGetSuspense(
        workspace,
        safeSynthesizerJobName,
        {
          query: {
            refetchInterval: (query) => {
              const latestJob = query.state.data;
              const shouldPoll = !isJobTerminated(latestJob?.status);
              return shouldPoll ? SAFE_SYNTHESIZER_POLLING_INTERVAL_MS : false;
            },
          },
        }
      );

      const isSuccessful = isJobSuccessful(job.status);

      const { data: jobResultSummary } =
        useDownloadJobResultSummaryV1beta1GenerateJobsJobIdResultsSummaryDownloadGet(
          workspace,
          safeSynthesizerJobName || '',
          {
            query: {
              enabled: !!safeSynthesizerJobName && isSuccessful,
            },
          }
        );

      const {
        data: logs,
        isLoading,
        loadProgress,
      } = useJobLogs({
        workspace,
        name: safeSynthesizerJobName,
        jobStatus: job.status,
      });

      // Extract error message from logs when job status is error
      const errorMessage = useMemo(() => {
        if (job.status !== HelixJobStatus.error || !logs || logs.length === 0) {
          return undefined;
        }

        // Search backwards through logs to find the first ERROR level log
        for (let i = logs.length - 1; i >= 0; i--) {
          const log = logs[i];

          try {
            const parsed = JSON.parse(log.message);
            if (parsed.level === 'ERROR' && parsed.message) {
              return parsed.message;
            }
          } catch {
            // Not JSON or invalid JSON, skip
          }
        }

        return undefined;
      }, [job.status, logs]);

      useBreadcrumbs({
        items: [
          {
            href: getSafeSynthesizerRoute(workspace),
            slotLabel: 'Safe Synthesizer',
          },
          {
            slotLabel: 'Job Details',
          },
        ],
      });

      return (
        <AccessibleTitle title={`Safe Synthesizer Job - ${safeSynthesizerJobName}`}>
          <Stack className="h-full w-full overflow-auto" gap="density-2xl" padding="density-2xl">
            <SafeSynthesizerNavigation selected="summary" jobName={safeSynthesizerJobName} />
            <Stack gap="density-2xl" className="w-full min-w-0">
              <Grid cols={{ base: 1, xl: 2 }} gap="density-2xl">
                <JobDetailsPanel job={job} errorMessage={errorMessage} />
                <ReportSummaryPanel
                  jobId={safeSynthesizerJobName}
                  jobResultSummary={jobResultSummary}
                />
              </Grid>
              <ProgressSection
                jobId={safeSynthesizerJobName}
                isLoading={isLoading}
                logs={logs}
                loadProgress={loadProgress}
              />
            </Stack>
          </Stack>
        </AccessibleTitle>
      );
    }
  : null;
