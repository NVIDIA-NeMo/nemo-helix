// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { LogViewer } from '@nemo/common/src/components/LogViewer';
import {
  useAnonymizerGetRunJob,
  useAnonymizerGetRunJobLogs,
  useAnonymizerListRunJobResults,
} from '@nemo/sdk/generated/anonymizer/anonymizer';
import { Banner, Grid, PageHeader, Panel, Stack } from '@nvidia/foundations-react-core';
import { AnonymizerJobActionsMenu } from '@studio/components/AnonymizerJobActionsMenu';
import { ANONYMIZER_ENABLED } from '@studio/constants/environment';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { FailedRecordsPanel } from '@studio/routes/AnonymizerJobDetailRoute/components/FailedRecordsPanel';
import { JobSummaryPanel } from '@studio/routes/AnonymizerJobDetailRoute/components/JobSummaryPanel';
import { ResultsPanel } from '@studio/routes/AnonymizerJobDetailRoute/components/ResultsPanel';
import { ResultsPreviewPanel } from '@studio/routes/AnonymizerJobDetailRoute/components/ResultsPreviewPanel';
import { ANONYMIZER_POLLING_INTERVAL_MS } from '@studio/routes/AnonymizerJobDetailRoute/util';
import { getWorkspaceAnonymizerRoute } from '@studio/routes/utils';
import { useRequiredPathParams } from '@studio/util/hooks/useRequiredPathParams';
import { isJobTerminated } from '@studio/util/safeSynthesizer';
import type { FC } from 'react';
import { useNavigate } from 'react-router';

export const AnonymizerJobDetailRoute: FC | null = ANONYMIZER_ENABLED
  ? () => {
      const workspace = useWorkspaceFromPath();
      const navigate = useNavigate();
      const { anonymizerJobName } = useRequiredPathParams([ROUTE_PARAMS.anonymizerJobName]);

      const {
        data: job,
        isLoading: isLoadingJob,
        error,
      } = useAnonymizerGetRunJob(workspace, anonymizerJobName, {
        query: {
          refetchInterval: (query) =>
            isJobTerminated(query.state.data?.status) ? false : ANONYMIZER_POLLING_INTERVAL_MS,
        },
      });

      const isTerminal = isJobTerminated(job?.status);

      const {
        data: results,
        isLoading: isLoadingResults,
        error: resultsError,
      } = useAnonymizerListRunJobResults(workspace, anonymizerJobName, {
        query: { enabled: isTerminal },
      });

      const {
        data: logs,
        isLoading: isLoadingLogs,
        error: logsError,
      } = useAnonymizerGetRunJobLogs(workspace, anonymizerJobName, undefined, {
        query: { refetchInterval: isTerminal ? false : ANONYMIZER_POLLING_INTERVAL_MS },
      });

      const artifactUrl = results?.data?.[0]?.artifact_url;

      useBreadcrumbs({
        items: [
          { href: getWorkspaceAnonymizerRoute(workspace), slotLabel: 'Anonymizer' },
          { slotLabel: anonymizerJobName },
        ],
      });

      return (
        <AccessibleTitle title={`Anonymizer Job - ${anonymizerJobName}`}>
          <Stack className="h-full w-full overflow-auto" gap="density-2xl" padding="density-2xl">
            <PageHeader
              className="p-0"
              slotHeading={anonymizerJobName}
              slotActions={
                job ? (
                  <AnonymizerJobActionsMenu
                    job={job}
                    onDeleted={() => navigate(getWorkspaceAnonymizerRoute(workspace))}
                  />
                ) : null
              }
            />

            {error ? (
              <Banner kind="inline" status="error">
                Could not load this job.
              </Banner>
            ) : (
              <Stack gap="density-2xl" className="w-full min-w-0">
                <Grid cols={{ base: 1, xl: 2 }} gap="density-2xl">
                  {job && !isLoadingJob ? <JobSummaryPanel job={job} /> : null}
                  <ResultsPanel
                    workspace={workspace}
                    jobName={anonymizerJobName}
                    results={results?.data ?? []}
                    isLoading={isLoadingResults}
                    isTerminal={isTerminal}
                    loadError={!!resultsError}
                  />
                </Grid>
                {isTerminal && artifactUrl ? (
                  <>
                    <ResultsPreviewPanel workspace={workspace} artifactUrl={artifactUrl} />
                    <FailedRecordsPanel workspace={workspace} artifactUrl={artifactUrl} />
                  </>
                ) : null}

                <Panel slotHeading="Logs" elevation="high" density="compact">
                  {logsError ? (
                    <Banner kind="inline" status="error">
                      Could not load logs for this job.
                    </Banner>
                  ) : (
                    <LogViewer
                      logs={logs?.data ?? []}
                      isLoading={isLoadingLogs}
                      downloadFilename={`anonymizer-${anonymizerJobName}-logs.txt`}
                    />
                  )}
                </Panel>
              </Stack>
            )}
          </Stack>
        </AccessibleTitle>
      );
    }
  : null;
