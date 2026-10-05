// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LogViewer } from '@nemo/common/src/components/LogViewer';
import { HelixJobTerminalStatuses } from '@nemo/common/src/constants/query';
import { useJobLogs } from '@nemo/common/src/hooks/useJobLogs';
import { Banner, Card, Flex, Spinner, Stack, Text } from '@nvidia/foundations-react-core';
import { useDataDesignerJobFromRoute } from '@studio/routes/DataDesignerJobDetailsRoute/useDataDesignerJobFromRoute';
import type { FC } from 'react';

export const JobLogsSection: FC = () => {
  const { workspace, jobName, job } = useDataDesignerJobFromRoute();

  const isRunning = !(job?.status != null && HelixJobTerminalStatuses.includes(job.status));

  const {
    data: logs,
    isLoading,
    error,
    loadProgress,
  } = useJobLogs({
    workspace,
    name: jobName,
    jobStatus: job?.status,
  });

  return (
    <Card className="w-full">
      <Stack gap="density-lg" className="w-full min-w-0">
        <Flex gap="density-md" align="center" justify="between" className="flex-wrap">
          <Text kind="body/bold/md">Job logs</Text>
          {isRunning && (
            <Flex gap="density-sm" align="center">
              <Spinner size="small" aria-label="Job running" />
            </Flex>
          )}
        </Flex>

        {error ? (
          <Banner kind="inline" status="error">
            Could not load logs for this job.
          </Banner>
        ) : (
          <LogViewer
            logs={logs}
            isLoading={isLoading && logs.length === 0}
            loadProgress={loadProgress}
            downloadFilename={`data-designer-${jobName}-logs.txt`}
            emptyMessage={
              isRunning
                ? 'Waiting for the job to emit its first log lines...'
                : 'No logs available for this job'
            }
          />
        )}
      </Stack>
    </Card>
  );
};
