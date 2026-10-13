// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LogViewer } from '@nemo/common/src/components/LogViewer';
import { useJobLogs } from '@nemo/common/src/hooks/useJobLogs';
import type { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import { FC } from 'react';

interface StatusLogsContentProps {
  workspace: string;
  jobName: string;
  /** When provided, logs poll while the job runs and stop once it's terminal. */
  jobStatus?: HelixJobStatus;
}

export const StatusLogsContent: FC<StatusLogsContentProps> = ({
  workspace,
  jobName,
  jobStatus,
}) => {
  const {
    data: logs,
    isLoading,
    loadProgress,
  } = useJobLogs({
    workspace,
    name: jobName,
    jobStatus,
    enabled: !!jobName,
  });

  return (
    <LogViewer
      logs={logs}
      isLoading={isLoading}
      loadProgress={loadProgress}
      downloadFilename={`${jobName}-logs.txt`}
      emptyMessage="No status logs available for this job."
    />
  );
};
