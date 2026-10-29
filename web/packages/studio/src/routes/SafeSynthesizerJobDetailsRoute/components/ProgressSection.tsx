// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LogViewer } from '@nemo/common/src/components/LogViewer';
import type { LogLoadProgress, LogStatus } from '@nemo/common/src/utils/logs';
import type { HelixJobLog } from '@nemo/sdk/generated/platform/schema';
import { Panel } from '@nvidia/foundations-react-core';
import { FC } from 'react';

interface ProgressSectionProps {
  jobId: string;
  isLoading: boolean;
  logs: HelixJobLog[];
  loadProgress?: LogLoadProgress | null;
  logStatus?: LogStatus;
}

export const ProgressSection: FC<ProgressSectionProps> = ({
  jobId,
  isLoading,
  logs,
  loadProgress,
  logStatus,
}) => {
  return (
    <Panel slotHeading="Progress" elevation="high" density="compact">
      <LogViewer
        logs={logs}
        isLoading={isLoading}
        loadProgress={loadProgress}
        logStatus={logStatus}
        downloadFilename={`job-${jobId}-logs.txt`}
      />
    </Panel>
  );
};
