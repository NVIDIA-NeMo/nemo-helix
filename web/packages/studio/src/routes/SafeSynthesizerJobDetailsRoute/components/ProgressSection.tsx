// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LogViewer } from '@nemo/common/src/components/LogViewer';
import type { LogLoadProgress } from '@nemo/common/src/utils/logs';
import type { HelixJobLog } from '@nemo/sdk/generated/platform/schema';
import { Panel } from '@nvidia/foundations-react-core';
import { FC } from 'react';

interface ProgressSectionProps {
  jobId: string;
  isLoading: boolean;
  logs: HelixJobLog[];
  loadProgress?: LogLoadProgress | null;
}

export const ProgressSection: FC<ProgressSectionProps> = ({
  jobId,
  isLoading,
  logs,
  loadProgress,
}) => {
  return (
    <Panel slotHeading="Progress" elevation="high" density="compact">
      <LogViewer
        logs={logs}
        isLoading={isLoading}
        loadProgress={loadProgress}
        downloadFilename={`job-${jobId}-logs.txt`}
      />
    </Panel>
  );
};
