// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LogViewer } from '@nemo/common/src/components/LogViewer';
import { useJobLogs } from '@nemo/common/src/hooks/useJobLogs';
import type { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import { Flex } from '@nvidia/foundations-react-core';
import type { FC } from 'react';

interface Props {
  customizationJobName: string;
  workspace: string;
  jobStatus?: HelixJobStatus;
}

export const LogsTab: FC<Props> = ({ customizationJobName, workspace, jobStatus }) => {
  const {
    data: logs,
    isLoading,
    loadProgress,
  } = useJobLogs({
    workspace,
    name: customizationJobName,
    jobStatus,
  });

  return (
    <Flex className="min-h-0 w-full flex-1 pt-4">
      <LogViewer
        logs={logs ?? []}
        isLoading={isLoading}
        loadProgress={loadProgress}
        downloadFilename={`${customizationJobName}.log`}
        fillHeight
      />
    </Flex>
  );
};
