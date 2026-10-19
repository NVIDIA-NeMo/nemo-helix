// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useJobLogs } from '@nemo/common/src/hooks/useJobLogs';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { triggerDownload } from '@nemo/common/src/utils/file';
import { formatLogs } from '@nemo/common/src/utils/logs';
import { useJobsPageJobLogs } from '@nemo/sdk/generated/platform/jobs';
import { Button, ButtonProps } from '@nvidia/foundations-react-core';
import { Download } from 'lucide-react';
import { FC, useCallback, useState } from 'react';

interface Props {
  workspace: string;
  jobName: string;
  compact?: boolean;
  size?: ButtonProps['size'];
  kind?: ButtonProps['kind'];
}

export const DownloadEvaluationLogsButton: FC<Props> = ({
  workspace,
  jobName,
  compact = false,
  size = undefined,
  kind = 'tertiary',
}) => {
  const toast = useToast();
  const [isDownloading, setIsDownloading] = useState(false);

  const { data: logsData } = useJobsPageJobLogs(
    workspace,
    jobName,
    { limit: 1 },
    {
      query: { enabled: !!jobName },
    }
  );
  const hasLogs = (logsData?.total ?? 0) > 0;

  const { refetch } = useJobLogs({ workspace, name: jobName, enabled: false, maxPages: Infinity });

  const handleDownloadLogs = useCallback(async () => {
    setIsDownloading(true);
    try {
      const { data } = await refetch();
      if (data?.logs) {
        triggerDownload(formatLogs(data.logs), `${jobName}-logs.txt`);
      }
    } catch (error) {
      toast.error(
        `Failed to download logs: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    } finally {
      setIsDownloading(false);
    }
  }, [refetch, jobName, toast]);

  if (!hasLogs) return null;

  return (
    <Button kind={kind} size={size} onClick={handleDownloadLogs} disabled={isDownloading}>
      <Download width="16px" height="16px" />
      {compact ? null : 'Download Logs'}
    </Button>
  );
};
