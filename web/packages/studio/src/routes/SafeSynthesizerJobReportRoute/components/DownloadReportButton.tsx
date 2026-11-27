// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { triggerDownload } from '@nemo/common/src/utils/file';
import { Download } from '@nemo/icons/icons';
import { useSafeSynthesizerDownloadJobResultEvaluationReport } from '@nemo/sdk/generated/safe-synthesizer/safe-synthesizer';
import { Button } from '@nvidia/foundations-react-core';
import type { FC } from 'react';

interface DownloadReportButtonProps {
  workspace: string;
  jobName: string;
  enabled: boolean;
}

/** Downloads the library's HTML report; hidden until the job has one. */
export const DownloadReportButton: FC<DownloadReportButtonProps> = ({
  workspace,
  jobName,
  enabled,
}) => {
  const { data: report } = useSafeSynthesizerDownloadJobResultEvaluationReport(workspace, jobName, {
    query: { enabled },
  });
  if (!report) {
    return null;
  }
  return (
    <Button color="brand" onClick={() => triggerDownload(report, `${jobName}-report.html`)}>
      <Download />
      Download Report
    </Button>
  );
};
