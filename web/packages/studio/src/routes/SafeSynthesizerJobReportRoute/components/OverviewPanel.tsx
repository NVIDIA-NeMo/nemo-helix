// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { triggerDownload } from '@nemo/common/src/utils/file';
import {
  useSafeSynthesizerDownloadJobResultEvaluationReport as useDownloadJobResultEvaluationReportV1beta1GenerateJobsJobIdResultsEvaluationReportDownloadGet,
  useSafeSynthesizerGetJobSuspense as useGetJobV1beta1GenerateJobsJobIdGetSuspense,
} from '@nemo/sdk/generated/safe-synthesizer/safe-synthesizer';
import { Button, Panel } from '@nvidia/foundations-react-core';
import { SafeSynthesizerFilesetPreview } from '@studio/components/SafeSynthesizerFilesetPreview';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { isJobSuccessful } from '@studio/util/safeSynthesizer';
import { Download } from 'lucide-react';
import { FC } from 'react';

interface OverviewPanelProps {
  jobId: string;
  title: string;
  icon: React.ReactNode;
}

export const OverviewPanel: FC<OverviewPanelProps> = ({ jobId, title, icon }) => {
  const workspace = useWorkspaceFromPath();
  const { data: job } = useGetJobV1beta1GenerateJobsJobIdGetSuspense(workspace, jobId);

  const isSuccessful = isJobSuccessful(job.status);

  const { data: report } =
    useDownloadJobResultEvaluationReportV1beta1GenerateJobsJobIdResultsEvaluationReportDownloadGet(
      workspace,
      jobId,
      {
        query: {
          enabled: isSuccessful,
        },
      }
    );

  const handleDownloadReport = () => {
    if (!report || !job) return;
    triggerDownload(report, `${job.name}-report.html`);
  };

  return (
    <Panel slotHeading={title} slotIcon={icon} elevation="high" density="standard">
      <SafeSynthesizerFilesetPreview job={job} />
      {report && (
        <Button className="mt-density-xl" color="brand" onClick={handleDownloadReport}>
          <Download />
          Download Report
        </Button>
      )}
    </Panel>
  );
};
