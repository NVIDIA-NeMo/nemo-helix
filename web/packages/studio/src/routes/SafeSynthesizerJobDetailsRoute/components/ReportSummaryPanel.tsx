// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ScoreGauge } from '@nemo/common/src/components/ScoreGauge';
import { SafeSynthesizerSummary } from '@nemo/sdk/generated/safe-synthesizer/schema';
import { Button, Flex, Panel, Stack, Text } from '@nvidia/foundations-react-core';
import { IconDocument } from '@nvidia/react-gui-icons/icons';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { getGenerateJobReportRoute } from '@studio/routes/utils';
import { FC } from 'react';
import { useNavigate } from 'react-router';

interface ReportSummaryPanelProps {
  jobId: string;
  jobResultSummary?: SafeSynthesizerSummary;
}

export const ReportSummaryPanel: FC<ReportSummaryPanelProps> = ({ jobId, jobResultSummary }) => {
  const navigate = useNavigate();
  const workspace = useWorkspaceFromPath();

  const sqsScore = jobResultSummary?.synthetic_data_quality_score;
  const dpsScore = jobResultSummary?.data_privacy_score;

  return (
    <Panel
      slotHeading="Report Summary"
      slotIcon={<IconDocument />}
      elevation="high"
      density="compact"
    >
      <Stack gap="density-2xl" padding="density-xl" justify="between" className="overflow-hidden">
        <Flex gap="density-md" align="center" justify="around" className="w-full h-full">
          <Stack align="center" justify="center" gap="density-lg">
            <Text kind="body/semibold/md">Quality (SQS)</Text>
            <ScoreGauge score={sqsScore} size="lg" scaleToFit />
          </Stack>
          <Stack align="center" justify="center" gap="density-lg">
            <Text kind="body/semibold/md">Privacy (DPS)</Text>
            <ScoreGauge score={dpsScore} size="lg" scaleToFit />
          </Stack>
        </Flex>
        {jobResultSummary && (
          <Flex justify="end">
            <Button
              kind="secondary"
              onClick={() => navigate(getGenerateJobReportRoute(workspace, jobId))}
            >
              View Report
            </Button>
          </Flex>
        )}
      </Stack>
    </Panel>
  );
};
