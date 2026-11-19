// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PageHeader, Stack, TabsList, TabsRoot, TabsTrigger } from '@nvidia/foundations-react-core';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { getGenerateJobRoute, getGenerateJobReportRoute } from '@studio/routes/utils';
import type { FC } from 'react';
import { useNavigate } from 'react-router';

interface SafeSynthesizerNavigationProps {
  selected: 'summary' | 'report';
  jobName: string;
}

export const SafeSynthesizerNavigation: FC<SafeSynthesizerNavigationProps> = ({
  selected,
  jobName,
}) => {
  const workspace = useWorkspaceFromPath();
  const navigate = useNavigate();

  return (
    <Stack gap="density-2xl">
      <PageHeader slotHeading="Safe Synthesizer Job" />
      <TabsRoot value={selected}>
        <TabsList>
          <TabsTrigger
            value="summary"
            onClick={() => navigate(getGenerateJobRoute(workspace, jobName))}
          >
            Summary
          </TabsTrigger>
          <TabsTrigger
            value="report"
            onClick={() => navigate(getGenerateJobReportRoute(workspace, jobName))}
          >
            Report
          </TabsTrigger>
        </TabsList>
      </TabsRoot>
    </Stack>
  );
};
