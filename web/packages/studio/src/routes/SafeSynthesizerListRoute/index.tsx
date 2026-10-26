// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { Button, PageHeader, Stack } from '@nvidia/foundations-react-core';
import { GenerateJobsDataView } from '@studio/components/dataViews/SafeSynthesizerJobsDataView';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { getNewSafeSynthesizerRoute } from '@studio/routes/utils';
import { FC } from 'react';
import { Link, Outlet } from 'react-router';

export const SafeSynthesizerListRoute: FC = () => {
  const workspace = useWorkspaceFromPath();

  useBreadcrumbs({ items: [{ slotLabel: 'Safe Synthesizer' }] });

  return (
    <AccessibleTitle title="Safe Synthesizer">
      <Stack className="h-full" gap="density-2xl" padding="density-2xl">
        <PageHeader
          className="p-0"
          slotHeading="Safe Synthesizer"
          slotDescription="Create and monitor safe, synthetic data jobs for fine-tuning, sharing, and analysis."
          slotActions={
            <Button asChild color="brand">
              <Link to={getNewSafeSynthesizerRoute(workspace)}>Synthesize Data</Link>
            </Button>
          }
        />
        <GenerateJobsDataView />
      </Stack>
      <Outlet />
    </AccessibleTitle>
  );
};
