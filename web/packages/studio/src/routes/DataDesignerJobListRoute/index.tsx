// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { Button, PageHeader, Stack } from '@nvidia/foundations-react-core';
import { DataDesignerJobsDataView } from '@studio/components/dataViews/DataDesignerJobsDataView';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { getNewDataDesignerJobRoute } from '@studio/routes/utils';
import type { FC } from 'react';
import { Link, Outlet } from 'react-router';

export const DataDesignerJobListRoute: FC = () => {
  const workspace = useWorkspaceFromPath();

  useBreadcrumbs({ items: [{ slotLabel: 'Data Designer' }] });

  return (
    <AccessibleTitle title="Data Designer">
      <Stack className="h-full overflow-auto" gap="density-2xl" padding="density-2xl">
        <PageHeader
          className="p-0"
          slotHeading="Data Designer"
          slotDescription="Create and manage data designer jobs to generate or transform datasets."
          slotActions={
            <Button asChild color="brand">
              <Link to={getNewDataDesignerJobRoute(workspace)}>New Job</Link>
            </Button>
          }
        />
        <DataDesignerJobsDataView />
      </Stack>
      <Outlet />
    </AccessibleTitle>
  );
};
