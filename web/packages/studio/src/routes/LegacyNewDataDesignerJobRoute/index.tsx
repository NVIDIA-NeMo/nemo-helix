// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { PageHeader, Stack } from '@nvidia/foundations-react-core';
import { NewDataDesignerJobForm } from '@studio/components/NewDataDesignerJobForm';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { getDataDesignerJobListRoute } from '@studio/routes/utils';
import type { FC } from 'react';

/**
 * Legacy job-creation form, superseded by {@link CreateFilesetStart} and the DAG canvas
 * builder. Kept for now but not linked from any UI — reachable only by typing the URL.
 */
export const LegacyNewDataDesignerJobRoute: FC = () => {
  const workspace = useWorkspaceFromPath();

  useBreadcrumbs({
    items: [
      { href: getDataDesignerJobListRoute(workspace), slotLabel: 'Data Designer' },
      { slotLabel: 'New Job' },
    ],
  });

  return (
    <AccessibleTitle title="New Data Designer Job">
      <Stack className="h-full overflow-auto" gap="density-2xl" padding="density-2xl">
        <PageHeader
          slotHeading="New Data Designer Job"
          slotDescription="Describe what type of content you want to generate. A model will create the job specification for you."
        />
        <NewDataDesignerJobForm />
      </Stack>
    </AccessibleTitle>
  );
};
