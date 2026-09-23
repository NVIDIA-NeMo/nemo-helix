// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { GradientBackground } from '@nemo/common/src/components/GradientBackground';
import { PageHeader, Stack } from '@nvidia/foundations-react-core';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { QuickstartSection } from '@studio/routes/WorkspaceDashboardHomeRoute/QuickstartSection';
import { StatTileRow } from '@studio/routes/WorkspaceDashboardHomeRoute/StatTileRow';
import { useRef, type FC } from 'react';

export const WorkspaceDashboardHomeRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const getStartedRef = useRef<HTMLDivElement>(null);

  useBreadcrumbs({
    items: [{ slotLabel: 'Dashboard' }],
  });

  return (
    <GradientBackground>
      <AccessibleTitle title="Dashboard">
        <Stack gap="density-3xl" padding="density-2xl" className="relative">
          <PageHeader slotHeading="Dashboard" />
          <Stack
            gap="density-3xl"
            data-tour="dashboard-get-started"
            ref={getStartedRef}
            tabIndex={-1}
          >
            <StatTileRow workspace={workspace} />
            <QuickstartSection
              workspace={workspace}
              onDismiss={() => getStartedRef.current?.focus()}
            />
          </Stack>
        </Stack>
      </AccessibleTitle>
    </GradientBackground>
  );
};
