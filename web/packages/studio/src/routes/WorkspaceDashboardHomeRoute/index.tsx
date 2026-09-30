// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { GradientBackground } from '@nemo/common/src/components/GradientBackground';
import { Banner, PageHeader, Stack, Text } from '@nvidia/foundations-react-core';
import { AGENTS_ENABLED } from '@studio/constants/environment';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { QuickstartSamplePanel } from '@studio/routes/WorkspaceDashboardHomeRoute/QuickstartSamplePanel';
import { QuickstartSection } from '@studio/routes/WorkspaceDashboardHomeRoute/QuickstartSection';
import { StatTileRow } from '@studio/routes/WorkspaceDashboardHomeRoute/StatTileRow';
import {
  SAMPLE_WORKSPACE,
  useSampleQuickstartAgent,
} from '@studio/routes/WorkspaceDashboardHomeRoute/useSampleQuickstartAgent';
import { TriangleAlert } from 'lucide-react';
import { useRef, type FC } from 'react';

const SANDBOX_WORKSPACE = 'sample';

export const WorkspaceDashboardHomeRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const getStartedRef = useRef<HTMLDivElement>(null);
  const isSampleWorkspace = workspace === SAMPLE_WORKSPACE;
  // The panel returns null without an agent or with agents disabled, so gating the fetch on
  // AGENTS_ENABLED also keeps the heading below from outliving it.
  const sampleAgent = useSampleQuickstartAgent(workspace, isSampleWorkspace && AGENTS_ENABLED);

  useBreadcrumbs({
    items: [{ slotLabel: 'Dashboard' }],
  });

  return (
    <GradientBackground>
      {workspace === SANDBOX_WORKSPACE && (
        <Banner kind="global" status="warning" slotIcon={<TriangleAlert role="img" aria-hidden />}>
          Sample Sandbox. This sandbox can be reset at any time with nemo CLI.
        </Banner>
      )}
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
            {isSampleWorkspace ? (
              sampleAgent && (
                <Stack gap="density-lg">
                  <Text kind="title/md">Quickstart</Text>
                  <Text kind="body/regular/sm" className="text-secondary">
                    A complete sample workload, already run end to end. Inspect what shipped, or run
                    any step yourself.
                  </Text>
                  <QuickstartSamplePanel workspace={workspace} agent={sampleAgent} />
                </Stack>
              )
            ) : (
              <QuickstartSection
                workspace={workspace}
                onDismiss={() => getStartedRef.current?.focus()}
              />
            )}
          </Stack>
        </Stack>
      </AccessibleTitle>
    </GradientBackground>
  );
};
