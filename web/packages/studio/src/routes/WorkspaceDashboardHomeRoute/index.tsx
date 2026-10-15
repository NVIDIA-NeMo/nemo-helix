// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { GradientBackground } from '@nemo/common/src/components/GradientBackground';
import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import { Banner, PageHeader, Stack, Text } from '@nvidia/foundations-react-core';
import { AGENTS_ENABLED } from '@studio/constants/environment';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { getWorkspaceDetailsDefaultRoute } from '@studio/routes/utils';
import { QuickstartSamplePanel } from '@studio/routes/WorkspaceDashboardHomeRoute/QuickstartSamplePanel';
import { QuickstartSamplePanelError } from '@studio/routes/WorkspaceDashboardHomeRoute/QuickstartSamplePanel/QuickstartSamplePanelError';
import { QuickstartSamplePanelSkeleton } from '@studio/routes/WorkspaceDashboardHomeRoute/QuickstartSamplePanel/QuickstartSamplePanelSkeleton';
import { QuickstartSection } from '@studio/routes/WorkspaceDashboardHomeRoute/QuickstartSection';
import { StatTileRow } from '@studio/routes/WorkspaceDashboardHomeRoute/StatTileRow';
import {
  isSampleWorkspace,
  useSampleQuickstartAgent,
} from '@studio/routes/WorkspaceDashboardHomeRoute/useSampleQuickstartAgent';
import { TriangleAlert } from 'lucide-react';
import { useRef, type FC } from 'react';
import { useNavigate } from 'react-router';

export const WorkspaceDashboardHomeRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const navigate = useNavigate();
  const getStartedRef = useRef<HTMLDivElement>(null);
  const isSample = isSampleWorkspace(workspace);
  // The panel returns null with agents disabled, so gating the fetch on AGENTS_ENABLED sends
  // that case to the regular Quickstart rather than leaving the heading below with no panel.
  const sample = useSampleQuickstartAgent(workspace, isSample && AGENTS_ENABLED);

  useBreadcrumbs({
    items: [{ slotLabel: 'Dashboard' }],
  });

  return (
    <GradientBackground>
      {isSample && (
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
            {/* Every state but `disabled` is a sample workspace, so the panel's place is held
                by a skeleton while loading and by an error if the lookup fails. */}
            {sample.state !== 'disabled' && (
              <Stack gap="density-lg">
                <Text kind="title/md">Quickstart</Text>
                <Text kind="body/regular/sm" className="text-secondary">
                  A sample workload with an agent and dataset already loaded. Inspect what shipped,
                  or run any step yourself.
                </Text>
                {sample.state === 'ready' && (
                  <QuickstartSamplePanel
                    workspace={workspace}
                    agent={sample.agent}
                    // `default` is the workspace every user shares; the sample is a sandbox.
                    onSwitchWorkspace={() =>
                      navigate(getWorkspaceDetailsDefaultRoute(DEFAULT_WORKSPACE))
                    }
                  />
                )}
                {sample.state === 'loading' && <QuickstartSamplePanelSkeleton />}
                {sample.state === 'error' && <QuickstartSamplePanelError onRetry={sample.retry} />}
              </Stack>
            )}
            {/* Never a fallback for a failed sample lookup, which shows its own error above. */}
            {sample.state === 'disabled' && (
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
