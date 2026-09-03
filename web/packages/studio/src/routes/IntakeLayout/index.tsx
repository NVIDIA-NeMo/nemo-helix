// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { Button, PageHeader, Stack, Tabs } from '@nvidia/foundations-react-core';
import { FeatureFlagBadge } from '@studio/components/FeatureFlagBadge';
import { ImportTracesModal } from '@studio/components/ImportTracesModal';
import { Loading } from '@studio/components/Layouts/Loading';
import { ROUTES } from '@studio/constants/routes';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { getIntakeSpansRoute, getIntakeTracesRoute } from '@studio/routes/utils';
import { Upload } from 'lucide-react';
import { FC, Suspense, useState } from 'react';
import { Link, Outlet, matchPath, useLocation } from 'react-router';

export const INTAKE_FILTER_ACTION_TARGET_ID = 'intake-filter-action-target';

/**
 * Layout component for the Intake section.
 * Provides shared header and tab navigation for trace and span telemetry views.
 * Child routes are rendered via <Outlet />.
 */
export const IntakeLayout: FC = () => {
  const workspace = useWorkspaceFromPath();
  const [importOpen, setImportOpen] = useState(false);

  const location = useLocation();
  const match = matchPath(
    { path: `${ROUTES.workspace.intake}/:selectedTab`, end: false },
    location.pathname
  );
  const {
    params: { selectedTab },
  } = match ?? { params: { selectedTab: 'traces' } };

  const tracesRoute = getIntakeTracesRoute(workspace);
  const spansRoute = getIntakeSpansRoute(workspace);

  useBreadcrumbs({
    items: [
      {
        slotLabel: 'Traces',
      },
    ],
  });

  return (
    <AccessibleTitle title="Traces">
      <Stack gap="density-2xl" padding="density-2xl" className="h-full">
        <PageHeader
          className="p-0"
          slotHeading={
            <>
              Trace explorer
              <FeatureFlagBadge flag="intakeEnabled" />
            </>
          }
          slotActions={
            <Button kind="secondary" onClick={() => setImportOpen(true)}>
              <Upload />
              Import traces
            </Button>
          }
        />
        <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-density-lg">
          <Tabs
            // Override KUI's default overflow:hidden since we're using Tabs purely for
            // navigation (with renderLink), not for containing tab panel content.
            className="min-w-0 flex-1 overflow-visible"
            value={selectedTab}
            items={[
              { value: 'traces', children: 'Trace runs', href: tracesRoute },
              { value: 'spans', children: 'Spans', href: spansRoute },
            ]}
            renderLink={(item) => <Link to={item.href!}>{item.children}</Link>}
          />
          <div
            id={INTAKE_FILTER_ACTION_TARGET_ID}
            className="flex shrink-0 items-center justify-end gap-density-xl"
          />
        </div>
        <Suspense fallback={<Loading description="Loading..." />}>
          <Outlet />
        </Suspense>
        <ImportTracesModal
          open={importOpen}
          onClose={() => setImportOpen(false)}
          workspace={workspace}
        />
      </Stack>
    </AccessibleTitle>
  );
};
