// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { PageHeader, Stack, Tabs } from '@nvidia/foundations-react-core';
import { SuspenseOutlet } from '@studio/components/Layouts/SuspenseOutlet';
import { ROUTE_PARAMS, ROUTES } from '@studio/constants/routes';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import {
  getVirtualModelChatRoute,
  getVirtualModelDetailsRoute,
  getWorkspaceVirtualModelsRoute,
} from '@studio/routes/utils';
import { useRequiredPathParams } from '@studio/util/hooks/useRequiredPathParams';
import { type FC } from 'react';
import { Link, matchPath, useLocation } from 'react-router';

/**
 * Detail page for a single virtual model. Replaces the former details side panel: the name lives
 * in the path, so heading and tabs render without waiting on a request and there is no dialog
 * open/close state to keep in sync with the URL.
 */
export const VirtualModelDetailRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const { virtualModelName } = useRequiredPathParams([ROUTE_PARAMS.virtualModelName]);

  const location = useLocation();
  const match = matchPath(
    { path: `${ROUTES.workspace.virtualModelDetail}/:selectedTab`, end: false },
    location.pathname
  );
  const {
    params: { selectedTab },
  } = match ?? { params: { selectedTab: 'details' } };

  useBreadcrumbs({
    items: [
      { href: getWorkspaceVirtualModelsRoute(workspace), slotLabel: 'Virtual Models' },
      { slotLabel: virtualModelName },
    ],
  });

  return (
    <AccessibleTitle title={`Virtual model ${virtualModelName}`}>
      <Stack className="w-full min-h-full p-density-2xl" gap="density-xl">
        <PageHeader
          slotHeading={
            <span className="min-w-0 truncate" title={virtualModelName}>
              {virtualModelName}
            </span>
          }
        />

        <Tabs
          // Tabs used purely for navigation (renderLink); override KUI's default overflow:hidden.
          className="overflow-visible"
          value={selectedTab}
          items={[
            {
              value: 'details',
              children: 'Details',
              href: getVirtualModelDetailsRoute(workspace, virtualModelName),
            },
            {
              value: 'chat',
              children: 'Chat',
              href: getVirtualModelChatRoute(workspace, virtualModelName),
            },
          ]}
          renderLink={(item) => <Link to={item.href!}>{item.children}</Link>}
        />

        <SuspenseOutlet />
      </Stack>
    </AccessibleTitle>
  );
};
