// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AgentHardenerManifestsDataView } from '@agent-hardener/dataViews/AgentHardenerManifestsDataView';
import { useBreadcrumbs, useWorkspace } from '@agent-hardener/host';
import { getAgentHardenerRunListRoute, getNewAgentHardenerManifestRoute } from '@agent-hardener/paths';
import { AccessibleTitle } from '@nemo/common';
import { Button, PageHeader, Stack } from '@nvidia/foundations-react-core';
import { FC } from 'react';
import { Link, Outlet } from 'react-router';

export const AgentHardenerManifestListRoute: FC = () => {
  const workspace = useWorkspace();
  useBreadcrumbs({
    items: [
      { href: getAgentHardenerRunListRoute(workspace), slotLabel: 'Agent Hardener' },
      { slotLabel: 'Manifests' },
    ],
  });

  return (
    <AccessibleTitle title="Agent Hardener Manifests">
      <Stack className="h-full overflow-auto" gap="density-2xl" padding="density-2xl">
        <PageHeader
          className="p-0"
          slotHeading="Manifests"
          slotDescription="Reusable war-game targets scaffolded from a deployed agent. Create one, then run the war-game against it."
          slotActions={
            <Button asChild color="brand">
              <Link to={getNewAgentHardenerManifestRoute(workspace)}>New Manifest</Link>
            </Button>
          }
        />
        <AgentHardenerManifestsDataView />
      </Stack>
      <Outlet />
    </AccessibleTitle>
  );
};
