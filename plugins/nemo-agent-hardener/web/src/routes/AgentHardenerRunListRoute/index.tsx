// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AgentHardenerRunsDataView } from '@agent-hardener/dataViews/AgentHardenerRunsDataView';
import { useBreadcrumbs, useWorkspace } from '@agent-hardener/host';
import { getAgentHardenerManifestListRoute } from '@agent-hardener/paths';
import { AccessibleTitle } from '@nemo/common';
import { Button, PageHeader, Stack } from '@nvidia/foundations-react-core';
import { FC } from 'react';
import { Link, Outlet } from 'react-router';

export const AgentHardenerRunListRoute: FC = () => {
  const workspace = useWorkspace();
  useBreadcrumbs({ items: [{ slotLabel: 'Agent Hardener' }] });

  return (
    <AccessibleTitle title="Agent Hardener">
      <Stack className="h-full overflow-auto" gap="density-2xl" padding="density-2xl">
        <PageHeader
          className="p-0"
          slotHeading="Agent Hardener"
          slotDescription="Attack, defend, and validate war-game runs that harden your deployed agents."
          slotActions={
            <Button asChild color="brand">
              <Link to={getAgentHardenerManifestListRoute(workspace)}>Manifests</Link>
            </Button>
          }
        />
        <AgentHardenerRunsDataView />
      </Stack>
      <Outlet />
    </AccessibleTitle>
  );
};
