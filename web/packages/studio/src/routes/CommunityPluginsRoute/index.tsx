// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { Button, PageHeader, Stack } from '@nvidia/foundations-react-core';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { AddPluginModal } from '@studio/routes/CommunityPluginsRoute/AddPluginModal';
import { CatalogSuggestions } from '@studio/routes/CommunityPluginsRoute/CatalogSuggestions';
import { InstalledPluginsDataView } from '@studio/routes/CommunityPluginsRoute/InstalledPluginsDataView';
import { getCommunityPluginsRoute, getWorkspaceSettingsRoute } from '@studio/routes/utils';
import { type FC, useState } from 'react';

/**
 * Registry of community plugins: external APIs the platform proxies to, with an optional
 * Studio UI bundle. Registering one needs a HelixAdmin; the backend enforces that and
 * its error is shown as-is.
 */
export const CommunityPluginsRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  // `undefined` = closed; a string (possibly empty) = open with that URL prefilled.
  const [addUrl, setAddUrl] = useState<string>();

  useBreadcrumbs({
    items: [
      { href: getWorkspaceSettingsRoute(workspace), slotLabel: 'Settings' },
      { href: getCommunityPluginsRoute(workspace), slotLabel: 'Community Plugins' },
    ],
  });

  return (
    <AccessibleTitle title="Community Plugins">
      <Stack className="h-full" gap="density-2xl" padding="density-2xl">
        <PageHeader
          className="p-0"
          slotHeading="Community Plugins"
          slotDescription="Register external APIs, and optionally a Studio UI, from a plugin manifest URL. Registering a plugin requires the HelixAdmin role."
          slotActions={
            <Button color="brand" onClick={() => setAddUrl('')}>
              Add Plugin
            </Button>
          }
        />
        <CatalogSuggestions onAdd={setAddUrl} />
        <InstalledPluginsDataView
          onAdd={() => setAddUrl('')}
          attributes={{ Stack: { className: 'flex-1 min-h-0' } }}
        />
      </Stack>
      {addUrl !== undefined && (
        <AddPluginModal initialUrl={addUrl} onClose={() => setAddUrl(undefined)} />
      )}
    </AccessibleTitle>
  );
};
