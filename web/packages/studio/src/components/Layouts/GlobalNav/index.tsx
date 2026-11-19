// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Collapse, Expand } from '@nemo/icons/icons';
import { AppBar, Button, Flex, Stack, Text, Tooltip } from '@nvidia/foundations-react-core';
import { Breadcrumbs } from '@studio/components/Breadcrumbs';
import { DocumentationLink } from '@studio/components/Layouts/GlobalNav/components/DocumentationLink';
import { ThemeSwitch } from '@studio/components/Layouts/GlobalNav/components/ThemeSwitch';
import { UserPopover } from '@studio/components/UserPopover';
import { FLAG_OVERRIDES_ENABLED } from '@studio/constants/environment';
import { useWorkspaceFromPathIfExists } from '@studio/hooks/useWorkspaceFromPath';
import { getWorkspaceDetailsDefaultRoute } from '@studio/routes/utils';
import { useSidebarState } from '@studio/util/hooks/useSidebarState';
import { lazy, Suspense, type FC, type ReactNode } from 'react';
import { Link } from 'react-router';

const WelcomeTour = lazy(() =>
  import('@studio/components/WelcomeTour').then((m) => ({ default: m.WelcomeTour }))
);

// Lazy so the flag editor never enters a bundle for a deployment that has
// overrides turned off.
const FeatureFlagPanel = lazy(() =>
  import('@studio/components/FeatureFlagPanel').then((m) => ({ default: m.FeatureFlagPanel }))
);

interface Props {
  sideNav?: (collapsed: boolean) => ReactNode;
}

export const GlobalNav: FC<Props> = ({ sideNav }) => {
  const workspace = useWorkspaceFromPathIfExists();
  const { expanded, toggle } = useSidebarState(true);

  const toggleLabel = expanded ? 'Collapse sidebar' : 'Expand sidebar';
  const ToggleSidebarButton = (
    <Tooltip slotContent={toggleLabel} side="right">
      <Button
        kind="tertiary"
        size="small"
        aria-label={toggleLabel}
        onClick={toggle}
        disabled={!sideNav}
      >
        {expanded ? <Collapse /> : <Expand />}
      </Button>
    </Tooltip>
  );

  return (
    <>
      <Flex
        className={`[grid-area:logobar] bg-surface-navigation transition-colors border-r border-b border-base ${expanded ? 'pl-4 pr-2' : ''}`}
        align="center"
        gap="density-md"
        justify={expanded ? 'between' : 'center'}
      >
        {expanded && (
          <Stack direction="row" align="center" gap="density-sm" asChild>
            <Link to={workspace ? getWorkspaceDetailsDefaultRoute(workspace) : '/'}>
              <Text kind="label/bold/md">NeMo Studio</Text>
            </Link>
          </Stack>
        )}
        {ToggleSidebarButton}
      </Flex>
      <AppBar
        className="[grid-area:navbar] transition-colors"
        slotStart={
          <Flex gap="density-md" align="center" data-tour="nav-workspace">
            <Breadcrumbs />
          </Flex>
        }
        slotEnd={
          <Flex gap="density-md" align="center">
            <Suspense>
              <WelcomeTour />
            </Suspense>
            {FLAG_OVERRIDES_ENABLED && (
              <Suspense>
                <FeatureFlagPanel />
              </Suspense>
            )}
            <ThemeSwitch />
            <DocumentationLink />
            <UserPopover />
          </Flex>
        }
      />
      {sideNav && (
        <div
          className="h-full max-h-[calc(100vh-var(--nv-app-bar-height))] overflow-y-auto [grid-area:sidebar]"
          data-tour="sidebar"
        >
          {sideNav(!expanded)}
        </div>
      )}
    </>
  );
};
