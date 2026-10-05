// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AppBar, Button, Flex, Stack, Text, Tooltip } from '@nvidia/foundations-react-core';
import { Breadcrumbs } from '@studio/components/Breadcrumbs';
import { DocumentationLink } from '@studio/components/Layouts/GlobalNav/components/DocumentationLink';
import { ThemeSwitch } from '@studio/components/Layouts/GlobalNav/components/ThemeSwitch';
import { UserPopover } from '@studio/components/UserPopover';
import { TOUR_ENABLED } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { useWorkspaceFromPathIfExists } from '@studio/hooks/useWorkspaceFromPath';
import { AssistantTopBarChat } from '@studio/routes/agents/AssistantChatRoute/AssistantTopBarChat';
import { getWorkspaceDetailsDefaultRoute } from '@studio/routes/utils';
import { useSidebarState } from '@studio/util/hooks/useSidebarState';
import { PanelLeftClose, PanelLeftOpen } from 'lucide-react';
import { lazy, Suspense, type FC, type ReactNode } from 'react';
import { Link, matchPath, useLocation } from 'react-router';

const WelcomeTour = lazy(() =>
  import('@studio/components/WelcomeTour').then((m) => ({ default: m.WelcomeTour }))
);

interface Props {
  sideNav?: (collapsed: boolean) => ReactNode;
}

interface GlobalNavContentProps extends Props {
  isDashboardRoute: boolean;
  isAssistantChatRoute: boolean;
}

const GlobalNavContent: FC<GlobalNavContentProps> = ({
  sideNav,
  isDashboardRoute,
  isAssistantChatRoute,
}) => {
  const workspace = useWorkspaceFromPathIfExists();
  const { expanded, toggle } = useSidebarState(!isAssistantChatRoute);
  const shouldMountAssistantTopBarChat = !isDashboardRoute && !isAssistantChatRoute;
  const sidebarBackground = isAssistantChatRoute
    ? 'bg-surface-sunken dark:bg-surface-base'
    : 'bg-surface-navigation';

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
        {expanded ? <PanelLeftClose /> : <PanelLeftOpen />}
      </Button>
    </Tooltip>
  );

  return (
    <>
      <Flex
        className={`[grid-area:logobar] ${sidebarBackground} transition-colors border-r border-b border-base ${expanded ? 'pl-4 pr-2' : ''}`}
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
            {TOUR_ENABLED && (
              <Suspense>
                <WelcomeTour />
              </Suspense>
            )}
            {shouldMountAssistantTopBarChat && <AssistantTopBarChat />}
            <ThemeSwitch />
            <DocumentationLink />
            <UserPopover />
          </Flex>
        }
      />
      {sideNav && (
        <div
          className={`h-full max-h-[calc(100vh-var(--nv-app-bar-height))] overflow-y-auto [grid-area:sidebar] ${isAssistantChatRoute ? `${sidebarBackground} [&_.nv-vertical-nav-root]:bg-transparent!` : ''}`}
          data-tour="sidebar"
        >
          {sideNav(!expanded)}
        </div>
      )}
    </>
  );
};

export const GlobalNav: FC<Props> = ({ sideNav }) => {
  const location = useLocation();
  const isDashboardRoute =
    matchPath({ path: ROUTES.workspace.dashboard, end: true }, location.pathname) !== null;
  const isAssistantChatRoute =
    matchPath({ path: ROUTES.workspace.assistantChat, end: true }, location.pathname) !== null;

  return (
    <GlobalNavContent
      key={isAssistantChatRoute ? 'assistant' : 'default'}
      sideNav={sideNav}
      isDashboardRoute={isDashboardRoute}
      isAssistantChatRoute={isAssistantChatRoute}
    />
  );
};
