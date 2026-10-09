// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, Text } from '@nvidia/foundations-react-core';
import { GlobalNav } from '@studio/components/Layouts/GlobalNav';
import { ASSISTANT_STUDIO_ENABLED } from '@studio/constants/environment';
import { useWorkspaceFromPathIfExists } from '@studio/hooks/useWorkspaceFromPath';
import { useAuthAutoLogin } from '@studio/providers/auth';
import { useAuthTokenStatus } from '@studio/providers/auth/useAuthTokenStatus';
import { useSelectedWorkspace } from '@studio/providers/workspace';
import { AssistantChatProvider } from '@studio/routes/agents/AssistantChatRoute/context/AssistantChatProvider';
import { WorkspaceGuard } from '@studio/routes/RootLayout/WorkspaceGuard';
import { LogIn } from 'lucide-react';
import { ReactNode } from 'react';
import { Outlet } from 'react-router';

const SignedOutView = ({ onSignIn }: { onSignIn: () => void }) => (
  <main className="flex min-h-screen items-center justify-center bg-surface-sunken px-6 text-primary">
    <Flex className="w-full max-w-[420px] flex-col items-center gap-4 text-center">
      <Text asChild kind="body/bold/2xl">
        <h1>Signed out</h1>
      </Text>
      <Text kind="body/regular/md" color="secondary">
        Your Studio session has ended.
      </Text>
      <Button color="brand" onClick={onSignIn}>
        <LogIn size={16} />
        Sign in
      </Button>
    </Flex>
  </main>
);

export const PageLayout = ({ sideNav }: { sideNav?: (collapsed: boolean) => ReactNode }) => {
  const { isAuthPending, isSignedOut, signIn } = useAuthAutoLogin();
  const { isTokenActive } = useAuthTokenStatus();
  const { selectedWorkspace, isWorkspaceUnauthorized, isWorkspaceLoading } = useSelectedWorkspace();
  const workspace = useWorkspaceFromPathIfExists();

  if (isAuthPending) {
    return null;
  }

  if (isSignedOut) {
    return <SignedOutView onSignIn={signIn} />;
  }

  const workspaceCheckActive = isTokenActive && !!selectedWorkspace;
  const hideSideNav = workspaceCheckActive && (isWorkspaceLoading || isWorkspaceUnauthorized);

  const gridAreas = hideSideNav
    ? "[grid-template-areas:'logobar_navbar''content_content']"
    : "[grid-template-areas:'logobar_navbar''sidebar_content']";

  // Rendered as direct grid children; the provider adds no DOM of its own, so
  // it can wrap both the nav (pop-out chat) and the outlet (full chat) while
  // keeping a single shared chat runtime alive across workspace navigation.
  const layout = (
    <>
      <GlobalNav sideNav={hideSideNav ? undefined : sideNav} />
      <div className="bg-surface-sunken transition-colors relative h-full max-h-[calc(100vh-var(--nv-app-bar-height))] overflow-y-auto [grid-area:content]">
        <WorkspaceGuard>
          <Outlet />
        </WorkspaceGuard>
      </div>
    </>
  );

  return (
    <div
      className={`min-h-screen relative grid size-full text-primary grid-cols-[auto_minmax(0,1fr)] grid-rows-[auto_1fr] ${gridAreas}`}
    >
      {ASSISTANT_STUDIO_ENABLED && workspace ? (
        <AssistantChatProvider key={workspace} workspace={workspace}>
          {layout}
        </AssistantChatProvider>
      ) : (
        layout
      )}
    </div>
  );
};
