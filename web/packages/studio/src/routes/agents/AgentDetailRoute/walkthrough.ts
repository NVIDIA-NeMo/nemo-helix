// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type WalkthroughStep = 'deploy' | 'switch-to-chat' | 'wait' | 'chat';

export interface WalkthroughState {
  active: boolean;
  dismissed: boolean;
  createDeploymentOpen: boolean;
  selectedTab: string;
  hasDeployment: boolean;
  hasHealthyDeployment: boolean;
}

export const deriveWalkthroughStep = (s: WalkthroughState): WalkthroughStep | null => {
  if (!s.active || s.dismissed || s.createDeploymentOpen) return null;
  if (!s.hasDeployment) return s.selectedTab === 'summary' ? 'deploy' : null;
  if (s.selectedTab !== 'chat') return 'switch-to-chat';
  return s.hasHealthyDeployment ? 'chat' : 'wait';
};

export const WALKTHROUGH_COPY: Record<WalkthroughStep, { title: string; body: string }> = {
  deploy: {
    title: 'Deploy your agent',
    body: 'An agent needs a running deployment before you can use it. Click "Deploy this Agent" to start one.',
  },
  'switch-to-chat': {
    title: 'Open the chat',
    body: 'Your deployment is starting. Switch to the Chat tab to watch it come up and talk to your agent.',
  },
  wait: {
    title: 'Deployment starting',
    body: 'Hang tight — the deployment needs a moment to finish starting. The chat unlocks as soon as it is running.',
  },
  chat: {
    title: 'Your agent is live',
    body: 'The deployment is running. Send it a message to start chatting.',
  },
};
