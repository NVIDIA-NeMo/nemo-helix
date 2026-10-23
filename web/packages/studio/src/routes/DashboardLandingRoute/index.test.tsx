// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ROUTES } from '@studio/constants/routes';
import { getAssistantActiveSessionStorageKey } from '@studio/routes/agents/AssistantChatRoute/activeSessionStorage';
import { DashboardLandingRoute } from '@studio/routes/DashboardLandingRoute';
import { mockFeatureFlags } from '@studio/tests/util/mockFeatureFlags';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { createMemoryRouter, generatePath, RouterProvider, useLocation } from 'react-router';

vi.mock('@studio/routes/agents/AssistantChatRoute/api', async (importOriginal) => {
  const actual =
    await importOriginal<typeof import('@studio/routes/agents/AssistantChatRoute/api')>();

  return {
    ...actual,
    listAssistantHistorySessions: vi.fn(async () => []),
  };
});

const workspace = 'default';
const CHAT_ROUTE_TEST_ID = 'chat-route';

const ChatRouteProbe = () => {
  const location = useLocation();
  const state = location.state as { initialPrompt?: string } | null;

  return (
    <div data-testid={CHAT_ROUTE_TEST_ID}>
      {location.pathname}|{state?.initialPrompt}
    </div>
  );
};

const renderRoute = () => {
  const route = generatePath(ROUTES.workspace.dashboard, { workspace });
  const router = createMemoryRouter(
    [
      { path: ROUTES.workspace.dashboard, element: <DashboardLandingRoute /> },
      { path: ROUTES.workspace.assistantChat, element: <ChatRouteProbe /> },
    ],
    {
      initialEntries: [route],
    }
  );

  return render(
    <TestProviders>
      <RouterProvider router={router} />
    </TestProviders>
  );
};

describe('DashboardLandingRoute', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
    mockFeatureFlags({
      guardrailsEnabled: true,
      inferenceProviderEnabled: true,
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders the dashboard landing page', async () => {
    renderRoute();

    expect(await screen.findByText('What would you like to do?')).toBeInTheDocument();
    const composer = screen.getByRole('textbox', { name: 'Message NeMo Assistant' });
    expect(composer).toBeInTheDocument();
    expect(screen.getByTestId('dashboard-landing-composer')).toHaveClass('rounded-lg');
    expect(screen.getByTestId('dashboard-landing-composer')).not.toHaveClass('rounded-2xl');
    expect(composer).toHaveClass(
      '[&&]:focus:outline-none',
      '[&&]:focus-visible:outline-none',
      '[&&]:focus-visible:ring-0'
    );
    expect(composer).not.toHaveClass('[&&]:focus-visible:outline-accent');
    expect(screen.queryByLabelText('Skill action suggestions')).not.toBeInTheDocument();
    expect(screen.queryByTestId('skill-actions-empty')).not.toBeInTheDocument();
    expect(screen.queryByText('No skills found')).not.toBeInTheDocument();
  });

  it('only enables the send affordance once the composer has text', async () => {
    const user = userEvent.setup();
    renderRoute();

    const composer = await screen.findByRole('textbox', { name: 'Message NeMo Assistant' });
    const sendButton = screen.getByRole('button', { name: 'Send message' });

    expect(sendButton).toBeDisabled();

    await user.type(composer, 'Sketch a dashboard');

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Send message' })).toBeEnabled();
    });
  });

  it('navigates to the NeMo Assistant chat with the submitted prompt', async () => {
    const user = userEvent.setup();
    renderRoute();

    await user.type(
      await screen.findByRole('textbox', { name: 'Message NeMo Assistant' }),
      'Check repo'
    );
    await user.click(screen.getByRole('button', { name: 'Send message' }));

    expect(await screen.findByTestId(CHAT_ROUTE_TEST_ID)).toHaveTextContent(
      `${generatePath(ROUTES.workspace.assistantChat, { workspace })}|Check repo`
    );
  });

  it('clears the active NeMo Assistant session before starting from the landing composer', async () => {
    const user = userEvent.setup();
    localStorage.setItem(getAssistantActiveSessionStorageKey(workspace), 'session-existing');
    renderRoute();

    await user.type(
      await screen.findByRole('textbox', { name: 'Message NeMo Assistant' }),
      'Check repo'
    );
    await user.click(screen.getByRole('button', { name: 'Send message' }));

    expect(localStorage.getItem(getAssistantActiveSessionStorageKey(workspace))).toBeNull();
  });

  it('submits the landing composer when Enter is pressed', async () => {
    const user = userEvent.setup();
    renderRoute();

    await user.type(
      await screen.findByRole('textbox', { name: 'Message NeMo Assistant' }),
      'Check repo'
    );
    await user.keyboard('{Enter}');

    expect(await screen.findByTestId(CHAT_ROUTE_TEST_ID)).toHaveTextContent(
      `${generatePath(ROUTES.workspace.assistantChat, { workspace })}|Check repo`
    );
  });

  it('keeps Shift Enter as a new line in the landing composer', async () => {
    const user = userEvent.setup();
    renderRoute();

    const composer = await screen.findByRole('textbox', { name: 'Message NeMo Assistant' });

    await user.type(composer, 'Line one');
    await user.keyboard('{Shift>}{Enter}{/Shift}');
    await user.type(composer, 'Line two');

    expect(screen.queryByTestId(CHAT_ROUTE_TEST_ID)).not.toBeInTheDocument();
    expect(composer).toHaveValue('Line one\nLine two');
  });
});
